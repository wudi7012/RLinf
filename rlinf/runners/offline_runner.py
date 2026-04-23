# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import queue
import threading
import time
from collections import defaultdict
from typing import Any

from omegaconf.dictconfig import DictConfig

from rlinf.algorithms.offline_rl import is_pure_offline_dataset_enabled
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress


class OfflineRunner:
    """Offline RL runner: train on local dataset, keep env/rollout only for eval."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: Any,
        env: Any | None,
        rollout: Any | None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.env = env
        self.rollout = rollout

        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.use_pure_offline_dataset = is_pure_offline_dataset_enabled(cfg)

        self.global_step = 0
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        while not self.stop_logging:
            try:
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Logging error: %s", e)
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        enable_eval = (
            self.cfg.runner.val_check_interval > 0 or self.cfg.runner.only_eval
        )
        if enable_eval:
            if self.env is None or self.rollout is None:
                raise RuntimeError(
                    "Evaluation is enabled but env/rollout worker groups are missing."
                )
            self.env.init_worker().wait()
        if self.rollout is not None:
            self.rollout.init_worker().wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info("Resuming training from checkpoint directory %s.", resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        if self.rollout is None:
            raise RuntimeError("Rollout worker group is not initialized.")
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def _prepare_pure_offline_batches_for_actor(self):
        if not self.use_pure_offline_dataset:
            return
        if self.rollout is None:
            raise RuntimeError(
                "Pure offline training requires a rollout worker group to preprocess "
                "adapter context batches."
            )
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        rollout_results = self.rollout.prepare_pure_offline_train_batches(
            num_batches=update_epoch
        ).wait()
        actor_world_size = len(self.actor.worker_info_list)
        rollout_world_size = len(rollout_results)
        if rollout_world_size != actor_world_size:
            raise RuntimeError(
                "Pure offline preprocess currently requires rollout and actor world "
                f"sizes to match, got rollout={rollout_world_size}, actor={actor_world_size}."
            )
        set_handles: list[Handle] = []
        for rank, rank_batches in enumerate(rollout_results):
            set_handles.append(
                self.actor.execute_on(rank).set_pure_offline_batches(rank_batches)
            )
        for handle in set_handles:
            handle.wait()

    def evaluate(self):
        if self.env is None or self.rollout is None:
            raise RuntimeError("Env/Rollout worker groups are not initialized.")
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        return compute_evaluate_metrics(eval_metrics_list)

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        log_interval = int(self.cfg.runner.log_interval)
        if log_interval < 1:
            raise ValueError(f"runner.log_interval must be >= 1, got {log_interval}.")
        worker_step_synced = False

        while self.global_step < self.max_steps:
            current_step = self.global_step
            next_step = current_step + 1
            if not worker_step_synced:
                self.actor.set_global_step(current_step)
            if self.use_pure_offline_dataset:
                self._prepare_pure_offline_batches_for_actor()

            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_metrics_per_rank = actor_training_handle.wait()
                if not isinstance(actor_metrics_per_rank, list):
                    actor_metrics_per_rank = [actor_metrics_per_rank]
                actor_time_metrics = actor_training_handle.consume_durations()

            ranked_actor_training_results = [
                {"rank": rank, "train": rank_metrics}
                for rank, rank_metrics in enumerate(actor_metrics_per_rank)
                if rank_metrics is not None
            ]
            metrics, actor_training_metrics_per_rank = (
                self._process_ranked_numeric_results(
                    ranked_actor_training_results,
                    metric_field="train",
                )
            )
            metrics.pop("__global_step", None)
            global_steps = []
            for rank_metrics in actor_training_metrics_per_rank:
                if "__global_step" in rank_metrics:
                    global_steps.append(int(rank_metrics.pop("__global_step")))

            if global_steps:
                self.global_step = max(global_steps)
                worker_step_synced = True
            else:
                self.global_step = next_step
                worker_step_synced = False

            run_val, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )

            eval_metrics: dict[str, Any] = {}
            if run_val and self.env is not None and self.rollout is not None:
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = {f"eval/{k}": v for k, v in self.evaluate().items()}
                    self.metric_logger.log(data=eval_metrics, step=self.global_step)

            if save_model:
                self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )
            training_metrics = {f"train/{k}": v for k, v in metrics.items()}

            if (
                self.global_step == start_step + 1
                or self.global_step % log_interval == 0
            ):
                self.metric_logger.log(time_metrics, self.global_step)
                self.metric_logger.log(training_metrics, self.global_step)
                self._log_ranked_metrics(
                    metrics_list=actor_training_metrics_per_rank,
                    step=self.global_step,
                    prefix="train",
                    worker_group_name=self.actor.worker_group_name,
                )
                logging_metrics = dict(time_metrics)
                logging_metrics.update(training_metrics)
                logging_metrics.update(eval_metrics)
                self.print_metrics_table_async(
                    self.global_step - 1,
                    self.max_steps,
                    start_time,
                    logging_metrics,
                    start_step,
                )

        self.metric_logger.finish()

        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info("Saving checkpoint at step %s.", self.global_step)
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
