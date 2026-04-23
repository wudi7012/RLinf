# Copyright 2025 The RLinf Authors.
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

import copy
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from rlinf.algorithms.offline_rl import is_pure_offline_dataset_enabled
from rlinf.config import SupportedModel
from rlinf.data.datasets.libero_offline_rl import (
    build_libero_chunk_transition_dataset_from_cfg,
    libero_offline_transition_collate_fn,
)
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    Trajectory,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import get_model_weights_id


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.model_weights_id = ""
        self.count_update = 0

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval
        self.actor_split_num = self.get_actor_split_num()
        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.collect_versions = self.cfg.algorithm.loss_type == "decoupled_actor_critic"
        self.version = 0
        self.finished_episodes = None
        self.use_pure_offline_dataset = is_pure_offline_dataset_enabled(cfg)
        self.offline_dataset = None
        self.offline_data_loader = None
        self.offline_data_iter = None
        self._offline_data_epoch = 0
        self._offline_data_iter_offset = 0

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.dst_ranks = {}
        self.src_ranks = {}
        use_train_env_comm = not self.use_pure_offline_dataset
        if use_train_env_comm:
            self.dst_ranks["train"] = self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
            self.src_ranks["train"] = self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.use_pure_offline_dataset:
            self.build_pure_offline_dataloader()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self,
        env_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        if env_output["rewards"] is None:
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        if bootstrap_type == "standard":
            last_step_truncations = env_output["truncations"].cpu().contiguous()[:, -1]
        else:
            last_step_truncations = dones[:, -1]

        # Handle auto_reset: add bootstrap value ONLY for truncated episodes (not terminated)
        if last_step_truncations.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                # bootstrap only on the truncated episode
                final_values[last_step_truncations] = _final_values[:, 0][
                    last_step_truncations
                ]
                # Add bootstrap value to the last step of truncated episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        return dones, rewards

    def _get_sampling_kwargs_for_mode(
        self, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        return (
            dict(self._train_sampling_params)
            if mode == "train"
            else dict(self._eval_sampling_params)
        )

    def _prepare_transition_obs_from_result(
        self,
        result: dict[str, Any],
        raw_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> dict[str, Any]:
        if hasattr(self.hf_model, "build_transition_obs"):
            return self.hf_model.build_transition_obs(
                forward_inputs=result["forward_inputs"],
            )
        raw_obs = dict(raw_obs)
        raw_obs.pop("task_descriptions", None)
        return raw_obs

    def _prepare_transition_obs_from_env(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> dict[str, Any]:
        if hasattr(self.hf_model, "build_transition_obs"):
            return self.hf_model.build_transition_obs(
                env_obs=env_obs,
                **self._get_sampling_kwargs_for_mode(mode),
            )
        env_obs = dict(env_obs)
        env_obs.pop("task_descriptions", None)
        return env_obs

    def _pure_offline_dataset_cfg(self):
        return self.cfg.algorithm.offline_rl.dataset

    def build_pure_offline_dataloader(self) -> None:
        dataset_cfg = self._pure_offline_dataset_cfg()
        per_rank_batch_size = self.cfg.actor.global_batch_size // self._world_size
        if per_rank_batch_size < 1:
            raise ValueError(
                f"Per-rank offline batch size must be >= 1, got {per_rank_batch_size}."
            )

        self.offline_dataset = build_libero_chunk_transition_dataset_from_cfg(self.cfg)
        if len(self.offline_dataset) < per_rank_batch_size:
            raise ValueError(
                f"Offline dataset size ({len(self.offline_dataset)}) must be >= "
                f"per-rank batch size ({per_rank_batch_size})."
            )

        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(
                self.offline_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=bool(dataset_cfg.get("shuffle", True)),
                seed=int(dataset_cfg.get("seed", self.cfg.actor.get("seed", 1234))),
                drop_last=True,
            )

        num_workers = int(dataset_cfg.get("num_workers", 0))
        pin_memory = bool(dataset_cfg.get("pin_memory", True))
        persistent_workers = num_workers > 0 and bool(
            dataset_cfg.get("persistent_workers", True)
        )

        self.offline_data_loader = DataLoader(
            self.offline_dataset,
            batch_size=per_rank_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=libero_offline_transition_collate_fn,
        )
        self._offline_data_epoch = 0
        self._offline_data_iter_offset = 0
        if sampler is not None:
            sampler.set_epoch(self._offline_data_epoch)
        self.offline_data_iter = iter(self.offline_data_loader)
        self.log_info(
            "Pure offline preprocess dataloader setup: "
            f"rank={self._rank} dataset_size={len(self.offline_dataset)} "
            f"per_rank_batch_size={per_rank_batch_size} num_workers={num_workers}"
        )

    def _build_pure_offline_transition_kwargs(self) -> dict[str, Any]:
        dataset_cfg = self._pure_offline_dataset_cfg()
        sampling_cfg = self.cfg.algorithm.sampling_params

        do_sample = bool(
            dataset_cfg.get("base_do_sample", sampling_cfg.get("do_sample", True))
        )
        temperature = float(
            dataset_cfg.get(
                "base_temperature", sampling_cfg.get("temperature_train", 1.0)
            )
        )
        if temperature <= 0:
            do_sample = False
            temperature = 1.0

        kwargs: dict[str, Any] = {"do_sample": do_sample}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs.update(
                {
                    "temperature": temperature,
                    "top_k": int(
                        dataset_cfg.get("base_top_k", sampling_cfg.get("top_k", -1))
                    ),
                    "top_p": float(
                        dataset_cfg.get("base_top_p", sampling_cfg.get("top_p", 1.0))
                    ),
                    "repetition_penalty": float(
                        dataset_cfg.get(
                            "base_repetition_penalty",
                            sampling_cfg.get("repetition_penalty", 1.0),
                        )
                    ),
                }
            )
        return kwargs

    def _next_pure_offline_raw_batch(self) -> dict[str, Any]:
        if self.offline_data_iter is None:
            raise RuntimeError("Pure offline preprocess DataLoader is not initialized.")
        try:
            batch = next(self.offline_data_iter)
            self._offline_data_iter_offset += 1
        except StopIteration:
            self._offline_data_epoch += 1
            sampler = getattr(self.offline_data_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._offline_data_epoch)
            self.offline_data_iter = iter(self.offline_data_loader)
            batch = next(self.offline_data_iter)
            self._offline_data_iter_offset = 1
        return batch

    def _transition_obs_to_cpu(
        self,
        transition_obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in transition_obs.items()
        }

    def _prepare_pure_offline_batch(
        self,
        raw_batch: dict[str, Any],
    ) -> dict[str, Any]:
        transition_kwargs = self._build_pure_offline_transition_kwargs()
        with torch.no_grad():
            curr_transition_obs = self.hf_model.build_transition_obs(
                env_obs=raw_batch["curr_env_obs"],
                **transition_kwargs,
            )
            next_transition_obs = self.hf_model.build_transition_obs(
                env_obs=raw_batch["next_env_obs"],
                **transition_kwargs,
            )

        actions = raw_batch["actions"].to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if hasattr(self.hf_model, "_delta_to_policy_action"):
            base_flat = curr_transition_obs["base_actions"].flatten(start_dim=1).to(
                dtype=actions.dtype
            )
            delta_actions = actions.view(actions.shape[0], -1) - base_flat
            actions = self.hf_model._delta_to_policy_action(
                delta_actions,
                curr_transition_obs["base_actions"],
            ).to(dtype=torch.float32)

        batch = {
            "curr_obs": self._transition_obs_to_cpu(curr_transition_obs),
            "next_obs": self._transition_obs_to_cpu(next_transition_obs),
            "actions": actions.detach().cpu().contiguous(),
            "rewards": raw_batch["rewards"].to(dtype=torch.float32).cpu().contiguous(),
            "terminations": raw_batch["terminations"].cpu().contiguous(),
            "truncations": raw_batch["truncations"].cpu().contiguous(),
            "dones": raw_batch["dones"].cpu().contiguous(),
        }
        if "returns_to_go" in raw_batch:
            batch["returns_to_go"] = (
                raw_batch["returns_to_go"]
                .to(dtype=torch.float32)
                .cpu()
                .contiguous()
            )
        return batch

    def prepare_pure_offline_train_batches(self, num_batches: int) -> list[dict[str, Any]]:
        if not self.use_pure_offline_dataset:
            raise RuntimeError(
                "prepare_pure_offline_train_batches is only available when "
                "algorithm.offline_rl.dataset.enable=true."
            )
        if self.enable_offload:
            self.reload_model()

        self.hf_model.eval()
        prepared_batches: list[dict[str, Any]] = []
        for batch_idx in range(int(num_batches)):
            prepared_batches.append(
                self._prepare_pure_offline_batch(self._next_pure_offline_raw_batch())
            )
            self.log_info(
                "Prepared pure offline training batch: "
                f"rank={self._rank} batch={batch_idx + 1}/{num_batches} "
                f"data_epoch={self._offline_data_epoch} "
                f"iter_offset={self._offline_data_iter_offset}"
            )

        if self.enable_offload:
            self.offload_model()
        return prepared_batches

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        self.hf_model.load_state_dict(param_state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1
        del param_state_dict
        gc.collect()
        self.torch_platform.empty_cache()

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        last_transition_obs = [None for i in range(self.num_pipeline_stages)]
        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                if env_output["intervene_actions"] is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output["intervene_actions"],
                        env_output["intervene_flags"],
                    )

                dones, rewards = self.get_dones_and_rewards(env_output)

                actions, result = self.predict(env_output["obs"])
                current_transition_obs = None
                if self.collect_transitions:
                    current_transition_obs = self._prepare_transition_obs_from_result(
                        result,
                        env_output["obs"],
                        mode="train",
                    )

                next_obs_env = (
                    env_output["final_obs"]
                    if dones.any() and self.cfg.env.train.auto_reset
                    else env_output["obs"]
                )
                next_transition_obs = None
                if self.collect_transitions and last_transition_obs[stage_id] is not None:
                    next_transition_obs = self._prepare_transition_obs_from_env(
                        next_obs_env,
                        mode="train",
                    )

                env_output["obs"].pop("task_descriptions", None)
                if env_output["final_obs"] is not None:
                    env_output["final_obs"].pop("task_descriptions", None)
                chunk_step_result = ChunkStepResult(
                    actions=result["forward_inputs"].get("action", None),
                    dones=dones,
                    rewards=rewards,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    )
                    if self.collect_versions
                    else None,
                )

                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if self.collect_transitions and last_transition_obs[stage_id] is not None:
                    curr_obs = last_transition_obs[stage_id]
                    next_obs = next_transition_obs
                    self.rollout_results[stage_id].append_transitions(
                        curr_obs, next_obs
                    )

                last_transition_obs[stage_id] = current_transition_obs

                self.send_chunk_actions(output_channel, actions)

        for stage_id in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)

            if env_output["intervene_actions"] is not None:
                self.rollout_results[stage_id].update_last_actions(
                    env_output["intervene_actions"], env_output["intervene_flags"]
                )

            dones, rewards = self.get_dones_and_rewards(env_output)

            _, result = self.predict(env_output["obs"])

            env_output["obs"].pop("task_descriptions", None)
            if env_output["final_obs"] is not None:
                env_output["final_obs"].pop("task_descriptions", None)

            chunk_step_result = ChunkStepResult(
                dones=dones,
                rewards=rewards,
                truncations=env_output["truncations"],
                terminations=env_output["terminations"],
                prev_logprobs=None,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                forward_inputs=None,
            )

            self.rollout_results[stage_id].append_step_result(chunk_step_result)
            if self.collect_transitions and last_transition_obs[stage_id] is not None:
                curr_obs = last_transition_obs[stage_id]
                next_obs_env = (
                    env_output["final_obs"]
                    if dones.any() and self.cfg.env.train.auto_reset
                    else env_output["obs"]
                )
                next_obs = self._prepare_transition_obs_from_env(
                    next_obs_env,
                    mode="train",
                )
                self.rollout_results[stage_id].append_transitions(curr_obs, next_obs)

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        # rollout_results[stage_id]
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
                model_weights_id=self.model_weights_id,
            )
            for _ in range(self.num_pipeline_stages)
        ]

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        for stage_id in range(self.num_pipeline_stages):
            await self.send_rollout_trajectories(
                self.rollout_results[stage_id], actor_channel
            )

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, torch.Tensor]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        env_outputs = []
        for src_rank, expected_size in src_ranks_and_sizes:
            env_output = await input_channel.get(
                key=CommMapper.build_channel_key(src_rank, self._rank, extra=mode),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(env_output)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            env_outputs.append(env_output)
        env_output = EnvOutput.merge_env_outputs(env_outputs)
        return env_output

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        if isinstance(actions, dict):
            split_payloads = [dict() for _ in sizes]
            for key, value in actions.items():
                split_values = self._split_actions(value, sizes)
                for idx, split_value in enumerate(split_values):
                    split_payloads[idx][key] = split_value
            return split_payloads
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(env_output: dict[str, Any]) -> int:
        dones = env_output.get("dones")
        if isinstance(dones, torch.Tensor):
            return dones.shape[0]

        obs = env_output["obs"]
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env output.")

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            chunk_action_i = self._move_actions_to_cpu(chunk_action_i)
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(self._rank, dst_rank, extra=mode),
                async_op=True,
            )

    def _move_actions_to_cpu(self, actions):
        if isinstance(actions, torch.Tensor):
            return actions.detach().cpu()
        if isinstance(actions, dict):
            return {
                key: self._move_actions_to_cpu(value) for key, value in actions.items()
            }
        return actions

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step: int):
        self.version = global_step
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
