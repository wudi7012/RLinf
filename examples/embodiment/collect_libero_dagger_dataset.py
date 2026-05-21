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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.data.lerobot_writer import merge_distributed_datasets
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _collect_raw_lerobot_dataset(cfg, rollout_group, env_group) -> str:
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
    runner.init_workers()
    runner.run()
    env_group.finalize_data_collection().wait()

    collection_cfg = cfg.runner.collection
    shard_root = str(cfg.env.eval.data_collection.save_dir)
    if not bool(collection_cfg.get("merge_shards", True)):
        return shard_root

    output_dataset_dir = str(collection_cfg.output_dataset_dir)
    merge_distributed_datasets(
        base_dir=shard_root,
        output_dir=output_dataset_dir,
        pattern=str(
            collection_cfg.get("shard_pattern", "collected_data_stage*_rank*")
        ),
        robot_type=str(cfg.env.eval.data_collection.get("robot_type", "panda")),
        fps=int(cfg.env.eval.data_collection.get("fps", 10)),
    )
    return output_dataset_dir


def _materialize_preprocessed_sidecars(cfg, rollout_group, dataset_root: str) -> None:
    collection_cfg = cfg.runner.collection
    if not bool(collection_cfg.get("preprocess_after_collection", True)):
        return

    preprocess_results = rollout_group.materialize_pure_offline_dataset(
        dataset_root=dataset_root,
        output_root=str(collection_cfg.preprocessed_root),
        batch_size=int(collection_cfg.get("preprocess_batch_size", 32)),
        overwrite=bool(collection_cfg.get("overwrite_preprocessed", False)),
    ).wait()
    print(json.dumps({"preprocess": preprocess_results}, indent=2))


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_collect_base_vla_dataset",
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    dataset_root = _collect_raw_lerobot_dataset(cfg, rollout_group, env_group)
    _materialize_preprocessed_sidecars(cfg, rollout_group, dataset_root)


if __name__ == "__main__":
    main()
