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
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_cql_openvlaoft_adapter_offline",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    else:
        raise NotImplementedError(
            f"Unsupported offline algorithm.loss_type={cfg.algorithm.loss_type!r}. "
            "Current train_offline_rl entry only supports 'embodied_sac'."
        )
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )

    enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval
    env_group = None
    rollout_group = None
    if enable_eval:
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=env_placement,
        )

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement,
        )

    runner = OfflineRunner(
        cfg=cfg,
        actor=actor_group,
        env=env_group,
        rollout=rollout_group,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
