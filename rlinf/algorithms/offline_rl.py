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

import torch


SUPPORTED_OFFLINE_RL_ALGOS = {"sac", "cql", "calql", "iql"}


def get_offline_rl_algo_name(cfg) -> str:
    offline_cfg = cfg.algorithm.get("offline_rl", None)
    if offline_cfg is None:
        return "sac"
    return offline_cfg.get("name", "sac").lower()


def is_pure_offline_dataset_enabled(cfg) -> bool:
    offline_cfg = cfg.algorithm.get("offline_rl", None)
    if offline_cfg is None:
        return False
    dataset_cfg = offline_cfg.get("dataset", None)
    if dataset_cfg is None:
        return False
    return bool(dataset_cfg.get("enable", False))


def iql_expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff >= 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


def iql_advantage_weights(
    advantages: torch.Tensor,
    weight_temp: float,
    max_weight: float,
) -> torch.Tensor:
    return torch.exp(weight_temp * advantages).clamp(max=max_weight)
