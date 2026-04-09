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

"""Frozen VLA + residual chunk adapter policy."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.vla_adapter.base_feature_extractor import (
    extract_features_from_env_obs,
    extract_features_from_forward_inputs,
)


def _get_cfg_value(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ResidualGaussianActor(nn.Module):
    """Small Gaussian residual policy on top of frozen VLA features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_layernorm: bool = True,
        std_mode: str = "global_learnable",
        init_log_std: float = -2.0,
    ) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dim, output_dim)
        self.std_mode = std_mode
        if std_mode == "global_learnable":
            self.log_std = nn.Parameter(torch.full((output_dim,), float(init_log_std)))
            self.log_std_head = None
        elif std_mode == "state_dependent":
            self.log_std = None
            self.log_std_head = nn.Linear(dim, output_dim)
        else:
            raise ValueError(
                f"Unsupported adapter std_mode '{std_mode}'. "
                "Supported: ['global_learnable', 'state_dependent']"
            )

    def forward(self, adapter_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(adapter_inputs)
        mean = self.mean_head(hidden)
        if self.log_std_head is None:
            log_std = self.log_std.unsqueeze(0).expand_as(mean)
        else:
            log_std = self.log_std_head(hidden)
        return mean, torch.clamp(log_std, min=-20.0, max=2.0)


class ResidualChunkAdapterPolicy(nn.Module, BasePolicy):
    """Frozen VLA with a trainable residual Gaussian adapter."""

    def __init__(self, base_vla: nn.Module, cfg) -> None:
        super().__init__()
        self.base_vla = base_vla
        self.cfg = cfg
        self.adapter_cfg = _get_cfg_value(cfg, "adapter", {})
        self.vla_cfg = _get_cfg_value(cfg, "vla", {})

        self.action_dim = base_vla.action_dim
        self.num_action_chunks = base_vla.num_action_chunks
        self.proprio_dim = _get_cfg_value(cfg, "proprio_dim", self.action_dim)
        self.residual_bound = float(
            _get_cfg_value(self.adapter_cfg, "residual_bound", 0.5)
        )

        self._freeze_base_vla()

        feature_dim = self._infer_feature_dim()
        adapter_input_dim = feature_dim + self.proprio_dim + (
            self.num_action_chunks * self.action_dim
        )
        self.adapter_actor = ResidualGaussianActor(
            input_dim=adapter_input_dim,
            output_dim=self.num_action_chunks * self.action_dim,
            hidden_dim=int(_get_cfg_value(self.adapter_cfg, "hidden_dim", 512)),
            num_layers=int(_get_cfg_value(self.adapter_cfg, "num_layers", 3)),
            use_layernorm=bool(_get_cfg_value(self.adapter_cfg, "use_layernorm", True)),
            std_mode=str(
                _get_cfg_value(self.adapter_cfg, "std_mode", "global_learnable")
            ),
            init_log_std=float(
                _get_cfg_value(self.adapter_cfg, "init_log_std", -2.0)
            ),
        )

    def _freeze_base_vla(self) -> None:
        freeze_backbone = bool(_get_cfg_value(self.vla_cfg, "freeze_backbone", True))
        freeze_action_head = bool(
            _get_cfg_value(self.vla_cfg, "freeze_action_head", True)
        )
        if freeze_backbone and freeze_action_head:
            self.base_vla.requires_grad_(False)
        else:
            for name, param in self.base_vla.named_parameters():
                if freeze_action_head and (
                    "lm_head" in name or "action_head" in name or "value_head" in name
                ):
                    param.requires_grad = False
                elif freeze_backbone:
                    param.requires_grad = False
        self.base_vla.eval()

    def _infer_feature_dim(self) -> int:
        if hasattr(self.base_vla, "hidden_size"):
            return int(self.base_vla.hidden_size)
        if hasattr(self.base_vla.config, "text_config"):
            return int(self.base_vla.config.text_config.hidden_size)
        if hasattr(self.base_vla.config, "hidden_size"):
            return int(self.base_vla.config.hidden_size)
        raise ValueError(
            f"Unable to infer feature dimension from backbone {type(self.base_vla).__name__}"
        )

    def _build_adapter_inputs(
        self,
        features: torch.Tensor,
        states: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        base_actions_flat = base_actions.flatten(start_dim=1)
        return torch.cat(
            [features, states.to(dtype=features.dtype), base_actions_flat.to(features.dtype)],
            dim=-1,
        )

    def _sample_delta(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
        delta_pre_tanh: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if delta_pre_tanh is None:
            delta_pre_tanh = mean if deterministic else normal.rsample()
        squashed = torch.tanh(delta_pre_tanh)
        delta_actions = self.residual_bound * squashed
        logprobs = normal.log_prob(delta_pre_tanh)
        logprobs = logprobs - torch.log(
            self.residual_bound * (1.0 - squashed.pow(2)) + 1e-7
        )
        entropy = normal.entropy()
        return delta_pre_tanh, delta_actions, logprobs.to(dtype=torch.float32), entropy.to(
            dtype=torch.float32
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del calculate_values
        deterministic = not kwargs.get("do_sample", True)
        extracted = extract_features_from_env_obs(self.base_vla, env_obs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        mean, log_std = self.adapter_actor(adapter_inputs)
        delta_pre_tanh, delta_actions, logprobs, _ = self._sample_delta(
            mean,
            log_std,
            deterministic=deterministic,
        )

        final_actions = extracted.base_actions + delta_actions.view_as(extracted.base_actions)
        final_actions = final_actions.to(dtype=torch.float32)
        delta_actions = delta_actions.view_as(extracted.base_actions).to(dtype=torch.float32)
        base_actions = extracted.base_actions.to(dtype=torch.float32)

        forward_inputs = {
            **extracted.forward_inputs,
            "states": extracted.states.to(dtype=torch.float32),
            "delta_pre_tanh": delta_pre_tanh.view_as(base_actions).to(dtype=torch.float32),
            "action": final_actions,
        }
        result = {
            "prev_logprobs": logprobs.to(dtype=torch.float32),
            "prev_values": torch.zeros(
                (final_actions.shape[0], 1),
                device=final_actions.device,
                dtype=torch.float32,
            ),
            "forward_inputs": forward_inputs,
            "adapter_debug": {
                "base_actions": base_actions,
                "delta_actions": delta_actions,
            },
        }

        action_payload = {
            "actions": final_actions,
            "base_actions": base_actions,
            "delta_actions": delta_actions,
        }
        if not calculate_logprobs:
            result["prev_logprobs"] = None
        return action_payload, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ):
        del kwargs
        extracted = extract_features_from_forward_inputs(self.base_vla, forward_inputs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        mean, log_std = self.adapter_actor(adapter_inputs)
        delta_pre_tanh = forward_inputs["delta_pre_tanh"].to(
            device=mean.device,
            dtype=mean.dtype,
        ).flatten(start_dim=1)
        _, _, logprobs, entropy = self._sample_delta(
            mean,
            log_std,
            deterministic=False,
            delta_pre_tanh=delta_pre_tanh,
        )

        result = {"logprobs": None, "entropy": None, "values": None}
        if compute_logprobs:
            result["logprobs"] = logprobs
        if compute_entropy:
            result["entropy"] = entropy
        if compute_values:
            result["values"] = torch.zeros(
                (mean.shape[0], 1),
                device=mean.device,
                dtype=torch.float32,
            )
        return result
