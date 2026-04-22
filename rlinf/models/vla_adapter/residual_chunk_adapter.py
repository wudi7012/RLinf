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
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.vla_adapter.base_feature_extractor import (
    attach_base_actions,
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
        self.flat_action_dim = self.num_action_chunks * self.action_dim
        self.proprio_dim = self._infer_proprio_dim(cfg, base_vla)
        self.residual_bound = float(
            _get_cfg_value(self.adapter_cfg, "residual_bound", 0.5)
        )
        self.offline_rl_action_space = str(
            _get_cfg_value(self.adapter_cfg, "offline_rl_action_space", "residual")
        )
        if self.offline_rl_action_space not in {"residual", "final"}:
            raise ValueError(
                "ResidualChunkAdapterPolicy only supports "
                "`adapter.offline_rl_action_space` in ['residual', 'final']."
            )

        self._freeze_base_vla()

        feature_dim = self._infer_feature_dim()
        adapter_input_dim = feature_dim + self.proprio_dim + (
            self.num_action_chunks * self.action_dim
        )
        self.adapter_input_dim = adapter_input_dim
        self.adapter_actor = ResidualGaussianActor(
            input_dim=adapter_input_dim,
            output_dim=self.flat_action_dim,
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

        q_hidden_dims = list(
            _get_cfg_value(self.adapter_cfg, "q_head_hidden_dims", [512, 256, 256])
        )
        if bool(_get_cfg_value(self.adapter_cfg, "add_q_head", False)):
            q_head_type = str(_get_cfg_value(self.adapter_cfg, "q_head_type", "default"))
            if q_head_type != "default":
                raise ValueError(
                    "ResidualChunkAdapterPolicy currently supports only "
                    "`adapter.q_head_type=default`."
                )
            self.q_head = MultiQHead(
                hidden_size=adapter_input_dim,
                action_feature_dim=self.flat_action_dim,
                hidden_dims=q_hidden_dims,
                num_q_heads=int(_get_cfg_value(self.adapter_cfg, "num_q_heads", 2)),
            )
        value_hidden_dims = tuple(
            _get_cfg_value(self.adapter_cfg, "value_head_hidden_dims", [512, 256])
        )
        if bool(_get_cfg_value(self.adapter_cfg, "add_value_head", False)):
            self.value_head = ValueHead(
                input_dim=adapter_input_dim,
                hidden_sizes=value_hidden_dims,
                output_dim=1,
                activation=str(
                    _get_cfg_value(self.adapter_cfg, "value_head_activation", "gelu")
                ),
            )
        base_param = next(self.base_vla.parameters())
        self.to(device=base_param.device, dtype=base_param.dtype)

    def __getattr__(self, name: str):
        """Delegate missing attributes to the wrapped base VLA."""
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            modules = object.__getattribute__(self, "_modules")
            base_vla = modules.get("base_vla")
            if base_vla is not None and hasattr(base_vla, name):
                return getattr(base_vla, name)
            raise exc

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

    def train(self, mode: bool = True):
        """Keep the frozen VLA in eval mode while training the adapter."""
        super().train(mode)
        self.base_vla.eval()
        return self

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_vla, "gradient_checkpointing_enable"):
            return self.base_vla.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self, *args, **kwargs):
        if hasattr(self.base_vla, "gradient_checkpointing_disable"):
            return self.base_vla.gradient_checkpointing_disable(*args, **kwargs)
        return None

    def enable_torch_compile(
        self,
        mode: str = "max-autotune-no-cudagraphs",
    ):
        if hasattr(self.base_vla, "enable_torch_compile"):
            return self.base_vla.enable_torch_compile(mode=mode)
        raise NotImplementedError(
            "torch compile is not supported for the wrapped base VLA"
        )

    def capture_cuda_graph(self, train_batch_size: int, eval_batch_size: int):
        if hasattr(self.base_vla, "capture_cuda_graph"):
            return self.base_vla.capture_cuda_graph(
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
            )
        raise NotImplementedError(
            "cuda graph is not supported for the wrapped base VLA"
        )

    def release_cuda_graph(self):
        if hasattr(self.base_vla, "release_cuda_graph"):
            return self.base_vla.release_cuda_graph()
        return None

    def is_cuda_graph_enabled(self) -> bool:
        if hasattr(self.base_vla, "is_cuda_graph_enabled"):
            return bool(self.base_vla.is_cuda_graph_enabled())
        return False

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

    def _infer_proprio_dim(self, cfg, base_vla: nn.Module) -> int:
        configured_dim = _get_cfg_value(cfg, "proprio_dim", None)
        if configured_dim is not None:
            return int(configured_dim)
        if hasattr(base_vla, "proprio_dim"):
            return int(base_vla.proprio_dim)
        if hasattr(base_vla, "config") and hasattr(base_vla.config, "proprio_dim"):
            return int(base_vla.config.proprio_dim)
        raise ValueError(
            "ResidualChunkAdapterPolicy requires an explicit proprio/state dimension. "
            "Please set `actor.model.proprio_dim` in the config."
        )

    def _build_adapter_inputs(
        self,
        features: torch.Tensor,
        states: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        base_actions_flat = base_actions.flatten(start_dim=1)
        adapter_inputs = torch.cat(
            [features, states.to(dtype=features.dtype), base_actions_flat.to(features.dtype)],
            dim=-1,
        )
        adapter_param = next(self.adapter_actor.parameters())
        return adapter_inputs.to(
            device=adapter_param.device,
            dtype=adapter_param.dtype,
        )

    def _get_transition_obs_from_forward_inputs(
        self,
        forward_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in forward_inputs.items()
            if key not in {"delta_pre_tanh", "action"}
        }

    @torch.no_grad()
    def build_transition_obs(
        self,
        env_obs: Optional[dict[str, Any]] = None,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if forward_inputs is not None:
            return self._get_transition_obs_from_forward_inputs(forward_inputs)

        if env_obs is None:
            raise ValueError("build_transition_obs requires either env_obs or forward_inputs.")

        base_chunk_actions, base_result = self.base_vla.predict_action_batch(
            env_obs=env_obs,
            calculate_logprobs=False,
            calculate_values=False,
            **kwargs,
        )
        states = env_obs["states"]
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)
        device = next(self.base_vla.parameters()).device
        states = states.to(device=device, dtype=torch.float32)
        base_actions = torch.as_tensor(
            base_chunk_actions,
            device=device,
            dtype=torch.float32,
        )
        transition_obs = {
            **base_result["forward_inputs"],
            "states": states,
            "base_actions": base_actions,
        }
        return transition_obs

    def _extract_adapter_context(
        self,
        obs: dict[str, torch.Tensor],
        **kwargs,
    ):
        if "input_ids" in obs:
            return extract_features_from_forward_inputs(self.base_vla, obs)
        base_chunk_actions, _ = self.base_vla.predict_action_batch(
            env_obs=obs,
            calculate_logprobs=False,
            calculate_values=False,
            **kwargs,
        )
        return attach_base_actions(
            extract_features_from_env_obs(self.base_vla, obs),
            base_chunk_actions,
        )

    def _delta_to_policy_action(
        self,
        delta_actions: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        delta_actions = delta_actions.view(base_actions.shape[0], -1)
        if self.offline_rl_action_space == "residual":
            return delta_actions
        base_actions = base_actions.flatten(start_dim=1).to(dtype=delta_actions.dtype)
        return base_actions + delta_actions

    def _policy_action_to_delta(
        self,
        policy_actions: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        policy_actions = policy_actions.view(policy_actions.shape[0], -1)
        if self.offline_rl_action_space == "residual":
            return policy_actions
        base_flat = base_actions.flatten(start_dim=1).to(dtype=policy_actions.dtype)
        return policy_actions - base_flat

    def _delta_from_pre_tanh(self, delta_pre_tanh: torch.Tensor) -> torch.Tensor:
        return self.residual_bound * torch.tanh(delta_pre_tanh)

    def _invert_policy_action(
        self,
        policy_actions: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_bound <= 0:
            raise ValueError(
                "IQL/CQL log-prob computation requires adapter.residual_bound > 0."
            )
        delta_actions = self._policy_action_to_delta(policy_actions, base_actions)
        normalized = (delta_actions / self.residual_bound).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(normalized)

    def _compute_logprob_from_policy_action(
        self,
        policy_actions: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        delta_pre_tanh = self._invert_policy_action(policy_actions, base_actions).to(
            device=mean.device,
            dtype=mean.dtype,
        )
        _, _, logprobs, _ = self._sample_delta(
            mean,
            log_std,
            deterministic=False,
            delta_pre_tanh=delta_pre_tanh,
        )
        return logprobs

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
        base_chunk_actions, base_result = self.base_vla.predict_action_batch(
            env_obs=env_obs,
            calculate_logprobs=False,
            calculate_values=False,
            **kwargs,
        )
        extracted = attach_base_actions(
            extract_features_from_env_obs(self.base_vla, env_obs),
            base_chunk_actions,
        )
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
        policy_actions = self._delta_to_policy_action(delta_actions, base_actions).to(
            dtype=torch.float32
        )

        forward_inputs = {
            **base_result["forward_inputs"],
            "states": extracted.states.to(dtype=torch.float32),
            "base_actions": base_actions,
            "delta_pre_tanh": delta_pre_tanh.view_as(base_actions).to(dtype=torch.float32),
            "action": policy_actions,
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
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.IQL_V:
            return self.iql_v_forward(**kwargs)
        if forward_type == ForwardType.IQL_LOGPROB:
            return self.iql_logprob_forward(**kwargs)
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

    def sac_forward(self, obs, **kwargs):
        extracted = self._extract_adapter_context(obs, **kwargs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        mean, log_std = self.adapter_actor(adapter_inputs)
        _, delta_actions, logprobs, _ = self._sample_delta(
            mean,
            log_std,
            deterministic=False,
        )
        policy_actions = self._delta_to_policy_action(delta_actions, extracted.base_actions)
        return policy_actions, logprobs, None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False, **kwargs):
        del shared_feature
        if not hasattr(self, "q_head"):
            raise NotImplementedError(
                "Offline RL on ResidualChunkAdapterPolicy requires adapter.add_q_head=True."
            )
        extracted = self._extract_adapter_context(obs, **kwargs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        if detach_encoder:
            adapter_inputs = adapter_inputs.detach()
        flat_actions = actions.view(actions.shape[0], -1).to(
            device=adapter_inputs.device,
            dtype=adapter_inputs.dtype,
        )
        return self.q_head(adapter_inputs, flat_actions)

    def iql_v_forward(self, obs, **kwargs):
        if not hasattr(self, "value_head"):
            raise NotImplementedError(
                "IQL on ResidualChunkAdapterPolicy requires adapter.add_value_head=True."
            )
        extracted = self._extract_adapter_context(obs, **kwargs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        return self.value_head(adapter_inputs)

    def iql_logprob_forward(self, obs, actions, **kwargs):
        extracted = self._extract_adapter_context(obs, **kwargs)
        adapter_inputs = self._build_adapter_inputs(
            extracted.features,
            extracted.states,
            extracted.base_actions,
        )
        mean, log_std = self.adapter_actor(adapter_inputs)
        return self._compute_logprob_from_policy_action(
            actions,
            mean=mean,
            log_std=log_std,
            base_actions=extracted.base_actions,
        )
