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

"""Reward helpers for offline VLA adapter training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reduce_error(error: torch.Tensor, per_action: bool) -> torch.Tensor:
    if per_action:
        return error.mean(dim=-1)
    return error.mean(dim=(-1, -2))


def mae_error(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    per_action: bool = False,
) -> torch.Tensor:
    """Compute MAE between action chunks."""
    return _reduce_error((pred_actions - gt_actions).abs(), per_action=per_action)


def mse_error(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    per_action: bool = False,
) -> torch.Tensor:
    """Compute MSE between action chunks."""
    return _reduce_error((pred_actions - gt_actions).pow(2), per_action=per_action)


def cosine_error(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    per_action: bool = False,
) -> torch.Tensor:
    """Convert cosine similarity into an error quantity."""
    if per_action:
        cosine = F.cosine_similarity(pred_actions, gt_actions, dim=-1)
        return 1.0 - cosine

    pred_flat = pred_actions.flatten(start_dim=1)
    gt_flat = gt_actions.flatten(start_dim=1)
    cosine = F.cosine_similarity(pred_flat, gt_flat, dim=-1)
    return 1.0 - cosine


def compute_error(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    error_fn: str = "mae",
    *,
    per_action: bool = False,
) -> torch.Tensor:
    """Compute an action-space error metric."""
    if error_fn == "mae":
        return mae_error(pred_actions, gt_actions, per_action=per_action)
    if error_fn == "mse":
        return mse_error(pred_actions, gt_actions, per_action=per_action)
    if error_fn == "cosine":
        return cosine_error(pred_actions, gt_actions, per_action=per_action)
    raise ValueError(
        f"Unsupported adapter reward error_fn '{error_fn}'. "
        "Supported: ['mae', 'mse', 'cosine']"
    )


def improvement_reward(
    base_actions: torch.Tensor,
    final_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    *,
    error_fn: str = "mae",
    per_action: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward the adapter for improving on the frozen base action."""
    base_error = compute_error(
        base_actions,
        gt_actions,
        error_fn=error_fn,
        per_action=per_action,
    )
    final_error = compute_error(
        final_actions,
        gt_actions,
        error_fn=error_fn,
        per_action=per_action,
    )
    return base_error - final_error, base_error, final_error


def residual_penalty(
    delta_actions: torch.Tensor,
    *,
    penalty_type: str = "l1",
    per_action: bool = False,
) -> torch.Tensor:
    """Regularize adapter residual magnitude."""
    if penalty_type in {"none", None}:
        shape = delta_actions.shape[:2] if per_action else delta_actions.shape[:1]
        return torch.zeros(shape, dtype=delta_actions.dtype, device=delta_actions.device)

    if penalty_type == "l1":
        penalty = delta_actions.abs()
    elif penalty_type == "l2":
        penalty = delta_actions.pow(2)
    else:
        raise ValueError(
            f"Unsupported residual_penalty '{penalty_type}'. "
            "Supported: ['none', 'l1', 'l2']"
        )
    return _reduce_error(penalty, per_action=per_action)


def compose_reward(
    *,
    base_actions: torch.Tensor,
    final_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    delta_actions: torch.Tensor | None = None,
    error_fn: str = "mae",
    residual_penalty_type: str = "none",
    residual_coef: float = 0.0,
    per_action: bool = False,
) -> dict[str, torch.Tensor]:
    """Compose adapter improvement reward and diagnostics."""
    improvement, base_error, final_error = improvement_reward(
        base_actions=base_actions,
        final_actions=final_actions,
        gt_actions=gt_actions,
        error_fn=error_fn,
        per_action=per_action,
    )

    if delta_actions is None:
        penalty = torch.zeros_like(improvement)
    else:
        penalty = residual_penalty(
            delta_actions,
            penalty_type=residual_penalty_type,
            per_action=per_action,
        )

    reward = improvement - residual_coef * penalty
    return {
        "reward": reward,
        "improvement": improvement,
        "base_error": base_error,
        "final_error": final_error,
        "penalty": penalty,
    }
