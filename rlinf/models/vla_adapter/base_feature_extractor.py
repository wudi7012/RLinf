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

"""Extract frozen-VLA features and deterministic base actions for adapters."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class VLAAdapterFeatures:
    """Frozen VLA outputs consumed by the residual adapter."""

    features: torch.Tensor
    base_actions: torch.Tensor
    states: torch.Tensor
    forward_inputs: dict[str, torch.Tensor]


def _env_states_to_device(env_obs: dict[str, Any], device: torch.device) -> torch.Tensor:
    states = env_obs["states"]
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states)
    return states.to(device=device, dtype=torch.float32)


def _pool_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.mean(dim=1)


def _to_base_action_tensor(
    base_actions: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(base_actions, np.ndarray):
        return torch.from_numpy(base_actions).to(device=device, dtype=torch.float32)
    return base_actions.to(device=device, dtype=torch.float32)


def _prepare_rlinf_inputs(base_model, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    task_descriptions = [
        f"In: What action should the robot take to {t.lower()}?\nOut: "
        for t in env_obs["task_descriptions"]
    ]
    main_images = env_obs["main_images"]
    if isinstance(main_images, np.ndarray):
        main_images = torch.from_numpy(main_images)
    if main_images.ndim == 4:
        main_images = main_images.unsqueeze(1)
    assert main_images.ndim == 5

    all_images = [main_images.permute(0, 1, 4, 2, 3)]
    if base_model.vision_backbone.get_num_images_in_input() > 1:
        wrist_images = env_obs["wrist_images"]
        if isinstance(wrist_images, np.ndarray):
            wrist_images = torch.from_numpy(wrist_images)
        if wrist_images.ndim == 4:
            wrist_images = wrist_images.unsqueeze(1)
        wrist_images = wrist_images.permute(0, 1, 4, 2, 3)
        all_images.extend([wrist_images[:, i] for i in range(wrist_images.shape[1])])

    primary_image = all_images.pop(0)
    inputs = base_model.input_processor(
        text=task_descriptions,
        images={"images": primary_image},
        proprio_states=env_obs["states"],
        padding="max_length",
        max_length=base_model.max_prompt_length,
    )

    if all_images:
        all_wrist_inputs = [
            base_model.input_processor(
                text=task_descriptions,
                images={"images": wrist_image.unsqueeze(1)},
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=base_model.max_prompt_length,
            )
            for wrist_image in all_images
        ]
        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [
            wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
        ]
        inputs["pixel_values"] = torch.cat(
            [primary_pixel_values] + all_wrist_pixel_values,
            dim=1,
        )

    device = next(base_model.parameters()).device
    precision = next(base_model.parameters()).dtype

    input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.bool)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)
    bsz, num_images, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.reshape(bsz, num_images * channels, height, width)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }


def _extract_rlinf_from_forward_inputs(base_model, forward_inputs: dict[str, torch.Tensor]):
    input_ids = forward_inputs["input_ids"]
    attention_mask = forward_inputs["attention_mask"]
    pixel_values = forward_inputs["pixel_values"]

    n_prompt_tokens = input_ids.shape[-1] - 1
    n_patches = (
        base_model.vision_backbone.get_num_patches()
        * base_model.vision_backbone.get_num_images_in_input()
    )

    input_ids_act, attention_mask_act = base_model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    mm_embeddings, mm_attention_mask = base_model._build_embedding(
        input_ids_act, attention_mask_act, pixel_values
    )
    mm_position_ids = mm_attention_mask.cumsum(dim=1) - 1
    outputs = base_model.language_model(
        input_ids=None,
        attention_mask=mm_attention_mask,
        position_ids=mm_position_ids,
        past_key_values=None,
        inputs_embeds=mm_embeddings,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    logits = outputs.logits[
        :,
        n_patches + n_prompt_tokens : n_patches
        + n_prompt_tokens
        + base_model.action_dim * base_model.num_action_chunks,
        :,
    ]
    logits = logits.clone()
    logits[..., : base_model.vocab_size - base_model.config.n_action_bins] = -torch.inf
    logits[..., base_model.vocab_size :] = -torch.inf
    action_token_ids = logits.argmax(dim=-1)
    chunk_action_tokens = action_token_ids.reshape(-1, base_model.action_dim)
    predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
    discretized_actions = base_model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=base_model.bin_centers.shape[0] - 1,
    )
    normalized_actions = np.asarray(
        [base_model.bin_centers[da] for da in discretized_actions]
    ).reshape(-1, base_model.action_dim)
    actions = base_model._unnormalize_actions(normalized_actions, base_model.unnorm_key)
    base_actions = torch.as_tensor(
        actions.reshape(-1, base_model.num_action_chunks, base_model.action_dim),
        device=logits.device,
        dtype=torch.float32,
    )

    hidden_states = outputs.hidden_states[-1][
        :,
        -base_model.action_dim * base_model.num_action_chunks - 1 : -1,
    ]
    features = _pool_hidden_states(hidden_states)
    return features, base_actions


def _extract_official_from_forward_inputs(base_model, forward_inputs: dict[str, torch.Tensor]):
    input_ids = forward_inputs["input_ids"]
    attention_mask = forward_inputs["attention_mask"]
    pixel_values = forward_inputs["pixel_values"]
    proprio = forward_inputs.get("proprio")

    labels = input_ids.clone()
    labels[:] = -100
    num_prompt_tokens = (
        input_ids.ne(base_model.processor.tokenizer.pad_token_id).sum(dim=1) - 1
    )
    mm_embeddings, mm_attention_mask, mm_position_ids = base_model._build_embedding(
        input_ids,
        attention_mask.to(torch.long),
        pixel_values,
        labels,
        proprio,
    )

    num_patches = (
        base_model.vision_backbone.get_num_patches()
        * base_model.vision_backbone.get_num_images_in_input()
    )
    if base_model.proprio_projector is not None and proprio is not None:
        num_patches += 1

    outputs = base_model.language_model(
        input_ids=None,
        attention_mask=mm_attention_mask,
        position_ids=mm_position_ids,
        past_key_values=None,
        inputs_embeds=mm_embeddings,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    batch_size = outputs.logits.shape[0]
    device = outputs.logits.device
    start_indices = (num_patches + num_prompt_tokens).unsqueeze(1)
    position_offsets = torch.arange(
        base_model.action_dim * base_model.num_action_chunks,
        device=device,
    ).unsqueeze(0)
    seq_indices = start_indices + position_offsets
    response_logits = outputs.logits[
        torch.arange(batch_size, device=device).unsqueeze(-1),
        seq_indices,
        :,
    ]
    action_logits = response_logits[
        ...,
        -base_model.config.n_action_bins
        - base_model.config.pad_to_multiple_of : -base_model.config.pad_to_multiple_of,
    ]
    action_token_ids = action_logits.argmax(dim=-1)
    final_response_ids = action_token_ids + (
        base_model.vocab_size - base_model.config.n_action_bins
    )
    predicted_action_token_ids = final_response_ids.cpu().numpy()
    discretized_actions = base_model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=base_model.bin_centers.shape[0] - 1,
    )
    normalized_actions = base_model.bin_centers[discretized_actions].reshape(
        -1, base_model.action_dim
    )
    actions = base_model._unnormalize_actions(normalized_actions, base_model.unnorm_key)
    base_actions = torch.as_tensor(
        actions.reshape(-1, base_model.num_action_chunks, base_model.action_dim),
        device=device,
        dtype=torch.float32,
    )

    hidden_states = outputs.hidden_states[-1][
        torch.arange(batch_size, device=device).unsqueeze(-1),
        seq_indices,
        :,
    ]
    features = _pool_hidden_states(hidden_states)
    return features, base_actions


def extract_features_from_env_obs(base_model, env_obs: dict[str, Any]) -> VLAAdapterFeatures:
    """Run deterministic frozen-VLA inference from raw observations."""
    device = next(base_model.parameters()).device
    states = _env_states_to_device(env_obs, device)

    if hasattr(base_model, "prepare_inputs"):
        input_ids, attention_mask, pixel_values, proprio = base_model.prepare_inputs(
            env_obs
        )
        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "proprio": proprio,
        }
        features, base_actions = _extract_official_from_forward_inputs(
            base_model, forward_inputs
        )
    elif hasattr(base_model, "input_processor"):
        forward_inputs = _prepare_rlinf_inputs(base_model, env_obs)
        features, base_actions = _extract_rlinf_from_forward_inputs(
            base_model, forward_inputs
        )
    else:
        raise NotImplementedError(
            f"Unsupported frozen VLA backbone type: {type(base_model).__name__}"
        )

    return VLAAdapterFeatures(
        features=features.to(dtype=torch.float32),
        base_actions=base_actions.to(dtype=torch.float32),
        states=states,
        forward_inputs=forward_inputs,
    )


def attach_base_actions(
    extracted: VLAAdapterFeatures,
    base_actions: np.ndarray | torch.Tensor,
) -> VLAAdapterFeatures:
    """Override deterministic base actions with externally supplied rollout actions."""
    base_actions_t = _to_base_action_tensor(
        base_actions,
        device=extracted.features.device,
    )
    return VLAAdapterFeatures(
        features=extracted.features,
        base_actions=base_actions_t,
        states=extracted.states,
        forward_inputs=extracted.forward_inputs,
    )


def extract_features_from_prediction_result(
    base_model,
    env_obs: dict[str, Any],
    base_actions: np.ndarray | torch.Tensor,
    prediction_result: dict[str, Any],
) -> VLAAdapterFeatures:
    """Reuse adapter features returned by base VLA rollout inference when available."""
    adapter_features = prediction_result.get("adapter_features")
    if adapter_features is None:
        return attach_base_actions(
            extract_features_from_env_obs(base_model, env_obs),
            base_actions,
        )

    device = next(base_model.parameters()).device
    features = adapter_features.to(device=device, dtype=torch.float32)
    return VLAAdapterFeatures(
        features=features,
        base_actions=_to_base_action_tensor(base_actions, device=device),
        states=_env_states_to_device(env_obs, device),
        forward_inputs=prediction_result["forward_inputs"],
    )


def extract_features_from_forward_inputs(
    base_model,
    forward_inputs: dict[str, torch.Tensor],
) -> VLAAdapterFeatures:
    """Recompute deterministic frozen-VLA features from cached forward inputs."""
    device = next(base_model.parameters()).device
    states = forward_inputs["states"].to(device=device, dtype=torch.float32)
    if "features" in forward_inputs:
        cached_base_actions = forward_inputs.get("base_actions")
        if cached_base_actions is None:
            raise ValueError(
                "Cached adapter forward inputs with `features` must also provide "
                "`base_actions`."
            )
        features = forward_inputs["features"].to(device=device, dtype=torch.float32)
        base_actions = _to_base_action_tensor(
            cached_base_actions,
            device=device,
        )
        cached_forward_inputs = {
            key: value
            for key, value in forward_inputs.items()
            if key not in {"delta_pre_tanh", "action"}
        }
        return VLAAdapterFeatures(
            features=features,
            base_actions=base_actions.to(dtype=torch.float32),
            states=states,
            forward_inputs=cached_forward_inputs,
        )
    vla_inputs = {
        key: value
        for key, value in forward_inputs.items()
        if key not in {"states", "delta_pre_tanh", "action", "base_actions"}
    }
    cached_base_actions = forward_inputs.get("base_actions")
    if "proprio" in vla_inputs:
        features, base_actions = _extract_official_from_forward_inputs(
            base_model, vla_inputs
        )
    else:
        features, base_actions = _extract_rlinf_from_forward_inputs(
            base_model, vla_inputs
        )
    if cached_base_actions is not None:
        base_actions = _to_base_action_tensor(
            cached_base_actions,
            device=features.device,
        )
    return VLAAdapterFeatures(
        features=features.to(dtype=torch.float32),
        base_actions=base_actions.to(dtype=torch.float32),
        states=states,
        forward_inputs=vla_inputs,
    )
