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

"""Pure offline RL dataset utilities for LIBERO LeRobot parquet trajectories."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class LiberoOfflineEpisode:
    episode_index: int
    max_episode_length: int
    curr_env_obs: dict[str, Any]
    next_env_obs: dict[str, Any]
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor


@dataclass
class LiberoOfflineTransition:
    curr_env_obs: dict[str, Any]
    next_env_obs: dict[str, Any]
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor
    returns_to_go: Optional[torch.Tensor] = None
    next_returns_to_go: Optional[torch.Tensor] = None


def _decode_image_cell(image_cell: Any) -> np.ndarray:
    if isinstance(image_cell, dict) and image_cell.get("bytes"):
        pil_img = Image.open(io.BytesIO(image_cell["bytes"])).convert("RGB")
        return np.asarray(pil_img, dtype=np.uint8)

    if isinstance(image_cell, np.ndarray):
        image = image_cell
        if image.dtype in (np.float32, np.float64):
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        return image

    raise ValueError(f"Unsupported image cell type: {type(image_cell)!r}")


class LiberoChunkOfflineDataset:
    """Chunk-level offline RL dataset built from LeRobot parquet episodes.

    Each expert episode is converted into chunk transitions:
    ``(s_t, a_t:t+K-1, r_t, s_{t+K}, done_t)``.

    The reward is intentionally sparse:
    - reward = ``terminal_reward`` on the last valid chunk
    - reward = ``intermediate_reward`` on every earlier chunk
    """

    def __init__(
        self,
        dataset_root: str,
        chunk_size: int = 8,
        sample_stride: int = 1,
        terminal_reward: float = 1.0,
        intermediate_reward: float = 0.0,
        camera_name: str = "image",
        state_key: str = "state",
        max_episodes: Optional[int] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.data_root = self._resolve_data_root(self.dataset_root)
        self.meta_root = self._resolve_meta_root(self.dataset_root)
        self.chunk_size = int(chunk_size)
        self.sample_stride = max(1, int(sample_stride))
        self.terminal_reward = float(terminal_reward)
        self.intermediate_reward = float(intermediate_reward)
        self.camera_name = camera_name
        self.state_key = state_key

        self._episode_files = sorted(self.data_root.glob("**/*.parquet"))
        if max_episodes is not None:
            self._episode_files = self._episode_files[: int(max_episodes)]
        if not self._episode_files:
            raise FileNotFoundError(
                f"No parquet episodes found under dataset root '{self.dataset_root}'."
            )
        self._task_index_to_description = self._load_task_descriptions()

    @staticmethod
    def _resolve_data_root(dataset_root: Path) -> Path:
        if (dataset_root / "data").is_dir():
            return dataset_root / "data"
        if dataset_root.is_dir():
            return dataset_root
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    @staticmethod
    def _resolve_meta_root(dataset_root: Path) -> Path:
        if (dataset_root / "meta").is_dir():
            return dataset_root / "meta"
        parent_meta = dataset_root.parent / "meta"
        if parent_meta.is_dir():
            return parent_meta
        return dataset_root / "meta"

    def _load_task_descriptions(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        tasks_path = self.meta_root / "tasks.jsonl"
        if tasks_path.exists():
            with tasks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    task_idx = record.get("task_index", None)
                    task_text = record.get("task", None)
                    if task_idx is None or task_text is None:
                        continue
                    mapping[int(task_idx)] = str(task_text)
        return mapping

    def __len__(self) -> int:
        return len(self._episode_files)

    def _resolve_task_description(self, df: pd.DataFrame) -> str:
        for key in ("instruction", "task"):
            if key in df.columns:
                for value in df[key].tolist():
                    if value is None:
                        continue
                    if isinstance(value, float) and np.isnan(value):
                        continue
                    return str(value)

        if "task_index" in df.columns and len(df) > 0:
            task_idx = int(df.iloc[0]["task_index"])
            return self._task_index_to_description.get(task_idx, f"task_{task_idx}")

        return ""

    def _build_batched_env_obs(
        self,
        main_images: list[np.ndarray],
        wrist_images: list[Optional[np.ndarray]],
        states: list[np.ndarray],
        task_description: str,
    ) -> dict[str, Any]:
        wrist_images_tensor: Optional[torch.Tensor] = None
        if all(image is not None for image in wrist_images):
            wrist_stack = np.stack(wrist_images, axis=0)
            wrist_images_tensor = torch.from_numpy(wrist_stack)

        return {
            "main_images": torch.from_numpy(np.stack(main_images, axis=0)),
            "wrist_images": wrist_images_tensor,
            "states": torch.from_numpy(np.stack(states, axis=0).astype(np.float32)),
            "task_descriptions": [task_description for _ in range(len(main_images))],
        }

    def load_episode(self, episode_offset: int) -> Optional[LiberoOfflineEpisode]:
        file_path = self._episode_files[episode_offset]
        df = pd.read_parquet(file_path)
        num_frames = len(df)
        if num_frames < self.chunk_size:
            return None

        task_description = self._resolve_task_description(df)
        action_array = np.stack(df["actions"].tolist(), axis=0).astype(np.float32)
        state_array = np.stack(df[self.state_key].tolist(), axis=0).astype(np.float32)

        image_column = self.camera_name if self.camera_name in df.columns else "image"
        main_images = [_decode_image_cell(cell) for cell in df[image_column].tolist()]

        wrist_images_raw: list[Optional[np.ndarray]] = []
        if "wrist_image" in df.columns:
            for wrist_cell in df["wrist_image"].tolist():
                if wrist_cell is None:
                    wrist_images_raw.append(None)
                else:
                    wrist_images_raw.append(_decode_image_cell(wrist_cell))
        else:
            wrist_images_raw = [None for _ in range(num_frames)]

        last_start = num_frames - self.chunk_size
        start_indices = list(range(0, last_start + 1, self.sample_stride))
        if not start_indices or start_indices[-1] != last_start:
            start_indices.append(last_start)

        curr_main_images: list[np.ndarray] = []
        curr_wrist_images: list[Optional[np.ndarray]] = []
        curr_states: list[np.ndarray] = []
        next_main_images: list[np.ndarray] = []
        next_wrist_images: list[Optional[np.ndarray]] = []
        next_states: list[np.ndarray] = []
        chunk_actions: list[np.ndarray] = []
        rewards: list[list[float]] = []
        terminations: list[list[bool]] = []

        for transition_idx, start_idx in enumerate(start_indices):
            next_idx = min(start_idx + self.chunk_size, num_frames - 1)
            is_terminal = transition_idx == len(start_indices) - 1

            curr_main_images.append(main_images[start_idx])
            curr_wrist_images.append(wrist_images_raw[start_idx])
            curr_states.append(state_array[start_idx])

            next_main_images.append(main_images[next_idx])
            next_wrist_images.append(wrist_images_raw[next_idx])
            next_states.append(state_array[next_idx])

            chunk_actions.append(
                action_array[start_idx : start_idx + self.chunk_size].reshape(-1)
            )
            rewards.append(
                [self.terminal_reward if is_terminal else self.intermediate_reward]
            )
            terminations.append([bool(is_terminal)])

        curr_env_obs = self._build_batched_env_obs(
            curr_main_images,
            curr_wrist_images,
            curr_states,
            task_description,
        )
        next_env_obs = self._build_batched_env_obs(
            next_main_images,
            next_wrist_images,
            next_states,
            task_description,
        )

        episode_index = episode_offset
        if "episode_index" in df.columns and len(df) > 0:
            episode_index = int(df.iloc[0]["episode_index"])

        terminations_tensor = torch.as_tensor(terminations, dtype=torch.bool)
        return LiberoOfflineEpisode(
            episode_index=episode_index,
            max_episode_length=len(start_indices),
            curr_env_obs=curr_env_obs,
            next_env_obs=next_env_obs,
            actions=torch.as_tensor(np.stack(chunk_actions, axis=0), dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            terminations=terminations_tensor,
            truncations=torch.zeros_like(terminations_tensor),
            dones=terminations_tensor.clone(),
        )


def _compute_returns_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    returns_to_go = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.zeros_like(rewards[0], dtype=torch.float32)
    for step_id in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[step_id].to(torch.float32) + gamma * running * (
            ~dones[step_id]
        ).to(torch.float32)
        returns_to_go[step_id] = running
    return returns_to_go


class LiberoChunkTransitionDataset(Dataset):
    """Map-style transition dataset for actor-local offline training.

    This keeps the dataset in raw environment space and materializes model-ready
    observations on the actor at training time. That matches classic offline RL
    more closely and avoids startup-time replay-buffer preload.
    """

    def __init__(
        self,
        dataset_root: str,
        chunk_size: int = 8,
        sample_stride: int = 1,
        terminal_reward: float = 1.0,
        intermediate_reward: float = 0.0,
        camera_name: str = "image",
        state_key: str = "state",
        max_episodes: Optional[int] = None,
        returns_to_go_gamma: Optional[float] = None,
        episode_cache_size: int = 2,
    ) -> None:
        self.episode_dataset = LiberoChunkOfflineDataset(
            dataset_root=dataset_root,
            chunk_size=chunk_size,
            sample_stride=sample_stride,
            terminal_reward=terminal_reward,
            intermediate_reward=intermediate_reward,
            camera_name=camera_name,
            state_key=state_key,
            max_episodes=max_episodes,
        )
        self.returns_to_go_gamma = returns_to_go_gamma
        self.episode_cache_size = max(1, int(episode_cache_size))
        self._episode_cache: OrderedDict[
            int, tuple[LiberoOfflineEpisode, Optional[torch.Tensor]]
        ] = OrderedDict()
        self._transition_index: list[tuple[int, int]] = []
        self._build_transition_index()

    def _get_num_frames(self, file_path: Path) -> int:
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(file_path).metadata.num_rows)
        except Exception:
            df = pd.read_parquet(file_path, columns=["actions"])
            return int(len(df))

    def _build_transition_index(self) -> None:
        chunk_size = self.episode_dataset.chunk_size
        sample_stride = self.episode_dataset.sample_stride
        for episode_offset, file_path in enumerate(self.episode_dataset._episode_files):
            num_frames = self._get_num_frames(file_path)
            if num_frames < chunk_size:
                continue
            last_start = num_frames - chunk_size
            start_indices = list(range(0, last_start + 1, sample_stride))
            if not start_indices or start_indices[-1] != last_start:
                start_indices.append(last_start)
            for transition_idx in range(len(start_indices)):
                self._transition_index.append((episode_offset, transition_idx))

    def __len__(self) -> int:
        return len(self._transition_index)

    def _get_cached_episode(
        self,
        episode_offset: int,
    ) -> tuple[LiberoOfflineEpisode, Optional[torch.Tensor]]:
        cached = self._episode_cache.get(episode_offset, None)
        if cached is not None:
            self._episode_cache.move_to_end(episode_offset)
            return cached

        episode = self.episode_dataset.load_episode(episode_offset)
        if episode is None:
            raise RuntimeError(
                f"Episode {episode_offset} is invalid for transition sampling."
            )

        returns_to_go = None
        if self.returns_to_go_gamma is not None:
            returns_to_go = _compute_returns_to_go(
                rewards=episode.rewards,
                dones=episode.dones,
                gamma=float(self.returns_to_go_gamma),
            )

        cached = (episode, returns_to_go)
        self._episode_cache[episode_offset] = cached
        self._episode_cache.move_to_end(episode_offset)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return cached

    def _slice_transition_env_obs(
        self,
        env_obs: dict[str, Any],
        transition_idx: int,
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[transition_idx].clone()
            elif isinstance(value, list):
                sliced[key] = value[transition_idx]
            else:
                sliced[key] = value
        return sliced

    def __getitem__(self, index: int) -> LiberoOfflineTransition:
        episode_offset, transition_idx = self._transition_index[index]
        episode, returns_to_go = self._get_cached_episode(episode_offset)
        next_returns_to_go = None
        if returns_to_go is not None:
            if transition_idx + 1 < returns_to_go.shape[0]:
                next_returns_to_go = returns_to_go[transition_idx + 1].clone()
            else:
                next_returns_to_go = torch.zeros_like(returns_to_go[transition_idx])
        return LiberoOfflineTransition(
            curr_env_obs=self._slice_transition_env_obs(
                episode.curr_env_obs, transition_idx
            ),
            next_env_obs=self._slice_transition_env_obs(
                episode.next_env_obs, transition_idx
            ),
            actions=episode.actions[transition_idx].clone(),
            rewards=episode.rewards[transition_idx].clone(),
            terminations=episode.terminations[transition_idx].clone(),
            truncations=episode.truncations[transition_idx].clone(),
            dones=episode.dones[transition_idx].clone(),
            returns_to_go=(
                returns_to_go[transition_idx].clone()
                if returns_to_go is not None
                else None
            ),
            next_returns_to_go=next_returns_to_go,
        )


def _collate_env_obs_batch(env_obs_batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    keys = env_obs_batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in env_obs_batch]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif first_value is None:
            if not all(value is None for value in values):
                raise ValueError(
                    f"Mixed None / tensor values encountered for env obs key '{key}'."
                )
            collated[key] = None
        elif isinstance(first_value, str):
            collated[key] = list(values)
        else:
            collated[key] = list(values)
    return collated


def libero_offline_transition_collate_fn(
    batch: list[LiberoOfflineTransition],
) -> dict[str, Any]:
    collated = {
        "curr_env_obs": _collate_env_obs_batch(
            [transition.curr_env_obs for transition in batch]
        ),
        "next_env_obs": _collate_env_obs_batch(
            [transition.next_env_obs for transition in batch]
        ),
        "actions": torch.stack([transition.actions for transition in batch], dim=0),
        "rewards": torch.stack([transition.rewards for transition in batch], dim=0),
        "terminations": torch.stack(
            [transition.terminations for transition in batch], dim=0
        ),
        "truncations": torch.stack(
            [transition.truncations for transition in batch], dim=0
        ),
        "dones": torch.stack([transition.dones for transition in batch], dim=0),
    }
    if batch[0].returns_to_go is not None:
        collated["returns_to_go"] = torch.stack(
            [transition.returns_to_go for transition in batch], dim=0
        )
    if batch[0].next_returns_to_go is not None:
        collated["next_returns_to_go"] = torch.stack(
            [transition.next_returns_to_go for transition in batch], dim=0
        )
    return collated


def build_libero_chunk_offline_dataset_from_cfg(cfg) -> LiberoChunkOfflineDataset:
    dataset_cfg = cfg.algorithm.offline_rl.dataset
    return LiberoChunkOfflineDataset(
        dataset_root=str(dataset_cfg.dataset_root),
        chunk_size=int(dataset_cfg.get("chunk_size", cfg.actor.model.num_action_chunks)),
        sample_stride=int(dataset_cfg.get("sample_stride", 1)),
        terminal_reward=float(dataset_cfg.get("terminal_reward", 1.0)),
        intermediate_reward=float(dataset_cfg.get("intermediate_reward", 0.0)),
        camera_name=str(dataset_cfg.get("camera_name", "image")),
        state_key=str(dataset_cfg.get("state_key", "state")),
        max_episodes=dataset_cfg.get("max_episodes", None),
    )


def build_libero_chunk_transition_dataset_from_cfg(cfg) -> LiberoChunkTransitionDataset:
    dataset_cfg = cfg.algorithm.offline_rl.dataset
    returns_to_go_gamma = None
    if cfg.algorithm.offline_rl.get("name", "sac").lower() == "calql":
        returns_to_go_gamma = cfg.algorithm.offline_rl.get("calql", {}).get(
            "returns_to_go_gamma", 1.0
        )
    return LiberoChunkTransitionDataset(
        dataset_root=str(dataset_cfg.dataset_root),
        chunk_size=int(dataset_cfg.get("chunk_size", cfg.actor.model.num_action_chunks)),
        sample_stride=int(dataset_cfg.get("sample_stride", 1)),
        terminal_reward=float(dataset_cfg.get("terminal_reward", 1.0)),
        intermediate_reward=float(dataset_cfg.get("intermediate_reward", 0.0)),
        camera_name=str(dataset_cfg.get("camera_name", "image")),
        state_key=str(dataset_cfg.get("state_key", "state")),
        max_episodes=dataset_cfg.get("max_episodes", None),
        returns_to_go_gamma=returns_to_go_gamma,
        episode_cache_size=int(dataset_cfg.get("episode_cache_size", 2)),
    )
