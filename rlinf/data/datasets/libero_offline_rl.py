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
import re
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
        task_descriptions: Optional[list[str]] = None,
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
        self._task_index_to_description = self._load_task_descriptions()
        self._task_description_filter = self._normalize_task_description_filter(
            task_descriptions
        )

        self._episode_files = sorted(self.data_root.glob("**/*.parquet"))
        if self._task_description_filter is not None:
            self._episode_files = self._filter_episode_files_by_task_descriptions(
                self._episode_files
            )
        if max_episodes is not None:
            self._episode_files = self._episode_files[: int(max_episodes)]
        if not self._episode_files:
            filter_msg = ""
            if self._task_description_filter is not None:
                filter_msg = (
                    f" matching tasks {sorted(self._task_description_filter)!r}"
                )
            raise FileNotFoundError(
                f"No parquet episodes{filter_msg} found under dataset root "
                f"'{self.dataset_root}'."
            )

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

    @staticmethod
    def _normalize_task_description(task_description: str) -> str:
        return " ".join(str(task_description).strip().lower().split())

    def _normalize_task_description_filter(
        self,
        task_descriptions: Optional[list[str]],
    ) -> Optional[set[str]]:
        if task_descriptions is None:
            return None
        normalized = {
            self._normalize_task_description(task_description)
            for task_description in task_descriptions
            if str(task_description).strip()
        }
        return normalized or None

    @staticmethod
    def _episode_index_from_path(file_path: Path) -> Optional[int]:
        match = re.search(r"episode_(\d+)$", file_path.stem)
        if match is None:
            return None
        return int(match.group(1))

    def _episode_file_map(self, episode_files: list[Path]) -> dict[int, Path]:
        mapped_files: dict[int, Path] = {}
        for file_path in episode_files:
            episode_index = self._episode_index_from_path(file_path)
            if episode_index is not None:
                mapped_files[episode_index] = file_path
        return mapped_files

    def _filter_episode_files_by_task_descriptions(
        self,
        episode_files: list[Path],
    ) -> list[Path]:
        if self._task_description_filter is None:
            return episode_files

        episodes_path = self.meta_root / "episodes.jsonl"
        if episodes_path.exists():
            episode_file_by_index = self._episode_file_map(episode_files)
            filtered_episode_files: list[Path] = []
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    episode_index = record.get("episode_index", None)
                    if episode_index is None:
                        continue
                    file_path = episode_file_by_index.get(int(episode_index), None)
                    if file_path is None:
                        continue
                    tasks = record.get("tasks", [])
                    if isinstance(tasks, str):
                        tasks = [tasks]
                    normalized_tasks = {
                        self._normalize_task_description(task) for task in tasks
                    }
                    if normalized_tasks & self._task_description_filter:
                        filtered_episode_files.append(file_path)
            return filtered_episode_files

        filtered_episode_files = []
        for file_path in episode_files:
            try:
                import pyarrow.parquet as pq

                available_columns = set(
                    pq.ParquetFile(file_path).schema_arrow.names
                )
                metadata_columns = [
                    column
                    for column in ("instruction", "task", "task_index")
                    if column in available_columns
                ]
                df = pd.read_parquet(file_path, columns=metadata_columns)
            except Exception:
                df = pd.read_parquet(file_path)
                metadata_columns = [
                    column
                    for column in ("instruction", "task", "task_index")
                    if column in df.columns
                ]
                df = df[metadata_columns]
            task_description = self._resolve_task_description(df)
            if (
                self._normalize_task_description(task_description)
                in self._task_description_filter
            ):
                filtered_episode_files.append(file_path)
        return filtered_episode_files

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
        task_descriptions: Optional[list[str]] = None,
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
            task_descriptions=task_descriptions,
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


class LiberoPreprocessedTransitionDataset(Dataset):
    """Transition dataset backed by pre-materialized model-ready sidecars.

    The sidecar format is intentionally separate from the LeRobot ``data/`` tree:
    raw episodes remain compatible with existing LIBERO offline datasets, while
    ``preprocessed/`` can be regenerated for a particular model/preprocessing setup.
    """

    def __init__(
        self,
        preprocessed_root: str,
        max_episodes: Optional[int] = None,
        episode_cache_size: int = 2,
    ) -> None:
        self.preprocessed_root = Path(preprocessed_root).expanduser().resolve()
        self.episode_cache_size = max(1, int(episode_cache_size))
        self._indexed_num_transitions: dict[Path, int] = {}
        self._episode_files = self._discover_episode_files()
        if max_episodes is not None:
            self._episode_files = self._episode_files[: int(max_episodes)]
        if not self._episode_files:
            raise FileNotFoundError(
                f"No preprocessed transition sidecars found under "
                f"'{self.preprocessed_root}'."
            )
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._transition_index: list[tuple[int, int]] = []
        self._build_transition_index()

    def _discover_episode_files(self) -> list[Path]:
        index_path = self.preprocessed_root / "index.jsonl"
        if index_path.exists():
            episode_files: list[Path] = []
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rel_path = record.get("path", None)
                    if rel_path is None:
                        continue
                    file_path = self.preprocessed_root / str(rel_path)
                    episode_files.append(file_path)
                    num_transitions = record.get("num_transitions", None)
                    if num_transitions is not None:
                        self._indexed_num_transitions[file_path] = int(
                            num_transitions
                        )
            return [path for path in episode_files if path.exists()]
        return sorted(self.preprocessed_root.glob("**/episode_*.pt"))

    @staticmethod
    def _torch_load(path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _get_num_transitions(self, file_path: Path) -> int:
        indexed_num_transitions = self._indexed_num_transitions.get(file_path, None)
        if indexed_num_transitions is not None:
            return indexed_num_transitions
        payload = self._torch_load(file_path)
        return int(payload["actions"].shape[0])

    def _build_transition_index(self) -> None:
        for episode_offset, file_path in enumerate(self._episode_files):
            for transition_idx in range(self._get_num_transitions(file_path)):
                self._transition_index.append((episode_offset, transition_idx))

    def __len__(self) -> int:
        return len(self._transition_index)

    def _get_cached_episode(self, episode_offset: int) -> dict[str, Any]:
        cached = self._episode_cache.get(episode_offset, None)
        if cached is not None:
            self._episode_cache.move_to_end(episode_offset)
            return cached

        payload = self._torch_load(self._episode_files[episode_offset])
        self._episode_cache[episode_offset] = payload
        self._episode_cache.move_to_end(episode_offset)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return payload

    def _slice_value(self, value: Any, transition_idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[transition_idx].clone()
        if isinstance(value, dict):
            return {
                key: self._slice_value(sub_value, transition_idx)
                for key, sub_value in value.items()
            }
        if isinstance(value, list):
            return value[transition_idx]
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_offset, transition_idx = self._transition_index[index]
        episode = self._get_cached_episode(episode_offset)
        transition = {
            "curr_obs": self._slice_value(episode["curr_obs"], transition_idx),
            "next_obs": self._slice_value(episode["next_obs"], transition_idx),
            "actions": episode["actions"][transition_idx].clone(),
            "rewards": episode["rewards"][transition_idx].clone(),
            "terminations": episode["terminations"][transition_idx].clone(),
            "truncations": episode["truncations"][transition_idx].clone(),
            "dones": episode["dones"][transition_idx].clone(),
        }
        if "returns_to_go" in episode:
            transition["returns_to_go"] = episode["returns_to_go"][
                transition_idx
            ].clone()
        if "next_returns_to_go" in episode:
            transition["next_returns_to_go"] = episode["next_returns_to_go"][
                transition_idx
            ].clone()
        return transition


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


def _collate_nested_batch(values: list[Any]) -> Any:
    first_value = values[0]
    if isinstance(first_value, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first_value, dict):
        return {
            key: _collate_nested_batch([value[key] for value in values])
            for key in first_value
        }
    if isinstance(first_value, str):
        return list(values)
    if first_value is None:
        if not all(value is None for value in values):
            raise ValueError("Mixed None / non-None values in preprocessed batch.")
        return None
    return list(values)


def libero_offline_transition_collate_fn(
    batch: list[LiberoOfflineTransition | dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(batch[0], dict):
        collated = {
            "curr_obs": _collate_nested_batch(
                [transition["curr_obs"] for transition in batch]
            ),
            "next_obs": _collate_nested_batch(
                [transition["next_obs"] for transition in batch]
            ),
            "actions": torch.stack(
                [transition["actions"] for transition in batch], dim=0
            ),
            "rewards": torch.stack(
                [transition["rewards"] for transition in batch], dim=0
            ),
            "terminations": torch.stack(
                [transition["terminations"] for transition in batch], dim=0
            ),
            "truncations": torch.stack(
                [transition["truncations"] for transition in batch], dim=0
            ),
            "dones": torch.stack(
                [transition["dones"] for transition in batch], dim=0
            ),
        }
        if "returns_to_go" in batch[0]:
            collated["returns_to_go"] = torch.stack(
                [transition["returns_to_go"] for transition in batch], dim=0
            )
        if "next_returns_to_go" in batch[0]:
            collated["next_returns_to_go"] = torch.stack(
                [transition["next_returns_to_go"] for transition in batch], dim=0
            )
        return collated

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


def default_libero_preprocessed_root(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if dataset_root.name == "data" and (dataset_root.parent / "meta").is_dir():
        return dataset_root.parent / "preprocessed"
    return dataset_root / "preprocessed"


def _has_preprocessed_sidecars(preprocessed_root: str | Path) -> bool:
    root = Path(preprocessed_root).expanduser().resolve()
    if not root.exists():
        return False
    index_path = root / "index.jsonl"
    if index_path.exists() and index_path.stat().st_size > 0:
        return True
    return any(root.glob("**/episode_*.pt"))


def _resolve_preprocessed_root(dataset_cfg) -> Path:
    preprocessed_root = dataset_cfg.get("preprocessed_root", None)
    if preprocessed_root is None:
        preprocessed_root = default_libero_preprocessed_root(dataset_cfg.dataset_root)
    return Path(preprocessed_root).expanduser().resolve()


def _should_use_preprocessed_dataset(dataset_cfg, preprocessed_root: Path) -> bool:
    mode = dataset_cfg.get("use_preprocessed", "auto")
    if isinstance(mode, str):
        mode = mode.lower()
    if mode in (False, "false", "never", "off", "raw"):
        return False

    has_sidecars = _has_preprocessed_sidecars(preprocessed_root)
    if mode in (True, "true", "always", "on"):
        if not has_sidecars:
            raise FileNotFoundError(
                f"algorithm.offline_rl.dataset.use_preprocessed=true but no "
                f"sidecars were found under '{preprocessed_root}'."
            )
        return True
    if mode in ("auto", None):
        return has_sidecars
    raise ValueError(
        "algorithm.offline_rl.dataset.use_preprocessed must be one of "
        "auto/true/false."
    )


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def resolve_libero_specific_reset_task_descriptions(cfg) -> Optional[list[str]]:
    """Resolve offline task filtering from the LIBERO train specific reset id."""
    dataset_cfg = cfg.algorithm.offline_rl.dataset
    explicit_task_descriptions = _cfg_get(dataset_cfg, "task_descriptions", None)
    if explicit_task_descriptions is not None:
        return [str(task) for task in explicit_task_descriptions]

    specific_reset_id = _cfg_get(cfg.env.train, "specific_reset_id", None)
    if specific_reset_id is None:
        return None

    task_suite_name = _cfg_get(
        cfg.env.train,
        "task_suite_name",
        _cfg_get(cfg.env.eval, "task_suite_name", None),
    )
    if task_suite_name is None:
        raise ValueError(
            "env.train.task_suite_name is required when env.train.specific_reset_id "
            "is used for pure offline LIBERO task filtering."
        )

    from rlinf.envs.libero.utils import get_benchmark_overridden

    task_suite = get_benchmark_overridden(task_suite_name)()
    reset_state_id = int(specific_reset_id)
    start_pivot = 0
    for task_id in range(task_suite.get_num_tasks()):
        end_pivot = start_pivot + len(task_suite.get_task_init_states(task_id))
        if start_pivot <= reset_state_id < end_pivot:
            return [str(task_suite.get_task(task_id).language)]
        start_pivot = end_pivot

    raise ValueError(
        f"env.train.specific_reset_id={reset_state_id} is outside the valid "
        f"reset-state range [0, {start_pivot}) for task suite {task_suite_name!r}."
    )


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
        task_descriptions=resolve_libero_specific_reset_task_descriptions(cfg),
    )


def build_libero_chunk_transition_dataset_from_cfg(cfg) -> LiberoChunkTransitionDataset:
    dataset_cfg = cfg.algorithm.offline_rl.dataset
    preprocessed_root = _resolve_preprocessed_root(dataset_cfg)
    if _should_use_preprocessed_dataset(dataset_cfg, preprocessed_root):
        return LiberoPreprocessedTransitionDataset(
            preprocessed_root=str(preprocessed_root),
            max_episodes=dataset_cfg.get("max_episodes", None),
            episode_cache_size=int(dataset_cfg.get("episode_cache_size", 2)),
        )

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
        task_descriptions=resolve_libero_specific_reset_task_descriptions(cfg),
    )
