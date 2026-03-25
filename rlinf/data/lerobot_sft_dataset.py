from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


class ActionTokenizer:
    """Minimal OpenVLA action tokenizer used for SFT."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> str | list[str]:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            a_min=float(self.min_action),
            a_max=float(self.max_action),
        )
        discretized_action = np.digitize(action, self.bins)
        if discretized_action.ndim == 1:
            return self.tokenizer.decode(
                list(self.tokenizer.vocab_size - discretized_action)
            )
        return self.tokenizer.batch_decode(
            (self.tokenizer.vocab_size - discretized_action).tolist()
        )

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = (
            [instance["dataset_name"] for instance in instances]
            if "dataset_name" in instances[0]
            else None
        )

        assert self.padding_side == "right", f"Invalid tokenizer padding side: {self.padding_side}"

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        if not all(isinstance(pv, torch.Tensor) for pv in pixel_values):
            raise ValueError("OpenVLA SFT expects all pixel_values to be torch tensors.")

        pixel_values = torch.stack(pixel_values)
        if "pixel_values_wrist" in instances[0]:
            pixel_values_wrist = torch.stack(
                [instance["pixel_values_wrist"] for instance in instances]
            )
            pixel_values = torch.cat((pixel_values, pixel_values_wrist), dim=1)

        actions = torch.stack(
            [
                torch.from_numpy(np.asarray(instance["actions"], dtype=np.float32).copy())
                for instance in instances
            ]
        )

        proprio = None
        if "proprio" in instances[0]:
            proprio = torch.as_tensor(
                np.stack(
                    [
                        np.asarray(instance["proprio"], dtype=np.float32)
                        for instance in instances
                    ]
                ),
                dtype=torch.float32,
            )

        batch = {
            "pixel_values": pixel_values,
            "proprio": proprio,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
        }
        if dataset_names is not None:
            batch["dataset_names"] = dataset_names
        return batch


class LeRobotSFTDataset(IterableDataset):
    """Minimal LeRobot parquet dataset for OpenVLA-OFT SFT inside RLinf."""

    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform,
        *,
        num_action_chunks: int,
        train: bool = True,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        predict_stop_token: bool = True,
        val_ratio: float = 0.0,
        seed: int = 7,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.num_action_chunks = int(num_action_chunks)
        self.train = train
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.predict_stop_token = predict_stop_token
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.meta_dir = self.dataset_root / "meta"
        self.data_dir = self.dataset_root / "data"
        self._validate_dataset_root()

        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()
        self.episode_tasks = self._load_episode_tasks()
        self.dataset_statistics = {
            self.dataset_name: self._build_dataset_statistics(),
        }
        self.sample_index = self._build_sample_index()

        if len(self.sample_index) == 0:
            raise ValueError(f"No samples found in LeRobot dataset: {self.dataset_root}")

    def _validate_dataset_root(self) -> None:
        required_paths = [
            self.meta_dir / "info.json",
            self.meta_dir / "episodes.jsonl",
            self.meta_dir / "tasks.jsonl",
            self.meta_dir / "stats.json",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if not self.data_dir.is_dir():
            missing.append(str(self.data_dir))
        if missing:
            raise FileNotFoundError(f"LeRobot dataset is missing required files: {missing}")

    def _load_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        with open(self.meta_dir / "episodes.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    def _load_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        with open(self.meta_dir / "tasks.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tasks[int(row["task_index"])] = str(row["task"])
        return tasks

    def _load_episode_tasks(self) -> dict[int, str]:
        episode_tasks: dict[int, str] = {}
        for ep in self.episodes:
            episode_index = int(ep["episode_index"])
            task_value = ""
            if isinstance(ep.get("tasks"), list) and ep["tasks"]:
                task_value = str(ep["tasks"][0])
            elif ep.get("task") is not None:
                task_value = str(ep["task"])
            if task_value:
                episode_tasks[episode_index] = task_value
        return episode_tasks

    def _build_dataset_statistics(self) -> dict[str, Any]:
        with open(self.meta_dir / "stats.json", "r") as f:
            stats = json.load(f)

        action_stats = stats.get("actions") or stats.get("action")
        if action_stats is None:
            raise KeyError(f"Could not find action statistics in {self.meta_dir / 'stats.json'}")

        proprio_stats = stats.get("state") or stats.get("proprio")
        action_dim = len(action_stats["mean"])
        action_mask = [True] * max(action_dim - 1, 0) + ([False] if action_dim > 0 else [])
        dataset_stats = {
            "action": {
                "mean": action_stats["mean"],
                "std": action_stats["std"],
                "max": action_stats["max"],
                "min": action_stats["min"],
                "q01": action_stats.get("q01", action_stats["min"]),
                "q99": action_stats.get("q99", action_stats["max"]),
                "mask": action_stats.get("mask", action_mask),
            },
            "num_trajectories": len(self.episodes),
            "num_transitions": int(sum(int(ep["length"]) for ep in self.episodes)),
        }

        if proprio_stats is not None:
            proprio_dim = len(proprio_stats["mean"])
            dataset_stats["proprio"] = {
                "mean": proprio_stats["mean"],
                "std": proprio_stats["std"],
                "max": proprio_stats["max"],
                "min": proprio_stats["min"],
                "q01": proprio_stats.get("q01", proprio_stats["min"]),
                "q99": proprio_stats.get("q99", proprio_stats["max"]),
                "mask": proprio_stats.get("mask", [True] * proprio_dim),
            }
        elif self.use_proprio:
            raise KeyError(
                f"Dataset {self.dataset_root} does not have proprio/state statistics required for use_proprio=True."
            )

        return dataset_stats

    def _build_sample_index(self) -> list[tuple[int, int]]:
        split_idx = int(round(len(self.episodes) * (1.0 - self.val_ratio)))
        selected_episodes = self.episodes[:split_idx] if self.train else self.episodes[split_idx:]
        return [
            (int(ep["episode_index"]), int(step_idx))
            for ep in selected_episodes
            for step_idx in range(int(ep["length"]))
        ]

    @staticmethod
    def _normalize_with_bounds_q99(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        normalized = np.asarray(values, dtype=np.float32).copy()
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", [True] * normalized.shape[-1]), dtype=bool)
        normalized[..., mask] = np.clip(
            2.0 * (normalized[..., mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0,
            -1.0,
            1.0,
        )
        zero_mask = np.asarray(stats["min"], dtype=np.float32) == np.asarray(stats["max"], dtype=np.float32)
        normalized[..., zero_mask] = 0.0
        return normalized

    @staticmethod
    def _decode_image(image_struct: dict[str, Any]) -> Image.Image:
        image_bytes = image_struct.get("bytes")
        if image_bytes is None:
            raise ValueError("Expected image bytes in LeRobot parquet row.")
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_episode_rows(episode_path: str) -> dict[str, list[Any]]:
        import pandas as pd

        return pd.read_parquet(episode_path).to_dict(orient="list")

    def _resolve_episode_path(self, episode_index: int) -> Path:
        episode_name = f"episode_{episode_index:06d}.parquet"
        candidate_chunk_ids = [episode_index // 1000, episode_index // 1000 + 1]
        for chunk_id in dict.fromkeys(candidate_chunk_ids):
            candidate = self.data_dir / f"chunk-{chunk_id:03d}" / episode_name
            if candidate.exists():
                return candidate

        matches = list(self.data_dir.glob(f"chunk-*/{episode_name}"))
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"Could not locate parquet for episode_index={episode_index} under {self.data_dir}."
        )

    def _make_action_chunk(self, actions: np.ndarray, step_idx: int) -> np.ndarray:
        future_indices = np.clip(
            np.arange(step_idx, step_idx + self.num_action_chunks),
            0,
            len(actions) - 1,
        )
        return actions[future_indices]

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return text.replace("<image>", "").strip()

    @staticmethod
    def _maybe_get_text(rows: dict[str, list[Any]], key: str, step_idx: int) -> str | None:
        values = rows.get(key)
        if values is None:
            return None
        value = values[step_idx]
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        return text or None

    def _resolve_instruction(
        self,
        rows: dict[str, list[Any]],
        episode_index: int,
        step_idx: int,
    ) -> str:
        instruction = self._maybe_get_text(rows, "instruction", step_idx)
        if instruction is None:
            instruction = self._maybe_get_text(rows, "task", step_idx)

        if instruction is None and "task_index" in rows:
            task_idx = rows["task_index"][step_idx]
            if task_idx is not None and not (isinstance(task_idx, float) and np.isnan(task_idx)):
                instruction = self.tasks.get(int(task_idx))

        if instruction is None:
            instruction = self.episode_tasks.get(episode_index)

        if instruction is None:
            raise ValueError(
                f"Could not resolve language instruction for episode_index={episode_index}, step_idx={step_idx}."
            )

        return self._sanitize_text(instruction)

    def _make_prompt(self, instruction: str, action_chunk: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_instruction = instruction.lower()
        current_action_string = self.action_tokenizer(action_chunk[0])
        future_actions_string = "".join(self.action_tokenizer(action_chunk[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt = (
            f"In: What action should the robot take to {normalized_instruction}?\n"
            f"Out: {action_chunk_string}</s>"
        )
        input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return input_ids, labels

    def _build_instance(self, episode_index: int, step_idx: int) -> dict[str, Any]:
        episode_path = self._resolve_episode_path(episode_index)
        rows = self._load_episode_rows(str(episode_path))

        action_chunk_raw = self._make_action_chunk(
            np.asarray(rows["actions"], dtype=np.float32),
            step_idx,
        )
        action_chunk = self._normalize_with_bounds_q99(
            action_chunk_raw,
            self.dataset_statistics[self.dataset_name]["action"],
        )

        instruction = self._resolve_instruction(rows, episode_index, step_idx)
        input_ids, labels = self._make_prompt(instruction, action_chunk)

        image = self._decode_image(rows["image"][step_idx])
        pixel_values = self.image_transform(image)

        instance = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "actions": action_chunk,
        }

        if self.use_wrist_image:
            if "wrist_image" not in rows:
                raise KeyError(
                    "Dataset does not contain `wrist_image`, but use_wrist_image=True."
                )
            wrist_image = self._decode_image(rows["wrist_image"][step_idx])
            instance["pixel_values_wrist"] = self.image_transform(wrist_image)

        if self.use_proprio:
            if "state" not in rows:
                raise KeyError("Dataset does not contain `state`, but use_proprio=True.")
            proprio = np.asarray(rows["state"][step_idx], dtype=np.float32)
            instance["proprio"] = self._normalize_with_bounds_q99(
                proprio,
                self.dataset_statistics[self.dataset_name]["proprio"],
            )

        return instance

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = np.random.default_rng(self.seed + (0 if self.train else 10_000))
        indices = np.arange(len(self.sample_index))
        if self.train:
            while True:
                rng.shuffle(indices)
                for idx in indices:
                    episode_index, step_idx = self.sample_index[idx]
                    yield self._build_instance(episode_index, step_idx)
            return

        for episode_index, step_idx in self.sample_index:
            yield self._build_instance(episode_index, step_idx)

    def __len__(self) -> int:
        return len(self.sample_index)
