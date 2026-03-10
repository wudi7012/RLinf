# Copyright 2025 The RLinf Authors.
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

"""DatasetEnv: wraps an SFT dataset as a virtual environment for offline reward training.

Supports two dataset formats:

* **npy** – directory of ``*.npy`` files, each an array of frame dicts
  (legacy world-model format used elsewhere in RLinf).
* **parquet** – LeRobot-style directory tree where each
  ``episode_XXXXXX.parquet`` file contains columns ``image``, ``actions``,
  ``state``, etc.  Images are stored as ``{bytes, path}`` dicts with
  PNG-encoded bytes.

Each episode consists of a single chunk step:
  1. ``reset()`` draws (observation, ground_truth_actions) from the dataset.
  2. ``chunk_step(predicted_actions)`` computes a rule-based reward
     (e.g. negative MAE) between the predicted actions and the ground truth,
     then immediately terminates the episode.

This enables GRPO-style training where ``group_size`` environments share the
same observation and different action predictions are compared via relative
advantages.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import logging
import os
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from rlinf.envs.utils import to_tensor

logger = logging.getLogger(__name__)

__all__ = ["DatasetEnv"]


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _reward_mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Negative MAE per environment, averaged over chunk and action dims."""
    return -torch.mean(torch.abs(pred - gt), dim=(1, 2))  # [num_envs]


def _reward_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Negative MSE per environment."""
    return -torch.mean((pred - gt) ** 2, dim=(1, 2))


def _reward_exp_mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """exp(-MAE) reward in [0, 1]."""
    mae = torch.mean(torch.abs(pred - gt), dim=(1, 2))
    return torch.exp(-mae)


def _reward_cosine(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Cosine similarity averaged over chunk steps."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)
    return torch.nn.functional.cosine_similarity(pred_flat, gt_flat, dim=1)


_REWARD_FN_MAP = {
    "mae": _reward_mae,
    "mse": _reward_mse,
    "exp_mae": _reward_exp_mae,
    "cosine": _reward_cosine,
}


# ---------------------------------------------------------------------------
# Dataset backends
# ---------------------------------------------------------------------------


class _NpyBackend:
    """Loads episodes from a flat directory of ``.npy`` trajectory files."""

    def __init__(self, data_path: str, **kwargs):
        files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        if not files:
            raise ValueError(f"No .npy files found in {data_path}")
        self._files = files

    def __len__(self) -> int:
        return len(self._files)

    def load_episode(self, index: int) -> list[dict]:
        """Return list-of-dicts, one dict per frame."""
        data = np.load(
            self._files[index % len(self._files)], allow_pickle=True
        )
        return list(data)


class _ParquetBackend:
    """Loads episodes from a LeRobot-style parquet directory tree.

    Expected layout::

        data_path/
          chunk-000/
            episode_000000.parquet
            episode_000001.parquet
            ...

    Each parquet has columns: image (dict{bytes,path}), wrist_image,
    state (ndarray), actions (ndarray), timestamp, frame_index,
    episode_index, index, task_index.
    """

    def __init__(self, data_path: str, **kwargs):
        import pandas as pd  # noqa: F811 – lazy import

        self._pd = pd
        # Collect all parquet files recursively
        files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
        if not files:
            raise ValueError(f"No .parquet files found under {data_path}")
        self._files = files

    def __len__(self) -> int:
        return len(self._files)

    def load_episode(self, index: int) -> list[dict]:
        """Return list-of-dicts compatible with the npy format."""
        df = self._pd.read_parquet(self._files[index % len(self._files)])
        frames: list[dict] = []
        for _, row in df.iterrows():
            frame: dict = {}

            # --- image ---
            img_cell = row.get("image")
            if (
                isinstance(img_cell, dict)
                and "bytes" in img_cell
                and img_cell["bytes"]
            ):
                pil_img = Image.open(io.BytesIO(img_cell["bytes"])).convert("RGB")
                frame["image"] = np.asarray(pil_img, dtype=np.uint8)  # [H,W,3]
            elif isinstance(img_cell, np.ndarray):
                frame["image"] = img_cell

            # --- wrist_image ---
            wrist_cell = row.get("wrist_image")
            if (
                isinstance(wrist_cell, dict)
                and "bytes" in wrist_cell
                and wrist_cell["bytes"]
            ):
                pil_img = Image.open(io.BytesIO(wrist_cell["bytes"])).convert("RGB")
                frame["wrist_image"] = np.asarray(pil_img, dtype=np.uint8)
            elif isinstance(wrist_cell, np.ndarray):
                frame["wrist_image"] = wrist_cell

            # --- actions ---
            actions = row.get("actions")
            if actions is not None:
                frame["actions"] = np.asarray(actions, dtype=np.float32)

            # --- state ---
            state = row.get("state")
            if state is not None:
                frame["state"] = np.asarray(state, dtype=np.float32)

            # --- optional language fields ---
            for text_key in ("instruction", "task"):
                text_value = row.get(text_key)
                if text_value is None:
                    continue
                # Pandas may surface missing string as NaN float.
                if isinstance(text_value, float) and np.isnan(text_value):
                    continue
                frame[text_key] = text_value

            # --- task metadata ---
            task_idx = row.get("task_index")
            if task_idx is not None:
                frame["task_index"] = int(task_idx)

            frames.append(frame)
        return frames


def _make_backend(data_path: str, data_format: str, **kwargs):
    fmt = data_format.lower()
    if fmt == "auto":
        has_npy = bool(glob.glob(os.path.join(data_path, "*.npy")))
        has_pq = bool(
            glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True)
        )
        if has_pq and not has_npy:
            fmt = "parquet"
        else:
            fmt = "npy"

    if fmt == "npy":
        return _NpyBackend(data_path, **kwargs)
    elif fmt == "parquet":
        return _ParquetBackend(data_path, **kwargs)
    else:
        raise ValueError(f"Unknown data_format '{data_format}'. Use 'npy', 'parquet', or 'auto'.")


# ---------------------------------------------------------------------------
# DatasetEnv
# ---------------------------------------------------------------------------


class DatasetEnv:
    """Virtual environment backed by an SFT trajectory dataset.

    Compatible with :class:`EnvWorker` – exposes the same interface as
    :class:`LiberoEnv` (``reset``, ``chunk_step``, ``update_reset_state_ids``,
    ``is_start``, ``elapsed_steps``, ``info_logging_keys``, etc.).

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Environment configuration (see ``dataset_env_libero.yaml``).
    num_envs : int
        Number of parallel environments managed by this worker.
    seed_offset : int
        Per-worker seed offset (identical to ``LiberoEnv``).
    total_num_processes : int
        Total number of environment workers across all ranks.
    worker_info : WorkerInfo
        Ray worker metadata.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        self.seed = cfg.seed + seed_offset
        self._is_start = True

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        assert self.num_envs % self.group_size == 0, (
            f"num_envs ({self.num_envs}) must be divisible by "
            f"group_size ({self.group_size})"
        )

        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = getattr(cfg, "use_rel_reward", False)
        self.ignore_terminations = getattr(cfg, "ignore_terminations", False)

        # Reward configuration
        reward_fn_name = getattr(cfg, "reward_fn", "mae")
        if reward_fn_name not in _REWARD_FN_MAP:
            raise ValueError(
                f"Unknown reward_fn '{reward_fn_name}'. "
                f"Supported: {list(_REWARD_FN_MAP.keys())}"
            )
        self._reward_fn = _REWARD_FN_MAP[reward_fn_name]
        self.reward_coef = getattr(cfg, "reward_coef", 1.0)

        # Random generator for sampling
        self._generator = np.random.default_rng(seed=self.seed)

        # Build dataset backend
        self._build_dataset(cfg)

        # Video cfg (needed by EnvWorker / RecordVideo wrapper)
        self.video_cfg = cfg.video_cfg

        # Runtime state --------------------------------------------------
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs)

        # Cached per-env data (set during reset)
        self._current_obs: Optional[dict] = None
        self._gt_actions: Optional[np.ndarray] = None  # [num_envs, chunk, action_dim]
        self._current_episode_indices: Optional[np.ndarray] = None
        self.task_descriptions: list[str] = [""] * self.num_envs

        self._init_metrics()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg):
        data_path = cfg.data_path
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"DatasetEnv data_path does not exist: {data_path}"
            )

        data_format = getattr(cfg, "data_format", "auto")
        self._backend = _make_backend(data_path, data_format)

        self._action_key = getattr(cfg, "action_key", "actions")
        self._state_key = getattr(cfg, "state_key", "state")
        self._camera_name = getattr(cfg, "camera_name", "image")
        self._chunk_size = getattr(cfg, "chunk_size", 8)
        self._task_suite_name = getattr(cfg, "task_suite_name", None)
        self._task_index_to_description_path = getattr(
            cfg, "task_index_to_description_path", None
        )
        self._task_index_to_description = self._build_task_description_map(cfg)
        self._missing_task_description_indices: set[int] = set()

        logger.info(
            "DatasetEnv: loaded %d episodes from %s "
            "(format=%s, chunk_size=%d, action_key=%s)",
            len(self._backend),
            data_path,
            type(self._backend).__name__,
            self._chunk_size,
            self._action_key,
        )

    def _build_task_description_map(self, cfg) -> dict[int, str]:
        """Build task_index -> natural language description mapping.

        Priority:
        1) user-provided mapping file (JSON or JSONL)
        2) auto-discovered LeRobot ``meta/tasks.jsonl``
        3) LIBERO benchmark task language by ``task_suite_name``
        """
        mapping: dict[int, str] = {}

        def _merge_mapping_from_tasks_jsonl(path: str) -> int:
            added = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    task_idx = record.get("task_index")
                    if task_idx is None:
                        continue
                    task_text = record.get("task")
                    if task_text is None:
                        continue
                    try:
                        idx = int(task_idx)
                    except (TypeError, ValueError):
                        continue
                    if idx in mapping:
                        continue
                    mapping[idx] = str(task_text)
                    added += 1
            return added

        def _merge_mapping_from_json(path: str) -> int:
            added = 0
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                items = data.items()
            elif isinstance(data, list):
                # Support [{"task_index": 0, "task": "..."}] style.
                items = []
                for row in data:
                    if isinstance(row, dict) and "task_index" in row and "task" in row:
                        items.append((row["task_index"], row["task"]))
            else:
                items = []
            for key, value in items:
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if value is None or idx in mapping:
                    continue
                mapping[idx] = str(value)
                added += 1
            return added

        def _merge_mapping_from_path(path: str) -> int:
            if path.endswith(".jsonl"):
                return _merge_mapping_from_tasks_jsonl(path)
            return _merge_mapping_from_json(path)

        mapping_path = self._task_index_to_description_path
        if mapping_path:
            if not os.path.exists(mapping_path):
                logger.warning(
                    "DatasetEnv: task_index_to_description_path does not exist: %s",
                    mapping_path,
                )
            else:
                try:
                    added = _merge_mapping_from_path(mapping_path)
                    logger.info(
                        "DatasetEnv: loaded %d task descriptions from mapping path %s",
                        added,
                        mapping_path,
                    )
                except Exception as e:
                    logger.warning(
                        "DatasetEnv: failed to load task description mapping from %s: %s",
                        mapping_path,
                        e,
                    )

        # Auto-discover LeRobot meta/tasks.jsonl near data_path.
        data_path_abs = os.path.abspath(cfg.data_path)
        candidate_paths = [
            os.path.join(data_path_abs, "meta", "tasks.jsonl"),
            os.path.join(os.path.dirname(data_path_abs), "meta", "tasks.jsonl"),
            os.path.join(os.path.dirname(os.path.dirname(data_path_abs)), "meta", "tasks.jsonl"),
        ]
        for candidate in candidate_paths:
            if not os.path.exists(candidate):
                continue
            try:
                added = _merge_mapping_from_tasks_jsonl(candidate)
                logger.info(
                    "DatasetEnv: loaded %d task descriptions from dataset meta %s",
                    added,
                    candidate,
                )
            except Exception as e:
                logger.warning(
                    "DatasetEnv: failed to parse dataset meta task descriptions from %s: %s",
                    candidate,
                    e,
                )
            break

        suite_name = getattr(cfg, "task_suite_name", None)
        if suite_name:
            try:
                from rlinf.envs.libero.utils import get_benchmark_overridden

                task_suite = get_benchmark_overridden(suite_name)()
                for task_id in range(task_suite.get_num_tasks()):
                    if task_id in mapping:
                        continue
                    task = task_suite.get_task(task_id)
                    mapping[task_id] = str(task.language)
                logger.info(
                    "DatasetEnv: resolved %d task descriptions from LIBERO suite '%s'",
                    len(mapping),
                    suite_name,
                )
            except Exception as e:
                logger.warning(
                    "DatasetEnv: failed to build task description map from suite '%s': %s",
                    suite_name,
                    e,
                )
        return mapping

    def _resolve_task_description(self, first_frame: dict) -> str:
        # Prefer explicit natural language fields from dataset.
        for key in ("instruction", "task"):
            if key in first_frame:
                td = first_frame[key]
                if isinstance(td, (bytes, np.bytes_)):
                    td = td.decode("utf-8")
                task_desc = str(td)
                if task_desc:
                    # If legacy placeholder text was stored, try mapping from task_index.
                    if task_desc.startswith("task_") and "task_index" in first_frame:
                        try:
                            task_idx = int(first_frame["task_index"])
                            return self._task_index_to_description.get(
                                task_idx, task_desc
                            )
                        except (TypeError, ValueError):
                            return task_desc
                    return task_desc

        # Fall back to task_index mapping.
        task_idx = first_frame.get("task_index", None)
        if task_idx is not None:
            try:
                task_idx = int(task_idx)
                if task_idx in self._task_index_to_description:
                    return self._task_index_to_description[task_idx]
                if task_idx not in self._missing_task_description_indices:
                    logger.warning(
                        "DatasetEnv: missing natural-language task description for task_index=%d. "
                        "Falling back to placeholder 'task_%d'.",
                        task_idx,
                        task_idx,
                    )
                    self._missing_task_description_indices.add(task_idx)
                return f"task_{task_idx}"
            except (TypeError, ValueError):
                pass
        return ""

    @property
    def _num_episodes(self) -> int:
        return len(self._backend)

    # ------------------------------------------------------------------
    # Properties expected by EnvWorker
    # ------------------------------------------------------------------

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    # ------------------------------------------------------------------
    # Metrics (mirrors LiberoEnv)
    # ------------------------------------------------------------------

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0.0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = np.where(
            episode_info["episode_len"] > 0,
            episode_info["return"] / episode_info["episode_len"],
            0.0,
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    # ------------------------------------------------------------------
    # Reset state management (group_size aware)
    # ------------------------------------------------------------------

    def update_reset_state_ids(self):
        """Sample ``num_group`` unique dataset indices; repeat for group_size."""
        group_ids = self._generator.integers(
            low=0, high=self._num_episodes, size=(self.num_group,)
        )
        self.reset_state_ids = np.repeat(group_ids, self.group_size)

    # ------------------------------------------------------------------
    # Observation & action extraction
    # ------------------------------------------------------------------

    def _extract_single(self, ep_idx: int):
        """Extract obs and ground-truth action chunk from one episode.

        Returns
        -------
        image : np.ndarray [H, W, C] uint8
        wrist_image : np.ndarray [H, W, C] uint8 | None
        state : np.ndarray [state_dim] float32
        task_desc : str
        gt_actions : np.ndarray [chunk_size, action_dim] float32
        """
        frames = self._backend.load_episode(int(ep_idx))
        first_frame = frames[0]

        # ---- image ----
        image = first_frame.get(self._camera_name)
        if image is None:
            image = first_frame.get("image")
        if image is None:
            raise ValueError(
                f"No image key '{self._camera_name}' or 'image' in episode {ep_idx}"
            )
        if isinstance(image, np.ndarray):
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).clip(0, 255).astype(np.uint8)
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

        # ---- wrist image (optional) ----
        wrist_image = first_frame.get("wrist_image")
        if wrist_image is not None and isinstance(wrist_image, np.ndarray):
            if wrist_image.dtype in (np.float32, np.float64):
                wrist_image = (wrist_image * 255).clip(0, 255).astype(np.uint8)
            if wrist_image.ndim == 3 and wrist_image.shape[0] == 3:
                wrist_image = np.transpose(wrist_image, (1, 2, 0))  # CHW -> HWC

        # ---- state ----
        state = None
        for key in (self._state_key, "state", "init_ee_pose", "abs_action"):
            if key in first_frame:
                state = np.asarray(first_frame[key], dtype=np.float32)
                break
        if state is None:
            state = np.zeros(7, dtype=np.float32)

        # ---- task description ----
        task_desc = self._resolve_task_description(first_frame)

        # ---- ground-truth action chunk ----
        gt_chunk = []
        for step_idx in range(self._chunk_size):
            if step_idx < len(frames):
                frame = frames[step_idx]
                action = None
                for akey in (self._action_key, "actions", "delta_action", "abs_action"):
                    if akey in frame:
                        action = np.asarray(frame[akey], dtype=np.float32)
                        break
                if action is None:
                    raise ValueError(
                        f"No action key '{self._action_key}' in episode {ep_idx} frame {step_idx}"
                    )
            else:
                action = np.zeros_like(gt_chunk[0])
            gt_chunk.append(action)

        gt_actions = np.stack(gt_chunk, axis=0)  # [chunk_size, action_dim]
        return image, wrist_image, state, task_desc, gt_actions

    def _extract_batch(self, episode_indices: np.ndarray):
        """Extract observations and ground-truth actions for a batch.

        Returns
        -------
        obs_dict : dict  (main_images, wrist_images, states, task_descriptions)
        gt_actions : np.ndarray [num_envs, chunk_size, action_dim]
        """
        images, wrist_images, states, descs, gts = [], [], [], [], []
        has_wrist = True
        for ep_idx in episode_indices:
            img, wrist_img, st, desc, gt = self._extract_single(ep_idx)
            images.append(img)
            wrist_images.append(wrist_img)
            if wrist_img is None:
                has_wrist = False
            states.append(st)
            descs.append(desc)
            gts.append(gt)

        wrist_images_tensor = None
        if has_wrist:
            wrist_images_tensor = torch.from_numpy(np.stack(wrist_images, axis=0))

        obs_dict = {
            "main_images": torch.from_numpy(np.stack(images, axis=0)),
            "wrist_images": wrist_images_tensor,
            "states": torch.from_numpy(np.stack(states, axis=0)),
            "task_descriptions": descs,
        }
        gt_actions = np.stack(gts, axis=0)
        return obs_dict, gt_actions

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        """Reset the virtual environment by sampling new dataset episodes.

        Returns
        -------
        obs : dict
        infos : dict
        """
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx)

        if self.is_start:
            self.update_reset_state_ids()
            self._is_start = False

        if reset_state_ids is None:
            num_groups_needed = max(len(env_idx) // self.group_size, 1)
            group_ids = self._generator.integers(
                low=0, high=self._num_episodes, size=(num_groups_needed,)
            )
            reset_state_ids = np.repeat(group_ids, self.group_size)[
                : len(env_idx)
            ]

        # Full reset (all envs)
        if len(env_idx) == self.num_envs:
            obs_dict, gt_actions = self._extract_batch(reset_state_ids)
            self._current_obs = obs_dict
            self._gt_actions = gt_actions
            self.task_descriptions = obs_dict["task_descriptions"]
            self._current_episode_indices = reset_state_ids.copy()
        else:
            # Partial reset
            partial_obs, partial_gt = self._extract_batch(reset_state_ids)

            if self._gt_actions is None:
                self.update_reset_state_ids()
                full_obs, full_gt = self._extract_batch(self.reset_state_ids)
                self._current_obs = full_obs
                self._gt_actions = full_gt
                self.task_descriptions = full_obs["task_descriptions"]
                self._current_episode_indices = self.reset_state_ids.copy()

            for i, idx in enumerate(env_idx):
                self._gt_actions[idx] = partial_gt[i]
                self.task_descriptions[idx] = partial_obs["task_descriptions"][i]
                self._current_episode_indices[idx] = reset_state_ids[i]

            obs_dict, _ = self._extract_batch(self._current_episode_indices)
            self._current_obs = obs_dict

        self._reset_metrics(env_idx)
        infos = {}
        return obs_dict, infos

    def step(self, actions=None, auto_reset=True):
        """Single-step interface — not used; ``chunk_step`` is the primary API."""
        raise NotImplementedError(
            "DatasetEnv uses chunk_step only. Use chunk_step() instead."
        )

    def chunk_step(self, chunk_actions):
        """Execute one chunk step and compute the offline reward.

        Parameters
        ----------
        chunk_actions : np.ndarray or torch.Tensor
            Predicted actions, shape ``[num_envs, chunk_size, action_dim]``.

        Returns
        -------
        tuple of (obs_list, chunk_rewards, chunk_terminations,
                  chunk_truncations, infos_list)
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions_t = chunk_actions.detach().float()
        else:
            chunk_actions_t = torch.from_numpy(
                np.asarray(chunk_actions, dtype=np.float32)
            )

        chunk_size = chunk_actions_t.shape[1]
        assert self._gt_actions is not None, "Must call reset() before chunk_step()"

        gt_actions_t = torch.from_numpy(self._gt_actions).float()

        # Align chunk sizes
        min_chunk = min(chunk_size, gt_actions_t.shape[1])
        pred = chunk_actions_t[:, :min_chunk, :]
        gt = gt_actions_t[:, :min_chunk, :]

        # Compute per-environment scalar reward
        env_rewards = self._reward_fn(pred, gt) * self.reward_coef  # [num_envs]

        # Spread reward: only the last step carries the reward
        chunk_rewards = torch.zeros(self.num_envs, chunk_size, dtype=torch.float32)
        chunk_rewards[:, -1] = env_rewards

        # Update elapsed steps
        self._elapsed_steps += chunk_size

        # Every episode terminates after one chunk step
        terminations = np.ones(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)

        chunk_terminations = torch.zeros(self.num_envs, chunk_size, dtype=torch.bool)
        chunk_terminations[:, -1] = True

        chunk_truncations = torch.zeros(self.num_envs, chunk_size, dtype=torch.bool)

        # Record metrics
        step_reward_np = env_rewards.numpy()
        infos = {}
        infos = self._record_metrics(step_reward_np, terminations, infos)

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # Handle auto-reset
        dones = terminations | truncations
        if dones.any() and self.auto_reset:
            final_obs = copy.deepcopy(self._current_obs)
            final_info = copy.deepcopy(infos)

            done_idx = np.where(dones)[0]
            obs, _ = self.reset(env_idx=done_idx)

            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
        else:
            obs = self._current_obs

        return (
            [obs],
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    # ------------------------------------------------------------------
    # Offload / onload stubs (compatibility with EnvWorker)
    # ------------------------------------------------------------------

    def offload(self):
        """No-op: DatasetEnv has no GPU models to offload."""

    def onload(self):
        """No-op: DatasetEnv has no GPU models to onload."""
