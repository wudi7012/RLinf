import json

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.libero_offline_rl import (
    LiberoChunkOfflineDataset,
    LiberoChunkTransitionDataset,
    LiberoPreprocessedTransitionDataset,
    build_libero_chunk_transition_dataset_from_cfg,
    libero_offline_transition_collate_fn,
)


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_episode(path, episode_index: int, task_index: int, length: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "episode_index": [episode_index] * length,
            "task_index": [task_index] * length,
            "actions": [np.zeros(7, dtype=np.float32).tolist()] * length,
            "state": [np.zeros(8, dtype=np.float32).tolist()] * length,
        }
    ).to_parquet(path)


def test_libero_offline_dataset_filters_episodes_by_task_description(tmp_path):
    dataset_root = tmp_path / "dataset"
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True)
    _write_jsonl(
        meta_root / "tasks.jsonl",
        [
            {"task_index": 0, "task": "target task"},
            {"task_index": 1, "task": "other task"},
        ],
    )
    _write_jsonl(
        meta_root / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["target task"], "length": 5},
            {"episode_index": 1, "tasks": ["other task"], "length": 5},
            {"episode_index": 2, "tasks": ["target task"], "length": 3},
        ],
    )
    _write_episode(dataset_root / "data/chunk-000/episode_000000.parquet", 0, 0, 5)
    _write_episode(dataset_root / "data/chunk-000/episode_000001.parquet", 1, 1, 5)
    _write_episode(dataset_root / "data/chunk-000/episode_000002.parquet", 2, 0, 3)

    episode_dataset = LiberoChunkOfflineDataset(
        dataset_root=str(dataset_root),
        chunk_size=2,
        task_descriptions=[" target   task "],
    )
    transition_dataset = LiberoChunkTransitionDataset(
        dataset_root=str(dataset_root),
        chunk_size=2,
        sample_stride=1,
        task_descriptions=["target task"],
    )

    assert [path.stem for path in episode_dataset._episode_files] == [
        "episode_000000",
        "episode_000002",
    ]
    assert len(episode_dataset) == 2
    assert len(transition_dataset) == 6


def test_libero_preprocessed_dataset_loads_sidecars(tmp_path):
    preprocessed_root = tmp_path / "dataset/preprocessed"
    episode_dir = preprocessed_root / "chunk-000"
    episode_dir.mkdir(parents=True)
    payload = {
        "episode_index": 0,
        "curr_obs": {
            "features": torch.arange(6, dtype=torch.float32).view(3, 2),
            "base_actions": torch.zeros(3, 2, 7),
        },
        "next_obs": {
            "features": torch.arange(6, 12, dtype=torch.float32).view(3, 2),
            "base_actions": torch.ones(3, 2, 7),
        },
        "actions": torch.zeros(3, 14),
        "rewards": torch.zeros(3, 1),
        "terminations": torch.zeros(3, 1, dtype=torch.bool),
        "truncations": torch.zeros(3, 1, dtype=torch.bool),
        "dones": torch.zeros(3, 1, dtype=torch.bool),
    }
    torch.save(payload, episode_dir / "episode_000000.pt")
    _write_jsonl(
        preprocessed_root / "index.jsonl",
        [{"episode_index": 0, "path": "chunk-000/episode_000000.pt"}],
    )

    dataset = LiberoPreprocessedTransitionDataset(str(preprocessed_root))
    batch = libero_offline_transition_collate_fn([dataset[0], dataset[1]])

    assert len(dataset) == 3
    assert batch["curr_obs"]["features"].shape == (2, 2)
    assert batch["next_obs"]["base_actions"].shape == (2, 2, 7)
    assert batch["actions"].shape == (2, 14)


def test_libero_transition_builder_uses_preprocessed_auto(tmp_path):
    dataset_root = tmp_path / "dataset"
    preprocessed_root = dataset_root / "preprocessed/chunk-000"
    preprocessed_root.mkdir(parents=True)
    torch.save(
        {
            "curr_obs": {"features": torch.zeros(1, 2)},
            "next_obs": {"features": torch.ones(1, 2)},
            "actions": torch.zeros(1, 14),
            "rewards": torch.zeros(1, 1),
            "terminations": torch.zeros(1, 1, dtype=torch.bool),
            "truncations": torch.zeros(1, 1, dtype=torch.bool),
            "dones": torch.zeros(1, 1, dtype=torch.bool),
        },
        preprocessed_root / "episode_000000.pt",
    )
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2}},
            "algorithm": {
                "offline_rl": {
                    "name": "cql",
                    "dataset": {
                        "dataset_root": str(dataset_root),
                        "use_preprocessed": "auto",
                    },
                }
            },
        }
    )

    dataset = build_libero_chunk_transition_dataset_from_cfg(cfg)

    assert isinstance(dataset, LiberoPreprocessedTransitionDataset)
    assert len(dataset) == 1
