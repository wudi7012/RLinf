import json

import numpy as np
import pandas as pd

from rlinf.data.datasets.libero_offline_rl import (
    LiberoChunkOfflineDataset,
    LiberoChunkTransitionDataset,
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
