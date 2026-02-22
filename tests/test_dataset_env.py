#!/usr/bin/env python3
"""Standalone test for DatasetEnv — run INSIDE the Docker container.

Usage (3 phases, run in order):

    # Phase 1: DatasetEnv unit test (no model, no Ray)
    python tests/test_dataset_env.py --phase 1 --data_path /workspace/data/dataset/data

    # Phase 2: DatasetEnv + config dry-run (validates Hydra config)
    python tests/test_dataset_env.py --phase 2

    # Phase 3: Full pipeline single-step (model + env + rollout, 1 GPU)
    #   → Use run_embodiment.sh with the offline config instead.

Set breakpoints in any of these for line-by-line debugging via:
    python -m debugpy --listen 5678 --wait-for-client tests/test_dataset_env.py --phase 1 ...

Or simply add `breakpoint()` calls where you want to pause.
"""

import argparse
import sys
import os
import time

import numpy as np
import torch


# ============================================================
# Phase 1: Pure DatasetEnv unit test
# ============================================================
def phase1(args):
    """Test DatasetEnv in isolation — no Ray, no model, no Hydra."""
    from omegaconf import OmegaConf

    # Dynamically add RLinf root to PYTHONPATH
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from rlinf.envs.dataset_env.dataset_env import DatasetEnv

    # Build a minimal config dict
    cfg = OmegaConf.create({
        "data_path": args.data_path,
        "data_format": "auto",
        "reward_fn": "mae",
        "reward_coef": 1.0,
        "action_key": "actions",
        "state_key": "state",
        "camera_name": "image",
        "chunk_size": args.chunk_size,
        "group_size": args.group_size,
        "auto_reset": True,
        "use_rel_reward": False,
        "ignore_terminations": False,
        "seed": 42,
        "video_cfg": {"save_video": False, "info_on_video": False, "video_base_dir": "/tmp"},
    })

    num_envs = args.group_size * args.num_groups
    print(f"\n{'='*60}")
    print(f"Phase 1: DatasetEnv unit test")
    print(f"  data_path    = {args.data_path}")
    print(f"  num_envs     = {num_envs} ({args.num_groups} groups × {args.group_size})")
    print(f"  chunk_size   = {args.chunk_size}")
    print(f"{'='*60}\n")

    # ---- Step 1: Construct ----
    print("[1/5] Constructing DatasetEnv...")
    env = DatasetEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    print(f"  Backend type  : {type(env._backend).__name__}")
    print(f"  Num episodes  : {env._num_episodes}")
    print(f"  OK\n")

    # ---- Step 2: Reset ----
    print("[2/5] Calling env.reset()...")
    t0 = time.time()
    obs, infos = env.reset()
    dt = time.time() - t0
    print(f"  Time: {dt:.3f}s")
    print(f"  obs keys      : {list(obs.keys())}")
    print(f"  main_images   : {obs['main_images'].shape}, dtype={obs['main_images'].dtype}")
    print(f"  states        : {obs['states'].shape}, dtype={obs['states'].dtype}")
    print(f"  task_descs    : {obs['task_descriptions'][:2]}...")
    print(f"  gt_actions    : {env._gt_actions.shape}")
    print(f"  OK\n")

    # Verify group_size: same group should have identical gt_actions
    if args.group_size > 1:
        print("[2b/5] Verifying group_size consistency...")
        for g in range(args.num_groups):
            start = g * args.group_size
            base = env._gt_actions[start]
            for i in range(1, args.group_size):
                assert np.array_equal(env._gt_actions[start + i], base), (
                    f"Group {g}: env {start} != env {start+i}"
                )
        print(f"  All {args.num_groups} groups consistent ✓\n")

    # ---- Step 3: chunk_step with zeros (worst case) ----
    print("[3/5] chunk_step with zero actions (worst-case reward)...")
    fake_actions = np.zeros((num_envs, args.chunk_size, env._gt_actions.shape[-1]), dtype=np.float32)
    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(fake_actions)
    print(f"  rewards shape : {rewards.shape}")
    print(f"  rewards[:4,-1]: {rewards[:4, -1].tolist()}")
    print(f"  terms[:4,-1]  : {terms[:4, -1].tolist()}")
    print(f"  auto_reset    : final_observation in infos = {'final_observation' in infos_list[0]}")
    print(f"  OK\n")

    # ---- Step 4: chunk_step with gt actions (perfect reward) ----
    print("[4/5] chunk_step with ground-truth actions (perfect reward)...")
    # Need to reset first since auto_reset already happened
    obs, _ = env.reset()
    gt_copy = env._gt_actions.copy()
    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(gt_copy)
    print(f"  rewards[:4,-1]: {rewards[:4, -1].tolist()}")
    print(f"  (should be ~0.0 for MAE since pred == gt)")
    print(f"  OK\n")

    # ---- Step 5: chunk_step with noisy actions ----
    print("[5/5] chunk_step with noisy actions...")
    obs, _ = env.reset()
    noisy_actions = env._gt_actions + np.random.randn(*env._gt_actions.shape).astype(np.float32) * 0.1
    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(noisy_actions)
    print(f"  rewards[:4,-1]: {rewards[:4, -1].tolist()}")
    print(f"  (should be small negative values)")
    print(f"  OK\n")

    print("=" * 60)
    print("Phase 1 PASSED — DatasetEnv works correctly in isolation.")
    print("=" * 60)


# ============================================================
# Phase 2: Config loading dry-run
# ============================================================
def phase2(args):
    """Load the offline YAML config via Hydra, validate it prints correctly."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    config_dir = os.path.join(repo_root, "examples", "embodiment", "config")
    os.environ.setdefault("EMBODIED_PATH", os.path.join(repo_root, "examples", "embodiment"))

    print(f"\n{'='*60}")
    print(f"Phase 2: Hydra config dry-run")
    print(f"  config_dir = {config_dir}")
    print(f"{'='*60}\n")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="libero_grpo_openvlaoft_offline")

    print("[1/2] Config loaded successfully.")
    print(f"  env.train.env_type    = {cfg.env.train.env_type}")
    print(f"  env.train.data_path   = {cfg.env.train.data_path}")
    print(f"  env.train.reward_fn   = {cfg.env.train.get('reward_fn', 'N/A')}")
    print(f"  env.train.group_size  = {cfg.env.train.group_size}")
    print(f"  algorithm.adv_type    = {cfg.algorithm.adv_type}")
    print(f"  algorithm.group_size  = {cfg.algorithm.group_size}")
    print()

    # Instantiate DatasetEnv from the loaded config
    if args.data_path:
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.env.train.data_path = args.data_path
            cfg.env.train.total_num_envs = 8

    print("[2/2] Instantiating DatasetEnv from Hydra config...")
    from rlinf.envs import get_env_cls
    env_cls = get_env_cls(cfg.env.train.env_type, cfg.env.train)
    print(f"  env_cls = {env_cls.__name__}")

    if args.data_path:
        env = env_cls(
            cfg=cfg.env.train,
            num_envs=cfg.env.train.total_num_envs,
            seed_offset=0,
            total_num_processes=1,
        )
        obs, _ = env.reset()
        print(f"  reset() OK, main_images shape = {obs['main_images'].shape}")

    print()
    print("=" * 60)
    print("Phase 2 PASSED — Config loads and DatasetEnv instantiates.")
    print("=" * 60)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetEnv debug test")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="1=unit test, 2=config dry-run")
    parser.add_argument("--data_path", type=str, default="/workspace/data/dataset/data",
                        help="Path to dataset directory")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=2)

    args = parser.parse_args()

    if args.phase == 1:
        phase1(args)
    elif args.phase == 2:
        phase2(args)
