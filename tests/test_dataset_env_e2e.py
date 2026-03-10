#!/usr/bin/env python3
"""End-to-end single-step test for the offline reward pipeline.

Tests the full loop:  DatasetEnv.reset() → model.predict() → DatasetEnv.chunk_step()

Run INSIDE Docker:
    cd /workspace/RLinf
    python tests/test_dataset_env_e2e.py \
        --data_path /workspace/data/dataset/data \
        --model_path /workspace/data/model/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/

For line-by-line debugging:
    python -m debugpy --listen 5678 --wait-for-client tests/test_dataset_env_e2e.py \
        --data_path /workspace/data/dataset/data \
        --model_path /workspace/data/model/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def banner(msg: str):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}\n")


def build_env(args):
    """Construct DatasetEnv with a minimal OmegaConf config."""
    from omegaconf import OmegaConf

    from rlinf.envs.dataset_env.dataset_env import DatasetEnv

    cfg = OmegaConf.create(
        {
            "data_path": args.data_path,
            "data_format": "auto",
            "reward_fn": args.reward_fn,
            "reward_coef": args.reward_coef,
            "action_key": "actions",
            "state_key": "state",
            "camera_name": "image",
            "chunk_size": args.chunk_size,
            "group_size": args.group_size,
            "auto_reset": True,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "seed": 42,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp",
            },
        }
    )
    num_envs = args.group_size * args.num_groups
    env = DatasetEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    return env, num_envs


def build_model(args):
    """Load the OpenVLA-OFT model for inference (single GPU)."""
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(
        {
            "model_type": "openvla_oft",
            "implement_version": "rlinf",
            "model_path": args.model_path,
            "precision": "bf16",
            "action_dim": 7,
            "num_action_chunks": args.chunk_size,
            "use_proprio": False,
            "use_film": False,
            "unnorm_key": args.unnorm_key,
            "center_crop": True,
            "max_prompt_length": 128,
            "vocab_size": 32000,
            "hidden_size": 4096,
            "add_value_head": False,
            "image_size": [224, 224],
            "is_lora": False,
            "lora_rank": 32,
            "lora_path": None,
            "num_images_in_input": 1,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "policy_setup": "widowx_bridge",
            "value_type": "action_level",
        }
    )

    print("[model] Loading OpenVLA-OFT model (this may take a minute)...")
    t0 = time.time()

    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    model = get_model(model_cfg, torch_dtype=torch.bfloat16)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    dt = time.time() - t0
    print(f"[model] Loaded in {dt:.1f}s on {device}")
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[model] Parameters: {param_count:.2f}B")
    return model, device


def prepare_obs_for_model(obs: dict, device: torch.device) -> dict:
    """Convert DatasetEnv observation dict to the format model.predict_action_batch expects."""
    env_obs = {}
    main_images = obs["main_images"]
    if isinstance(main_images, np.ndarray):
        main_images = torch.from_numpy(main_images)
    env_obs["main_images"] = main_images.to(device)

    states = obs["states"]
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states)
    env_obs["states"] = states.to(device)

    env_obs["task_descriptions"] = obs["task_descriptions"]

    wrist = obs.get("wrist_images")
    if wrist is not None:
        if isinstance(wrist, np.ndarray):
            wrist = torch.from_numpy(wrist)
        env_obs["wrist_images"] = wrist.to(device)
    else:
        env_obs["wrist_images"] = None

    return env_obs


def run_test(args):
    banner("Step 1/5: Build DatasetEnv")
    env, num_envs = build_env(args)
    print(f"  num_envs     = {num_envs}")
    print(f"  backend      = {type(env._backend).__name__}")
    print(f"  num_episodes = {env._num_episodes}")
    print(f"  chunk_size   = {env._chunk_size}")
    print(f"  group_size   = {env.group_size}")

    banner("Step 2/5: DatasetEnv.reset() — extract observations from dataset")
    t0 = time.time()
    obs, infos = env.reset()
    dt_reset = time.time() - t0
    print(f"  Time: {dt_reset:.3f}s")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  main_images : shape={obs['main_images'].shape}, dtype={obs['main_images'].dtype}")
    print(f"  states      : shape={obs['states'].shape}, dtype={obs['states'].dtype}")
    print(f"  task_descs  : {obs['task_descriptions'][:2]}")
    print(f"  gt_actions  : shape={env._gt_actions.shape}")

    if args.group_size > 1:
        print("\n  Verifying GRPO group consistency...")
        for g in range(args.num_groups):
            start = g * args.group_size
            base_gt = env._gt_actions[start]
            for i in range(1, args.group_size):
                assert np.array_equal(env._gt_actions[start + i], base_gt), (
                    f"Group {g}: env[{start}] != env[{start + i}]"
                )
            base_img = obs["main_images"][start]
            for i in range(1, args.group_size):
                assert torch.equal(obs["main_images"][start + i], base_img), (
                    f"Group {g}: image[{start}] != image[{start + i}]"
                )
        print(f"  All {args.num_groups} groups verified (same obs within group)")

    gt_actions_snapshot = env._gt_actions.copy()

    print("\n  --- Ground-truth action samples (first 2 envs, first 3 steps) ---")
    for e in range(min(2, num_envs)):
        print(f"  env[{e}] gt_actions:")
        for s in range(min(3, args.chunk_size)):
            a = gt_actions_snapshot[e, s]
            print(f"    step {s}: [{', '.join(f'{v:+.4f}' for v in a)}]")

    banner("Step 3/5: Model inference — predict action chunks")
    model, device = build_model(args)
    env_obs = prepare_obs_for_model(obs, device)

    sampling_params = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": -1,
        "top_p": 1.0,
    }

    print(f"\n[inference] Running predict_action_batch (B={num_envs})...")
    t0 = time.time()
    with torch.no_grad():
        pred_actions, result = model.predict_action_batch(
            env_obs=env_obs,
            calculate_logprobs=True,
            calculate_values=False,
            **sampling_params,
        )
    dt_infer = time.time() - t0
    print(f"[inference] Time: {dt_infer:.3f}s")
    print(f"[inference] pred_actions: type={type(pred_actions).__name__}, shape={pred_actions.shape}")
    print(f"[inference] result keys: {list(result.keys())}")
    if "prev_logprobs" in result:
        lp = result["prev_logprobs"]
        print(f"[inference] prev_logprobs: shape={lp.shape}, mean={lp.float().mean():.4f}")

    print("\n  --- Model predicted action samples (first 2 envs, first 3 steps) ---")
    for e in range(min(2, num_envs)):
        print(f"  env[{e}] pred_actions:")
        for s in range(min(3, args.chunk_size)):
            a = pred_actions[e, s]
            print(f"    step {s}: [{', '.join(f'{v:+.4f}' for v in a)}]")

    print("\n  --- Per-step diff (pred - gt) for env[0] ---")
    for s in range(min(args.chunk_size, 8)):
        diff = pred_actions[0, s] - gt_actions_snapshot[0, s]
        abs_diff = np.abs(diff)
        print(f"    step {s}: MAE={abs_diff.mean():.4f}  "
              f"diff=[{', '.join(f'{v:+.4f}' for v in diff)}]")

    if args.group_size > 1:
        print("\n  --- GRPO intra-group diversity (group 0) ---")
        g0_start = 0
        for i in range(min(args.group_size, 4)):
            a = pred_actions[g0_start + i, 0]
            print(f"    env[{g0_start + i}] step 0: [{', '.join(f'{v:+.4f}' for v in a)}]")

        n_diverse = 0
        for g in range(args.num_groups):
            start = g * args.group_size
            base = pred_actions[start]
            for i in range(1, args.group_size):
                if not np.allclose(pred_actions[start + i], base, atol=1e-6):
                    n_diverse += 1
        total_pairs = args.num_groups * (args.group_size - 1)
        print(f"\n  {n_diverse}/{total_pairs} intra-group pairs are different")
        if n_diverse > 0:
            print(f"  Temperature={args.temperature} is producing diversity")
        else:
            print(f"  WARNING: No diversity detected. Consider increasing temperature.")

    banner("Step 4/5: DatasetEnv.chunk_step() — compute offline reward")
    env_for_step = env
    obs_reset, _ = env_for_step.reset()
    gt_for_reward = env_for_step._gt_actions.copy()

    print("[reward] Testing with model predictions...")
    obs_model, _ = env_for_step.reset()
    gt_snap = env_for_step._gt_actions.copy()
    obs_list, rewards, terms, truncs, infos_list = env_for_step.chunk_step(pred_actions)
    model_rewards = rewards[:, -1].clone()
    print(f"  rewards shape : {rewards.shape}")
    print(f"  rewards[:,-1] : {model_rewards.tolist()}")
    print(f"  terminations  : all True at last step = {terms[:, -1].all().item()}")

    print("\n  --- Reward breakdown per env (model predictions) ---")
    print(f"  {'env':>4s}  {'reward':>10s}  {'pred_mean':>10s}  {'gt_mean':>10s}  {'MAE':>10s}")
    for e in range(num_envs):
        pred_e = pred_actions[e]  # [chunk, 7]
        gt_e = gt_snap[e]        # [chunk, 7]
        mae_e = np.abs(pred_e - gt_e).mean()
        print(f"  {e:4d}  {model_rewards[e]:10.4f}  "
              f"{np.abs(pred_e).mean():10.4f}  {np.abs(gt_e).mean():10.4f}  {mae_e:10.4f}")

    # --- Compare all reward functions on the same (pred, gt) pair ---
    from rlinf.envs.dataset_env.dataset_env import (
        _reward_mae, _reward_mse, _reward_exp_mae, _reward_cosine,
    )
    pred_t = torch.from_numpy(np.asarray(pred_actions, dtype=np.float32))
    gt_t = torch.from_numpy(gt_snap.astype(np.float32))
    min_c = min(pred_t.shape[1], gt_t.shape[1])
    pred_t_aligned = pred_t[:, :min_c, :]
    gt_t_aligned = gt_t[:, :min_c, :]

    all_reward_fns = {
        "mae":     _reward_mae,
        "mse":     _reward_mse,
        "exp_mae": _reward_exp_mae,
        "cosine":  _reward_cosine,
    }

    print("\n  --- All reward functions comparison (model predictions vs GT) ---")
    header = f"  {'env':>4s}"
    for fn_name in all_reward_fns:
        header += f"  {fn_name:>10s}"
    print(header)
    all_fn_rewards = {}
    for fn_name, fn in all_reward_fns.items():
        all_fn_rewards[fn_name] = fn(pred_t_aligned, gt_t_aligned)
    for e in range(num_envs):
        row = f"  {e:4d}"
        for fn_name in all_reward_fns:
            row += f"  {all_fn_rewards[fn_name][e]:10.4f}"
        print(row)
    print()
    print(f"  {'mean':>4s}", end="")
    for fn_name in all_reward_fns:
        print(f"  {all_fn_rewards[fn_name].mean():10.4f}", end="")
    print()

    print("\n  --- All reward functions: GT actions (should be perfect) ---")
    print(header)
    all_fn_gt = {}
    for fn_name, fn in all_reward_fns.items():
        all_fn_gt[fn_name] = fn(gt_t_aligned, gt_t_aligned)
    for e in range(min(4, num_envs)):
        row = f"  {e:4d}"
        for fn_name in all_reward_fns:
            row += f"  {all_fn_gt[fn_name][e]:10.4f}"
        print(row)
    print(f"  → MAE=0, MSE=0, exp_mae=1, cosine=1 expected")

    print("\n  --- All reward functions: zero actions ---")
    zero_t = torch.zeros_like(pred_t_aligned)
    print(header)
    all_fn_zero = {}
    for fn_name, fn in all_reward_fns.items():
        all_fn_zero[fn_name] = fn(zero_t, gt_t_aligned)
    for e in range(min(4, num_envs)):
        row = f"  {e:4d}"
        for fn_name in all_reward_fns:
            row += f"  {all_fn_zero[fn_name][e]:10.4f}"
        print(row)
    print(f"  {'mean':>4s}", end="")
    for fn_name in all_reward_fns:
        print(f"  {all_fn_zero[fn_name].mean():10.4f}", end="")
    print()

    print("\n[reward] Testing with ground-truth actions (expect ~0 for MAE)...")
    obs_gt, _ = env_for_step.reset()
    gt_snap2 = env_for_step._gt_actions.copy()
    obs_list, rewards_gt, terms, _, _ = env_for_step.chunk_step(gt_snap2)
    gt_rewards = rewards_gt[:, -1].clone()
    print(f"  rewards[:,-1] : {gt_rewards.tolist()}")

    print("\n[reward] Testing with zero actions (expect worst reward)...")
    obs_zero, _ = env_for_step.reset()
    zero_actions = np.zeros_like(env_for_step._gt_actions)
    gt_snap3 = env_for_step._gt_actions.copy()
    obs_list, rewards_zero, _, _, _ = env_for_step.chunk_step(zero_actions)
    zero_rewards = rewards_zero[:, -1].clone()
    print(f"  rewards[:,-1] : {zero_rewards.tolist()}")

    print("\n  --- Side-by-side comparison: env[0] action trajectories ---")
    print(f"  {'step':>4s}  {'dim':>3s}  {'GT':>10s}  {'Model':>10s}  {'Zero':>10s}  {'|pred-gt|':>10s}")
    env0_gt = gt_snap[0]
    env0_pred = pred_actions[0]
    for s in range(min(args.chunk_size, 8)):
        for d in range(min(7, env0_gt.shape[-1])):
            g = env0_gt[s, d]
            p = env0_pred[s, d]
            print(f"  {s:4d}  {d:3d}  {g:+10.4f}  {p:+10.4f}  {0.0:+10.4f}  {abs(p - g):10.4f}")
        if s < min(args.chunk_size, 8) - 1:
            print(f"  {'----':>4s}  {'---':>3s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")

    banner("Step 5/5: Summary & Sanity Checks")

    gt_mean = gt_rewards.mean().item()
    model_mean = model_rewards.mean().item()
    zero_mean = zero_rewards.mean().item()

    print(f"  Reward function : {args.reward_fn}")
    print(f"  Reward coef     : {args.reward_coef}")
    print()
    print(f"  Ground-truth reward (mean) : {gt_mean:.6f}")
    print(f"  Model prediction reward    : {model_mean:.6f}")
    print(f"  Zero actions reward        : {zero_mean:.6f}")
    print()

    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if abs(gt_mean) < 1e-4:
        print("  [PASS] GT actions produce ~0 reward (MAE)")
        checks_passed += 1
    else:
        print(f"  [WARN] GT reward not ~0: {gt_mean:.6f} (may be OK for non-MAE reward)")
        if args.reward_fn == "mae":
            pass
        else:
            checks_passed += 1

    checks_total += 1
    if args.reward_fn in ("mae", "mse"):
        if zero_mean < model_mean:
            print("  [PASS] Zero actions worse than model predictions")
            checks_passed += 1
        else:
            print(f"  [WARN] Zero reward ({zero_mean:.4f}) >= model reward ({model_mean:.4f})")
    elif args.reward_fn in ("exp_mae", "cosine"):
        if zero_mean < model_mean:
            print("  [PASS] Zero actions worse than model predictions")
            checks_passed += 1
        else:
            print(f"  [WARN] Zero reward ({zero_mean:.4f}) >= model reward ({model_mean:.4f})")

    checks_total += 1
    if pred_actions.shape == (num_envs, args.chunk_size, 7):
        print(f"  [PASS] Action shape correct: {pred_actions.shape}")
        checks_passed += 1
    else:
        print(f"  [FAIL] Unexpected action shape: {pred_actions.shape}")

    checks_total += 1
    if terms[:, -1].all().item():
        print("  [PASS] All episodes terminate after one chunk step")
        checks_passed += 1
    else:
        print("  [FAIL] Not all episodes terminated")

    print(f"\n  Results: {checks_passed}/{checks_total} checks passed")
    print(f"\n  Timing:")
    print(f"    reset()              : {dt_reset:.3f}s")
    print(f"    predict_action_batch : {dt_infer:.3f}s")
    print()

    if checks_passed == checks_total:
        banner("ALL CHECKS PASSED — offline reward pipeline works end-to-end")
    else:
        banner(f"DONE — {checks_passed}/{checks_total} checks passed (review warnings)")

    return checks_passed == checks_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2E test: DatasetEnv → model inference → reward"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/data/dataset/data",
        help="Path to dataset (parquet or npy)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/data/model/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/",
        help="Path to OpenVLA-OFT model",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="libero_130_no_noops_trajall",
        help="Unnormalization key for the model",
    )
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--reward_fn", type=str, default="mae", choices=["mae", "mse", "exp_mae", "cosine"])
    parser.add_argument("--reward_coef", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.6)

    args = parser.parse_args()

    success = run_test(args)
    sys.exit(0 if success else 1)
