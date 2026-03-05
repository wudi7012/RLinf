"""
End-to-end test that closely mirrors the actual training pipeline.

Tests:
  1. LiberoEnv (with ReconfigureSubprocEnv) at training scale (16 envs)
  2. chunk_step (the exact call path that fails in training)
  3. Nested process: simulate Ray actor by running everything inside a spawn child
  4. Full scale: multiple "workers" each with their own LiberoEnv (like 4 EnvWorkers)

Usage:
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python test_libero_e2e.py
"""

import os
import sys
import time
import traceback
import multiprocessing

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf


def make_libero_cfg(num_envs=16, task_suite_name="libero_130"):
    """Create a minimal OmegaConf that LiberoEnv expects."""
    cfg = OmegaConf.create({
        "env_type": "libero",
        "task_suite_name": task_suite_name,
        "total_num_envs": num_envs,
        "auto_reset": False,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 512,
        "max_episode_steps": 512,
        "use_rel_reward": True,
        "reward_coef": 5.0,
        "reset_gripper_open": False,
        "is_eval": False,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "use_ordered_reset_state_ids": True,
        "specific_reset_id": None,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "init_params": {
            "camera_heights": 256,
            "camera_widths": 256,
        },
    })
    return cfg


# ---------------------------------------------------------------------------
# Test 1: LiberoEnv with ReconfigureSubprocEnv — same as actual training
# ---------------------------------------------------------------------------
def test_libero_env(num_envs=16):
    print("=" * 60)
    print(f"[Test 1] LiberoEnv with {num_envs} envs (ReconfigureSubprocEnv)")
    print("=" * 60)
    from rlinf.envs.libero.libero_env import LiberoEnv

    cfg = make_libero_cfg(num_envs=num_envs)
    t0 = time.time()
    env = LiberoEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    print(f"  LiberoEnv created in {time.time()-t0:.1f}s")

    t0 = time.time()
    obs, infos = env.reset()
    print(f"  reset OK in {time.time()-t0:.1f}s, obs keys={list(obs.keys())}")

    t0 = time.time()
    actions = torch.zeros(num_envs, 7)
    result = env.step(actions)
    print(f"  step OK in {time.time()-t0:.1f}s, result has {len(result)} elements")

    for i in range(5):
        actions = torch.randn(num_envs, 7) * 0.01
        result = env.step(actions)
    print(f"  5 more steps OK")

    env.env.close()
    print("[Test 1] PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 2: chunk_step — exact call path that triggers the error in training
# ---------------------------------------------------------------------------
def test_chunk_step(num_envs=16, chunk_size=8):
    print("=" * 60)
    print(f"[Test 2] chunk_step: {num_envs} envs, chunk_size={chunk_size}")
    print("=" * 60)
    from rlinf.envs.libero.libero_env import LiberoEnv

    cfg = make_libero_cfg(num_envs=num_envs)
    env = LiberoEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    obs, infos = env.reset()
    print(f"  reset OK")

    t0 = time.time()
    chunk_actions = torch.randn(num_envs, chunk_size, 7) * 0.01
    result = env.chunk_step(chunk_actions)
    obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = result
    print(f"  chunk_step OK in {time.time()-t0:.1f}s")
    print(f"    obs_list len={len(obs_list)}, chunk_rewards shape={chunk_rewards.shape}")

    for epoch in range(3):
        chunk_actions = torch.randn(num_envs, chunk_size, 7) * 0.01
        result = env.chunk_step(chunk_actions)
        print(f"  chunk_step epoch {epoch+1} OK")

    env.env.close()
    print("[Test 2] PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 3: Nested process — simulate what happens inside a Ray actor
# Ray actors are separate processes; LiberoEnv is created inside them.
# This tests: main process -> spawn child -> LiberoEnv -> spawn grandchildren
# ---------------------------------------------------------------------------
def _nested_worker(result_queue, num_envs):
    """Runs inside a spawned process, simulating a Ray actor."""
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        from rlinf.envs.libero.libero_env import LiberoEnv

        cfg = make_libero_cfg(num_envs=num_envs)
        env = LiberoEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        obs, infos = env.reset()

        chunk_actions = torch.randn(num_envs, 8, 7) * 0.01
        result = env.chunk_step(chunk_actions)

        for i in range(3):
            chunk_actions = torch.randn(num_envs, 8, 7) * 0.01
            env.chunk_step(chunk_actions)

        env.env.close()
        result_queue.put(("OK", None))
    except Exception:
        result_queue.put(("FAIL", traceback.format_exc()))


def test_nested_process(num_envs=16):
    print("=" * 60)
    print(f"[Test 3] Nested process: spawn -> LiberoEnv({num_envs}) -> spawn children")
    print("=" * 60)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_nested_worker, args=(q, num_envs), daemon=False)
    p.start()
    print(f"  Outer process spawned (pid={p.pid}), waiting...")

    p.join(timeout=300)
    if p.is_alive():
        p.kill()
        print("  TIMEOUT after 300s")
        print("[Test 3] FAILED\n")
        return False

    if p.exitcode != 0:
        print(f"  Child exited with code {p.exitcode}")
        if p.exitcode and p.exitcode < 0:
            import signal
            sig = -p.exitcode
            sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
            print(f"  Killed by signal: {sig_name} ({sig})")
        if not q.empty():
            status, tb = q.get_nowait()
            if tb:
                print(f"  Exception:\n{tb}")
        print("[Test 3] FAILED\n")
        return False

    status, tb = q.get_nowait()
    if status == "OK":
        print("  Nested process completed successfully")
        print("[Test 3] PASSED\n")
        return True
    else:
        print(f"  Exception:\n{tb}")
        print("[Test 3] FAILED\n")
        return False


# ---------------------------------------------------------------------------
# Test 4: Multiple concurrent workers — simulate 4 EnvWorkers each owning
# a LiberoEnv with num_envs/4 envs (actual training topology)
# ---------------------------------------------------------------------------
def _multi_worker(result_queue, worker_id, num_envs, seed_offset, total_workers):
    """Simulates one EnvWorker Ray actor."""
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        from rlinf.envs.libero.libero_env import LiberoEnv

        cfg = make_libero_cfg(num_envs=num_envs)
        env = LiberoEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_workers,
            worker_info=None,
        )
        obs, infos = env.reset()

        for i in range(4):
            chunk_actions = torch.randn(num_envs, 8, 7) * 0.01
            env.chunk_step(chunk_actions)

        env.env.close()
        result_queue.put((worker_id, "OK", None))
    except Exception:
        result_queue.put((worker_id, "FAIL", traceback.format_exc()))


def test_multiple_workers(num_workers=4, total_num_envs=16):
    envs_per_worker = total_num_envs // num_workers
    print("=" * 60)
    print(f"[Test 4] {num_workers} concurrent workers, {envs_per_worker} envs each")
    print("=" * 60)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()

    processes = []
    for i in range(num_workers):
        p = ctx.Process(
            target=_multi_worker,
            args=(q, i, envs_per_worker, i, num_workers),
            daemon=False,
        )
        p.start()
        processes.append(p)
        print(f"  Worker {i} spawned (pid={p.pid})")

    t0 = time.time()
    for p in processes:
        remaining = max(1, 600 - (time.time() - t0))
        p.join(timeout=remaining)

    all_ok = True
    results = {}
    while not q.empty():
        worker_id, status, tb = q.get_nowait()
        results[worker_id] = (status, tb)

    for i, p in enumerate(processes):
        if p.is_alive():
            print(f"  Worker {i} TIMEOUT, killing")
            p.kill()
            all_ok = False
        elif p.exitcode != 0:
            print(f"  Worker {i} exited with code {p.exitcode}")
            if p.exitcode and p.exitcode < 0:
                import signal
                sig = -p.exitcode
                sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
                print(f"    Killed by signal: {sig_name}")
            all_ok = False
        elif i in results:
            status, tb = results[i]
            if status == "OK":
                print(f"  Worker {i} OK")
            else:
                print(f"  Worker {i} FAILED:\n{tb}")
                all_ok = False
        else:
            print(f"  Worker {i} exited (code={p.exitcode}) but no result in queue")
            all_ok = False

    if all_ok:
        print(f"[Test 4] PASSED\n")
    else:
        print(f"[Test 4] FAILED\n")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"PID: {os.getpid()}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")
    print()

    passed = []
    failed = []

    for name, fn in [
        ("Test 1: LiberoEnv 16 envs", lambda: test_libero_env(num_envs=16)),
        ("Test 2: chunk_step", lambda: test_chunk_step(num_envs=16, chunk_size=8)),
        ("Test 3: Nested process (simulates Ray actor)", lambda: test_nested_process(num_envs=16)),
        ("Test 4: 4 concurrent workers x 4 envs", lambda: test_multiple_workers(num_workers=4, total_num_envs=16)),
    ]:
        try:
            if fn():
                passed.append(name)
            else:
                failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    for t in passed:
        print(f"  [PASS] {t}")
    for t in failed:
        print(f"  [FAIL] {t}")
    print("=" * 60)

    if failed:
        print("\nIf Test 1/2 pass but Test 3/4 fail, the issue is nested/multi-process.")
        print("If all pass here but training fails, the issue is Ray-specific.")
        sys.exit(1)
    else:
        print("\nAll tests passed. The env layer is functional.")
        print("If training still fails, the issue is likely Ray actor lifecycle or resource pressure at full training scale.")
