"""
Isolated test: only one LiberoEnv, with detailed per-step monitoring.
Diagnoses whether the crash is caused by resource exhaustion over time.

Usage:
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python test_libero_isolated.py
"""

import os
import sys
import time
import subprocess
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf


def gpu_mem_usage():
    """Return (used_MB, total_MB) from nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().split("\n")[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return None, None


def shm_usage():
    """Return (used_MB, total_MB) of /dev/shm."""
    import shutil
    usage = shutil.disk_usage("/dev/shm")
    return usage.used // (1024 * 1024), usage.total // (1024 * 1024)


def make_libero_cfg(num_envs=16):
    return OmegaConf.create({
        "env_type": "libero",
        "task_suite_name": "libero_130",
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


def run_test(num_envs, total_chunks, chunk_size=8):
    from rlinf.envs.libero.libero_env import LiberoEnv

    print(f"Config: num_envs={num_envs}, chunk_size={chunk_size}, total_chunks={total_chunks}")
    gpu_used, gpu_total = gpu_mem_usage()
    shm_used, shm_total = shm_usage()
    print(f"  Before env creation: GPU={gpu_used}/{gpu_total}MB, /dev/shm={shm_used}/{shm_total}MB")

    cfg = make_libero_cfg(num_envs=num_envs)
    env = LiberoEnv(
        cfg=cfg, num_envs=num_envs, seed_offset=0,
        total_num_processes=1, worker_info=None,
    )

    gpu_used, _ = gpu_mem_usage()
    shm_used, _ = shm_usage()
    print(f"  After env creation: GPU={gpu_used}MB, /dev/shm={shm_used}MB")

    obs, infos = env.reset()

    gpu_used, _ = gpu_mem_usage()
    shm_used, _ = shm_usage()
    print(f"  After reset (15 steps): GPU={gpu_used}MB, /dev/shm={shm_used}MB")

    total_steps = 0
    for epoch in range(total_chunks):
        chunk_actions = torch.randn(num_envs, chunk_size, 7) * 0.01
        env.chunk_step(chunk_actions)
        total_steps += chunk_size

        if epoch % 5 == 0 or epoch == total_chunks - 1:
            gpu_used, _ = gpu_mem_usage()
            shm_used, _ = shm_usage()
            print(f"  chunk {epoch+1}/{total_chunks} OK (total_steps={total_steps}), GPU={gpu_used}MB, /dev/shm={shm_used}MB")

    print(f"  All {total_chunks} chunks completed ({total_steps} total steps)")

    gpu_used, _ = gpu_mem_usage()
    print(f"  Before close: GPU={gpu_used}MB")

    env.env.close()

    gpu_used, _ = gpu_mem_usage()
    print(f"  After close: GPU={gpu_used}MB")


if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print()

    # Parse args: python test_libero_isolated.py [num_envs] [total_chunks]
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    total_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    try:
        run_test(num_envs=num_envs, total_chunks=total_chunks)
        print("\nPASSED")
    except Exception:
        traceback.print_exc()
        print("\nFAILED")
        sys.exit(1)
