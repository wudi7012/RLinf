"""
Diagnose exactly how the subprocess dies: signal? OOM? segfault?
Also tests with num_envs=1 to eliminate multi-process interference.

Usage:
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python test_libero_signal.py [num_envs] [total_steps]
"""

import os
import sys
import signal
import time
import traceback
import multiprocessing

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf


def make_libero_cfg(num_envs=1):
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


def check_worker_status(env):
    """Check the status of all subprocess workers."""
    dead = []
    for i, w in enumerate(env.env.workers):
        proc = w.process
        if not proc.is_alive():
            exitcode = proc.exitcode
            if exitcode is not None and exitcode < 0:
                sig = -exitcode
                try:
                    sig_name = signal.Signals(sig).name
                except (ValueError, AttributeError):
                    sig_name = f"signal({sig})"
                dead.append((i, exitcode, sig_name))
            else:
                dead.append((i, exitcode, "exited"))
    return dead


def run_test(num_envs, total_steps):
    from rlinf.envs.libero.libero_env import LiberoEnv

    print(f"\n{'='*60}")
    print(f"Testing: num_envs={num_envs}, total_steps={total_steps}")
    print(f"{'='*60}")

    cfg = make_libero_cfg(num_envs=num_envs)
    env = LiberoEnv(
        cfg=cfg, num_envs=num_envs, seed_offset=0,
        total_num_processes=1, worker_info=None,
    )

    worker_pids = [w.process.pid for w in env.env.workers]
    print(f"  Subprocess PIDs: {worker_pids}")

    obs, infos = env.reset()
    print(f"  reset OK (15 steps done)")

    dead = check_worker_status(env)
    if dead:
        print(f"  WARNING: Workers already dead after reset: {dead}")
        return False

    for step in range(total_steps):
        try:
            actions = torch.randn(num_envs, 7) * 0.01
            env.step(actions)

            if step % 10 == 0:
                dead = check_worker_status(env)
                if dead:
                    print(f"  Step {step}: Workers died: {dead}")
                    return False
                print(f"  Step {step+1}/{total_steps} OK")

        except EOFError:
            print(f"\n  EOFError at step {step+1}!")
            dead = check_worker_status(env)
            if dead:
                for idx, exitcode, sig_name in dead:
                    print(f"  >>> Worker {idx} (pid={worker_pids[idx]}): exitcode={exitcode}, signal={sig_name}")
            else:
                print(f"  No dead workers found (race condition?)")
                time.sleep(0.5)
                dead = check_worker_status(env)
                if dead:
                    for idx, exitcode, sig_name in dead:
                        print(f"  >>> Worker {idx} (pid={worker_pids[idx]}): exitcode={exitcode}, signal={sig_name}")

            pid_pattern = "|".join(str(p) for p in worker_pids)
            print(f"\n  Check kernel logs: dmesg | grep -E '{pid_pattern}' | tail -10")
            return False

    print(f"  All {total_steps} steps OK")
    env.env.close()
    return True


if __name__ == "__main__":
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"PID: {os.getpid()}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    ok = run_test(num_envs=num_envs, total_steps=total_steps)

    if ok:
        print("\nPASSED")
    else:
        print("\nFAILED")
        print("\nPlease run on the server:")
        print("  dmesg | grep -i -E 'oom|kill|segfault|out.of.memory' | tail -30")
        sys.exit(1)
