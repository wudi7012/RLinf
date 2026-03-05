"""
Minimal test script to diagnose LIBERO subprocess env failures.
Tests each layer independently: single-process env, spawn subprocess, 
multiprocessing Pipe communication, and the full SubprocVectorEnv.

Usage:
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python test_libero_subprocess.py
"""

import os
import sys
import traceback
import multiprocessing
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def get_env_fn():
    """Return a callable that creates one LIBERO OffScreenRenderEnv."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark("libero_90")()
    task = benchmark.get_task(0)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    params = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    def _make():
        env = OffScreenRenderEnv(**params)
        env.seed(0)
        return env

    return _make


# ---------------------------------------------------------------------------
# Test 1: Single-process env creation + step
# ---------------------------------------------------------------------------
def test_single_process():
    print("=" * 60)
    print("[Test 1] Single-process: create env, reset, step")
    print("=" * 60)
    make_env = get_env_fn()
    env = make_env()
    print(f"  env created OK (type={type(env).__name__})")

    obs = env.reset()
    print(f"  reset OK, obs type={type(obs)}, keys={list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

    action = np.zeros(7)
    result = env.step(action)
    print(f"  step OK, result has {len(result)} elements")

    for _ in range(5):
        result = env.step(np.random.randn(7) * 0.01)
    print(f"  5 more steps OK")

    env.close()
    print("[Test 1] PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: spawn subprocess — can MuJoCo/EGL render inside a child process?
# ---------------------------------------------------------------------------
def _child_worker_simple(result_queue):
    """Run in a spawned child process."""
    try:
        make_env = get_env_fn()
        env = make_env()
        obs = env.reset()
        for _ in range(3):
            env.step(np.zeros(7))
        env.close()
        result_queue.put(("OK", None))
    except Exception:
        result_queue.put(("FAIL", traceback.format_exc()))


def test_spawn_subprocess():
    print("=" * 60)
    print("[Test 2] Spawn subprocess: create env + step in child process")
    print("=" * 60)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_worker_simple, args=(q,), daemon=True)
    p.start()
    p.join(timeout=120)

    if p.is_alive():
        p.kill()
        print("  TIMEOUT: child process hung for 120s, killed it")
        print("[Test 2] FAILED\n")
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
                print(f"  Exception in child:\n{tb}")
        print("[Test 2] FAILED\n")
        return False

    status, tb = q.get_nowait()
    if status == "OK":
        print("  Child process ran env.reset() + env.step() successfully")
        print("[Test 2] PASSED\n")
        return True
    else:
        print(f"  Exception in child:\n{tb}")
        print("[Test 2] FAILED\n")
        return False


# ---------------------------------------------------------------------------
# Test 3: Pipe communication — the exact pattern used by SubprocVectorEnv
# ---------------------------------------------------------------------------
def _child_worker_pipe(parent_conn, child_conn):
    """Mimics the _worker function in rlinf/envs/libero/venv.py."""
    parent_conn.close()
    try:
        make_env = get_env_fn()
        env = make_env()
        while True:
            try:
                cmd, data = child_conn.recv()
            except EOFError:
                break
            if cmd == "reset":
                obs = env.reset()
                child_conn.send("reset_ok")
            elif cmd == "step":
                result = env.step(data)
                child_conn.send(("step_ok", len(result)))
            elif cmd == "close":
                env.close()
                child_conn.send("closed")
                break
        child_conn.close()
    except Exception:
        import sys
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        child_conn.close()


def test_pipe_communication():
    print("=" * 60)
    print("[Test 3] Pipe communication: parent <-> child (spawn)")
    print("=" * 60)
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_child_worker_pipe, args=(parent_conn, child_conn), daemon=True)
    p.start()
    child_conn.close()

    try:
        print("  Sending reset...")
        parent_conn.send(("reset", None))
        reply = parent_conn.recv()
        print(f"  Got reply: {reply}")

        print("  Sending step (zero action)...")
        parent_conn.send(("step", np.zeros(7)))
        reply = parent_conn.recv()
        print(f"  Got reply: {reply}")

        for i in range(5):
            parent_conn.send(("step", np.random.randn(7) * 0.01))
            reply = parent_conn.recv()
        print(f"  5 more step-recv cycles OK")

        parent_conn.send(("close", None))
        reply = parent_conn.recv()
        print(f"  Close reply: {reply}")

        p.join(timeout=30)
        print(f"  Child exit code: {p.exitcode}")
        print("[Test 3] PASSED\n")
        return True

    except EOFError:
        print(f"  EOFError! Child process died. exit code: {p.exitcode}")
        if p.exitcode and p.exitcode < 0:
            import signal
            sig = -p.exitcode
            sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
            print(f"  Killed by signal: {sig_name} ({sig})")
        print("[Test 3] FAILED\n")
        return False
    except Exception:
        traceback.print_exc()
        print("[Test 3] FAILED\n")
        return False


# ---------------------------------------------------------------------------
# Test 4: Multiple subprocesses (simulates total_num_envs=4)
# ---------------------------------------------------------------------------
def test_multiple_subprocesses(n=4):
    print("=" * 60)
    print(f"[Test 4] Multiple subprocesses: {n} envs via Pipe (spawn)")
    print("=" * 60)
    ctx = multiprocessing.get_context("spawn")

    workers = []
    for i in range(n):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_child_worker_pipe, args=(parent_conn, child_conn), daemon=True)
        p.start()
        child_conn.close()
        workers.append((parent_conn, p, i))
        print(f"  Worker {i} spawned (pid={p.pid})")

    try:
        for conn, p, idx in workers:
            conn.send(("reset", None))
        for conn, p, idx in workers:
            reply = conn.recv()
            print(f"  Worker {idx} reset: {reply}")

        for step in range(3):
            for conn, p, idx in workers:
                conn.send(("step", np.random.randn(7) * 0.01))
            for conn, p, idx in workers:
                reply = conn.recv()
            print(f"  Step {step} done for all {n} workers")

        for conn, p, idx in workers:
            conn.send(("close", None))
            conn.recv()
        for conn, p, idx in workers:
            p.join(timeout=30)
            print(f"  Worker {idx} exited with code {p.exitcode}")

        print(f"[Test 4] PASSED\n")
        return True

    except EOFError as e:
        print(f"  EOFError! A child process died unexpectedly.")
        for conn, p, idx in workers:
            if not p.is_alive() and p.exitcode != 0:
                print(f"  Worker {idx} (pid={p.pid}) exit code: {p.exitcode}")
                if p.exitcode and p.exitcode < 0:
                    import signal
                    sig = -p.exitcode
                    sig_name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
                    print(f"    Killed by signal: {sig_name}")
        print(f"[Test 4] FAILED\n")
        return False
    except Exception:
        traceback.print_exc()
        print(f"[Test 4] FAILED\n")
        return False


# ---------------------------------------------------------------------------
# Test 5: System resource checks
# ---------------------------------------------------------------------------
def test_system_resources():
    print("=" * 60)
    print("[Test 5] System resource checks")
    print("=" * 60)

    import shutil
    import resource

    shm_usage = shutil.disk_usage("/dev/shm")
    print(f"  /dev/shm: total={shm_usage.total // (1024*1024)}MB, "
          f"used={shm_usage.used // (1024*1024)}MB, "
          f"free={shm_usage.free // (1024*1024)}MB")
    if shm_usage.total < 512 * 1024 * 1024:
        print("  WARNING: /dev/shm < 512MB, may cause subprocess crashes!")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"  File descriptors: soft={soft}, hard={hard}")

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"  Max processes: soft={soft}, hard={hard}")

    mem = shutil.disk_usage("/")
    print(f"  Disk /: total={mem.total // (1024**3)}GB, free={mem.free // (1024**3)}GB")

    try:
        with open("/proc/self/cgroup") as f:
            cgroup = f.read()
        mem_limit_paths = [
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory.max",
        ]
        for path in mem_limit_paths:
            if os.path.exists(path):
                with open(path) as f:
                    val = f.read().strip()
                if val != "max" and val.isdigit():
                    print(f"  Container memory limit: {int(val) // (1024**3)}GB ({path})")
                else:
                    print(f"  Container memory limit: {val} ({path})")
                break
    except Exception:
        pass

    print("[Test 5] DONE\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"PID: {os.getpid()}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")
    print()

    test_system_resources()

    try:
        test_single_process()
    except Exception:
        traceback.print_exc()
        print("[Test 1] FAILED\n")
        sys.exit(1)

    if not test_spawn_subprocess():
        print(">>> Test 2 failed: MuJoCo/EGL cannot work in spawn subprocess.")
        print(">>> This is the root cause of your EOFError.")
        sys.exit(1)

    if not test_pipe_communication():
        print(">>> Test 3 failed: Pipe communication with env subprocess broken.")
        sys.exit(1)

    if not test_multiple_subprocesses(n=4):
        print(">>> Test 4 failed: Multiple subprocesses crash.")
        print(">>> Likely resource exhaustion (GPU mem, /dev/shm, or cgroup limit).")
        sys.exit(1)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
