from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tasks2_6_preflight_r1 as r1


def docker_exec_r2(name: str, script: str, *, check=True, timeout=None, capture=False):
    kwargs = {"text": True, "timeout": timeout}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    p = subprocess.run(
        [
            "docker", "exec",
            "-e", "HF_HUB_DISABLE_XET=1",
            "-w", "/testbed",
            name, "bash", "-lc", script,
        ],
        **kwargs,
    )
    if check and p.returncode != 0:
        msg = f"command failed rc={p.returncode}"
        if capture:
            msg += f"\nstdout:\n{(p.stdout or '')[-6000:]}\nstderr:\n{(p.stderr or '')[-6000:]}"
        raise RuntimeError(msg)
    return p


r1.docker_exec = docker_exec_r2

if __name__ == "__main__":
    r1.main()
