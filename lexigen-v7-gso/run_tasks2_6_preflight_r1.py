from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


TIME_RE = re.compile(r"Execution time:\s*([0-9.eE+-]+)s")


def run(cmd, *, check=True, capture=False, timeout=None, cwd=None):
    kwargs = {"text": True, "cwd": cwd, "timeout": timeout}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    p = subprocess.run(cmd, **kwargs)
    if check and p.returncode != 0:
        msg = f"command failed rc={p.returncode}: {cmd}"
        if capture:
            msg += f"\nstdout:\n{(p.stdout or '')[-6000:]}\nstderr:\n{(p.stderr or '')[-6000:]}"
        raise RuntimeError(msg)
    return p


def docker_exec(name: str, script: str, *, check=True, timeout=None, capture=False):
    return run(
        [
            "docker", "exec",
            "-e", "HF_HUB_DISABLE_XET=1",
            "-e", "HF_HUB_ENABLE_HF_TRANSFER=1",
            "-w", "/testbed",
            name, "bash", "-lc", script,
        ],
        check=check,
        capture=capture,
        timeout=timeout,
    )


def parse_time(text: str) -> float:
    vals = [float(x) for x in TIME_RE.findall(text)]
    if not vals:
        raise RuntimeError(f"No execution time found in: {text[-2000:]}")
    return vals[-1]


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", x)


def create_container(image: str, name: str, cache_root: Path) -> str:
    hf = cache_root / "hf"
    uv = cache_root / "uv"
    hf.mkdir(parents=True, exist_ok=True)
    uv.mkdir(parents=True, exist_ok=True)
    run([
        "docker", "create", "--name", name,
        "-v", f"{hf.resolve()}:/root/.cache/huggingface",
        "-v", f"{uv.resolve()}:/root/.cache/uv",
        image, "sleep", "infinity",
    ])
    run(["docker", "start", name])
    return name


def cleanup_container(name: str) -> None:
    run(["docker", "rm", "-f", name], check=False, capture=True)


def copy_tests(container: str, tests_dir: Path, indexes: list[int]) -> None:
    for idx in indexes:
        run(["docker", "cp", str(tests_dir / f"gso_test_{idx}.py"), f"{container}:/gso_test_{idx}.py"])


def copy_refs_to_container(container: str, refs_dir: Path) -> None:
    for p in sorted(refs_dir.iterdir()):
        if p.is_file():
            run(["docker", "cp", str(p), f"{container}:/testbed/{p.name}"])


def baseline_reference(
    *, image: str, task: int, indexes: list[int], tests_dir: Path,
    refs_dir: Path, cache_root: Path, timeout_per_test: int,
) -> dict[str, float]:
    name = safe_name(f"lexigen-v7-preflight-t{task}-baseline-{os.getpid()}")
    refs_dir.mkdir(parents=True, exist_ok=True)
    create_container(image, name, cache_root)
    try:
        copy_tests(name, tests_dir, indexes)
        times: dict[str, float] = {}
        for idx in indexes:
            out = f"/tmp/base_{idx}.txt"
            cmd = (
                "source .venv/bin/activate && "
                f"python /gso_test_{idx}.py {out} --reference --file_prefix gso_{idx}"
            )
            p = docker_exec(name, cmd, check=False, timeout=timeout_per_test, capture=True)
            if p.returncode != 0:
                raise RuntimeError(
                    f"baseline test {idx} failed rc={p.returncode}\nstdout:\n{(p.stdout or '')[-6000:]}\nstderr:\n{(p.stderr or '')[-6000:]}"
                )
            cat = docker_exec(name, f"cat {out}", capture=True)
            times[str(idx)] = parse_time(cat.stdout)
            found = docker_exec(
                name,
                f"find /testbed -maxdepth 1 -type f -name 'gso_{idx}*' -printf '%f\\n' | sort",
                capture=True,
            ).stdout.splitlines()
            if not found:
                raise RuntimeError(f"baseline test {idx} produced no gso_{idx}* reference files")
            for fn in found:
                run(["docker", "cp", f"{name}:/testbed/{fn}", str(refs_dir / fn)])
        return times
    finally:
        cleanup_container(name)


def build_patch(
    *, task: int, candidate: str, target: str, container: str,
    work_root: Path, output_root: Path,
) -> dict:
    root = work_root / candidate
    dest = root / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "cp", f"{container}:/testbed/{target}", str(dest)])
    out = output_root / candidate
    out.mkdir(parents=True, exist_ok=True)
    if task == 2:
        cmd = [
            sys.executable, "lexigen-v7-gso/build_task2_candidate_r3.py",
            "--candidate", candidate, "--root", str(root), "--output", str(out),
        ]
    else:
        cmd = [
            sys.executable, "lexigen-v7-gso/build_candidate_r1.py",
            "--task", str(task), "--candidate", candidate,
            "--root", str(root), "--output", str(out),
        ]
    run(cmd)
    report_path = out / f"task{task}-{candidate}.json"
    report = json.loads(report_path.read_text())
    report["patch_path"] = str(out / f"task{task}-{candidate}.patch")
    return report


def evaluate_candidate(
    *, task: int, candidate: str, image: str, target: str,
    indexes: list[int], tests_dir: Path, refs_dir: Path,
    install_commands: list[str], expected_patch_sha: str,
    baseline_times: dict[str, float], cache_root: Path,
    work_root: Path, patch_root: Path, timeout_per_test: int,
) -> dict:
    name = safe_name(f"lexigen-v7-preflight-t{task}-{candidate}-{os.getpid()}")
    result = {
        "candidate": candidate,
        "correct": False,
        "install_ok": False,
        "tests_passed": 0,
        "test_count": len(indexes),
        "times": {},
        "speedups": {},
        "harmonic_speedup": 0.0,
        "minimum_speedup": 0.0,
        "error": None,
    }
    create_container(image, name, cache_root)
    try:
        copy_tests(name, tests_dir, indexes)
        report = build_patch(
            task=task, candidate=candidate, target=target, container=name,
            work_root=work_root, output_root=patch_root,
        )
        result["patch_sha256"] = report["patch_sha256"]
        if report["patch_sha256"] != expected_patch_sha:
            raise RuntimeError(
                f"patch hash drift for T{task} {candidate}: {report['patch_sha256']} != {expected_patch_sha}"
            )
        run(["docker", "cp", report["patch_path"], f"{name}:/tmp/patch.diff"])
        apply_cmd = (
            "git apply --verbose "
            "--exclude='.venv/*' --exclude='.git/*' --exclude='__pycache__/*' "
            "--exclude='*.egg-info/*' --exclude='*.json' --exclude='*.txt' "
            "--exclude='*.csv' --exclude='*.log' --exclude='*.pkl' /tmp/patch.diff"
        )
        ap = docker_exec(name, apply_cmd, check=False, timeout=120, capture=True)
        if ap.returncode != 0:
            raise RuntimeError(
                f"patch apply failed rc={ap.returncode}\nstdout:\n{(ap.stdout or '')[-4000:]}\nstderr:\n{(ap.stderr or '')[-4000:]}"
            )

        install_script = [
            "set -euo pipefail",
            "echo 'setuptools<82' > /tmp/uv_build_constraints.txt",
            "export UV_BUILD_CONSTRAINT=/tmp/uv_build_constraints.txt",
            "export HF_HUB_DISABLE_XET=1",
        ] + install_commands
        inst = docker_exec(
            name, "\n".join(install_script), check=False,
            timeout=2400 if task in {2,3,4} else 1800, capture=True,
        )
        if inst.returncode != 0:
            raise RuntimeError(
                f"install failed rc={inst.returncode}\nstdout:\n{(inst.stdout or '')[-7000:]}\nstderr:\n{(inst.stderr or '')[-7000:]}"
            )
        result["install_ok"] = True
        copy_refs_to_container(name, refs_dir)

        candidate_times = []
        speedups = []
        for idx in indexes:
            out = f"/tmp/result_{idx}.txt"
            cmd = (
                "source .venv/bin/activate && "
                f"python /gso_test_{idx}.py {out} --eqcheck --file_prefix gso_{idx}"
            )
            p = docker_exec(name, cmd, check=False, timeout=timeout_per_test, capture=True)
            if p.returncode != 0:
                raise RuntimeError(
                    f"eqcheck test {idx} failed rc={p.returncode}\nstdout:\n{(p.stdout or '')[-6000:]}\nstderr:\n{(p.stderr or '')[-6000:]}"
                )
            cat = docker_exec(name, f"cat {out}", capture=True)
            t = parse_time(cat.stdout)
            b = float(baseline_times[str(idx)])
            sp = b / t if t > 0 else 0.0
            result["times"][str(idx)] = t
            result["speedups"][str(idx)] = sp
            result["tests_passed"] += 1
            candidate_times.append(t)
            speedups.append(sp)
        result["correct"] = result["tests_passed"] == len(indexes)
        result["harmonic_speedup"] = harmonic(speedups) if result["correct"] else 0.0
        result["minimum_speedup"] = min(speedups) if speedups and result["correct"] else 0.0
    except Exception as exc:
        result["error"] = str(exc)[-12000:]
    finally:
        cleanup_container(name)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--execmeta", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    selection = json.loads(Path("lexigen-v7-gso/PREFLIGHT_SELECTION_R1.json").read_text())
    plan = json.loads(Path("lexigen-v7-gso/PREFLIGHT_PLAN_R1.json").read_text())
    hashes = json.loads(Path("lexigen-v7-gso/PREFLIGHT_TEST_HASH_LOCK_R1.json").read_text())
    materialized = json.loads(Path("lexigen-v7-gso/TASKS2_6_MATERIALIZATION_COMPLETE_R1.json").read_text())
    spec = json.loads((args.spec / "task-spec.json").read_text())
    execmeta = json.loads(args.execmeta.read_text())

    tkey = str(args.task)
    if spec["instance_id"] != args.instance_id or hashes["tasks"][tkey]["instance_id"] != args.instance_id:
        raise SystemExit("instance identity mismatch")
    indexes = list(plan["representative_test_indexes"][tkey])
    if indexes != list(hashes["tasks"][tkey]["indexes"]):
        raise SystemExit("representative-test index lock mismatch")

    tests_dir = args.output / "tests"
    refs_dir = args.output / "refs"
    work_root = args.output / "work"
    patch_root = args.output / "patches"
    cache_root = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / f"lexigen-v7-t{args.task}-cache"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for pos, idx in enumerate(indexes):
        text = spec["tests"][idx]
        got = hashlib.sha256(text.encode()).hexdigest()
        expected = hashes["tasks"][tkey]["sha256"][pos]
        if got != expected:
            raise SystemExit(f"test hash mismatch T{args.task} index {idx}: {got}")
        (tests_dir / f"gso_test_{idx}.py").write_text(text, encoding="utf-8")

    rows = [r for r in execmeta["records"] if r["instance_id"] == args.instance_id]
    if len(rows) != 1:
        raise SystemExit("execution metadata identity mismatch")
    install_commands = list(rows[0]["install_commands"])
    if execmeta.get("expert_opt_commit_accessed") or execmeta.get("expert_diff_accessed") or execmeta.get("hints_accessed"):
        raise SystemExit("forbidden expert metadata in execution projection")

    timeout_per_test = 1800 if args.task == 2 else 900
    args.output.mkdir(parents=True, exist_ok=True)
    campaign_credit = args.task != 5
    result = {
        "campaign": "LEXIGEN V7 GSO Real Causal-Transfer Pilot R1",
        "stage": "tasks2_6_preflight_r1",
        "task": args.task,
        "instance_id": args.instance_id,
        "campaign_credit_eligible": campaign_credit,
        "representative_test_indexes": indexes,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "candidate_results": [],
    }

    try:
        baseline_times = baseline_reference(
            image=args.image, task=args.task, indexes=indexes, tests_dir=tests_dir,
            refs_dir=refs_dir, cache_root=cache_root, timeout_per_test=timeout_per_test,
        )
        result["baseline_times"] = baseline_times
    except Exception as exc:
        result["status"] = "infrastructure_baseline_failure"
        result["error"] = str(exc)[-12000:]
        (args.output / "PREFLIGHT_RESULT_R1.json").write_text(json.dumps(result, indent=2) + "\n")
        raise

    sel = selection["tasks"][tkey]
    arms = ["v7_full", "v7_no_library", "v7_random_library"]
    candidate_arm = {}
    candidates = []
    for arm in arms:
        for cid in sel[arm]:
            candidate_arm[cid] = arm
            candidates.append(cid)

    for cid in candidates:
        expected = materialized["tasks"][tkey]["patch_sha256"][cid]
        rec = evaluate_candidate(
            task=args.task, candidate=cid, image=args.image, target=args.target,
            indexes=indexes, tests_dir=tests_dir, refs_dir=refs_dir,
            install_commands=install_commands, expected_patch_sha=expected,
            baseline_times=baseline_times, cache_root=cache_root,
            work_root=work_root, patch_root=patch_root,
            timeout_per_test=timeout_per_test,
        )
        rec["arm"] = candidate_arm[cid]
        result["candidate_results"].append(rec)
        (args.output / "PREFLIGHT_RESULT_R1.json").write_text(json.dumps(result, indent=2) + "\n")

    winners = {}
    for arm in arms:
        valid = [r for r in result["candidate_results"] if r["arm"] == arm and r["correct"]]
        valid.sort(key=lambda r: (-r["harmonic_speedup"], r["candidate"]))
        winners[arm] = valid[0]["candidate"] if valid else None
    result["winner_by_arm"] = winners
    result["winner_patch_sha256"] = {
        arm: next((r.get("patch_sha256") for r in result["candidate_results"] if r["candidate"] == cid), None)
        for arm, cid in winners.items()
    }
    full_hash = result["winner_patch_sha256"].get("v7_full")
    control_hashes = {
        result["winner_patch_sha256"].get("v7_no_library"),
        result["winner_patch_sha256"].get("v7_random_library"),
    }
    control_hashes.discard(None)
    result["full_winner_patch_equivalent_to_control_winner"] = full_hash in control_hashes if full_hash else False
    result["status"] = "completed"
    (args.output / "PREFLIGHT_RESULT_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "task": args.task,
        "status": result["status"],
        "winner_by_arm": winners,
        "winner_patch_sha256": result["winner_patch_sha256"],
        "full_winner_patch_equivalent_to_control_winner": result["full_winner_patch_equivalent_to_control_winner"],
        "campaign_credit_eligible": campaign_credit,
    }, indent=2))


if __name__ == "__main__":
    main()
