from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TIME_RE = re.compile(r"Execution time:\s*([0-9.eE+-]+)s")
EXPECTED_INSTANCE = "huggingface__datasets-ef3b5dd"
EXPECTED_IMAGE = "slimshetty/gso:gso.eval.x86_64.huggingface__datasets-ef3b5dd"
EXPECTED_TARGET = "src/datasets/load.py"
EXPECTED_BASE_BLOB = "13562ec82b01898334b2eaa455f4ce38bb7176da"
EXPECTED_CONTRACT_SHA256 = "f08f131a46c9a4d6a490e4a997f2c5045b4af67013bc705845e749ec63fb3e17"
EXPECTED_TEST_SHA256 = [
    "f00856f7f357c40a387fa978b3e228128f3eb8295ebfa768a12deec27c4affc5",
    "19468d8279123b5e783b1bc01915f9b1ef1f1a24df63578b9b99cdfba0c51dc8",
    "1ca37c818b7ad288d9a75b6d076e34aae3aa0bbb37214a04b54454c8ad45eb5e",
    "36235a3300dfc11a182f0f1a3ee375bc272141c27202f3297f563abfd94a666f",
    "852033ade92bb29704b597170473b3450f06441a1dcb644e2a73eff6737c96fb",
]
EXPECTED_PATCH = {
    "F1": ("26dd2fb85ae7aa9258f239a0b41747dfb22422d8a4f8d0a35629adb174b3358f", "f7be1e0efeba1b94c4042eec086395fbb8d54a983b8b07e1acd8ed5ed7ab1429"),
    "F2": ("eaac86b8a2fe2e38c312e395c20d70f5dcb5e4896fa1bd2b75cd6dab42abc8d6", "9f2379a662962c0cda2115670b0b5589ade5736baab11cfcaa8227805631a0aa"),
    "F4": ("2ac2cc8cd9c5f6053a5e00db241b5c552fc505f2a4e9c3bc917f2caa8d280699", "856c456a66f29d057831d76fbbdbc23df36d97b678b7e43a86fbe691788f76b8"),
    "N1": ("0c30eab2b3e2cab2c59c181d19834bb82ee74ee29ae41b03993da197a59b5255", "d4703144b1676d0994b54e586cb89e47f8e1e647dcda01b2e9662214708643ea"),
    "N3": ("3989a4ee72cb8a52f0e41abc94ddd190b7e7f0d2b94517f7d9927dd91710d322", "9f37ff47d2475de9fa9c256f7d5658a0bdd5b4f155065375d870e6823aa06956"),
    "N6": ("bf2df02102d28cb25320e8ab86c4ecd8091f789966dd0b8e70bdb1788b85dfec", "8b67d4fd2f34ac4541799d7f9307c33ac85fccd5b9e8aac61be9afb46db0e0db"),
    "R1": ("17eeb015a2991c8b0612259a577c7121db37b55a7fadd70e9a703628b62dac8e", "77960f9f0df6f4b024494daea22820db8fde2e8fabce328ac985a214459f7519"),
    "R3": ("9f3035a8148179f18be025ea6dc1306afe73aeeb57ea9915915e9d99d51bbe5a", "9970f84031d9e8c0f329ed90296d63015d06b9dee6c1a98ac551cef2d8a76a66"),
    "R6": ("8a0978096ca4a41e3a7680ef7faf44cf03e1e044cb74875dbc479c9342c3b9fb", "6ced4292341ebc551ea5eea2af39b7c23575ae7108c8ce55903d6f643ad0fc06"),
}
CANDIDATES = ("F1", "F2", "F4", "N1", "N3", "N6", "R1", "R3", "R6")


def run(cmd, *, check=True, capture=False, timeout=None, cwd=None):
    kwargs = {"text": True, "cwd": cwd, "timeout": timeout}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    p = subprocess.run(cmd, **kwargs)
    if check and p.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{(p.stdout or '')[-7000:]}\nstderr:\n{(p.stderr or '')[-7000:]}"
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd}{detail}")
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
        check=check, capture=capture, timeout=timeout,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_time(text: str) -> float:
    values = [float(x) for x in TIME_RE.findall(text)]
    if not values:
        raise RuntimeError(f"no execution time in output: {text[-2000:]}")
    return values[-1]


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def create_container(image: str, label: str) -> str:
    name = safe_name(f"lexigen-omega-r4-{label}-{os.getpid()}")
    run(["docker", "create", "--name", name, image, "sleep", "infinity"])
    run(["docker", "start", name])
    return name


def cleanup_container(name: str) -> None:
    run(["docker", "rm", "-f", name], check=False, capture=True)


def copy_tests(container: str, tests_dir: Path) -> None:
    for idx in range(len(EXPECTED_TEST_SHA256)):
        run(["docker", "cp", str(tests_dir / f"gso_test_{idx}.py"), f"{container}:/gso_test_{idx}.py"])


def copy_refs_to_container(container: str, refs_dir: Path) -> None:
    for path in sorted(refs_dir.iterdir()):
        if path.is_file():
            run(["docker", "cp", str(path), f"{container}:/testbed/{path.name}"])


def baseline_reference(*, image: str, tests_dir: Path, refs_dir: Path, timeout_per_test: int) -> dict[str, float]:
    name = create_container(image, "baseline")
    refs_dir.mkdir(parents=True, exist_ok=True)
    try:
        copy_tests(name, tests_dir)
        times: dict[str, float] = {}
        for idx in range(len(EXPECTED_TEST_SHA256)):
            out = f"/tmp/base_{idx}.txt"
            cmd = (
                "source .venv/bin/activate && "
                f"python /gso_test_{idx}.py {out} --reference --file_prefix gso_{idx}"
            )
            p = docker_exec(name, cmd, check=False, timeout=timeout_per_test, capture=True)
            if p.returncode != 0:
                raise RuntimeError(
                    f"baseline test {idx} failed rc={p.returncode}\n"
                    f"stdout:\n{(p.stdout or '')[-7000:]}\nstderr:\n{(p.stderr or '')[-7000:]}"
                )
            cat = docker_exec(name, f"cat {out}", capture=True)
            times[str(idx)] = parse_time(cat.stdout)
            found = docker_exec(
                name,
                f"find /testbed -maxdepth 1 -type f -name 'gso_{idx}*' -printf '%f\\n' | sort",
                capture=True,
            ).stdout.splitlines()
            if not found:
                raise RuntimeError(f"baseline test {idx} produced no reference files")
            for filename in found:
                run(["docker", "cp", f"{name}:/testbed/{filename}", str(refs_dir / filename)])
        return times
    finally:
        cleanup_container(name)


def materialize_patch(*, candidate: str, container: str, work_root: Path, patch_root: Path) -> Path:
    root = work_root / candidate
    dest = root / EXPECTED_TARGET
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "cp", f"{container}:/testbed/{EXPECTED_TARGET}", str(dest)])
    base_blob = run(["git", "hash-object", str(dest)], capture=True).stdout.strip()
    if base_blob != EXPECTED_BASE_BLOB:
        raise RuntimeError(f"{candidate}: base blob drift {base_blob}")
    out = patch_root / candidate
    out.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "lexigen-omega/tools/build_r4_candidates.py",
        "--candidate", candidate, "--root", str(root), "--output", str(out),
    ])
    report_path = out / f"R4-{candidate}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    patch_path = out / f"R4-{candidate}.patch"
    expected_patch, expected_after = EXPECTED_PATCH[candidate]
    if report["patch_sha256"] != expected_patch:
        raise RuntimeError(f"{candidate}: patch hash drift {report['patch_sha256']} != {expected_patch}")
    if report["after_sha256"] != expected_after:
        raise RuntimeError(f"{candidate}: after hash drift {report['after_sha256']} != {expected_after}")
    if sha256_bytes(patch_path.read_bytes()) != expected_patch:
        raise RuntimeError(f"{candidate}: patch byte hash mismatch")
    return patch_path


def evaluate_candidate(
    *, candidate: str, image: str, tests_dir: Path, refs_dir: Path,
    install_commands: list[str], baseline_times: dict[str, float],
    work_root: Path, patch_root: Path, timeout_per_test: int,
) -> dict:
    result = {
        "candidate": candidate,
        "correct": False,
        "install_ok": False,
        "tests_passed": 0,
        "test_count": len(EXPECTED_TEST_SHA256),
        "times": {},
        "speedups": {},
        "harmonic_speedup": 0.0,
        "minimum_speedup": 0.0,
        "error": None,
        "patch_sha256": EXPECTED_PATCH[candidate][0],
        "after_sha256": EXPECTED_PATCH[candidate][1],
    }
    name = create_container(image, candidate)
    try:
        copy_tests(name, tests_dir)
        patch_path = materialize_patch(
            candidate=candidate, container=name, work_root=work_root, patch_root=patch_root
        )
        run(["docker", "cp", str(patch_path), f"{name}:/tmp/r4.patch"])
        apply_cmd = (
            "git apply --verbose "
            "--exclude='.venv/*' --exclude='.git/*' --exclude='__pycache__/*' "
            "--exclude='*.egg-info/*' --exclude='*.json' --exclude='*.txt' "
            "--exclude='*.csv' --exclude='*.log' --exclude='*.pkl' /tmp/r4.patch"
        )
        applied = docker_exec(name, apply_cmd, check=False, timeout=120, capture=True)
        if applied.returncode != 0:
            raise RuntimeError(
                f"{candidate}: patch apply failed rc={applied.returncode}\n"
                f"stdout:\n{(applied.stdout or '')[-5000:]}\nstderr:\n{(applied.stderr or '')[-5000:]}"
            )
        got_after = docker_exec(
            name, f"sha256sum {EXPECTED_TARGET} | awk '{{print $1}}'", capture=True
        ).stdout.strip()
        if got_after != EXPECTED_PATCH[candidate][1]:
            raise RuntimeError(f"{candidate}: applied target hash drift {got_after}")

        install_script = [
            "set -euo pipefail",
            "echo 'setuptools<82' > /tmp/uv_build_constraints.txt",
            "export UV_BUILD_CONSTRAINT=/tmp/uv_build_constraints.txt",
            "export HF_HUB_DISABLE_XET=1",
        ] + install_commands
        inst = docker_exec(
            name, "\n".join(install_script), check=False, timeout=2400, capture=True
        )
        if inst.returncode != 0:
            raise RuntimeError(
                f"{candidate}: install failed rc={inst.returncode}\n"
                f"stdout:\n{(inst.stdout or '')[-8000:]}\nstderr:\n{(inst.stderr or '')[-8000:]}"
            )
        result["install_ok"] = True
        copy_refs_to_container(name, refs_dir)

        speedups: list[float] = []
        for idx in range(len(EXPECTED_TEST_SHA256)):
            out = f"/tmp/result_{idx}.txt"
            cmd = (
                "source .venv/bin/activate && "
                f"python /gso_test_{idx}.py {out} --eqcheck --file_prefix gso_{idx}"
            )
            p = docker_exec(name, cmd, check=False, timeout=timeout_per_test, capture=True)
            if p.returncode != 0:
                raise RuntimeError(
                    f"{candidate}: eqcheck test {idx} failed rc={p.returncode}\n"
                    f"stdout:\n{(p.stdout or '')[-7000:]}\nstderr:\n{(p.stderr or '')[-7000:]}"
                )
            cat = docker_exec(name, f"cat {out}", capture=True)
            elapsed = parse_time(cat.stdout)
            baseline = float(baseline_times[str(idx)])
            speedup = baseline / elapsed if elapsed > 0 else 0.0
            result["times"][str(idx)] = elapsed
            result["speedups"][str(idx)] = speedup
            result["tests_passed"] += 1
            speedups.append(speedup)

        result["correct"] = result["tests_passed"] == len(EXPECTED_TEST_SHA256)
        if result["correct"]:
            result["harmonic_speedup"] = harmonic(speedups)
            result["minimum_speedup"] = min(speedups)
    except Exception as exc:
        result["error"] = str(exc)[-14000:]
    finally:
        cleanup_container(name)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw_contract = args.contract.read_bytes()
    if sha256_bytes(raw_contract) != EXPECTED_CONTRACT_SHA256:
        raise SystemExit("R4 execution contract hash drift")
    contract = json.loads(raw_contract)
    if contract.get("instance_id") != EXPECTED_INSTANCE:
        raise SystemExit("R4 instance identity drift")
    for key in (
        "expert_opt_commit_accessed", "expert_gt_commit_message_accessed",
        "expert_diff_accessed", "hints_accessed",
        "candidate_timing_accessed_before_contract", "candidate_outcome_accessed_before_contract",
    ):
        if contract.get(key) is not False:
            raise SystemExit(f"forbidden/dirty contract field: {key}")

    tests = ast.literal_eval(contract["tests"])
    if not isinstance(tests, list) or len(tests) != len(EXPECTED_TEST_SHA256):
        raise SystemExit("R4 test cardinality drift")
    tests_dir = args.output / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(tests):
        if not isinstance(text, str):
            raise SystemExit(f"R4 test {idx} is not text")
        got = sha256_bytes(text.encode())
        if got != EXPECTED_TEST_SHA256[idx]:
            raise SystemExit(f"R4 test {idx} hash drift {got}")
        (tests_dir / f"gso_test_{idx}.py").write_text(text, encoding="utf-8")

    install_commands = [str(x) for x in contract["install_commands"]]
    refs_dir = args.output / "refs"
    work_root = args.output / "work"
    patch_root = args.output / "patches"
    timeout_per_test = 1800

    result = {
        "project": "LEXIGEN OMEGA",
        "stage": "omega3_R4_frozen_candidate_execution",
        "status": "running",
        "instance_id": EXPECTED_INSTANCE,
        "image": EXPECTED_IMAGE,
        "execution_policy": {
            "baseline_first": True,
            "candidate_order": list(CANDIDATES),
            "all_five_contract_tests": True,
            "cold_isolated_container_per_baseline_and_candidate": True,
            "shared_writable_cache_between_arms": False,
            "exact_patch_hash_required": True,
            "exact_after_hash_required": True,
            "single_scientific_execution": True
        },
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "test_sha256": EXPECTED_TEST_SHA256,
        "expert_opt_commit_accessed": False,
        "expert_gt_commit_message_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "R5_source_accessed": False,
        "candidate_results": [],
        "scientific_evidence_eligible": False
    }

    try:
        baseline_times = baseline_reference(
            image=EXPECTED_IMAGE, tests_dir=tests_dir, refs_dir=refs_dir,
            timeout_per_test=timeout_per_test
        )
        result["baseline_times"] = baseline_times
    except Exception as exc:
        result["status"] = "infrastructure_baseline_failure"
        result["error"] = str(exc)[-14000:]
        (args.output / "R4_EXECUTION_RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    for candidate in CANDIDATES:
        rec = evaluate_candidate(
            candidate=candidate, image=EXPECTED_IMAGE, tests_dir=tests_dir,
            refs_dir=refs_dir, install_commands=install_commands,
            baseline_times=baseline_times, work_root=work_root, patch_root=patch_root,
            timeout_per_test=timeout_per_test
        )
        result["candidate_results"].append(rec)

    result["status"] = "completed_frozen_R4_execution"
    result["scientific_evidence_eligible"] = True
    result["clean_candidate_count"] = sum(1 for r in result["candidate_results"] if not r["error"])
    result["correct_candidate_count"] = sum(1 for r in result["candidate_results"] if r["correct"])
    result["claim_boundary"] = (
        "This is one preregistered prospective-development R4 replication. "
        "It does not establish general transfer, AGI, or a breakthrough."
    )
    (args.output / "R4_EXECUTION_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "clean_candidate_count": result["clean_candidate_count"],
        "correct_candidate_count": result["correct_candidate_count"],
        "scientific_evidence_eligible": result["scientific_evidence_eligible"]
    }, indent=2))


if __name__ == "__main__":
    main()
