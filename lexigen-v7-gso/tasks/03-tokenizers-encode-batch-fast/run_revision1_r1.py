from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_tasks2_6_preflight_r1 as core
import run_tasks3_6_preflight_r2 as transport_r2  # noqa: F401; installs repaired docker_exec into core


TASK = 3
CANDIDATE_ARM = {
    "F2R1": "v7_full",
    "N4R1": "v7_no_library",
    "R4R1": "v7_random_library",
}


def build_patch_revision(
    *, task: int, candidate: str, target: str, container: str,
    work_root: Path, output_root: Path,
) -> dict:
    if task != TASK or candidate not in CANDIDATE_ARM:
        raise RuntimeError(f"unexpected revision build request T{task} {candidate}")
    lock = json.loads(Path(
        "lexigen-v7-gso/tasks/03-tokenizers-encode-batch-fast/REVISION1_MATERIALIZATION_LOCK_R1.json"
    ).read_text())
    root = work_root / candidate
    dest = root / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    core.run(["docker", "cp", f"{container}:/testbed/{target}", str(dest)])
    before_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    if before_sha != lock["base_target_sha256"]:
        raise RuntimeError(f"Task3 revision base source drift: {before_sha}")

    out = output_root / candidate
    out.mkdir(parents=True, exist_ok=True)
    core.run([
        sys.executable,
        "lexigen-v7-gso/tasks/03-tokenizers-encode-batch-fast/build_revision1_r1.py",
        "--candidate", candidate,
        "--root", str(root),
        "--output", str(out),
    ])
    report_path = out / f"task3-{candidate}.json"
    report = json.loads(report_path.read_text())
    expected = lock["patch_sha256"][candidate]
    if report["patch_sha256"] != expected:
        raise RuntimeError(
            f"Task3 revision patch hash drift {candidate}: {report['patch_sha256']} != {expected}"
        )
    if report["before_sha256"] != lock["base_target_sha256"]:
        raise RuntimeError(f"Task3 builder base hash drift for {candidate}")
    report["patch_path"] = str(out / f"task3-{candidate}.patch")
    return report


core.build_patch = build_patch_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--execmeta", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.instance_id != "huggingface__tokenizers-bfd9cde":
        raise SystemExit("Task3 revision instance mismatch")
    if args.target != "bindings/python/src/tokenizer.rs":
        raise SystemExit("Task3 revision target mismatch")

    hashes = json.loads(Path("lexigen-v7-gso/PREFLIGHT_TEST_HASH_LOCK_R1.json").read_text())
    plan = json.loads(Path("lexigen-v7-gso/PREFLIGHT_PLAN_R1.json").read_text())
    revplan = json.loads(Path(
        "lexigen-v7-gso/tasks/03-tokenizers-encode-batch-fast/REVISION1_PLAN_R1.json"
    ).read_text())
    matlock = json.loads(Path(
        "lexigen-v7-gso/tasks/03-tokenizers-encode-batch-fast/REVISION1_MATERIALIZATION_LOCK_R1.json"
    ).read_text())
    previous = json.loads(Path(
        "lexigen-v7-gso/tasks/03-tokenizers-encode-batch-fast/PREFLIGHT_R2_RESULT.json"
    ).read_text())
    spec = json.loads((args.spec / "task-spec.json").read_text())
    execmeta = json.loads(args.execmeta.read_text())

    if previous["revision_slots_used_per_arm"] != 0 or revplan["revision_slot"] != 1:
        raise SystemExit("Task3 revision budget mismatch")
    if any((revplan.get("expert_opt_commit_accessed"), revplan.get("expert_diff_accessed"), revplan.get("hints_accessed"))):
        raise SystemExit("Task3 revision plan crossed expert boundary")
    if matlock["candidate_execution_observed"] or matlock["candidate_timing_observed"]:
        raise SystemExit("Task3 candidates were not frozen before execution")

    tkey = "3"
    indexes = list(plan["representative_test_indexes"][tkey])
    if indexes != [0, 2, 4] or indexes != list(hashes["tasks"][tkey]["indexes"]):
        raise SystemExit("Task3 representative test lock mismatch")
    if spec["instance_id"] != args.instance_id:
        raise SystemExit("Task3 spec identity mismatch")

    tests_dir = args.output / "tests"
    refs_dir = args.output / "refs"
    work_root = args.output / "work"
    patch_root = args.output / "patches"
    cache_root = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "lexigen-v7-t3-revision1-cache"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for pos, idx in enumerate(indexes):
        text = spec["tests"][idx]
        got = hashlib.sha256(text.encode()).hexdigest()
        expected = hashes["tasks"][tkey]["sha256"][pos]
        if got != expected:
            raise SystemExit(f"Task3 revision test hash mismatch index {idx}: {got}")
        (tests_dir / f"gso_test_{idx}.py").write_text(text, encoding="utf-8")

    rows = [r for r in execmeta["records"] if r["instance_id"] == args.instance_id]
    if len(rows) != 1:
        raise SystemExit("Task3 execution metadata identity mismatch")
    if execmeta.get("expert_opt_commit_accessed") or execmeta.get("expert_diff_accessed") or execmeta.get("hints_accessed"):
        raise SystemExit("forbidden expert metadata in Task3 execution projection")
    install_commands = list(rows[0]["install_commands"])

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "campaign": "LEXIGEN V7 GSO Real Causal-Transfer Pilot R1",
        "task": 3,
        "instance_id": args.instance_id,
        "stage": "task3_revision1_r1",
        "revision_slot": 1,
        "representative_test_indexes": indexes,
        "candidate_results": [],
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "thresholds_changed": False,
    }

    baseline_times = core.baseline_reference(
        image=args.image,
        task=3,
        indexes=indexes,
        tests_dir=tests_dir,
        refs_dir=refs_dir,
        cache_root=cache_root,
        timeout_per_test=900,
    )
    result["baseline_times"] = baseline_times

    for cid in ["F2R1", "N4R1", "R4R1"]:
        rec = core.evaluate_candidate(
            task=3,
            candidate=cid,
            image=args.image,
            target=args.target,
            indexes=indexes,
            tests_dir=tests_dir,
            refs_dir=refs_dir,
            install_commands=install_commands,
            expected_patch_sha=matlock["patch_sha256"][cid],
            baseline_times=baseline_times,
            cache_root=cache_root,
            work_root=work_root,
            patch_root=patch_root,
            timeout_per_test=900,
        )
        rec["arm"] = CANDIDATE_ARM[cid]
        result["candidate_results"].append(rec)
        (args.output / "REVISION1_RESULT_R1.json").write_text(json.dumps(result, indent=2) + "\n")

    by_id = {r["candidate"]: r for r in result["candidate_results"]}
    result["full_no_library_patch_byte_identical"] = (
        by_id["F2R1"].get("patch_sha256") == by_id["N4R1"].get("patch_sha256")
    )
    result["full_no_library_both_correct"] = bool(
        by_id["F2R1"]["correct"] and by_id["N4R1"]["correct"]
    )
    result["causal_implication"] = (
        "If F2R1 and N4R1 are both correct, this revision cannot earn learned-library causal credit because the full and no-library executable patches are byte-identical."
    )
    result["status"] = "completed"
    (args.output / "REVISION1_RESULT_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "full_no_library_patch_byte_identical": result["full_no_library_patch_byte_identical"],
        "full_no_library_both_correct": result["full_no_library_both_correct"],
        "candidates": {r["candidate"]: {"correct": r["correct"], "harmonic_speedup": r["harmonic_speedup"]} for r in result["candidate_results"]},
    }, indent=2))


if __name__ == "__main__":
    main()
