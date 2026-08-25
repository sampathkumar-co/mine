from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from candidates import lowlevel_odr_exact
from train_shard import (
    BASE,
    EXPECTED_RECORDS,
    TEST_NAME,
    TEST_OID,
    TEST_SIZE,
    decode_problem,
    fetch,
    git_blob,
    reference_exact,
    timed,
    verify,
    flat,
)

SHARDS = 10
LOCK_NAME = "BLIND_R1_LOCK.json"
SELECTED_ARMS = (
    "v6_full",
    "v6_no_transfer",
    "random_search",
    "static_template",
    "v5_compatible",
    "strong_baseline",
)
ABLATION_ARM = "recipe_removal_ablation"


def load_lock() -> dict:
    lock = json.loads((Path(__file__).resolve().parent / LOCK_NAME).read_text())
    if lock.get("task") != "odr" or lock.get("stage") != "blind_r1_lock":
        raise RuntimeError("invalid blind lock")
    if lock.get("official_test_manifest_opened_before_lock") or int(lock.get("official_test_payloads_opened_before_lock", -1)) != 0:
        raise RuntimeError("blind boundary already crossed before lock")
    return lock


def selected_candidates(source_text: str, lock: dict):
    candidates = flat(source_text)
    by_name = {c.name: c for c in candidates}
    selected = []
    for arm in SELECTED_ARMS:
        spec = lock["selected_by_arm"][arm]
        name = spec["candidate"]
        if name not in by_name:
            raise RuntimeError(f"locked candidate missing: {name}")
        candidate = by_name[name]
        if candidate.arm != arm:
            raise RuntimeError(f"arm mismatch for {name}: {candidate.arm} != {arm}")
        if candidate.implementation_class != spec["implementation_class"]:
            raise RuntimeError(f"implementation mismatch for {name}")
        if list(candidate.operators) != spec["operators"]:
            raise RuntimeError(f"operator mismatch for {name}")
        if list(candidate.transfer_ids) != spec["transfer_ids"]:
            raise RuntimeError(f"transfer mismatch for {name}")
        if candidate.learned_template != spec.get("learned_template"):
            raise RuntimeError(f"template mismatch for {name}")
        if candidate.baseline_id != spec.get("baseline_id"):
            raise RuntimeError(f"baseline mismatch for {name}")
        selected.append(candidate)

    full = selected[0]
    ablation = lock["recipe_removal_ablation"]
    if ablation["source_candidate"] != full.name:
        raise RuntimeError("ablation source candidate mismatch")
    if ablation["removed_transfer_ids"] != list(full.transfer_ids):
        raise RuntimeError("ablation removed-transfer set mismatch")
    if ablation["resulting_transfer_ids"] != []:
        raise RuntimeError("ablation must remove all learned transfer IDs")
    if ablation["resulting_implementation_class"] != "lowlevel_odrpack_direct_weights":
        raise RuntimeError("unexpected frozen ablation implementation")
    if ablation["same_callable_as_full"]:
        raise RuntimeError("Task2 ablation must be the direct low-level callable, not the PBEB wrapper")
    if not ablation["preblind_semantic_equivalence_to_full"]:
        raise RuntimeError("Task2 frozen PBEB/direct semantic equivalence must be acknowledged")
    return selected, full


def evaluation_entries(selected, full, lock: dict):
    entries = []
    for candidate in selected:
        spec = lock["selected_by_arm"][candidate.arm]
        entries.append({
            "arm": candidate.arm,
            "candidate": candidate.name,
            "implementation_class": candidate.implementation_class,
            "semantic_implementation_key": spec["semantic_implementation_key"],
            "operators": list(candidate.operators),
            "transfer_ids": list(candidate.transfer_ids),
            "learned_template": candidate.learned_template,
            "baseline_id": candidate.baseline_id,
            "solve": candidate.solve,
            "is_ablation": False,
        })
    abl = lock["recipe_removal_ablation"]
    entries.append({
        "arm": ABLATION_ARM,
        "candidate": abl["candidate"],
        "implementation_class": abl["resulting_implementation_class"],
        "semantic_implementation_key": abl["resulting_semantic_implementation_key"],
        "operators": abl["retained_operators"],
        "transfer_ids": [],
        "learned_template": None,
        "baseline_id": None,
        "solve": lowlevel_odr_exact,
        "is_ablation": True,
    })
    return entries


def run_smoke(source_text: str, lock: dict) -> None:
    from synthetic import problems

    selected, full = selected_candidates(source_text, lock)
    entries = evaluation_entries(selected, full, lock)
    cases = problems()[:4]
    valid = 0
    for case_index, problem in enumerate(cases):
        ref = reference_exact(problem)
        for entry in entries:
            got = entry["solve"](problem)
            ok, reason, _ = verify(problem, got, ref)
            if not ok:
                raise RuntimeError(f"smoke invalid case={case_index} candidate={entry['candidate']} reason={reason}")
            valid += 1
    expected = len(cases) * len(entries)
    if valid != expected:
        raise RuntimeError(f"smoke row mismatch {valid} != {expected}")
    print(json.dumps({
        "stage": "preblind_synthetic_smoke",
        "cases": len(cases),
        "evaluations": expected,
        "valid": valid,
        "official_test_manifest_opened": False,
        "official_test_payloads_opened": 0,
    }, indent=2))


def run_official(shard: int, output: Path, source_text: str, source_sha: str, lock: dict) -> None:
    if not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    selected, full = selected_candidates(source_text, lock)
    entries = evaluation_entries(selected, full, lock)
    if len(entries) != 7:
        raise RuntimeError(f"expected 7 frozen blind entries got {len(entries)}")

    manifest = fetch(f"{BASE}/{TEST_NAME}?download=true")
    if len(manifest) != TEST_SIZE or git_blob(manifest) != TEST_OID:
        raise RuntimeError(f"test manifest identity mismatch size={len(manifest)} blob={git_blob(manifest)}")
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    records = [json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected 100 test records got {len(records)}")

    evidence = []
    for idx, row in ((i, r) for i, r in enumerate(records) if i % SHARDS == shard):
        problem = decode_problem(row["problem"])
        shift = idx % len(entries)
        ordered = entries[shift:] + entries[:shift]
        if idx % 2 == 0:
            ref, ref_ns, ref_error = timed(reference_exact, problem)
            candidate_runs = [(e, *timed(e["solve"], problem)) for e in ordered]
            execution_order = "reference_first"
        else:
            candidate_runs = [(e, *timed(e["solve"], problem)) for e in ordered]
            ref, ref_ns, ref_error = timed(reference_exact, problem)
            execution_order = "candidates_first"
        if ref is None or ref_ns is None or ref_error:
            raise RuntimeError(f"reference failed test record {idx + 1}: {ref_error}")

        for entry, got, candidate_ns, error in candidate_runs:
            if error is None:
                valid, reason, metrics = verify(problem, got, ref)
            else:
                valid, reason, metrics = False, "exception", {}
            evidence.append({
                "index": idx + 1,
                "seed": int(row.get("seed", idx + 1)),
                "arm": entry["arm"],
                "candidate": entry["candidate"],
                "implementation_class": entry["implementation_class"],
                "semantic_implementation_key": entry["semantic_implementation_key"],
                "operators": entry["operators"],
                "transfer_ids": entry["transfer_ids"],
                "learned_template": entry["learned_template"],
                "baseline_id": entry["baseline_id"],
                "recipe_removal_ablation": bool(entry["is_ablation"]),
                "valid": bool(valid and error is None),
                "semantic_and_official_certificate": bool(valid and error is None),
                "failure_reason": error or reason,
                "candidate_ns": candidate_ns,
                "reference_ns": ref_ns,
                "speedup": (ref_ns / candidate_ns) if candidate_ns and candidate_ns > 0 else 0.0,
                **metrics,
                "n": len(problem["x"]),
                "test_manifest_name": TEST_NAME,
                "test_manifest_git_blob_sha1": TEST_OID,
                "test_manifest_sha256": manifest_sha256,
                "source_sha256": source_sha,
                "execution_order": execution_order,
                "shard": shard,
                "invalid_output_retries": 0,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "verifier_capacity_loophole_exploited": False,
            })
        del problem, ref, candidate_runs
        gc.collect()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n")
    print(json.dumps({
        "stage": "official_blind_r1_shard",
        "shard": shard,
        "rows": len(evidence),
        "test_manifest_sha256": manifest_sha256,
        "invalid_output_retries": 0,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    source_raw = args.source.read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    if source_sha != "076efd6697175397912d5d8e3bc1b121ba7461db3fdbf04263fa6d57f81eb68c":
        raise RuntimeError("source identity mismatch")
    lock = load_lock()
    if lock["source_sha256"] != source_sha:
        raise RuntimeError("blind lock source mismatch")
    source_text = source_raw.decode("utf-8")

    if args.smoke:
        if args.shard is not None or args.output is not None:
            raise ValueError("smoke mode cannot accept shard/output")
        run_smoke(source_text, lock)
        return
    if args.shard is None or args.output is None:
        raise ValueError("official mode requires --shard and --output")
    run_official(args.shard, args.output, source_text, source_sha, lock)


if __name__ == "__main__":
    main()
