from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
SELECTED = {
    "v5_full": "v5_full_r2_41510e43e8fafb598496",
    "v5_no_transfer": "v5_no_transfer_r6_66c5848a3c8a4f51b562",
    "random_search": "random_search_r1_399ba5e6f15e49b3e885",
    "static_template": "static_template_r2_8fd871e046faa7e4d37c",
    "v4_compatible": "v4_compatible_r2_9f5f55df04a5ad23f542",
}


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0.0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == candidate), key=lambda r: int(r["index"]))
    if len(selected) != 100 or len({int(r["index"]) for r in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} lacks 100 unique blind records")
    valid = sum(1 for r in selected if bool(r["valid"]))
    retries = sum(int(r["invalid_output_retries"]) for r in selected)
    speeds = [float(r["speedup"]) for r in selected]
    result = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": valid,
        "invalid_outputs": 100-valid,
        "invalid_output_retries": retries,
        "harmonic_speedup": harmonic(speeds),
        "minimum_speedup": min(speeds),
        "median_speedup": statistics.median(speeds),
        "maximum_speedup": max(speeds),
    }
    result["passes_blind_gate"] = bool(
        valid == VALID_REQUIRED
        and retries == 0
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.rglob("blind-shard-*.jsonl"))
    if len(files) != 10:
        raise RuntimeError(f"expected 10 blind shards, got {len(files)}")
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 blind rows, got {len(rows)}")
    expected_pairs = set(SELECTED.items())
    seen_pairs = {(str(r["arm"]), str(r["candidate"])) for r in rows}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"blind identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index: dict[int,int] = {}
    for row in rows:
        idx=int(row["index"]); per_index[idx]=per_index.get(idx,0)+1
    if set(per_index) != set(range(1,101)) or any(v != 5 for v in per_index.values()):
        raise RuntimeError("each blind record must contain exactly five selected candidates")
    manifest_ids = {(
        str(r["test_manifest_name"]), str(r["test_manifest_tree_oid"]), str(r["test_manifest_git_blob_sha1"]), str(r["test_manifest_sha256"])
    ) for r in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("test manifest identity differs across shards")
    test_name,test_oid,test_blob,test_sha256 = next(iter(manifest_ids))
    arms = {arm:summarise(rows,arm,candidate) for arm,candidate in SELECTED.items()}
    full=arms["v5_full"]; no_transfer=arms["v5_no_transfer"]
    report = {
        "campaign":"LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index":4,
        "task":"quantile_regression",
        "stage":"blind_r1",
        "test_manifest_name":test_name,
        "test_manifest_tree_oid":test_oid,
        "test_manifest_git_blob_sha1":test_blob,
        "test_manifest_sha256":test_sha256,
        "blind_records":100,
        "arms":arms,
        "frozen_default_gate":{"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "v5_full_clean_unseen_win":bool(full["passes_blind_gate"]),
        "v5_over_v4_task_win":bool(full["passes_blind_gate"] and not arms["v4_compatible"]["passes_blind_gate"]),
        "selected_v5_uses_learned_transfer":True,
        "selected_v5_transfer_id":"TM-RRR-01",
        "semantic_distinctness_condition_passed_before_blind":False,
        "causal_transfer_win":False,
        "causal_transfer_credit_reason":"Selected v5_full and v5_no_transfer use implementation-equivalent free_parameter_lp mechanisms, so preregistered causal condition 3 fails regardless of blind timing.",
        "v5_over_no_transfer_harmonic_ratio":float(full["harmonic_speedup"])/max(float(no_transfer["harmonic_speedup"]),1e-12),
        "invalid_output_retries_total":sum(int(r["invalid_output_retries"]) for r in rows),
    }
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"blind-summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (args.output/"blind-results.jsonl").write_text("\n".join(json.dumps(r,separators=(",",":")) for r in sorted(rows,key=lambda r:(int(r["index"]),str(r["arm"]))))+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
