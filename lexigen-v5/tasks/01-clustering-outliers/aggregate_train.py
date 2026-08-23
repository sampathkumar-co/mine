from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED = {
    "v5_full": [
        "v5_full_r2_41510e43e8fafb598496",
        "v5_full_r3_a6102573c9f355414229",
        "v5_full_r4_4abf2b51384c560522e8",
        "v5_full_r5_c50e493c5549a408f3e5",
        "v5_full_r6_514b3e8a41ba1f8b73a1",
    ],
    "v5_no_transfer": [
        "v5_no_transfer_r1_91e027e622f2d9a98240",
        "v5_no_transfer_r2_b2614109e1a5ccc10c14",
        "v5_no_transfer_r3_20375ceceffce4d406a4",
        "v5_no_transfer_r4_4a4e1871b7f7b48b9485",
        "v5_no_transfer_r5_d69e86803f54c5a83d06",
        "v5_no_transfer_r6_66c5848a3c8a4f51b562",
    ],
    "random_search": [
        "random_search_r1_281d4a03f9bc5812f7af",
        "random_search_r2_0d6f85272e04650e490b",
        "random_search_r3_38776670db84b717ed92",
        "random_search_r4_c7951713adcdd3e83e67",
        "random_search_r5_57b17f2971c60d7d437b",
    ],
    "static_template": [
        "static_template_r2_8fd871e046faa7e4d37c",
        "static_template_r3_820b1c309b6117eb268d",
        "static_template_r4_8f1dafda0d3fbc099aa9",
        "static_template_r5_357e80313b8b9dc3cf36",
        "static_template_r6_d044a19fd4551034dc11",
    ],
    "v4_compatible": [
        "v4_compatible_r1_f9f3239b6866512e4f68",
        "v4_compatible_r2_9f5f55df04a5ad23f542",
        "v4_compatible_r3_ec4b9c17aaa3767d4f6",
        "v4_compatible_r4_7c30efb65d2c20ff8cc9",
        "v4_compatible_r5_3df5ed91505aea4ed6cb",
        "v4_compatible_r6_0dde88a4a159a3ad0e40",
    ],
}
LEARNED = {
    "v5_full_r2_41510e43e8fafb598496": {"causal_id": "TM-RRR-01", "learned_template": "reduced_representation_refinement", "learned_from_family": "linear_algebra"},
    "v5_full_r3_a6102573c9f355414229": {"causal_id": "TM-CAC-01", "learned_template": "certified_active_core", "learned_from_family": "numerical_optimization"},
}


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0.0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((row for row in rows if row["arm"] == arm and row["candidate"] == candidate), key=lambda row: int(row["index"]))
    if len(selected) != 100 or len({int(row["index"]) for row in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain 100 unique records")
    valid_rows = [row for row in selected if bool(row["valid"])]
    speeds = [float(row["speedup"]) for row in selected]
    candidate_times = [float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None]
    metrics = [row.get("metrics", {}) for row in valid_rows]
    result: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": len(valid_rows),
        "invalid_outputs": 100 - len(valid_rows),
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "total_candidate_s": sum(candidate_times),
        "maximum_cluster_deviation": max((float(m.get("cluster_deviation", 0.0)) for m in metrics), default=float("inf")),
        "maximum_noise_deviation": max((float(m.get("noise_deviation", 0.0)) for m in metrics), default=float("inf")),
        "minimum_ari": min((float(m.get("ari", 1.0)) for m in metrics), default=float("-inf")),
        "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected),
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "learned_transfer": LEARNED.get(candidate),
    }
    result["passes_training_correctness"] = len(valid_rows) == VALID_REQUIRED and int(result["invalid_output_retries"]) == 0
    result["passes_default_performance_gate_on_training"] = bool(
        result["passes_training_correctness"]
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
    )
    return result


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [row for row in summaries if row["arm"] == arm]
    correct = [row for row in arm_rows if bool(row["passes_training_correctness"])]
    pool = correct if correct else arm_rows
    selected = min(pool, key=lambda row: (-int(row["valid"]), -float(row["harmonic_speedup"]), -float(row["minimum_speedup"]), str(row["candidate"])))
    return {
        "arm": arm,
        "selected": selected,
        "candidate_count": len(arm_rows),
        "correct_candidate_count": len(correct),
        "performance_gate_candidate_count": sum(1 for row in arm_rows if bool(row["passes_default_performance_gate_on_training"])),
        "discovery_cost_total_candidate_s": sum(float(row["total_candidate_s"]) for row in arm_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shard_files = sorted(args.input.rglob("train-shard-*.jsonl"))
    if len(shard_files) != 10:
        raise RuntimeError(f"expected 10 shard files, got {len(shard_files)}")
    rows: list[dict[str, object]] = []
    for path in shard_files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected_count = sum(len(names) for names in EXPECTED.values())
    if expected_count != 27 or len(rows) != 2700:
        raise RuntimeError(f"expected 27 candidates/2700 rows, got {expected_count}/{len(rows)}")
    expected_pairs = {(arm, name) for arm, names in EXPECTED.items() for name in names}
    seen_pairs = {(str(row["arm"]), str(row["candidate"])) for row in rows}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"candidate identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index: dict[int, int] = {}
    for row in rows:
        index = int(row["index"])
        per_index[index] = per_index.get(index, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(count != 27 for count in per_index.values()):
        raise RuntimeError("each training record must contain exactly 27 candidate rows")
    manifest_ids = {(str(row["train_manifest_name"]), str(row["train_manifest_tree_oid"]), str(row["train_manifest_git_blob_sha1"]), str(row["train_manifest_sha256"]), str(row["expected_test_manifest_name"]), str(row["expected_test_manifest_tree_oid"])) for row in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("manifest identity differs across shards")
    train_name, train_oid, train_blob, train_sha256, test_name, test_oid = next(iter(manifest_ids))
    shapes = sorted({tuple(int(x) for x in row["dataset_shape"]) for row in rows})

    summaries = [summarise(rows, arm, name) for arm, names in EXPECTED.items() for name in names]
    arms = {arm: select_arm(summaries, arm) for arm in EXPECTED}
    full = arms["v5_full"]["selected"]
    no_transfer = arms["v5_no_transfer"]["selected"]
    random = arms["random_search"]["selected"]
    static = arms["static_template"]["selected"]
    v4 = arms["v4_compatible"]["selected"]
    full_learned = full.get("learned_transfer")
    comparison = {
        "v5_full_harmonic": full["harmonic_speedup"],
        "v5_no_transfer_harmonic": no_transfer["harmonic_speedup"],
        "random_search_harmonic": random["harmonic_speedup"],
        "static_template_harmonic": static["harmonic_speedup"],
        "v4_compatible_harmonic": v4["harmonic_speedup"],
        "v5_minus_no_transfer_harmonic": float(full["harmonic_speedup"]) - float(no_transfer["harmonic_speedup"]),
        "v5_over_no_transfer_ratio": float(full["harmonic_speedup"]) / max(float(no_transfer["harmonic_speedup"]), 1e-12),
        "v5_minus_random_harmonic": float(full["harmonic_speedup"]) - float(random["harmonic_speedup"]),
        "v5_minus_static_harmonic": float(full["harmonic_speedup"]) - float(static["harmonic_speedup"]),
        "v5_minus_v4_harmonic": float(full["harmonic_speedup"]) - float(v4["harmonic_speedup"]),
        "selected_v5_uses_learned_transfer": full_learned is not None,
        "selected_v5_semantically_distinct_from_no_transfer_by_construction": full_learned is not None,
        "training_causal_separation_threshold_crossed": bool(
            full_learned is not None
            and bool(full["passes_training_correctness"])
            and (
                not bool(no_transfer["passes_training_correctness"])
                or float(full["harmonic_speedup"]) >= 1.25 * float(no_transfer["harmonic_speedup"])
            )
        ),
        "causal_transfer_credit": False,
        "causal_transfer_credit_reason": "Blind result plus preregistered recipe-removal replay are required; training can only establish candidacy."
    }
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 1,
        "task": "clustering_outliers",
        "revision": 1,
        "stage": "official_training",
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_oid,
        "train_manifest_git_blob_sha1": train_blob,
        "train_manifest_sha256": train_sha256,
        "expected_test_manifest_name": test_name,
        "expected_test_manifest_tree_oid": test_oid,
        "training_records": 100,
        "candidate_count": 27,
        "dataset_shapes": [list(shape) for shape in shapes],
        "frozen_default_gate": {"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "all_candidates": summaries,
        "arms": arms,
        "architecture_comparison": comparison,
        "v5_full_has_correct_training_candidate": bool(arms["v5_full"]["correct_candidate_count"]),
        "blind_selection_ready": bool(arms["v5_full"]["correct_candidate_count"]),
        "training_revision_consumed": True,
        "official_test_manifest_contents_opened": False,
        "official_test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"]), str(row["candidate"])))) + "\n", encoding="utf-8")
    print(json.dumps({"selected_by_arm": {arm: value["selected"] for arm, value in arms.items()}, "architecture_comparison": comparison, "dataset_shapes": report["dataset_shapes"], "blind_selection_ready": report["blind_selection_ready"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
