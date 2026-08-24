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
        "v5_full_r1_3304c859d463a501bd86",
        "v5_full_r2_41510e43e8fafb598496",
        "v5_full_r3_a6102573c9f355414229",
        "v5_full_r4_4abf2b51384c560522e8",
        "v5_full_r5_c50e493c5549a408f3e5",
        "v5_full_r6_514b3e8a41ba1f8b73a1"
    ],
    "v5_no_transfer": [
        "v5_no_transfer_r1_91e027e622f2d9a98240",
        "v5_no_transfer_r2_b2614109e1a5ccc10c14",
        "v5_no_transfer_r3_20375ceceffce4d406a4",
        "v5_no_transfer_r4_4a4e1871b7f7b48b9485",
        "v5_no_transfer_r5_d69e86803f54c5a83d06",
        "v5_no_transfer_r6_66c5848a3c8a4f51b562"
    ],
    "random_search": [
        "random_search_r1_399ba5e6f15e49b3e885",
        "random_search_r2_281d4a03f9bc5812f7af",
        "random_search_r3_3667208f0eec7d49161f",
        "random_search_r4_d59864ed9ab0297d5542",
        "random_search_r5_83aa0674a4d1af4a6b66",
        "random_search_r6_f809298bda98f85e0e1b"
    ],
    "static_template": [
        "static_template_r1_dbfcd2af539b0b2636e7",
        "static_template_r2_8fd871e046faa7e4d37c",
        "static_template_r3_820b1c309b6117eb268d",
        "static_template_r4_8f1dafda0d3fbc099aa9",
        "static_template_r5_357e80313b8b9dc3cf36",
        "static_template_r6_d044a19fd4551034dc11"
    ],
    "v4_compatible": [
        "v4_compatible_r1_f9f3239b6866512e4f68",
        "v4_compatible_r2_9f5f55df04a5ad23f542",
        "v4_compatible_r3_ec4b9c17aaa3767d4f6d",
        "v4_compatible_r4_7c30efb65d2c20ff8cc9",
        "v4_compatible_r5_3df5ed91505aea4ed6cb",
        "v4_compatible_r6_0dde88a4a159a3ad0e40"
    ]
}

LEARNED = {
    "v5_full_r1_3304c859d463a501bd86": {"causal_id":"TM-BFR-01","learned_template":"bit_frontier_restriction","learned_from_family":"graph_discrete"},
    "v5_full_r2_41510e43e8fafb598496": {"causal_id":"TM-RRR-01","learned_template":"reduced_representation_refinement","learned_from_family":"linear_algebra"},
    "v5_full_r3_a6102573c9f355414229": {"causal_id":"TM-CAC-01","learned_template":"certified_active_core","learned_from_family":"numerical_optimization"}
}

IMPLEMENTATION_CLASS = {
    "v5_full_r1_3304c859d463a501bd86": "reference_exact_fallback",
    "v5_full_r2_41510e43e8fafb598496": "free_parameter_lp",
    "v5_full_r3_a6102573c9f355414229": "dual_certified_active_core",
    "v5_full_r4_4abf2b51384c560522e8": "split_parameter_highs_ds",
    "v5_full_r5_c50e493c5549a408f3e5": "split_parameter_highs_ds",
    "v5_full_r6_514b3e8a41ba1f8b73a1": "split_parameter_highs_ds",
    "v5_no_transfer_r1_91e027e622f2d9a98240": "split_parameter_highs_ds",
    "v5_no_transfer_r2_b2614109e1a5ccc10c14": "split_parameter_highs_ds",
    "v5_no_transfer_r3_20375ceceffce4d406a4": "split_parameter_highs_ds",
    "v5_no_transfer_r4_4a4e1871b7f7b48b9485": "split_parameter_highs_ds",
    "v5_no_transfer_r5_d69e86803f54c5a83d06": "split_parameter_highs_ds",
    "v5_no_transfer_r6_66c5848a3c8a4f51b562": "free_parameter_lp",
    "random_search_r1_399ba5e6f15e49b3e885": "free_parameter_lp",
    "random_search_r2_281d4a03f9bc5812f7af": "split_parameter_highs",
    "random_search_r3_3667208f0eec7d49161f": "split_parameter_highs",
    "random_search_r4_d59864ed9ab0297d5542": "split_parameter_highs",
    "random_search_r5_83aa0674a4d1af4a6b66": "split_parameter_highs",
    "random_search_r6_f809298bda98f85e0e1b": "split_parameter_highs",
    "static_template_r1_dbfcd2af539b0b2636e7": "reference_exact_fallback",
    "static_template_r2_8fd871e046faa7e4d37c": "free_parameter_lp",
    "static_template_r3_820b1c309b6117eb268d": "split_parameter_highs_ds",
    "static_template_r4_8f1dafda0d3fbc099aa9": "split_parameter_highs",
    "static_template_r5_357e80313b8b9dc3cf36": "split_parameter_highs",
    "static_template_r6_d044a19fd4551034dc11": "split_parameter_highs",
    "v4_compatible_r1_f9f3239b6866512e4f68": "split_parameter_highs_ds",
    "v4_compatible_r2_9f5f55df04a5ad23f542": "split_parameter_highs_ds",
    "v4_compatible_r3_ec4b9c17aaa3767d4f6d": "split_parameter_highs_ds",
    "v4_compatible_r4_7c30efb65d2c20ff8cc9": "split_parameter_highs_ds",
    "v4_compatible_r5_3df5ed91505aea4ed6cb": "split_parameter_highs_ds",
    "v4_compatible_r6_0dde88a4a159a3ad0e40": "reference_exact_fallback"
}


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0.0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == candidate), key=lambda r: int(r["index"]))
    if len(selected) != 100 or len({int(r["index"]) for r in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} lacks 100 unique training records")
    speeds = [float(r["speedup"]) for r in selected]
    valid = sum(1 for r in selected if bool(r["valid"]))
    retries = sum(int(r["invalid_output_retries"]) for r in selected)
    candidate_times = [float(r["candidate_s"]) for r in selected if r["candidate_s"] is not None]
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
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "total_candidate_s": sum(candidate_times),
        "implementation_class": IMPLEMENTATION_CLASS[candidate],
        "learned_transfer": LEARNED.get(candidate),
    }
    result["passes_training_correctness"] = valid == VALID_REQUIRED and retries == 0
    result["passes_default_performance_gate_on_training"] = bool(
        result["passes_training_correctness"]
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
    )
    return result


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [r for r in summaries if r["arm"] == arm]
    correct = [r for r in arm_rows if bool(r["passes_training_correctness"])]
    if correct:
        selected = min(correct, key=lambda r: (
            -int(bool(r["passes_default_performance_gate_on_training"])),
            -float(r["harmonic_speedup"]),
            -float(r["minimum_speedup"]),
            -float(r["median_speedup"]),
            str(r["candidate"]),
        ))
    else:
        selected = min(arm_rows, key=lambda r: (-int(r["valid"]), -float(r["harmonic_speedup"]), str(r["candidate"])))
    return {
        "arm": arm,
        "selected": selected,
        "candidate_count": len(arm_rows),
        "correct_candidate_count": len(correct),
        "performance_gate_candidate_count": sum(1 for r in arm_rows if bool(r["passes_default_performance_gate_on_training"])),
        "discovery_cost_total_candidate_s": sum(float(r["total_candidate_s"]) for r in arm_rows),
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
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 training rows, got {len(rows)}")
    expected_pairs = {(arm, name) for arm, names in EXPECTED.items() for name in names}
    seen_pairs = {(str(r["arm"]), str(r["candidate"])) for r in rows}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"candidate identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index: dict[int, int] = {}
    for row in rows:
        idx = int(row["index"]); per_index[idx] = per_index.get(idx, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(v != 30 for v in per_index.values()):
        raise RuntimeError("each training record must contain exactly 30 candidates")
    manifest_ids = {(
        str(r["train_manifest_name"]), str(r["train_manifest_tree_oid"]), str(r["train_manifest_git_blob_sha1"]),
        str(r["train_manifest_sha256"]), str(r["expected_test_manifest_name"]), str(r["expected_test_manifest_tree_oid"]),
        int(r["expected_test_manifest_size"])
    ) for r in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("manifest identity differs across shards")
    train_name, train_oid, train_blob, train_sha256, test_name, test_oid, test_size = next(iter(manifest_ids))

    summaries = [summarise(rows, arm, name) for arm, names in EXPECTED.items() for name in names]
    arms = {arm: select_arm(summaries, arm) for arm in EXPECTED}
    full = arms["v5_full"]["selected"]
    no_transfer = arms["v5_no_transfer"]["selected"]
    full_class = str(full["implementation_class"])
    no_transfer_classes = {IMPLEMENTATION_CLASS[name] for name in EXPECTED["v5_no_transfer"]}
    equivalent_available = full_class in no_transfer_classes
    full_learned = full.get("learned_transfer")
    ratio = float(full["harmonic_speedup"]) / max(float(no_transfer["harmonic_speedup"]), 1e-12)
    comparison = {
        "v5_full_harmonic": full["harmonic_speedup"],
        "v5_no_transfer_harmonic": no_transfer["harmonic_speedup"],
        "v5_over_no_transfer_ratio": ratio,
        "selected_v5_uses_learned_transfer": full_learned is not None,
        "selected_v5_implementation_class": full_class,
        "equivalent_implementation_class_available_in_no_transfer": equivalent_available,
        "selected_v5_semantically_distinct_from_no_transfer_by_construction": bool(full_learned is not None and not equivalent_available),
        "training_causal_separation_threshold_crossed": bool(
            full_learned is not None
            and not equivalent_available
            and bool(full["passes_training_correctness"])
            and (not bool(no_transfer["passes_training_correctness"]) or ratio >= 1.25)
        ),
        "causal_transfer_credit": false if False else False,
        "causal_transfer_credit_reason": "Blind result and preregistered recipe-removal replay are still required."
    }
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 4,
        "task": "quantile_regression",
        "revision": 1,
        "stage": "official_training",
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_oid,
        "train_manifest_git_blob_sha1": train_blob,
        "train_manifest_sha256": train_sha256,
        "expected_test_manifest_name": test_name,
        "expected_test_manifest_tree_oid": test_oid,
        "expected_test_manifest_size": test_size,
        "training_records": 100,
        "candidate_count": 30,
        "frozen_default_gate": {"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "selection_rule": "100% correctness; performance-gate pass first; then harmonic, minimum, median speedup, stable candidate name",
        "all_candidates": summaries,
        "arms": arms,
        "architecture_comparison": comparison,
        "blind_selection_ready": bool(arms["v5_full"]["correct_candidate_count"]),
        "training_revision_consumed": true if True else True,
        "official_test_manifest_contents_opened": False,
        "official_test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda r: (int(r["index"]), str(r["arm"]), str(r["candidate"])))) + "\n", encoding="utf-8")
    print(json.dumps({"selected_by_arm": {arm: value["selected"] for arm, value in arms.items()}, "architecture_comparison": comparison, "blind_selection_ready": report["blind_selection_ready"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
