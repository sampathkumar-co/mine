from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from candidates import PROVENANCE

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
ARMS = ("v5_full", "v5_no_transfer", "random_search", "static_template", "v4_compatible")


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0.0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def provenance_by_candidate() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        for row in PROVENANCE[arm]:
            result[str(row["candidate"])] = dict(row)
    if len(result) != 30:
        raise RuntimeError(f"expected 30 provenance rows, got {len(result)}")
    return result


def normalized_class(value: str) -> str:
    if value.startswith("free_parameter_lp"):
        return "free_parameter_lp"
    if value.startswith("split_parameter_highs_ds"):
        return "split_parameter_highs_ds"
    if value.startswith("split_parameter_highs"):
        return "split_parameter_highs"
    return value


def summarise(rows: list[dict[str, object]], arm: str, candidate: str, provenance: dict[str, object]) -> dict[str, object]:
    selected = sorted(
        (row for row in rows if row["arm"] == arm and row["candidate"] == candidate),
        key=lambda row: int(row["index"]),
    )
    if len(selected) != 100 or len({int(row["index"]) for row in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} lacks 100 unique records")
    valid = sum(1 for row in selected if bool(row["valid"]))
    retries = sum(int(row["invalid_output_retries"]) for row in selected)
    speeds = [float(row["speedup"]) for row in selected]
    candidate_times = [float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None]
    result: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": valid,
        "invalid_outputs": 100 - valid,
        "invalid_output_retries": retries,
        "harmonic_speedup": harmonic(speeds),
        "minimum_speedup": min(speeds),
        "median_speedup": statistics.median(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "total_candidate_s": sum(candidate_times),
        "implementation_class": provenance["implementation_class"],
        "normalized_implementation_class": normalized_class(str(provenance["implementation_class"])),
        "transfer_ids": provenance["transfer_ids"],
        "learned_template": provenance["learned_template"],
    }
    result["passes_training_correctness"] = valid == VALID_REQUIRED and retries == 0
    result["passes_default_performance_gate_on_training"] = bool(
        result["passes_training_correctness"]
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
    )
    return result


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [row for row in summaries if row["arm"] == arm]
    correct = [row for row in arm_rows if bool(row["passes_training_correctness"])]
    if correct:
        selected = min(
            correct,
            key=lambda row: (
                -int(bool(row["passes_default_performance_gate_on_training"])),
                -float(row["harmonic_speedup"]),
                -float(row["minimum_speedup"]),
                -float(row["median_speedup"]),
                str(row["candidate"]),
            ),
        )
    else:
        selected = min(
            arm_rows,
            key=lambda row: (-int(row["valid"]), -float(row["harmonic_speedup"]), str(row["candidate"])),
        )
    return {
        "arm": arm,
        "selected": selected,
        "candidate_count": len(arm_rows),
        "correct_candidate_count": len(correct),
        "performance_gate_candidate_count": sum(
            1 for row in arm_rows if bool(row["passes_default_performance_gate_on_training"])
        ),
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
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 training rows, got {len(rows)}")

    provenance = provenance_by_candidate()
    expected_pairs = {
        (arm, str(row["candidate"])) for arm in ARMS for row in PROVENANCE[arm]
    }
    seen_pairs = {(str(row["arm"]), str(row["candidate"])) for row in rows}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"candidate identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index: dict[int, int] = {}
    for row in rows:
        index = int(row["index"])
        per_index[index] = per_index.get(index, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(count != 30 for count in per_index.values()):
        raise RuntimeError("each official training record must contain exactly 30 candidate rows")

    manifest_ids = {
        (
            str(row["train_manifest_name"]),
            str(row["train_manifest_tree_oid"]),
            str(row["train_manifest_git_blob_sha1"]),
            str(row["train_manifest_sha256"]),
            str(row["expected_test_manifest_name"]),
            str(row["expected_test_manifest_tree_oid"]),
            int(row["expected_test_manifest_size"]),
        )
        for row in rows
    }
    if len(manifest_ids) != 1:
        raise RuntimeError("manifest identity differs across shards")
    train_name, train_oid, train_blob, train_sha256, test_name, test_oid, test_size = next(iter(manifest_ids))

    summaries = [
        summarise(rows, arm, str(row["candidate"]), provenance[str(row["candidate"])])
        for arm in ARMS
        for row in PROVENANCE[arm]
    ]
    arms = {arm: select_arm(summaries, arm) for arm in ARMS}
    full = arms["v5_full"]["selected"]
    no_transfer = arms["v5_no_transfer"]["selected"]
    full_class = str(full["normalized_implementation_class"])
    no_transfer_classes = {
        normalized_class(str(row["implementation_class"])) for row in PROVENANCE["v5_no_transfer"]
    }
    equivalent_available = full_class in no_transfer_classes
    full_uses_learned = bool(full["transfer_ids"])
    ratio = float(full["harmonic_speedup"]) / max(float(no_transfer["harmonic_speedup"]), 1e-12)
    comparison = {
        "v5_full_harmonic": full["harmonic_speedup"],
        "v5_no_transfer_harmonic": no_transfer["harmonic_speedup"],
        "v5_over_no_transfer_ratio": ratio,
        "selected_v5_uses_learned_transfer": full_uses_learned,
        "selected_v5_transfer_ids": full["transfer_ids"],
        "selected_v5_learned_template": full["learned_template"],
        "selected_v5_implementation_class": full_class,
        "equivalent_implementation_class_available_in_no_transfer": equivalent_available,
        "selected_v5_semantically_distinct_from_no_transfer_by_construction": bool(full_uses_learned and not equivalent_available),
        "training_causal_separation_threshold_crossed": bool(
            full_uses_learned
            and not equivalent_available
            and bool(full["passes_training_correctness"])
            and (not bool(no_transfer["passes_training_correctness"]) or ratio >= 1.25)
        ),
        "causal_transfer_credit": False,
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
        "frozen_default_gate": {"valid_required": 100, "harmonic_speedup_minimum": 1.5, "minimum_speedup": 1.05, "invalid_output_retries": 0},
        "selection_rule": "100% correctness; performance-gate pass first; then harmonic, minimum, median speedup, stable candidate name",
        "all_candidates": summaries,
        "arms": arms,
        "architecture_comparison": comparison,
        "blind_selection_ready": bool(arms["v5_full"]["correct_candidate_count"]),
        "training_revision_consumed": True,
        "official_test_manifest_contents_opened": False,
        "official_test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"]), str(row["candidate"])))
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_by_arm": {arm: value["selected"] for arm, value in arms.items()}, "architecture_comparison": comparison, "blind_selection_ready": report["blind_selection_ready"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
