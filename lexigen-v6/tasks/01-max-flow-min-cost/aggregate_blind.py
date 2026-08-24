from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_SHARDS = 10
EXPECTED_RECORDS = 100
EXPECTED_ENTRIES = 7
ARM_ORDER = (
    "v6_full",
    "v6_no_transfer",
    "random_search",
    "static_template",
    "v5_compatible",
    "strong_baseline",
    "recipe_removal_ablation",
)


def harmonic(values: list[float]) -> float:
    if not values or any((not math.isfinite(x) or x <= 0) for x in values):
        return 0.0
    return len(values) / sum(1.0 / x for x in values)


def summarize(rows: list[dict]) -> dict:
    speeds = [float(r["speedup"]) for r in rows]
    valid = sum(bool(r["valid"]) for r in rows)
    retries = sum(int(r["invalid_output_retries"]) for r in rows)
    out = {
        "arm": rows[0]["arm"],
        "candidate": rows[0]["candidate"],
        "implementation_class": rows[0]["implementation_class"],
        "operators": rows[0]["operators"],
        "transfer_ids": rows[0]["transfer_ids"],
        "learned_template": rows[0]["learned_template"],
        "baseline_id": rows[0]["baseline_id"],
        "recipe_removal_ablation": bool(rows[0].get("recipe_removal_ablation")),
        "records": len(rows),
        "valid": valid,
        "invalid_outputs": len(rows) - valid,
        "invalid_output_retries": retries,
        "harmonic_speedup": harmonic(speeds),
        "minimum_speedup": min(speeds),
        "median_speedup": statistics.median(speeds),
        "maximum_speedup": max(speeds),
    }
    out["passes_clean_blind_gate"] = (
        out["records"] == 100
        and out["valid"] == 100
        and out["invalid_output_retries"] == 0
        and out["harmonic_speedup"] >= 1.50
        and out["minimum_speedup"] >= 1.05
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.glob("blind-shard-*.jsonl"))
    if len(files) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} blind shards got {len(files)}")
    rows = []
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows) != EXPECTED_RECORDS * EXPECTED_ENTRIES:
        raise RuntimeError(f"expected 700 blind rows got {len(rows)}")
    if any(int(r["invalid_output_retries"]) != 0 for r in rows):
        raise RuntimeError("blind retries are forbidden")
    if any(bool(r.get("verifier_capacity_loophole_exploited")) for r in rows):
        raise RuntimeError("verifier loophole exploitation detected")

    manifest_names = {r["test_manifest_name"] for r in rows}
    manifest_oids = {r["test_manifest_git_blob_sha1"] for r in rows}
    manifest_hashes = {r["test_manifest_sha256"] for r in rows}
    source_hashes = {r["source_sha256"] for r in rows}
    if len(manifest_names) != 1 or len(manifest_oids) != 1 or len(manifest_hashes) != 1 or len(source_hashes) != 1:
        raise RuntimeError("blind identity disagreement across shards")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    if len(grouped) != EXPECTED_ENTRIES:
        raise RuntimeError(f"expected 7 blind candidates got {len(grouped)}")
    if any(len(v) != EXPECTED_RECORDS for v in grouped.values()):
        raise RuntimeError("every blind candidate must cover exactly 100 records")

    summaries = {name: summarize(group) for name, group in grouped.items()}
    by_arm = {}
    for arm in ARM_ORDER:
        matches = [v for v in summaries.values() if v["arm"] == arm]
        if len(matches) != 1:
            raise RuntimeError(f"expected one frozen candidate for arm {arm}, got {len(matches)}")
        by_arm[arm] = matches[0]

    full = by_arm["v6_full"]
    no_transfer = by_arm["v6_no_transfer"]
    strong = by_arm["strong_baseline"]
    ablation = by_arm["recipe_removal_ablation"]

    equal_validity_retries_no_transfer = (
        full["valid"] == no_transfer["valid"]
        and full["invalid_output_retries"] == no_transfer["invalid_output_retries"]
    )
    full_no_transfer_ratio = (
        full["harmonic_speedup"] / no_transfer["harmonic_speedup"]
        if no_transfer["harmonic_speedup"] > 0 else math.inf
    )
    causal_separation = (
        (not no_transfer["passes_clean_blind_gate"])
        or (full_no_transfer_ratio >= 1.25 and equal_validity_retries_no_transfer)
    )

    equal_validity_retries_ablation = (
        full["valid"] == ablation["valid"]
        and full["invalid_output_retries"] == ablation["invalid_output_retries"]
    )
    full_ablation_ratio = (
        full["harmonic_speedup"] / ablation["harmonic_speedup"]
        if ablation["harmonic_speedup"] > 0 else math.inf
    )
    recipe_removal_eliminates_advantage = (
        (not ablation["passes_clean_blind_gate"])
        or (full_ablation_ratio >= 1.25 and equal_validity_retries_ablation)
    )

    baseline_valid_same_denominator = strong["records"] == 100 and strong["valid"] == 100 and strong["invalid_output_retries"] == 0
    baseline_competitiveness = (
        full["harmonic_speedup"] / strong["harmonic_speedup"]
        if strong["harmonic_speedup"] > 0 else math.inf
    )
    baseline_competitiveness_passes = baseline_competitiveness >= 0.80

    lock = json.loads((Path(__file__).resolve().parent / "BLIND_R1_LOCK.json").read_text())
    selected_pair_semantically_distinct = bool(lock["causal_preblind_diagnostics"]["selected_full_no_transfer_semantically_distinct"])
    source_family_differs = bool(lock["causal_preblind_diagnostics"]["source_family_differs_from_current"])
    selected_full_uses_learned_transfer = len(full["transfer_ids"]) > 0

    causal_conditions = {
        "full_passes_clean_blind_gate": bool(full["passes_clean_blind_gate"]),
        "selected_full_uses_learned_transfer": selected_full_uses_learned_transfer,
        "selected_full_no_transfer_semantically_distinct": selected_pair_semantically_distinct,
        "source_family_differs_from_current": source_family_differs,
        "causal_separation": bool(causal_separation),
        "recipe_removal_eliminates_qualifying_advantage": bool(recipe_removal_eliminates_advantage),
        "strong_baseline_valid_same_denominator": bool(baseline_valid_same_denominator),
        "strong_baseline_competitiveness_passes": bool(baseline_competitiveness_passes),
    }
    baseline_qualified_causal_transfer_win = all(causal_conditions.values())

    clean_wins = {arm: bool(by_arm[arm]["passes_clean_blind_gate"]) for arm in ARM_ORDER if arm != "recipe_removal_ablation"}
    payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda x: (int(x["index"]), x["arm"], x["candidate"]))) + "\n"
    summary = {
        "campaign": "LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication",
        "task_index": 1,
        "task": "max_flow_min_cost",
        "family": "numerical_optimization",
        "stage": "official_blind_r1",
        "blind_records": EXPECTED_RECORDS,
        "blind_entries": EXPECTED_ENTRIES,
        "candidate_evaluations": len(rows),
        "test_manifest_name": next(iter(manifest_names)),
        "test_manifest_git_blob_sha1": next(iter(manifest_oids)),
        "test_manifest_sha256": next(iter(manifest_hashes)),
        "source_sha256": next(iter(source_hashes)),
        "frozen_clean_gate": {
            "valid_required": 100,
            "harmonic_speedup_minimum": 1.50,
            "minimum_speedup": 1.05,
            "invalid_output_retries": 0,
        },
        "by_arm": by_arm,
        "clean_wins": clean_wins,
        "full_no_transfer_harmonic_ratio": full_no_transfer_ratio,
        "full_recipe_removal_harmonic_ratio": full_ablation_ratio,
        "strong_baseline_time_over_full_time_harmonic": baseline_competitiveness,
        "causal_conditions": causal_conditions,
        "baseline_qualified_causal_transfer_win": baseline_qualified_causal_transfer_win,
        "causal_transfer_detected_but_baseline_uncompetitive": bool(all(v for k, v in causal_conditions.items() if k != "strong_baseline_competitiveness_passes") and not baseline_competitiveness_passes),
        "preblind_full_no_transfer_equivalence_flag": not selected_pair_semantically_distinct,
        "preblind_recipe_removal_equivalence_flag": bool(lock["recipe_removal_ablation"]["preblind_semantic_equivalence_to_full"]),
        "invalid_output_retries_total": sum(int(r["invalid_output_retries"]) for r in rows),
        "verifier_capacity_loophole_exploited": False,
        "results_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "blind_run_complete": True,
        "post_blind_candidate_revision_allowed": False,
        "post_blind_timing_rerun_allowed": False,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-results.jsonl").write_text(payload)
    (args.output / "blind-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "clean_wins": clean_wins,
        "full_harmonic": full["harmonic_speedup"],
        "full_minimum": full["minimum_speedup"],
        "full_no_transfer_ratio": full_no_transfer_ratio,
        "baseline_qualified_causal_transfer_win": baseline_qualified_causal_transfer_win,
        "results_sha256": summary["results_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
