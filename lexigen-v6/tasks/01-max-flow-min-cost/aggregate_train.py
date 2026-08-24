from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
CURRENT_FAMILY = "numerical_optimization"
ARM_ORDER = ("v6_full", "v6_no_transfer", "random_search", "static_template", "v5_compatible", "strong_baseline")


def harmonic(values: list[float]) -> float:
    return len(values) / sum(1.0 / x for x in values) if values and all(x > 0 for x in values) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.rglob("train-shard-*.jsonl"))
    if len(files) != 10:
        raise RuntimeError(f"expected 10 shards got {len(files)}")
    rows = [json.loads(line) for file in files for line in file.read_text().splitlines() if line.strip()]
    if len(rows) != 3100:
        raise RuntimeError(f"expected 3100 rows got {len(rows)}")
    if any(bool(r.get("verifier_capacity_loophole_exploited")) for r in rows):
        raise RuntimeError("forbidden verifier loophole exploitation recorded")

    identities = {(r["train_manifest_name"], r["train_manifest_git_blob_sha1"], r["train_manifest_sha256"], r["expected_test_manifest_name"], r["expected_test_manifest_tree_oid"], int(r["expected_test_manifest_size"]), r["source_sha256"]) for r in rows}
    if len(identities) != 1:
        raise RuntimeError("manifest/source identity mismatch across shards")
    train_name, train_oid, train_sha256, test_name, test_oid, test_size, source_sha256 = next(iter(identities))

    candidate_keys = sorted({(r["arm"], r["candidate"]) for r in rows}, key=lambda x: (ARM_ORDER.index(x[0]), x[1]))
    if len(candidate_keys) != 31:
        raise RuntimeError(f"expected 31 candidates got {len(candidate_keys)}")

    summaries = []
    for arm, name in candidate_keys:
        subset = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == name), key=lambda r: int(r["index"]))
        if len(subset) != 100 or len({int(r["index"]) for r in subset}) != 100:
            raise RuntimeError(f"coverage failure {arm}/{name}")
        meta_fields = ("implementation_class", "learned_template", "baseline_id")
        for field in meta_fields:
            if len({json.dumps(r.get(field), sort_keys=True) for r in subset}) != 1:
                raise RuntimeError(f"metadata mismatch {arm}/{name}/{field}")
        transfer_ids_set = {tuple(r.get("transfer_ids", [])) for r in subset}
        operators_set = {tuple(r.get("operators", [])) for r in subset}
        if len(transfer_ids_set) != 1 or len(operators_set) != 1:
            raise RuntimeError(f"proposal metadata mismatch {arm}/{name}")
        speeds = [float(r["speedup"]) for r in subset]
        valid = sum(bool(r["valid"]) for r in subset)
        retries = sum(int(r["invalid_output_retries"]) for r in subset)
        summary = {
            "arm":arm,
            "candidate":name,
            "valid":valid,
            "invalid_outputs":100 - valid,
            "invalid_output_retries":retries,
            "harmonic_speedup":harmonic(speeds),
            "minimum_speedup":min(speeds),
            "median_speedup":statistics.median(speeds),
            "maximum_speedup":max(speeds),
            "implementation_class":subset[0]["implementation_class"],
            "operators":list(next(iter(operators_set))),
            "transfer_ids":list(next(iter(transfer_ids_set))),
            "learned_template":subset[0].get("learned_template"),
            "baseline_id":subset[0].get("baseline_id"),
        }
        summary["passes_training_correctness"] = valid == VALID_REQUIRED and retries == 0
        summary["passes_gate"] = summary["passes_training_correctness"] and summary["harmonic_speedup"] >= HARMONIC_REQUIRED and summary["minimum_speedup"] >= MINIMUM_REQUIRED
        summaries.append(summary)

    arms = {}
    for arm in ARM_ORDER:
        pool = [x for x in summaries if x["arm"] == arm]
        expected = 1 if arm == "strong_baseline" else 6
        if len(pool) != expected:
            raise RuntimeError(f"expected {expected} candidates in {arm}, got {len(pool)}")
        correct = [x for x in pool if x["passes_training_correctness"]]
        selection_pool = correct or pool
        selected = min(selection_pool, key=lambda x: (-int(x["passes_gate"]), -x["valid"], -x["harmonic_speedup"], -x["minimum_speedup"], -x["median_speedup"], x["candidate"]))
        arms[arm] = {
            "selected":selected,
            "candidate_count":len(pool),
            "correct_candidate_count":len(correct),
            "performance_gate_candidate_count":sum(bool(x["passes_gate"]) for x in pool),
        }

    full = arms["v6_full"]["selected"]
    no_transfer = arms["v6_no_transfer"]["selected"]
    strong = arms["strong_baseline"]["selected"]
    full_nt_ratio = full["harmonic_speedup"] / max(no_transfer["harmonic_speedup"], 1e-12)
    full_strong_runtime_competitiveness = full["harmonic_speedup"] / max(strong["harmonic_speedup"], 1e-12)

    memory = json.loads((Path(__file__).resolve().parents[2] / "TRANSFER_MEMORY.json").read_text())
    source_family_by_id = {str(v["causal_id"]): str(v["learned_from_family"]) for v in memory["learned_templates"].values()}
    selected_ids = list(full.get("transfer_ids", []))
    selected_causal_id = selected_ids[0] if selected_ids else None
    learned_source_family = source_family_by_id.get(selected_causal_id) if selected_causal_id else None
    source_family_differs = bool(learned_source_family and learned_source_family != CURRENT_FAMILY)
    selected_pair_distinct = full["implementation_class"] != no_transfer["implementation_class"]
    equal_validity_retries = full["valid"] == no_transfer["valid"] and full["invalid_output_retries"] == no_transfer["invalid_output_retries"]
    causal_separation = bool(selected_causal_id and selected_pair_distinct and full["passes_training_correctness"] and ((not no_transfer["passes_gate"]) or (equal_validity_retries and full_nt_ratio >= 1.25)))
    equivalent_in_no_transfer = any(x["implementation_class"] == full["implementation_class"] for x in summaries if x["arm"] == "v6_no_transfer")

    comparison = {
        "v6_full_harmonic":full["harmonic_speedup"],
        "v6_no_transfer_harmonic":no_transfer["harmonic_speedup"],
        "strong_baseline_harmonic":strong["harmonic_speedup"],
        "v6_full_over_no_transfer_ratio":full_nt_ratio,
        "strong_baseline_time_over_full_time_harmonic":full_strong_runtime_competitiveness,
        "strong_baseline_competitiveness_threshold":0.80,
        "training_strong_baseline_competitiveness_passes":bool(strong["passes_training_correctness"] and full_strong_runtime_competitiveness >= 0.80),
        "selected_v6_uses_learned_transfer":selected_causal_id is not None,
        "selected_causal_id":selected_causal_id,
        "learned_recipe_source_family":learned_source_family,
        "current_family":CURRENT_FAMILY,
        "source_family_differs_from_current":source_family_differs,
        "selected_v6_implementation_class":full["implementation_class"],
        "selected_no_transfer_implementation_class":no_transfer["implementation_class"],
        "selected_pair_semantically_distinct":selected_pair_distinct,
        "equivalent_implementation_available_in_no_transfer":equivalent_in_no_transfer,
        "training_causal_separation_condition":causal_separation,
        "training_baseline_qualified_causal_diagnostic":bool(full["passes_gate"] and selected_causal_id and source_family_differs and causal_separation and strong["passes_training_correctness"] and full_strong_runtime_competitiveness >= 0.80),
        "causal_transfer_credit":False,
        "reason_no_credit":"Training evidence only; v6 causal credit is blind-only and additionally requires frozen recipe-removal ablation.",
    }

    report = {
        "campaign":"LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication",
        "task_index":1,
        "task":"max_flow_min_cost",
        "family":CURRENT_FAMILY,
        "stage":"official_training_r1",
        "training_records":100,
        "candidate_count":31,
        "candidate_evaluations":3100,
        "train_manifest_name":train_name,
        "train_manifest_git_blob_sha1":train_oid,
        "train_manifest_sha256":train_sha256,
        "expected_test_manifest_name":test_name,
        "expected_test_manifest_tree_oid":test_oid,
        "expected_test_manifest_size":test_size,
        "source_sha256":source_sha256,
        "frozen_default_gate":{"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "arms":arms,
        "all_candidates":summaries,
        "architecture_comparison":comparison,
        "blind_selection_ready":all(arms[k]["correct_candidate_count"] > 0 for k in ARM_ORDER),
        "official_test_manifest_contents_opened":False,
        "official_test_payloads_opened":0,
        "verifier_capacity_loophole_exploited":False,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    ordered_rows = sorted(rows, key=lambda r: (int(r["index"]), ARM_ORDER.index(r["arm"]), r["candidate"]))
    (args.output / "training-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in ordered_rows) + "\n")
    print(json.dumps({"selected_by_arm":{k:v["selected"] for k,v in arms.items()},"comparison":comparison}, indent=2))


if __name__ == "__main__":
    main()
