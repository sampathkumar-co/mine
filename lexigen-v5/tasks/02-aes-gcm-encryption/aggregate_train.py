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
        "v5_full_r2_a6102573c9f355414229",
        "v5_full_r3_b1ef08a2d68a248c0821",
        "v5_full_r4_7653e3865aa7a6def4dc",
        "v5_full_r5_c8350e5b9ffb9c400fc8",
        "v5_full_r6_ef8d1d6c3a4fa6d0be21"
    ],
    "v5_no_transfer": [
        "v5_no_transfer_r1_d14c06bd6ae45a8dd009",
        "v5_no_transfer_r2_14a8ffbc5159ff111ac9",
        "v5_no_transfer_r3_477905d60634240ebda9",
        "v5_no_transfer_r4_1b859d0377c9b2a19b53",
        "v5_no_transfer_r5_0a141855078f60fe2b98",
        "v5_no_transfer_r6_974e0d50c53bf65218c5"
    ],
    "random_search": [
        "random_search_r1_ed1f4796fdbc83a45b55",
        "random_search_r2_0520aab8496f3d685f92",
        "random_search_r3_e9c9180b0957bb3f7ca7",
        "random_search_r4_01baad563ce94255e3e2",
        "random_search_r5_ae3b52160647eaf9707e",
        "random_search_r6_8615e4a35db08222a26b"
    ],
    "static_template": [
        "static_template_r1_dbfcd2af539b0b2636e7",
        "static_template_r2_8fd871e046faa7e4d37c",
        "static_template_r3_820b1c309b6117eb268d",
        "static_template_r4_8f1dafda0d3fbc099aa9",
        "static_template_r5_d044a19fd4551034dc11",
        "static_template_r6_357e80313b8b9dc3cf36"
    ],
    "v4_compatible": [
        "v4_compatible_r1_bd9a928b0a959b433de2",
        "v4_compatible_r2_885bf4f21e819b330732",
        "v4_compatible_r3_695b907772d8a69a1186",
        "v4_compatible_r4_d9863922b850e9717a05",
        "v4_compatible_r5_cdae8cbf0d73bd4d047c",
        "v4_compatible_r6_af7d36f83a386b7726b9"
    ]
}

LEARNED = {
    "v5_full_r1_3304c859d463a501bd86": {"causal_id":"TM-BFR-01","learned_template":"bit_frontier_restriction","learned_from_family":"graph_discrete"},
    "v5_full_r2_a6102573c9f355414229": {"causal_id":"TM-CAC-01","learned_template":"certified_active_core","learned_from_family":"numerical_optimization"},
    "v5_full_r3_b1ef08a2d68a248c0821": {"causal_id":"TM-PBEB-01","learned_template":"precision_backend_error_budget","learned_from_family":"signal_processing"}
}

NATIVE = {
    "v5_full_r3_b1ef08a2d68a248c0821",
    "v5_no_transfer_r4_1b859d0377c9b2a19b53",
    "random_search_r3_e9c9180b0957bb3f7ca7",
    "random_search_r4_01baad563ce94255e3e2"
}


def implementation_class(candidate: str) -> str:
    return "native_cipher_exact" if candidate in NATIVE else "high_level_exact"


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
    result: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": len(valid_rows),
        "invalid_outputs": 100-len(valid_rows),
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "total_candidate_s": sum(candidate_times),
        "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected),
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "implementation_class": implementation_class(candidate),
        "learned_transfer": LEARNED.get(candidate)
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
        "discovery_cost_total_candidate_s": sum(float(row["total_candidate_s"]) for row in arm_rows)
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
    if expected_count != 30 or len(rows) != 3000:
        raise RuntimeError(f"expected 30 candidates/3000 rows, got {expected_count}/{len(rows)}")
    expected_pairs = {(arm,name) for arm,names in EXPECTED.items() for name in names}
    seen_pairs = {(str(row["arm"]),str(row["candidate"])) for row in rows}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"candidate identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index: dict[int,int] = {}
    for row in rows:
        index=int(row["index"]); per_index[index]=per_index.get(index,0)+1
    if set(per_index) != set(range(1,101)) or any(count != 30 for count in per_index.values()):
        raise RuntimeError("each training record must contain exactly 30 candidate rows")
    manifest_ids = {(str(row["train_manifest_name"]),str(row["train_manifest_tree_oid"]),str(row["train_manifest_git_blob_sha1"]),str(row["train_manifest_sha256"]),str(row["expected_test_manifest_name"]),str(row["expected_test_manifest_tree_oid"])) for row in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("manifest identity differs across shards")
    train_name,train_oid,train_blob,train_sha256,test_name,test_oid = next(iter(manifest_ids))
    plaintext_sizes = sorted({int(row["plaintext_size"]) for row in rows})
    key_sizes = sorted({int(row["key_size"]) for row in rows})
    aad_sizes = sorted({-1 if row["aad_size"] is None else int(row["aad_size"]) for row in rows})

    summaries=[summarise(rows,arm,name) for arm,names in EXPECTED.items() for name in names]
    arms={arm:select_arm(summaries,arm) for arm in EXPECTED}
    full=arms["v5_full"]["selected"]
    no_transfer=arms["v5_no_transfer"]["selected"]
    random=arms["random_search"]["selected"]
    static=arms["static_template"]["selected"]
    v4=arms["v4_compatible"]["selected"]
    full_class=str(full["implementation_class"])
    no_transfer_classes={implementation_class(name) for name in EXPECTED["v5_no_transfer"]}
    equivalent_in_no_transfer=full_class in no_transfer_classes
    full_learned=full.get("learned_transfer")
    comparison={
        "v5_full_harmonic":full["harmonic_speedup"],
        "v5_no_transfer_harmonic":no_transfer["harmonic_speedup"],
        "random_search_harmonic":random["harmonic_speedup"],
        "static_template_harmonic":static["harmonic_speedup"],
        "v4_compatible_harmonic":v4["harmonic_speedup"],
        "v5_minus_no_transfer_harmonic":float(full["harmonic_speedup"])-float(no_transfer["harmonic_speedup"]),
        "v5_over_no_transfer_ratio":float(full["harmonic_speedup"])/max(float(no_transfer["harmonic_speedup"]),1e-12),
        "v5_minus_random_harmonic":float(full["harmonic_speedup"])-float(random["harmonic_speedup"]),
        "v5_minus_static_harmonic":float(full["harmonic_speedup"])-float(static["harmonic_speedup"]),
        "v5_minus_v4_harmonic":float(full["harmonic_speedup"])-float(v4["harmonic_speedup"]),
        "selected_v5_uses_learned_transfer":full_learned is not None,
        "selected_v5_implementation_class":full_class,
        "equivalent_implementation_class_available_in_no_transfer":equivalent_in_no_transfer,
        "selected_v5_semantically_distinct_from_no_transfer_by_construction":bool(full_learned is not None and not equivalent_in_no_transfer),
        "training_causal_separation_threshold_crossed":bool(
            full_learned is not None
            and not equivalent_in_no_transfer
            and bool(full["passes_training_correctness"])
            and (not bool(no_transfer["passes_training_correctness"]) or float(full["harmonic_speedup"]) >= 1.25*float(no_transfer["harmonic_speedup"]))
        ),
        "causal_transfer_credit":False,
        "causal_transfer_credit_reason":"Blind result plus preregistered recipe-removal replay are required; implementation-equivalent paths receive zero credit."
    }
    report={
        "campaign":"LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index":2,
        "task":"aes_gcm_encryption",
        "revision":1,
        "stage":"official_training",
        "train_manifest_name":train_name,
        "train_manifest_tree_oid":train_oid,
        "train_manifest_git_blob_sha1":train_blob,
        "train_manifest_sha256":train_sha256,
        "expected_test_manifest_name":test_name,
        "expected_test_manifest_tree_oid":test_oid,
        "training_records":100,
        "candidate_count":30,
        "plaintext_sizes":plaintext_sizes,
        "key_sizes":key_sizes,
        "aad_sizes":aad_sizes,
        "frozen_default_gate":{"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "all_candidates":summaries,
        "arms":arms,
        "architecture_comparison":comparison,
        "v5_full_has_correct_training_candidate":bool(arms["v5_full"]["correct_candidate_count"]),
        "blind_selection_ready":bool(arms["v5_full"]["correct_candidate_count"]),
        "training_revision_consumed":True,
        "official_test_manifest_contents_opened":False,
        "official_test_payloads_opened":0,
        "reports_opened":False,
        "public_solvers_opened":False
    }
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"training-summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (args.output/"training-results.jsonl").write_text("\n".join(json.dumps(row,separators=(",",":")) for row in sorted(rows,key=lambda row:(int(row["index"]),str(row["arm"]),str(row["candidate"]))))+"\n",encoding="utf-8")
    print(json.dumps({"selected_by_arm":{arm:value["selected"] for arm,value in arms.items()},"architecture_comparison":comparison,"plaintext_sizes":plaintext_sizes,"blind_selection_ready":report["blind_selection_ready"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
