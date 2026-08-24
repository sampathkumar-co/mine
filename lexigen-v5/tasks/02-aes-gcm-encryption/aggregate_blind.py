from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED = {
    "v5_full": "v5_full_r4_7653e3865aa7a6def4dc",
    "v5_no_transfer": "v5_no_transfer_r5_0a141855078f60fe2b98",
    "random_search": "random_search_r5_ae3b52160647eaf9707e",
    "static_template": "static_template_r3_820b1c309b6117eb268d",
    "v4_compatible": "v4_compatible_r1_bd9a928b0a959b433de2",
}


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((row for row in rows if row["arm"] == arm and row["candidate"] == candidate), key=lambda row: int(row["index"]))
    if len(selected) != 100 or len({int(row["index"]) for row in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain 100 unique blind records")
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
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected),
        "implementation_class": "high_level_exact"
    }
    result["passes_blind_gate"] = bool(
        int(result["valid"]) == VALID_REQUIRED
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
        and int(result["invalid_output_retries"]) == 0
    )
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    shard_files=sorted(args.input.rglob("blind-shard-*.jsonl"))
    if len(shard_files)!=10:
        raise RuntimeError(f"expected 10 blind shard files, got {len(shard_files)}")
    rows=[]
    for path in shard_files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows)!=500:
        raise RuntimeError(f"expected 500 blind rows, got {len(rows)}")
    expected_pairs=set(EXPECTED.items())
    seen_pairs={(str(row["arm"]),str(row["candidate"])) for row in rows}
    if seen_pairs!=expected_pairs:
        raise RuntimeError(f"blind identity mismatch: missing={sorted(expected_pairs-seen_pairs)} extra={sorted(seen_pairs-expected_pairs)}")
    per_index={}
    for row in rows:
        index=int(row["index"]); per_index[index]=per_index.get(index,0)+1
    if set(per_index)!=set(range(1,101)) or any(count!=5 for count in per_index.values()):
        raise RuntimeError("each blind record must contain exactly five arm rows")
    manifest_ids={(str(row["test_manifest_name"]),str(row["test_manifest_git_blob_sha1"]),str(row["test_manifest_sha256"])) for row in rows}
    if len(manifest_ids)!=1:
        raise RuntimeError("blind manifest identity differs across shards")
    manifest_name,manifest_blob,manifest_sha256=next(iter(manifest_ids))
    dimensions={(int(row["key_size"]),int(row["nonce_size"]),int(row["plaintext_size"]),int(row["aad_size"])) for row in rows}
    if dimensions != {(16,12,298596352,32)}:
        raise RuntimeError(f"unexpected blind AES dimensions: {dimensions}")

    arms={arm:summarise(rows,arm,candidate) for arm,candidate in EXPECTED.items()}
    full=arms["v5_full"]; no_transfer=arms["v5_no_transfer"]; random=arms["random_search"]; static=arms["static_template"]; v4=arms["v4_compatible"]
    architecture={
        "v5_minus_no_transfer_harmonic":float(full["harmonic_speedup"])-float(no_transfer["harmonic_speedup"]),
        "v5_over_no_transfer_ratio":float(full["harmonic_speedup"])/max(float(no_transfer["harmonic_speedup"]),1e-12),
        "v5_minus_random_harmonic":float(full["harmonic_speedup"])-float(random["harmonic_speedup"]),
        "v5_minus_static_harmonic":float(full["harmonic_speedup"])-float(static["harmonic_speedup"]),
        "v5_minus_v4_harmonic":float(full["harmonic_speedup"])-float(v4["harmonic_speedup"]),
        "v5_beats_v4_by_task_win":bool(full["passes_blind_gate"] and not v4["passes_blind_gate"]),
        "all_selected_arms_implementation_equivalent":True,
        "selected_v5_uses_learned_transfer":False,
        "causal_transfer_credit":False,
        "causal_transfer_credit_reason":"Frozen blind selection uses no learned transfer causal ID and all five selected arms execute the same high_level_exact AESGCM implementation class. Implementation-equivalent timing differences receive zero transfer credit.",
        "recipe_removal_replay_required":False
    }
    report={
        "campaign":"LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index":2,
        "task":"aes_gcm_encryption",
        "family":"cryptography_encoding",
        "stage":"blind_r1",
        "test_manifest_name":manifest_name,
        "test_manifest_git_blob_sha1":manifest_blob,
        "test_manifest_sha256":manifest_sha256,
        "blind_records":100,
        "raw_record_count":len(rows),
        "key_size":16,"nonce_size":12,"plaintext_size":298596352,"aad_size":32,
        "selected_by_arm":EXPECTED,
        "frozen_gate":{"valid_required":100,"harmonic_speedup_minimum":1.5,"minimum_speedup":1.05,"invalid_output_retries":0},
        "arms":arms,
        "task_blind_status":"passed" if bool(full["passes_blind_gate"]) else "failed",
        "clean_unseen_task_win":bool(full["passes_blind_gate"]),
        "causal_transfer_win":False,
        "architecture_comparison":architecture,
        "invalid_output_retries":sum(int(row["invalid_output_retries"]) for row in rows),
        "blind_reruns":0,
        "thresholds_changed":False
    }
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"blind-summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (args.output/"blind-results.jsonl").write_text("\n".join(json.dumps(row,separators=(",",":")) for row in sorted(rows,key=lambda row:(int(row["index"]),str(row["arm"]))))+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
