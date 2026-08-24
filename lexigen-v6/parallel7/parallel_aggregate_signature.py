from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARM_ORDER=("v6_full","v6_no_transfer","random_search","static_template","v5_compatible","strong_baseline")
EXPECTED_RECORDS=100
EXPECTED_CANDIDATES=31
EXPECTED_SHARDS=5


def harmonic(values):
    vals=[float(x) for x in values]
    if not vals or any((not math.isfinite(x) or x<=0) for x in vals): return 0.0
    return len(vals)/sum(1/x for x in vals)


def signature_dict(r):
    return {
        "arm":r["arm"],
        "implementation_class":r["implementation_class"],
        "operators":list(r["operators"]),
        "transfer_ids":list(r["transfer_ids"]),
        "learned_template":r.get("learned_template"),
        "baseline_id":r.get("baseline_id"),
    }


def signature_key(r):
    d=signature_dict(r)
    return json.dumps(d,sort_keys=True,separators=(",",":"))


def canonical_candidate(task,key):
    d=json.loads(key); h=hashlib.sha256(key.encode()).hexdigest()[:16]
    if d.get("baseline_id")=="SB-GRAPH-BITSET-01" and task=="edge_expansion":
        return "strong_baseline_sb_graph_bitset_01_edge"
    return f"{task}_{d['arm']}_sig_{h}"


def summarize(task,key,rows):
    speeds=[float(r["speedup"]) for r in rows]; valid=sum(bool(r["valid"]) for r in rows); retries=sum(int(r["invalid_output_retries"]) for r in rows)
    spec=json.loads(key); cid=canonical_candidate(task,key)
    out={"arm":spec["arm"],"candidate":cid,"candidate_spec_sha256":hashlib.sha256(key.encode()).hexdigest(),"implementation_class":spec["implementation_class"],"operators":spec["operators"],"transfer_ids":spec["transfer_ids"],"learned_template":spec["learned_template"],"baseline_id":spec["baseline_id"],"records":len(rows),"valid":valid,"invalid_outputs":len(rows)-valid,"invalid_output_retries":retries,"harmonic_speedup":harmonic(speeds),"minimum_speedup":min(speeds),"median_speedup":statistics.median(speeds),"maximum_speedup":max(speeds),"shared_execution_class":all(bool(r.get("shared_execution_class")) for r in rows),"candidate_identity_recovered_from_frozen_signature":True}
    out["passes_frozen_performance_gate"]=(out["records"]==100 and valid==100 and retries==0 and out["harmonic_speedup"]>=1.50 and out["minimum_speedup"]>=1.05)
    return out


def choose(rows):
    return sorted(rows,key=lambda s:(-int(s["valid"]==100),-int(s["passes_frozen_performance_gate"]),-s["harmonic_speedup"],-s["minimum_speedup"],s["candidate_spec_sha256"]))[0]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",required=True); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    files=sorted(args.input.glob(f"{args.task}-shard-*.jsonl"))
    if len(files)!=EXPECTED_SHARDS: raise RuntimeError(f"{args.task}: expected {EXPECTED_SHARDS} shards got {len(files)}")
    rows=[]
    for p in files: rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    if len(rows)!=EXPECTED_RECORDS*EXPECTED_CANDIDATES: raise RuntimeError(f"{args.task}: expected {EXPECTED_RECORDS*EXPECTED_CANDIDATES} rows got {len(rows)}")
    grouped=defaultdict(list)
    for r in rows:
        if r["task"]!=args.task: raise RuntimeError("task contamination")
        grouped[signature_key(r)].append(r)
    if len(grouped)!=EXPECTED_CANDIDATES: raise RuntimeError(f"{args.task}: expected 31 frozen candidate signatures got {len(grouped)}")
    if any(len(v)!=EXPECTED_RECORDS for v in grouped.values()): raise RuntimeError(f"{args.task}: candidate signature denominator mismatch")
    per_shard=[]
    for shard in range(EXPECTED_SHARDS):
        ss={signature_key(r) for r in rows if int(r["shard"])==shard}
        per_shard.append(ss)
    if not all(s==per_shard[0] for s in per_shard): raise RuntimeError(f"{args.task}: frozen candidate signatures differ across shards")
    summaries=[summarize(args.task,key,group) for key,group in grouped.items()]
    by_arm={}
    for arm in ARM_ORDER:
        candidates=[s for s in summaries if s["arm"]==arm]; expected=1 if arm=="strong_baseline" else 6
        if len(candidates)!=expected: raise RuntimeError(f"{args.task}: arm {arm} expected {expected} signatures got {len(candidates)}")
        by_arm[arm]=choose(candidates)
    full=by_arm["v6_full"]; nt=by_arm["v6_no_transfer"]; strong=by_arm["strong_baseline"]
    full_nt_ratio=(full["harmonic_speedup"]/nt["harmonic_speedup"]) if nt["harmonic_speedup"]>0 else math.inf
    baseline_ratio=(full["harmonic_speedup"]/strong["harmonic_speedup"]) if strong["harmonic_speedup"]>0 else math.inf
    normalized=[]
    for r in rows:
        key=signature_key(r); nr=dict(r)
        overwritten_metric=nr.get("candidate")
        nr["verification_metric_candidate"]=overwritten_metric
        nr["candidate"]=canonical_candidate(args.task,key)
        nr["candidate_spec_sha256"]=hashlib.sha256(key.encode()).hexdigest()
        nr["candidate_identity_recovered_from_frozen_signature"]=True
        normalized.append(nr)
    payload="\n".join(json.dumps(r,separators=(",",":")) for r in sorted(normalized,key=lambda r:(int(r["index"]),r["arm"],r["candidate_spec_sha256"])))+"\n"
    manifest_hashes={r["train_manifest_sha256"] for r in rows}; source_hashes={r["source_sha256"] for r in rows}; manifest_names={r["train_manifest_name"] for r in rows}; test_names={r["expected_test_manifest_name"] for r in rows}
    train_meta={json.dumps(r.get("train_manifest_tree_metadata",{}),sort_keys=True) for r in rows}; test_meta={json.dumps(r.get("expected_test_manifest_tree_metadata",{}),sort_keys=True) for r in rows}
    if len(manifest_hashes)!=1 or len(source_hashes)!=1 or len(manifest_names)!=1 or len(test_names)!=1 or len(train_meta)!=1 or len(test_meta)!=1: raise RuntimeError("identity mismatch across shards")
    summary={"campaign":"LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication","stage":"official_training_r1_parallel7_sealed_signature_recovery","task":args.task,"task_index":rows[0]["task_index"],"family":rows[0]["family"],"training_records":100,"candidate_count":31,"candidate_evaluations":len(rows),"unique_implementation_classes":len({s["implementation_class"] for s in summaries}),"execution_deduplication_by_frozen_implementation_class":True,"candidate_identity_recovery":{"required":True,"reason":"verification metrics used reserved top-level key candidate in Task 6 evidence rows","frozen_signature_fields":["arm","implementation_class","operators","transfer_ids","learned_template","baseline_id"],"all_five_shards_have_identical_31_signature_set":True,"training_rerun_performed":False},"train_manifest_name":next(iter(manifest_names)),"train_manifest_sha256":next(iter(manifest_hashes)),"train_manifest_tree_metadata":json.loads(next(iter(train_meta))),"expected_test_manifest_name":next(iter(test_names)),"expected_test_manifest_tree_metadata":json.loads(next(iter(test_meta))),"official_test_manifest_opened":False,"official_test_payloads_opened":0,"source_sha256":next(iter(source_hashes)),"frozen_gate":{"valid_required":100,"harmonic_speedup_minimum":1.50,"minimum_speedup":1.05,"invalid_output_retries":0},"selected_by_arm":by_arm,"all_candidate_summaries":sorted(summaries,key=lambda s:(s["arm"],s["candidate_spec_sha256"])),"training_diagnostics":{"v6_full_over_no_transfer_harmonic_ratio":full_nt_ratio,"selected_full_uses_learned_transfer":bool(full["transfer_ids"]),"selected_full_no_transfer_implementation_class_equal":full["implementation_class"]==nt["implementation_class"],"selected_full_no_transfer_semantically_distinct":full["implementation_class"]!=nt["implementation_class"],"strong_baseline_time_over_full_time_harmonic":baseline_ratio,"full_passes_training_performance_gate":bool(full["passes_frozen_performance_gate"])},"candidates_passing_training_performance_gate":sum(bool(s["passes_frozen_performance_gate"]) for s in summaries),"results_sha256":hashlib.sha256(payload.encode()).hexdigest(),"scientific_candidate_revision_after_training":False,"threshold_changes":False}
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"train-results.jsonl").write_text(payload); (args.output/"train-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"task":args.task,"candidate_signatures":len(grouped),"full_harmonic":full["harmonic_speedup"],"full_minimum":full["minimum_speedup"],"full_passes":full["passes_frozen_performance_gate"],"full_no_transfer_distinct":full["implementation_class"]!=nt["implementation_class"],"results_sha256":summary["results_sha256"]},indent=2))

if __name__=="__main__": main()
