from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARM_ORDER=("v6_full","v6_no_transfer","random_search","static_template","v5_compatible","strong_baseline","recipe_removal_ablation")
SOURCE_FAMILY={"TM-BFR-01":"graph_discrete","TM-CAC-01":"numerical_optimization","TM-RRR-01":"linear_algebra","TM-PBEB-01":"signal_processing"}
EXPECTED_RECORDS=100
EXPECTED_ENTRIES=7
EXPECTED_SHARDS=5


def harmonic(values):
    vals=[float(x) for x in values]
    if not vals or any((not math.isfinite(x) or x<=0) for x in vals): return 0.0
    return len(vals)/sum(1/x for x in vals)


def summarize(rows):
    speeds=[float(r["speedup"]) for r in rows]; valid=sum(bool(r["valid"]) for r in rows); retries=sum(int(r["invalid_output_retries"]) for r in rows)
    out={"arm":rows[0]["arm"],"candidate":rows[0]["candidate"],"implementation_class":rows[0]["implementation_class"],"operators":rows[0]["operators"],"transfer_ids":rows[0]["transfer_ids"],"learned_template":rows[0]["learned_template"],"baseline_id":rows[0]["baseline_id"],"recipe_removal_ablation":bool(rows[0].get("recipe_removal_ablation")),"records":len(rows),"valid":valid,"invalid_outputs":len(rows)-valid,"invalid_output_retries":retries,"harmonic_speedup":harmonic(speeds),"minimum_speedup":min(speeds),"median_speedup":statistics.median(speeds),"maximum_speedup":max(speeds),"shared_execution_class":all(bool(r.get("shared_execution_class")) for r in rows)}
    out["passes_clean_blind_gate"]=(out["records"]==100 and valid==100 and retries==0 and out["harmonic_speedup"]>=1.50 and out["minimum_speedup"]>=1.05)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",required=True); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    files=sorted(args.input.glob(f"{args.task}-blind-shard-*.jsonl"))
    if len(files)!=EXPECTED_SHARDS: raise RuntimeError(f"{args.task}: expected 5 blind shards got {len(files)}")
    rows=[]
    for p in files: rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    if len(rows)!=EXPECTED_RECORDS*EXPECTED_ENTRIES: raise RuntimeError(f"{args.task}: expected 700 rows got {len(rows)}")
    grouped=defaultdict(list)
    for r in rows:
        if r["task"]!=args.task: raise RuntimeError("task contamination")
        grouped[r["candidate"]].append(r)
    if len(grouped)!=EXPECTED_ENTRIES or any(len(v)!=100 for v in grouped.values()): raise RuntimeError(f"{args.task}: blind denominator mismatch")
    summaries=[summarize(v) for v in grouped.values()]; by_arm={}
    for arm in ARM_ORDER:
        matches=[s for s in summaries if s["arm"]==arm]
        if len(matches)!=1: raise RuntimeError(f"{args.task}: expected one {arm}, got {len(matches)}")
        by_arm[arm]=matches[0]
    full=by_arm["v6_full"]; nt=by_arm["v6_no_transfer"]; strong=by_arm["strong_baseline"]; abl=by_arm["recipe_removal_ablation"]
    eq_nt=(full["valid"]==nt["valid"] and full["invalid_output_retries"]==nt["invalid_output_retries"]); eq_abl=(full["valid"]==abl["valid"] and full["invalid_output_retries"]==abl["invalid_output_retries"])
    full_nt_ratio=full["harmonic_speedup"]/nt["harmonic_speedup"] if nt["harmonic_speedup"]>0 else math.inf
    full_abl_ratio=full["harmonic_speedup"]/abl["harmonic_speedup"] if abl["harmonic_speedup"]>0 else math.inf
    baseline_ratio=full["harmonic_speedup"]/strong["harmonic_speedup"] if strong["harmonic_speedup"]>0 else math.inf
    source_families=[SOURCE_FAMILY[x] for x in full["transfer_ids"] if x in SOURCE_FAMILY]
    source_family_differs=bool(source_families) and all(f!=rows[0]["family"] for f in source_families)
    causal_conditions={
        "full_passes_clean_blind_gate":bool(full["passes_clean_blind_gate"]),
        "selected_full_uses_learned_transfer":bool(full["transfer_ids"]),
        "selected_full_no_transfer_semantically_distinct":full["implementation_class"]!=nt["implementation_class"],
        "source_family_differs_from_current":source_family_differs,
        "causal_separation":bool((not nt["passes_clean_blind_gate"]) or (full_nt_ratio>=1.25 and eq_nt)),
        "recipe_removal_eliminates_qualifying_advantage":bool((not abl["passes_clean_blind_gate"]) or (full_abl_ratio>=1.25 and eq_abl)),
        "strong_baseline_valid_same_denominator":bool(strong["records"]==100 and strong["valid"]==100 and strong["invalid_output_retries"]==0),
        "strong_baseline_competitiveness_passes":bool(baseline_ratio>=0.80),
    }
    causal_win=all(causal_conditions.values())
    payload="\n".join(json.dumps(r,separators=(",",":")) for r in sorted(rows,key=lambda r:(int(r["index"]),r["arm"],r["candidate"])))+"\n"
    manifest_hashes={r["test_manifest_sha256"] for r in rows}; source_hashes={r["source_sha256"] for r in rows}; train_hashes={r["training_results_sha256"] for r in rows}
    if len(manifest_hashes)!=1 or len(source_hashes)!=1 or len(train_hashes)!=1: raise RuntimeError("blind identity mismatch")
    summary={
      "campaign":"LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication","stage":"official_blind_r1_parallel7","task":args.task,"task_index":rows[0]["task_index"],"family":rows[0]["family"],"blind_records":100,"blind_entries":7,"candidate_evaluations":len(rows),"test_manifest_name":rows[0]["test_manifest_name"],"test_manifest_sha256":next(iter(manifest_hashes)),"source_sha256":next(iter(source_hashes)),"training_results_sha256":next(iter(train_hashes)),
      "frozen_clean_gate":{"valid_required":100,"harmonic_speedup_minimum":1.50,"minimum_speedup":1.05,"invalid_output_retries":0},"by_arm":by_arm,
      "clean_wins":{arm:bool(by_arm[arm]["passes_clean_blind_gate"]) for arm in ARM_ORDER if arm!="recipe_removal_ablation"},
      "full_no_transfer_harmonic_ratio":full_nt_ratio,"full_recipe_removal_harmonic_ratio":full_abl_ratio,"strong_baseline_time_over_full_time_harmonic":baseline_ratio,
      "causal_conditions":causal_conditions,"baseline_qualified_causal_transfer_win":causal_win,"causal_transfer_detected_but_baseline_uncompetitive":bool(all(v for k,v in causal_conditions.items() if k!="strong_baseline_competitiveness_passes") and not causal_conditions["strong_baseline_competitiveness_passes"]),
      "preblind_full_no_transfer_equivalence_flag":full["implementation_class"]==nt["implementation_class"],"preblind_recipe_removal_equivalence_flag":full["implementation_class"]==abl["implementation_class"],"blind_run_complete":True,"invalid_output_retries_total":sum(int(r["invalid_output_retries"]) for r in rows),"results_sha256":hashlib.sha256(payload.encode()).hexdigest(),"post_blind_candidate_revision_allowed":False,"post_blind_timing_rerun_allowed":False,"threshold_changes":False,
    }
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"blind-results.jsonl").write_text(payload); (args.output/"blind-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"task":args.task,"full_harmonic":full["harmonic_speedup"],"full_minimum":full["minimum_speedup"],"full_clean":full["passes_clean_blind_gate"],"causal_win":causal_win,"results_sha256":summary["results_sha256"]},indent=2))

if __name__=="__main__": main()
