from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from parallel_blind import preblind_smoke, selected_entries
from parallel_candidates import REFERENCE_SOLVERS
from parallel_train import EXPECTED_RECORDS, REVISION, SHARDS, SOURCE_COMMIT, TASKS, decode_value, fetch, timed, verify


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=sorted(TASKS),required=True)
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if not 0<=args.shard<SHARDS: raise ValueError("invalid shard")
    task=args.task; cfg=TASKS[task]
    lock=json.loads(Path("lexigen-v6/parallel7/PARALLEL7_BLIND_LOCK.json").read_text())
    if lock["official_test_manifests_opened_before_lock"] or lock["official_test_payloads_opened_before_lock"]:
        raise RuntimeError("parallel7 blind boundary crossed")
    if lock["blind_runs_completed_before_trigger"]!=0 or not lock["exactly_one_official_blind_run_per_task_allowed"]:
        raise RuntimeError("blind run budget mismatch")

    src_raw=fetch(f"https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}/{cfg['source_path']}")
    src_sha=hashlib.sha256(src_raw).hexdigest()
    if src_sha!=cfg["source_sha256"]: raise RuntimeError(f"{task}: source sha mismatch")
    source_text=src_raw.decode("utf-8")

    train_summary,entries=selected_entries(task,source_text)
    if train_summary.get("official_test_manifest_opened") is not False or int(train_summary.get("official_test_payloads_opened",-1))!=0:
        raise RuntimeError(f"{task}: training summary crossed test boundary")
    test_name=str(train_summary["expected_test_manifest_name"])
    if not test_name.endswith("_test.jsonl") or "/" in test_name or ".." in test_name:
        raise RuntimeError(f"{task}: unsafe sealed test manifest name {test_name}")
    expected_test_meta=train_summary.get("expected_test_manifest_tree_metadata",{})

    smoke=preblind_smoke(task,source_text,entries)
    class_members=defaultdict(list)
    for c in entries: class_members[c.implementation_class].append(c)
    reps={impl:members[0] for impl,members in class_members.items()}

    base=f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{task}"
    manifest=fetch(f"{base}/{test_name}?download=true",ua="LEXIGEN-v6-parallel7-blind-r1")
    if expected_test_meta.get("size") is not None and len(manifest)!=int(expected_test_meta["size"]):
        raise RuntimeError(f"{task}: sealed test manifest size mismatch {len(manifest)} != {expected_test_meta['size']}")
    manifest_sha=hashlib.sha256(manifest).hexdigest()
    records=[json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    if len(records)!=EXPECTED_RECORDS: raise RuntimeError(f"{task}: expected 100 test records got {len(records)}")

    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        problem=decode_value(row["problem"],base)
        classes=list(reps); shift=idx%len(classes); classes=classes[shift:]+classes[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem)
            class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}
            execution_order="reference_first"
        else:
            class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}
            ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem)
            execution_order="classes_first"
        if ref_err or ref is None or ref_ns is None:
            raise RuntimeError(f"{task}: blind reference failed record={idx+1}: {ref_err}")
        for impl,members in class_members.items():
            got,c_ns,c_err=class_runs[impl]
            if c_err is None: valid,reason,metrics=verify(task,problem,got,ref)
            else: valid,reason,metrics=False,"exception",{}
            for c in members:
                evidence.append({
                    "task":task,"task_index":cfg["index"],"family":cfg["family"],"index":idx+1,"seed":int(row.get("seed",idx+1)),
                    "arm":c.arm,"candidate":c.name,"implementation_class":c.implementation_class,"operators":list(c.operators),"transfer_ids":list(c.transfer_ids),"learned_template":c.learned_template,"baseline_id":c.baseline_id,
                    "recipe_removal_ablation":c.arm=="recipe_removal_ablation","valid":bool(valid and c_err is None),"failure_reason":c_err or reason,"candidate_ns":c_ns,"reference_ns":ref_ns,"speedup":(ref_ns/c_ns) if c_ns and c_ns>0 else 0.0,
                    "shared_execution_class":True,"class_candidate_count":len(members),"execution_order":execution_order,"shard":args.shard,"invalid_output_retries":0,
                    "test_manifest_name":test_name,"test_manifest_sha256":manifest_sha,"test_manifest_tree_metadata":expected_test_meta,"source_sha256":src_sha,"training_results_sha256":train_summary["results_sha256"],**metrics,
                })
        del problem,ref,class_runs; gc.collect()

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(json.dumps(x,separators=(",",":")) for x in evidence)+"\n")
    smoke_sha=hashlib.sha256(json.dumps(smoke,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    print(json.dumps({"task":task,"shard":args.shard,"rows":len(evidence),"implementation_classes":sorted(reps),"preblind_smoke_sha256":smoke_sha,"sealed_test_manifest_name":test_name,"test_manifest_sha256":manifest_sha},indent=2))

if __name__=="__main__": main()
