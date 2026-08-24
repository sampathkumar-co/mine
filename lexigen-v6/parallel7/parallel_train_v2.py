from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from parallel_candidates import REFERENCE_SOLVERS
from parallel_train import EXPECTED_RECORDS, REVISION, SHARDS, SOURCE_COMMIT, TASKS, decode_value, fetch, pretrain_certificate, timed, verify


def discover_manifest_pair(task: str) -> tuple[str,str,dict]:
    url=f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{task}?recursive=false&expand=false"
    raw=fetch(url,ua="LEXIGEN-v6-parallel7-metadata-r2")
    entries=json.loads(raw.decode("utf-8"))
    names=[]; metadata={}
    for entry in entries:
        path=str(entry.get("path", "")); name=path.rsplit("/",1)[-1]
        if name:
            names.append(name); metadata[name]={"size":entry.get("size"),"oid":entry.get("oid"),"type":entry.get("type")}
    tests=set(n for n in names if n.endswith("_test.jsonl"))
    pairs=[]
    for train in sorted(n for n in names if n.endswith("_train.jsonl")):
        test=train[:-len("_train.jsonl")]+"_test.jsonl"
        if test in tests: pairs.append((train,test))
    if len(pairs)!=1:
        size100=[p for p in pairs if "_size100_" in p[0] or p[0].endswith("_size100_train.jsonl")]
        if len(size100)==1: pairs=size100
    if len(pairs)!=1:
        raise RuntimeError(f"{task}: expected one deterministic train/test manifest pair, found {pairs}; names={sorted(names)}")
    train,test=pairs[0]
    return train,test,{"tree_url":url,"train":metadata.get(train,{}),"test":metadata.get(test,{}),"all_names":sorted(names)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=sorted(TASKS),required=True); ap.add_argument("--shard",type=int,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    if not 0<=args.shard<SHARDS: raise ValueError("invalid shard")
    task=args.task; cfg=TASKS[task]
    lock=json.loads(Path("lexigen-v6/parallel7/PARALLEL7_LOCK.json").read_text())
    if lock["official_training_manifests_opened_before_lock"] or lock["official_test_manifests_opened_before_lock"] or lock["official_payloads_opened_before_lock"]: raise RuntimeError("parallel7 data boundary crossed")
    if not lock["execution_deduplication_by_frozen_implementation_class"]: raise RuntimeError("dedup policy mismatch")
    src_url=f"https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}/{cfg['source_path']}"; src_raw=fetch(src_url); src_sha=hashlib.sha256(src_raw).hexdigest()
    if src_sha!=cfg["source_sha256"]: raise RuntimeError(f"source sha mismatch {task} {src_sha}")
    source_text=src_raw.decode("utf-8")
    candidates,synthetic_rows=pretrain_certificate(task,source_text)
    class_members=defaultdict(list)
    for c in candidates: class_members[c.implementation_class].append(c)
    reps={impl:members[0] for impl,members in class_members.items()}

    train_name,test_name,tree_meta=discover_manifest_pair(task)
    base=f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{task}"
    manifest=fetch(f"{base}/{train_name}?download=true"); manifest_sha=hashlib.sha256(manifest).hexdigest(); records=[json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    if len(records)!=EXPECTED_RECORDS: raise RuntimeError(f"{task}: expected 100 train records got {len(records)}")
    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        problem=decode_value(row["problem"],base)
        classes=list(reps); shift=idx%len(classes); classes=classes[shift:]+classes[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem); class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}; execution_order="reference_first"
        else:
            class_runs={impl:timed(reps[impl].solve,problem) for impl in classes}; ref,ref_ns,ref_err=timed(REFERENCE_SOLVERS[task],problem); execution_order="classes_first"
        if ref_err or ref is None or ref_ns is None: raise RuntimeError(f"{task} reference failed record={idx+1}: {ref_err}")
        for impl,members in class_members.items():
            got,c_ns,c_err=class_runs[impl]
            if c_err is None: valid,reason,metrics=verify(task,problem,got,ref)
            else: valid,reason,metrics=False,"exception",{}
            for c in members:
                evidence.append({
                    "task":task,"task_index":cfg["index"],"family":cfg["family"],"index":idx+1,"seed":int(row.get("seed",idx+1)),
                    "arm":c.arm,"candidate":c.name,"implementation_class":c.implementation_class,"operators":list(c.operators),"transfer_ids":list(c.transfer_ids),"learned_template":c.learned_template,"baseline_id":c.baseline_id,
                    "valid":bool(valid and c_err is None),"failure_reason":c_err or reason,"candidate_ns":c_ns,"reference_ns":ref_ns,"speedup":(ref_ns/c_ns) if c_ns and c_ns>0 else 0.0,
                    "shared_execution_class":True,"class_candidate_count":len(members),"execution_order":execution_order,"shard":args.shard,"invalid_output_retries":0,
                    "train_manifest_name":train_name,"train_manifest_sha256":manifest_sha,"train_manifest_tree_metadata":tree_meta["train"],"expected_test_manifest_name":test_name,"expected_test_manifest_tree_metadata":tree_meta["test"],"source_sha256":src_sha,"test_manifest_contents_opened":False,"test_payloads_opened":0,
                    **metrics,
                })
        del problem,ref,class_runs; gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("\n".join(json.dumps(x,separators=(",",":")) for x in evidence)+"\n")
    cert_sha=hashlib.sha256(json.dumps(synthetic_rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    print(json.dumps({"task":task,"shard":args.shard,"candidate_rows":len(evidence),"implementation_classes":sorted(reps),"synthetic_certificate_sha256":cert_sha,"train_manifest_name":train_name,"train_manifest_sha256":manifest_sha,"test_manifest_name_metadata_only":test_name,"test_opened":False},indent=2))

if __name__=="__main__": main()
