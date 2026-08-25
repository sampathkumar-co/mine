from __future__ import annotations

import argparse,gc,hashlib,json
from pathlib import Path

from train_shard import (
    SHARDS,EXPECTED_RECORDS,TEST_NAME,TEST_OID,TEST_SIZE,BASE,SOURCE_SHA256,
    decode_problem,fetch,git_blob,reference_exact,timed,verify,
)
from candidates import build_candidates

HERE=Path(__file__).resolve().parent
LOCK_NAME='BLIND_R1_LOCK.json'
SELECTED_ARMS=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')
ABLATION_ARM='recipe_removal_ablation'


def load_lock()->dict:
    lock=json.loads((HERE/LOCK_NAME).read_text())
    if lock.get('task')!='dct_type_I_scipy_fftpack' or lock.get('stage')!='blind_r1_lock':raise RuntimeError('invalid blind lock')
    if lock.get('official_test_manifest_opened_before_lock') or int(lock.get('official_test_payloads_opened_before_lock',-1))!=0:raise RuntimeError('blind boundary crossed before lock')
    if int(lock.get('blind_runs_completed_before_trigger',-1))!=0:raise RuntimeError('blind already consumed')
    return lock


def selected_candidates(source_text:str,lock:dict):
    arms=build_candidates(source_text);by_name={c.name:c for rows in arms.values() for c in rows};selected=[]
    for arm in SELECTED_ARMS:
        spec=lock['selected_by_arm'][arm];name=spec['candidate']
        if name not in by_name:raise RuntimeError(f'locked candidate missing: {name}')
        c=by_name[name]
        if c.arm!=arm:raise RuntimeError(f'arm mismatch {name}')
        if c.implementation_class!=spec['implementation_class']:raise RuntimeError(f'implementation mismatch {name}')
        if list(c.operators)!=spec['operators']:raise RuntimeError(f'operator mismatch {name}')
        if list(c.transfer_ids)!=spec['transfer_ids']:raise RuntimeError(f'transfer mismatch {name}')
        if c.learned_template!=spec.get('learned_template'):raise RuntimeError(f'template mismatch {name}')
        if c.baseline_id!=spec.get('baseline_id'):raise RuntimeError(f'baseline mismatch {name}')
        selected.append(c)
    full=selected[0];abl=lock['recipe_removal_ablation']
    if abl['source_candidate']!=full.name:raise RuntimeError('ablation source mismatch')
    if abl['removed_transfer_ids']!=list(full.transfer_ids):raise RuntimeError('ablation removed-transfer mismatch')
    if abl['resulting_transfer_ids']!=[]:raise RuntimeError('ablation must remove all transfer IDs')
    if abl['retained_operators']!=list(full.operators):raise RuntimeError('ablation retained-operator mismatch')
    if abl['resulting_implementation_class']!=full.implementation_class:raise RuntimeError('ablation implementation mismatch')
    if not abl['same_callable_as_full'] or not abl['preblind_semantic_equivalence_to_full']:raise RuntimeError('frozen recipe-removal equivalence not acknowledged')
    return selected,full


def evaluation_entries(selected,full,lock:dict):
    rows=[]
    for c in selected:
        rows.append({'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'solve':c.solve,'is_ablation':False})
    abl=lock['recipe_removal_ablation']
    rows.append({'arm':ABLATION_ARM,'candidate':abl['candidate'],'implementation_class':abl['resulting_implementation_class'],'operators':abl['retained_operators'],'transfer_ids':[],'learned_template':None,'baseline_id':None,'solve':full.solve,'is_ablation':True})
    return rows


def run_smoke(source_text:str,lock:dict)->None:
    from synthetic import problems
    selected,full=selected_candidates(source_text,lock);entries=evaluation_entries(selected,full,lock);cases=problems()[:4]
    valid=0
    for case_index,p in enumerate(cases):
        ref=reference_exact(p)
        for entry in entries:
            got=entry['solve'](p);ok,reason,_=verify(p,got,ref)
            if not ok:raise RuntimeError(f'smoke invalid case={case_index} candidate={entry["candidate"]} reason={reason}')
            valid+=1
    expected=len(cases)*len(entries)
    if valid!=expected:raise RuntimeError(f'smoke row mismatch {valid} != {expected}')
    print(json.dumps({'stage':'preblind_synthetic_smoke','cases':len(cases),'entries':len(entries),'evaluations':expected,'valid':valid,'official_test_manifest_opened':False,'official_test_payloads_opened':0},indent=2))


def run_official(shard:int,output:Path,source_text:str,source_sha:str,lock:dict)->None:
    if not 0<=shard<SHARDS:raise ValueError('invalid shard')
    selected,full=selected_candidates(source_text,lock);entries=evaluation_entries(selected,full,lock)
    if len(entries)!=7:raise RuntimeError(f'expected 7 blind entries got {len(entries)}')
    manifest=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(manifest)!=TEST_SIZE or git_blob(manifest)!=TEST_OID:raise RuntimeError(f'test manifest identity mismatch size={len(manifest)} blob={git_blob(manifest)}')
    manifest_sha=hashlib.sha256(manifest).hexdigest();records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 test records got {len(records)}')
    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==shard):
        p=decode_problem(row['problem']);shift=idx%len(entries);ordered=entries[shift:]+entries[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(reference_exact,p);runs=[(e,*timed(e['solve'],p)) for e in ordered];execution_order='reference_first'
        else:
            runs=[(e,*timed(e['solve'],p)) for e in ordered];ref,ref_ns,ref_err=timed(reference_exact,p);execution_order='candidates_first'
        if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed test record {idx+1}: {ref_err}')
        for entry,got,cand_ns,error in runs:
            valid,reason,metrics=verify(p,got,ref) if error is None else (False,'exception',{})
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':entry['arm'],'candidate':entry['candidate'],'implementation_class':entry['implementation_class'],'operators':entry['operators'],'transfer_ids':entry['transfer_ids'],'learned_template':entry['learned_template'],'baseline_id':entry['baseline_id'],'recipe_removal_ablation':bool(entry['is_ablation']),'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,**metrics,'shape':list(p.shape),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':manifest_sha,'source_sha256':source_sha,'execution_order':execution_order,'shard':shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'verifier_capacity_loophole_exploited':False})
        del p,ref,runs;gc.collect()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
    print(json.dumps({'stage':'official_blind_r1_shard','shard':shard,'rows':len(evidence),'test_manifest_sha256':manifest_sha,'invalid_output_retries':0},indent=2))


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--shard',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    raw=args.source.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    lock=load_lock();
    if lock['source_sha256']!=source_sha:raise RuntimeError('blind lock source mismatch')
    text=raw.decode('utf-8')
    if args.smoke:
        if args.shard is not None or args.output is not None:raise ValueError('smoke cannot accept shard/output')
        run_smoke(text,lock);return
    if args.shard is None or args.output is None:raise ValueError('official mode requires shard/output')
    run_official(args.shard,args.output,text,source_sha,lock)
if __name__=='__main__':main()
