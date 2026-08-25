from __future__ import annotations

import argparse,gc,hashlib,json
from pathlib import Path

from train_shard_r2 import SHARDS,EXPECTED_RECORDS,TEST_NAME,TEST_POINTER_OID,TEST_SIZE,TEST_SHA256,BASE,SOURCE_SHA256,decode_problem,fetch,reference_exact,timed,verify
from candidates import build_candidates

HERE=Path(__file__).resolve().parent;LOCK_NAME='BLIND_R1_LOCK.json'
SELECTED_ARMS=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline');ABLATION_ARM='recipe_removal_ablation'

def load_lock()->dict:
    lock=json.loads((HERE/LOCK_NAME).read_text())
    if lock.get('task')!='ode_lorenz96_nonchaotic' or lock.get('stage')!='blind_r1_lock':raise RuntimeError('invalid blind lock')
    if lock.get('official_test_manifest_opened_before_lock') or int(lock.get('official_test_payloads_opened_before_lock',-1))!=0:raise RuntimeError('blind boundary crossed before lock')
    if int(lock.get('blind_runs_completed_before_trigger',-1))!=0:raise RuntimeError('blind already consumed')
    return lock

def selected_candidates(source_text:str,lock:dict):
    arms=build_candidates(source_text);by_name={c.name:c for rows in arms.values() for c in rows};selected=[]
    for arm in SELECTED_ARMS:
        spec=lock['selected_by_arm'][arm];name=spec['candidate']
        if name not in by_name:raise RuntimeError(f'locked candidate missing: {name}')
        c=by_name[name]
        if c.arm!=arm or c.implementation_class!=spec['implementation_class']:raise RuntimeError(f'locked candidate mismatch {name}')
        if list(c.operators)!=spec['operators'] or list(c.transfer_ids)!=spec['transfer_ids']:raise RuntimeError(f'proposal metadata mismatch {name}')
        if c.learned_template!=spec.get('learned_template') or c.baseline_id!=spec.get('baseline_id'):raise RuntimeError(f'proposal provenance mismatch {name}')
        selected.append(c)
    full=selected[0];abl=lock['recipe_removal_ablation']
    if abl['source_candidate']!=full.name or abl['removed_transfer_ids']!=list(full.transfer_ids) or abl['resulting_transfer_ids']!=[]:raise RuntimeError('ablation transfer mismatch')
    if abl['retained_operators']!=list(full.operators) or abl['resulting_implementation_class']!=full.implementation_class:raise RuntimeError('ablation implementation mismatch')
    if not abl['same_callable_as_full'] or not abl['preblind_semantic_equivalence_to_full']:raise RuntimeError('ablation equivalence not frozen')
    return selected,full

def entries(selected,full,lock):
    out=[{'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'solve':c.solve,'is_ablation':False} for c in selected]
    a=lock['recipe_removal_ablation'];out.append({'arm':ABLATION_ARM,'candidate':a['candidate'],'implementation_class':a['resulting_implementation_class'],'operators':a['retained_operators'],'transfer_ids':[],'learned_template':None,'baseline_id':None,'solve':full.solve,'is_ablation':True});return out

def run_smoke(source_text,lock):
    from synthetic import problems
    selected,full=selected_candidates(source_text,lock);ee=entries(selected,full,lock);cases=problems()[:4];valid=0
    for ci,p in enumerate(cases):
        ref=reference_exact(p)
        for e in ee:
            got=e['solve'](p);ok,reason,_=verify(p,got,ref)
            if not ok:raise RuntimeError(f'smoke invalid case={ci} candidate={e["candidate"]} reason={reason}')
            valid+=1
    expected=len(cases)*len(ee)
    if valid!=expected:raise RuntimeError('smoke count mismatch')
    print(json.dumps({'stage':'preblind_synthetic_smoke','cases':len(cases),'entries':len(ee),'evaluations':expected,'valid':valid,'official_test_manifest_opened':False,'official_test_payloads_opened':0},indent=2))

def run_official(shard,output,source_text,source_sha,lock):
    if not 0<=shard<SHARDS:raise ValueError('invalid shard')
    selected,full=selected_candidates(source_text,lock);ee=entries(selected,full,lock)
    if len(ee)!=7:raise RuntimeError('expected 7 blind entries')
    manifest=fetch(f'{BASE}/{TEST_NAME}?download=true');manifest_sha=hashlib.sha256(manifest).hexdigest()
    if len(manifest)!=TEST_SIZE or manifest_sha!=TEST_SHA256:raise RuntimeError(f'test payload identity mismatch size={len(manifest)} sha256={manifest_sha}')
    records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 test records got {len(records)}')
    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==shard):
        p=decode_problem(row['problem']);shift=idx%len(ee);ordered=ee[shift:]+ee[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(reference_exact,p);runs=[(e,*timed(e['solve'],p)) for e in ordered];execution_order='reference_first'
        else:
            runs=[(e,*timed(e['solve'],p)) for e in ordered];ref,ref_ns,ref_err=timed(reference_exact,p);execution_order='candidates_first'
        if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed test record {idx+1}: {ref_err}')
        for e,got,cand_ns,error in runs:
            valid,reason,metrics=verify(p,got,ref) if error is None else (False,'exception',{})
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':e['arm'],'candidate':e['candidate'],'implementation_class':e['implementation_class'],'operators':e['operators'],'transfer_ids':e['transfer_ids'],'learned_template':e['learned_template'],'baseline_id':e['baseline_id'],'recipe_removal_ablation':bool(e['is_ablation']),'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,**metrics,'state_dimension':int(len(p['y0'])),'test_manifest_name':TEST_NAME,'test_manifest_pointer_oid':TEST_POINTER_OID,'test_manifest_payload_sha256':manifest_sha,'test_manifest_size':len(manifest),'source_sha256':source_sha,'execution_order':execution_order,'shard':shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'verifier_capacity_loophole_exploited':False})
        del p,ref,runs;gc.collect()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n');print(json.dumps({'stage':'official_blind_r1_shard','shard':shard,'rows':len(evidence),'test_manifest_sha256':manifest_sha},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--shard',type=int);ap.add_argument('--output',type=Path);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    raw=args.source.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    lock=load_lock()
    if lock['source_sha256']!=source_sha:raise RuntimeError('blind lock source mismatch')
    text=raw.decode('utf-8')
    if args.smoke:run_smoke(text,lock);return
    if args.shard is None or args.output is None:raise ValueError('official mode requires shard/output')
    run_official(args.shard,args.output,text,source_sha,lock)
if __name__=='__main__':main()
