from __future__ import annotations

import hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path

import numpy as np
from candidates import ARM_ORDER,build_candidates,independent_source_contract,numpy_source_svd_float64

SOURCE_SHA256='8f7771e5618509c0b8af73390440dd7258a253983cdf8cc03b0208fa4218b018'
SHAPES=((4,4),(4,7),(7,4),(12,12),(12,19),(19,12),(24,31),(31,24))


def _matrix(shape:tuple[int,int],mode:int,seed:int)->np.ndarray:
    m,n=shape;k=min(m,n);rng=np.random.default_rng(seed)
    if mode==0:return rng.standard_normal(shape)
    if mode==1:
        r=max(1,k-1);return rng.standard_normal((m,r))@rng.standard_normal((r,n))
    q1,_=np.linalg.qr(rng.standard_normal((m,k)));q2,_=np.linalg.qr(rng.standard_normal((n,k)));s=np.geomspace(1.0,1e-8,k)
    return (q1*s)@q2.T


def cases():
    out=[]
    for i,shape in enumerate(SHAPES):
        for mode in range(3):out.append({'case_id':f'{shape[0]}x{shape[1]}_m{mode}','shape':list(shape),'mode':mode,'problem':{'matrix':_matrix(shape,mode,7000+i*17+mode)}})
    if len(out)!=24:raise RuntimeError('case count mismatch')
    return out


def main():
    source=Path(sys.argv[1] if len(sys.argv)>1 else 'task-source.py').read_bytes();sha=hashlib.sha256(source).hexdigest()
    if sha!=SOURCE_SHA256:raise RuntimeError(f'source identity mismatch {sha}')
    arms=build_candidates(source.decode('utf-8'));flat=[c for arm in ARM_ORDER for c in arms[arm]]
    if len(flat)!=31:raise RuntimeError('candidate count mismatch')
    plan=[{'arm':c.arm,'candidate':c.name,'rank':c.rank,'proposal_id':c.proposal_id,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'baseline_id':c.baseline_id} for c in flat]
    rows=[];by_candidate=defaultdict(list)
    for item in cases():
        p=item['problem'];reference=numpy_source_svd_float64(p);ref_ok,ref_reason,ref_metrics=independent_source_contract(p,reference)
        if not ref_ok:raise RuntimeError(f'source reference failed independent contract on {item["case_id"]}: {ref_reason} {ref_metrics}')
        for c in flat:
            err=None;t0=time.perf_counter_ns()
            try:solution=c.solve(p)
            except Exception as exc:solution={};err=f'{type(exc).__name__}:{exc}'
            elapsed=time.perf_counter_ns()-t0
            valid,reason,metrics=independent_source_contract(p,solution) if err is None else (False,'exception',{})
            row={'case_id':item['case_id'],'shape':item['shape'],'mode':item['mode'],'arm':c.arm,'candidate':c.name,'proposal_id':c.proposal_id,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'baseline_id':c.baseline_id,'valid':bool(valid and err is None),'failure_reason':err or reason,'candidate_ns_diagnostic':elapsed,'invalid_output_retries':0,'official_data_opened':False,**metrics}
            rows.append(row);by_candidate[c.name].append(row)
    summary=[];eligible=[]
    for c in flat:
        sub=by_candidate[c.name];valid=sum(bool(r['valid']) for r in sub)
        x={'arm':c.arm,'candidate':c.name,'valid':valid,'required':24,'eligible':valid==24,'proposal_id':c.proposal_id,'transfer_ids':list(c.transfer_ids),'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'baseline_id':c.baseline_id,'first_failure':next((r['case_id'] for r in sub if not r['valid']),None)}
        summary.append(x)
        if x['eligible']:eligible.append(c.name)
    eligible_by_arm={a:sum(1 for x in summary if x['arm']==a and x['eligible']) for a in ARM_ORDER}
    if any(eligible_by_arm[a]<1 for a in ARM_ORDER):raise RuntimeError(f'required arm has no eligible candidate: {eligible_by_arm}')
    Path('synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    Path('synthetic-candidate-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    body={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':7,'task':'svd','stage':'synthetic_r1','source_sha256':sha,'case_count':24,'candidate_count':31,'evaluation_count':len(rows),'eligible_count':len(eligible),'eligible_names':eligible,'eligible_by_arm':eligible_by_arm,'candidate_summary':summary,'timing_is_diagnostic_only':True,'official_train_manifest_opened':False,'official_test_manifest_opened':False,'official_payloads_opened':0,'public_task_specific_solvers_opened':False,'threshold_changes':False}
    Path('synthetic-summary.json').write_text(json.dumps(body,indent=2)+'\n');print(json.dumps({'evaluation_count':len(rows),'eligible_count':len(eligible),'eligible_by_arm':eligible_by_arm},indent=2))
if __name__=='__main__':main()
