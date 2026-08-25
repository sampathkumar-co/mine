from __future__ import annotations

import hashlib,importlib.metadata,json,statistics,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lexigen-v5'));sys.path.insert(0,str(HERE))
from candidates import (
    CAMPAIGN_ELIGIBLE_TRANSFER_IDS,build_candidates,high_accuracy_independent,
    independent_semantic_certificate,source_reference,verify_against_reference,
)


def make_problem(size:int,seed:int)->dict:
    rng=np.random.default_rng(seed);N=2*size;p=size//10+2;m=size//10+2;n=size//5+4;tau=1.0;M=3.0
    Araw=rng.standard_normal((n,n));eigs=np.linalg.eigvals(Araw);A=Araw/(np.max(np.abs(eigs))+0.05)
    B=0.5*rng.standard_normal((n,p));C=rng.standard_normal((m,n));x0=rng.standard_normal(n)
    xtrue=np.zeros((N+1,n));xtrue[0]=x0;wtrue=0.1*rng.standard_normal((N,p));vtrue=0.1*rng.standard_normal((N,m))
    mask=rng.random(N)<=0.15
    if np.any(mask):vtrue[mask]=5.0*rng.standard_normal((int(np.sum(mask)),m))
    y=np.zeros((N,m))
    for t in range(N):xtrue[t+1]=A@xtrue[t]+B@wtrue[t];y[t]=C@xtrue[t]+vtrue[t]
    return {'A':A.tolist(),'B':B.tolist(),'C':C.tolist(),'y':y.tolist(),'x_initial':x0.tolist(),'tau':tau,'M':M}


def problems()->list[dict]:
    sizes=[2,3,4,5,6,8]
    return [make_problem(sizes[k%len(sizes)],85000+k) for k in range(24)]


def main()->None:
    lock=json.loads((HERE/'TASK_LOCK.json').read_text())
    if lock['task']!='robust_kalman_filter' or lock['source_sha256']!='3c589ba1f0d988f1d89db7a21d2f28a6d588334f881a407fbedc1a2c15a5bec2':raise RuntimeError('task lock mismatch')
    source_path=Path(sys.argv[1]) if len(sys.argv)>1 else Path('task-source.py')
    raw=source_path.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha!=lock['source_sha256']:raise RuntimeError('task source sha256 mismatch')
    arms=build_candidates(raw.decode('utf-8'));order=['v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline']
    meta=[]
    for arm in order:
        for c in arms[arm]:
            meta.append({'name':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'campaign_transfer_eligible':bool(c.transfer_ids) and all(x in CAMPAIGN_ELIGIBLE_TRANSFER_IDS for x in c.transfer_ids)})
    if len(meta)!=31:raise RuntimeError(f'expected 31 candidates got {len(meta)}')
    cases=problems();certs=[]
    for idx,p in enumerate(cases):
        ref=source_reference(p);ind=high_accuracy_independent(p)
        ok,reason,metrics=verify_against_reference(p,ind,ref,objective_factor=1.01,eps=1e-5)
        if not ok:raise RuntimeError(f'independent certificate baseline failed case={idx} reason={reason} metrics={metrics}')
        certs.append((ref,ind))
    rows=[]
    for arm in order:
        for c in arms[arm]:
            for idx,p in enumerate(cases):
                t0=time.perf_counter_ns();err=None;sol=None
                try:sol=c.solve(p)
                except Exception as exc:err=f'{type(exc).__name__}: {exc}'
                elapsed=time.perf_counter_ns()-t0
                if err is None:
                    official,reason,official_metrics=verify_against_reference(p,sol,certs[idx][0],objective_factor=1.01,eps=1e-5)
                    semantic,semantic_metrics=independent_semantic_certificate(p,sol,certs[idx][1])
                else:
                    official=False;semantic=False;reason='exception';official_metrics={};semantic_metrics={}
                rows.append({'candidate':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'case':idx,'official_valid':bool(official),'semantic_valid':bool(semantic),'failure_reason':err or reason,'elapsed_ns':elapsed,'objective_ratio':official_metrics.get('objective_ratio'),'independent_objective_ratio':semantic_metrics.get('independent_objective_ratio'),'max_dynamics_norm':official_metrics.get('max_dynamics_norm'),'max_measurement_norm':official_metrics.get('max_measurement_norm'),'error':err})
    by={}
    for m in meta:
        sub=[r for r in rows if r['candidate']==m['name']]
        by[m['name']]={**m,'cases':len(sub),'official_valid':sum(bool(r['official_valid']) for r in sub),'semantic_valid':sum(bool(r['semantic_valid']) for r in sub),'errors':sum(r['error'] is not None for r in sub),'median_elapsed_ns':statistics.median(r['elapsed_ns'] for r in sub),'eligible':all(r['official_valid'] and r['semantic_valid'] and r['error'] is None for r in sub)}
    arm_eligible={arm:sum(1 for m in meta if m['arm']==arm and by[m['name']]['eligible']) for arm in order}
    payload='\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n';plan=json.dumps(meta,sort_keys=True,separators=(',',':'))
    Path('synthetic-results.jsonl').write_text(payload);Path('synthetic-candidate-plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':5,'task':'robust_kalman_filter','family':'signal_processing','stage':'synthetic_r1','case_count':len(cases),'candidate_count':len(meta),'row_count':len(rows),'eligible_count':sum(v['eligible'] for v in by.values()),'arm_eligible_count':arm_eligible,'by_candidate':by,'candidate_plan_sha256':hashlib.sha256(plan.encode()).hexdigest(),'results_sha256':hashlib.sha256(payload.encode()).hexdigest(),'source_sha256':source_sha,'versions':{'python':sys.version.split()[0],'numpy':importlib.metadata.version('numpy'),'cvxpy':importlib.metadata.version('cvxpy'),'clarabel':importlib.metadata.version('clarabel')},'semantic_certificate':'independent tight CLARABEL solve of v-eliminated convex formulation; feasibility <=2e-6 and objective <=1.005x independent objective','official_verifier_equivalent':'full x/w/v shapes and finite; dynamics/measurement/initial norms <=1e-5; objective <=1.01x source-equivalent reference objective','reference_failure_fallback_exploited':False,'eligibility_uses_timing':False,'campaign_eligible_transfer_ids':sorted(CAMPAIGN_ELIGIBLE_TRANSFER_IDS),'official_train_manifest_opened':False,'official_test_manifest_opened':False,'official_payloads_opened':0,'public_task_specific_solvers_opened':False}
    Path('synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'candidate_count':len(meta),'eligible_count':summary['eligible_count'],'arm_eligible_count':arm_eligible,'candidate_plan_sha256':summary['candidate_plan_sha256'],'results_sha256':summary['results_sha256'],'versions':summary['versions']},indent=2))
    if any(arm_eligible[a]<1 for a in order):raise SystemExit(4)

if __name__=='__main__':main()
