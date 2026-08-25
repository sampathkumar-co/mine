from __future__ import annotations

import hashlib,importlib.metadata,json,statistics,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lexigen-v5'));sys.path.insert(0,str(HERE))
from candidates import build_candidates,independent_semantic_certificate,official_verifier_accepts


def make_case(n:int,seed:int,scale:float,hetero:float)->dict:
    rng=np.random.default_rng(seed)
    true=np.linspace(1.0,scale,n)**2
    slope=0.45+0.9*rng.random();intercept=-2.0+4.0*rng.random()
    sx=(0.03+hetero*np.linspace(0.2,1.0,n))*np.maximum(1.0,np.sqrt(true))
    sy=(0.04+0.8*hetero*np.linspace(1.0,0.3,n))*np.maximum(1.0,np.sqrt(true))
    x=true+rng.normal(0.0,sx)
    y=slope*true+intercept+rng.normal(0.0,sy)
    return {'x':x.tolist(),'y':y.tolist(),'sx':sx.tolist(),'sy':sy.tolist()}


def problems()->list[dict]:
    rows=[]
    ns=[4,6,8,12,18,26,38,52]
    for k in range(24):
        rows.append(make_case(ns[k%len(ns)],82000+k,4.0+1.5*(k%6),0.04+0.02*(k%5)))
    return rows


def main()->None:
    lock=json.loads((HERE/'TASK_LOCK.json').read_text())
    source_path=Path(sys.argv[1]) if len(sys.argv)>1 else Path('task-source.py')
    raw=source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=lock['source_sha256']:raise SystemExit('task source sha256 mismatch')
    arms=build_candidates(raw.decode())
    cases=problems();rows=[];meta=[]
    order=['v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline']
    for arm in order:
        for c in arms[arm]:
            meta.append({'name':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id})
            for idx,p in enumerate(cases):
                t0=time.perf_counter_ns();err=None
                try:
                    sol=c.solve(p);semantic=independent_semantic_certificate(p,sol);official=official_verifier_accepts(p,sol)
                except Exception as exc:
                    semantic=official=False;err=f'{type(exc).__name__}: {exc}'
                rows.append({'candidate':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'case':idx,'semantic_valid':semantic,'official_valid':official,'elapsed_ns':time.perf_counter_ns()-t0,'error':err})
    by={}
    for m in meta:
        sub=[r for r in rows if r['candidate']==m['name']]
        by[m['name']]={**m,'cases':len(sub),'semantic_valid':sum(bool(r['semantic_valid']) for r in sub),'official_valid':sum(bool(r['official_valid']) for r in sub),'errors':sum(r['error'] is not None for r in sub),'median_elapsed_ns':statistics.median(r['elapsed_ns'] for r in sub),'eligible':all(r['semantic_valid'] and r['official_valid'] and r['error'] is None for r in sub)}
    arm_eligible={arm:sum(1 for m in meta if m['arm']==arm and by[m['name']]['eligible']) for arm in order}
    payload='\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n';plan=json.dumps(meta,sort_keys=True,separators=(',',':'))
    Path('synthetic-results.jsonl').write_text(payload);Path('synthetic-candidate-plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':2,'task':'odr','stage':'synthetic_r1','case_count':len(cases),'candidate_count':len(meta),'row_count':len(rows),'eligible_count':sum(v['eligible'] for v in by.values()),'arm_eligible_count':arm_eligible,'by_candidate':by,'candidate_plan_sha256':hashlib.sha256(plan.encode()).hexdigest(),'results_sha256':hashlib.sha256(payload.encode()).hexdigest(),'source_sha256':hashlib.sha256(raw).hexdigest(),'versions':{'python':sys.version.split()[0],'numpy':importlib.metadata.version('numpy'),'scipy':importlib.metadata.version('scipy')},'official_train_manifest_opened':False,'official_test_manifest_opened':False,'public_task_specific_solvers_opened':False}
    Path('synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'candidate_count':len(meta),'eligible_count':summary['eligible_count'],'arm_eligible_count':arm_eligible,'candidate_plan_sha256':summary['candidate_plan_sha256'],'results_sha256':summary['results_sha256'],'versions':summary['versions']},indent=2))
    if any(arm_eligible[a]<1 for a in order):raise SystemExit(4)

if __name__=='__main__':main()
