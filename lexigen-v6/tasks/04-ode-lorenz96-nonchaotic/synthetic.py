from __future__ import annotations

import hashlib,importlib.metadata,json,statistics,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lexigen-v5'));sys.path.insert(0,str(HERE))
from candidates import build_candidates,reference_exact,high_accuracy_independent


def make_case(n:int,seed:int,perturb:float)->dict:
    rng=np.random.default_rng(seed);F=2.0;t0=0.0;t1=10.0
    y0=np.full(n,F)+rng.random(n)*perturb
    return {'F':F,'t0':t0,'t1':t1,'y0':y0.tolist()}


def problems()->list[dict]:
    ns=[5,8,12,20,32,48,72,96]
    return [make_case(ns[k%len(ns)],84000+k,0.004+0.003*(k%5)) for k in range(24)]


def main()->None:
    lock=json.loads((HERE/'TASK_LOCK.json').read_text());source_path=Path(sys.argv[1]) if len(sys.argv)>1 else Path('task-source.py')
    raw=source_path.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha!=lock['source_sha256']:raise SystemExit('task source sha256 mismatch')
    arms=build_candidates(raw.decode('utf-8'));order=['v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline']
    meta=[]
    for arm in order:
        for c in arms[arm]:meta.append({'name':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id})
    if len(meta)!=31:raise RuntimeError(f'expected 31 candidates got {len(meta)}')
    cases=problems();certs=[]
    for p in cases:
        ref=np.asarray(reference_exact(p),dtype=np.float64);ind=np.asarray(high_accuracy_independent(p),dtype=np.float64);certs.append((ref,ind))
    rows=[]
    by_name={m['name']:m for m in meta}
    for arm in order:
        for c in arms[arm]:
            for idx,p in enumerate(cases):
                ref,ind=certs[idx];t0=time.perf_counter_ns();err=None
                try:
                    sol=np.asarray(c.solve(p),dtype=np.float64);elapsed=time.perf_counter_ns()-t0
                    format_ok=sol.shape==ref.shape and np.all(np.isfinite(sol))
                    official=bool(format_ok and np.allclose(sol,ref,rtol=1e-5,atol=1e-8))
                    scale=np.maximum(1.0,np.abs(ind));semantic=bool(format_ok and np.max(np.abs(sol-ind)/scale)<=2.5e-4)
                    max_official_abs=float(np.max(np.abs(sol-ref))) if format_ok else None
                    max_independent_scaled=float(np.max(np.abs(sol-ind)/scale)) if format_ok else None
                except Exception as exc:
                    elapsed=time.perf_counter_ns()-t0;official=semantic=False;max_official_abs=max_independent_scaled=None;err=f'{type(exc).__name__}: {exc}'
                rows.append({'candidate':c.name,'arm':c.arm,'implementation_class':c.implementation_class,'case':idx,'semantic_valid':semantic,'official_valid':official,'elapsed_ns':elapsed,'max_official_abs_error':max_official_abs,'max_independent_scaled_error':max_independent_scaled,'error':err})
    by={}
    for m in meta:
        sub=[r for r in rows if r['candidate']==m['name']]
        by[m['name']]={**m,'cases':len(sub),'semantic_valid':sum(bool(r['semantic_valid']) for r in sub),'official_valid':sum(bool(r['official_valid']) for r in sub),'errors':sum(r['error'] is not None for r in sub),'median_elapsed_ns':statistics.median(r['elapsed_ns'] for r in sub),'eligible':all(r['semantic_valid'] and r['official_valid'] and r['error'] is None for r in sub)}
    arm_eligible={arm:sum(1 for m in meta if m['arm']==arm and by[m['name']]['eligible']) for arm in order}
    payload='\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n';plan=json.dumps(meta,sort_keys=True,separators=(',',':'))
    Path('synthetic-results.jsonl').write_text(payload);Path('synthetic-candidate-plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':4,'task':'ode_lorenz96_nonchaotic','stage':'synthetic_r1','case_count':len(cases),'candidate_count':len(meta),'row_count':len(rows),'eligible_count':sum(v['eligible'] for v in by.values()),'arm_eligible_count':arm_eligible,'by_candidate':by,'candidate_plan_sha256':hashlib.sha256(plan.encode()).hexdigest(),'results_sha256':hashlib.sha256(payload.encode()).hexdigest(),'source_sha256':source_sha,'versions':{'python':sys.version.split()[0],'numpy':importlib.metadata.version('numpy'),'scipy':importlib.metadata.version('scipy')},'semantic_certificate':'independent DOP853 rtol=atol=1e-11; max scaled final-state deviation <=2.5e-4','official_verifier_equivalent':'allclose to frozen RK45 reference, rtol=1e-5 atol=1e-8','eligibility_uses_timing':False,'official_train_manifest_opened':False,'official_test_manifest_opened':False,'public_task_specific_solvers_opened':False}
    Path('synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'candidate_count':len(meta),'eligible_count':summary['eligible_count'],'arm_eligible_count':arm_eligible,'candidate_plan_sha256':summary['candidate_plan_sha256'],'results_sha256':summary['results_sha256'],'versions':summary['versions']},indent=2))
    if any(arm_eligible[a]<1 for a in order):raise SystemExit(4)

if __name__=='__main__':main()
