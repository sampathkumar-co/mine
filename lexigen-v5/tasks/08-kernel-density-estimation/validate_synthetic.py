from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
KERNELS=['gaussian','tophat','epanechnikov','exponential','linear','cosine']
def cases():
    out=[]
    for i,(k,d,n,q,h) in enumerate(zip(KERNELS,[1,2,4,8,16,32],[160,180,220,260,300,340],[40,45,50,55,60,65],[0.45,0.8,1.2,1.8,2.5,3.4]),1):
        rng=np.random.default_rng(800+i);X=rng.normal(size=(n,d));Q=rng.normal(size=(q,d))
        if k in {'tophat','epanechnikov','linear','cosine'}:
            Q[-5:]+=12.0*np.sqrt(d)
        out.append({'name':k,'data_points':X.tolist(),'query_points':Q.tolist(),'kernel':k,'bandwidth':h})
    return out
def verify(problem,got,ref):
    try:
        if not isinstance(got,dict) or 'log_density' not in got:return False,'format'
        a=np.asarray(got['log_density'],dtype=float);b=np.asarray(ref['log_density'],dtype=float);q=len(problem['query_points'])
        a=np.squeeze(a);b=np.squeeze(b)
        if a.ndim==0:a=np.expand_dims(a,0)
        if b.ndim==0:b=np.expand_dims(b,0)
        if a.ndim!=1 or a.shape!=(q,) or b.shape!=(q,):return False,'shape'
        am=~np.isfinite(a);bm=~np.isfinite(b)
        if not np.array_equal(am,bm):return False,'nonfinite_mask'
        m=np.isfinite(b)
        if m.any() and not np.allclose(a[m],b[m],rtol=1e-4,atol=1e-6):return False,'tolerance'
        return True,None
    except Exception as e:return False,f'{type(e).__name__}:{e}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rows=[]
    for ci,p in enumerate(cases(),1):
        ref=reference_exact(p)
        for arm,cs in CANDIDATES_BY_ARM.items():
            for name,fn in cs:
                try:got=fn(p);ok,reason=verify(p,got,ref);err=None
                except Exception as e:got=None;ok=False;reason='exception';err=f'{type(e).__name__}:{e}'
                aa=np.asarray(got.get('log_density',[]),dtype=float) if isinstance(got,dict) else np.array([]);bb=np.asarray(ref['log_density'],dtype=float);m=np.isfinite(bb)&np.isfinite(aa) if aa.shape==bb.shape else np.zeros(bb.shape,dtype=bool)
                rows.append({'case_index':ci,'case':p['name'],'dims':len(p['data_points'][0]),'num_points':len(p['data_points']),'num_queries':len(p['query_points']),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'valid':bool(ok),'failure_reason':err or reason,'max_abs_error':float(np.max(np.abs(aa[m]-bb[m]))) if m.any() else 0.0})
    if len(rows)!=180:raise RuntimeError(f'expected 180 checks got {len(rows)}')
    eligible=[]
    for arm,cs in CANDIDATES_BY_ARM.items():
        for name,_ in cs:
            if all(r['valid'] for r in rows if r['candidate']==name):eligible.append(name)
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':8,'task':'kernel_density_estimation','stage':'synthetic_r1','checks':180,'valid_checks':sum(bool(r['valid']) for r in rows),'candidate_count':30,'eligible_candidate_count':len(eligible),'eligible_candidates':eligible,'official_training_manifest_opened':False,'official_training_payloads_opened':0,'official_test_manifest_opened':False,'official_test_payloads_opened':0,'threshold_changes':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'synthetic-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
