from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact,_safe_exp

def cases():
    rng=np.random.default_rng(9009);out=[]
    x=np.linspace(0,5,250);y=np.polyval([0.2,-1.1,2.0,0.7,-0.5],x)+rng.normal(0,0.05,len(x));out.append({'name':'poly4','model_type':'polynomial','degree':4,'x_data':x.tolist(),'y_data':y.tolist()})
    x=np.linspace(0,8,300);y=1.8*_safe_exp(0.13*x)-0.7+rng.normal(0,0.03,len(x));out.append({'name':'exponential','model_type':'exponential','x_data':x.tolist(),'y_data':y.tolist()})
    x=np.linspace(0.05,10,300);y=2.2*np.log(0.8*x+1.4)-0.4+rng.normal(0,0.03,len(x));out.append({'name':'logarithmic','model_type':'logarithmic','x_data':x.tolist(),'y_data':y.tolist()})
    x=np.linspace(0,12,300);y=4/(1+_safe_exp(-0.7*(x-6)))-0.5+rng.normal(0,0.03,len(x));out.append({'name':'sigmoid','model_type':'sigmoid','x_data':x.tolist(),'y_data':y.tolist()})
    x=np.linspace(0,12,350);y=2.4*np.sin(0.8*x+0.5)-0.3+rng.normal(0,0.03,len(x));out.append({'name':'sinusoidal','model_type':'sinusoidal','x_data':x.tolist(),'y_data':y.tolist()})
    x=np.linspace(-1,1,600);coef=rng.uniform(-1,1,9);y=np.polyval(coef,x)+rng.normal(0,0.02,len(x));out.append({'name':'poly8','model_type':'polynomial','degree':8,'x_data':x.tolist(),'y_data':y.tolist()})
    return out

def mse(problem,solution):
    x=np.asarray(problem['x_data']);y=np.asarray(problem['y_data']);p=np.asarray(solution['params'],dtype=float);m=problem['model_type']
    if m=='polynomial':yf=np.polyval(p,x)
    elif m=='exponential':a,b,c=p;yf=a*_safe_exp(b*x)+c
    elif m=='logarithmic':a,b,c,d=p;yf=a*np.log(b*x+c)+d
    elif m=='sigmoid':a,b,c,d=p;yf=a/(1+_safe_exp(-b*(x-c)))+d
    else:a,b,c,d=p;yf=a*np.sin(b*x+c)+d
    return float(np.mean((y-yf)**2))
def verify(problem,got,ref):
    try:
        if not isinstance(got,dict) or 'params' not in got:return False,'format',None,None
        p=np.asarray(got['params'],dtype=float)
        if p.ndim!=1 or p.size==0 or not np.all(np.isfinite(p)):return False,'params',None,None
        gm=mse(problem,got);rm=mse(problem,ref)
        return bool(np.isfinite(gm) and np.isfinite(rm) and gm<=1.05*rm),('mse' if gm>1.05*rm else None),gm,rm
    except Exception as e:return False,f'{type(e).__name__}:{e}',None,None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rows=[]
    for ci,p in enumerate(cases(),1):
        ref=reference_exact(p)
        for arm,cs in CANDIDATES_BY_ARM.items():
            for name,fn in cs:
                try:got=fn(p);ok,reason,gm,rm=verify(p,got,ref);err=None
                except Exception as e:ok=False;reason='exception';gm=rm=None;err=f'{type(e).__name__}:{e}'
                rows.append({'case_index':ci,'case':p['name'],'model_type':p['model_type'],'n':len(p['x_data']),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'valid':bool(ok),'failure_reason':err or reason,'candidate_mse':gm,'reference_mse':rm,'mse_ratio':(gm/rm if gm is not None and rm and rm>0 else None)})
    if len(rows)!=180:raise RuntimeError(f'expected 180 checks got {len(rows)}')
    eligible=[]
    for arm,cs in CANDIDATES_BY_ARM.items():
        for name,_ in cs:
            if all(r['valid'] for r in rows if r['candidate']==name):eligible.append(name)
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':9,'task':'least_squares','stage':'synthetic_r1','checks':180,'valid_checks':sum(bool(r['valid']) for r in rows),'candidate_count':30,'eligible_candidate_count':len(eligible),'eligible_candidates':eligible,'implementation_classes':sorted({CANDIDATE_META[n]['implementation_class'] for n in eligible}),'official_training_manifest_opened':False,'official_training_payloads_opened':0,'official_test_manifest_opened':False,'official_test_payloads_opened':0,'threshold_changes':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'synthetic-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
