from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact

def make_problem(n,seed):
    rng=np.random.default_rng(seed);n=max(1,n)
    base={'k':1.2,'e':0.95,'mu':1.78e-5,'rho':1.23,'tau':0.12,'N_ult':3.8,'V_min':22.0,'C_Lmax':1.5,'S_wetratio':2.05,'W_W_coeff1':8.71e-5,'W_W_coeff2':45.24,'CDA0':0.031,'W_0':4940.0}
    p={k:v*rng.uniform(0.95,1.05) for k,v in base.items()};conds=[]
    for i in range(n):
        c=p.copy();a=i/max(1,n-1);c['rho']=p['rho']*(1-0.7*a);c['mu']=p['mu']*(1-0.05*a);c['W_0']=p['W_0']*(1-0.3*a+0.1*rng.uniform(-1,1));c['V_min']=p['V_min']*(1+0.5*a+0.1*rng.uniform(-1,1));c['condition_id']=i;conds.append(c)
    return {'num_conditions':n,'conditions':conds}

def verify(problem,solution,ref):
    try:
        if not isinstance(solution,dict) or not {'A','S','avg_drag','condition_results'}.issubset(solution):return False,'format'
        if isinstance(solution['A'],list):return False,'empty'
        A=float(solution['A']);S=float(solution['S']);avg=float(solution['avg_drag']);results=solution['condition_results'];n=problem['num_conditions']
        if not (A>0 and S>0 and math.isfinite(A) and math.isfinite(S) and math.isfinite(avg)) or len(results)!=n:return False,'shape'
        eps=1e-5;total=0.0
        byid={r['condition_id']:r for r in results}
        for c in problem['conditions']:
            if c['condition_id'] not in byid:return False,'missing_condition'
            r=byid[c['condition_id']];V=float(r['V']);W=float(r['W']);Ww=float(r['W_w']);CL=float(r['C_L']);CD=float(r['C_D']);Cf=float(r['C_f']);Re=float(r['Re']);drag=float(r['drag'])
            vals=[V,W,Ww,CL,CD,Cf,Re,drag]
            if not all(math.isfinite(x) and x>0 for x in vals):return False,'nonpositive'
            exp_cd=float(c['CDA0'])/S+float(c['k'])*Cf*float(c['S_wetratio'])+CL**2/(np.pi*A*float(c['e']))
            if CD<exp_cd-eps:return False,'CD'
            if Cf<0.074/Re**0.2-eps:return False,'Cf'
            if Re<float(c['rho'])*V*np.sqrt(S/A)/float(c['mu'])-eps:return False,'Re'
            ww=float(c['W_W_coeff2'])*S+float(c['W_W_coeff1'])*float(c['N_ult'])*(A**1.5)*np.sqrt(float(c['W_0'])*W)/float(c['tau'])
            if Ww<ww-eps:return False,'Ww'
            if W<float(c['W_0'])+Ww-eps:return False,'W'
            if 0.5*float(c['rho'])*V**2*CL*S<W-eps:return False,'lift'
            if 2*W/(float(c['rho'])*float(c['V_min'])**2*S)>float(c['C_Lmax'])+eps:return False,'stall'
            de=0.5*float(c['rho'])*V**2*CD*S
            if abs(drag-de)>eps*max(1,de):return False,'drag'
            total+=drag
        ae=total/n
        if abs(avg-ae)>eps*max(1,ae):return False,'avg'
        if not isinstance(ref,dict) or isinstance(ref.get('A'),list) or 'avg_drag' not in ref:return True,None
        if avg>float(ref['avg_drag'])*1.01:return False,'optimality'
        return True,None
    except Exception as e:return False,f'{type(e).__name__}:{e}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cases=[(1,701),(2,702),(3,703),(5,704),(8,705),(12,706)];rows=[]
    for ci,(n,seed) in enumerate(cases,1):
        p=make_problem(n,seed);ref=reference_exact(p)
        if isinstance(ref.get('A'),list):raise RuntimeError(f'reference failed synthetic case {ci}')
        for arm,cs in CANDIDATES_BY_ARM.items():
            for name,fn in cs:
                try:got=fn(p);ok,reason=verify(p,got,ref);err=None
                except Exception as e:got=None;ok=False;reason='exception';err=f'{type(e).__name__}:{e}'
                rows.append({'case_index':ci,'num_conditions':n,'seed':seed,'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'valid':bool(ok),'failure_reason':err or reason,'avg_drag':got.get('avg_drag') if isinstance(got,dict) else None,'reference_avg_drag':ref.get('avg_drag')})
    if len(rows)!=180:raise RuntimeError(f'expected 180 checks got {len(rows)}')
    eligible=[]
    for arm,cs in CANDIDATES_BY_ARM.items():
        for name,_ in cs:
            if all(r['valid'] for r in rows if r['candidate']==name):eligible.append(name)
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':7,'task':'aircraft_wing_design','stage':'synthetic_r1','checks':180,'valid_checks':sum(bool(r['valid']) for r in rows),'candidate_count':30,'eligible_candidate_count':len(eligible),'eligible_candidates':eligible,'official_training_manifest_opened':False,'official_training_payloads_opened':0,'official_test_manifest_opened':False,'official_test_payloads_opened':0,'threshold_changes':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'synthetic-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
