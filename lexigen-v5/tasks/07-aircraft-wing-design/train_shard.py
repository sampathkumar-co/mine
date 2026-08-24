from __future__ import annotations
import argparse,gc,hashlib,json,math,time,urllib.error,urllib.request
from pathlib import Path
from typing import Callable
import numpy as np
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='aircraft_wing_design';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='aircraft_wing_design_T100ms_n10_size100_train.jsonl';TRAIN_OID='1ee8b5ce85490484e568e4359f710f7f59687b49';TRAIN_SIZE=384615
TEST_NAME='aircraft_wing_design_T100ms_n10_size100_test.jsonl';TEST_OID='adf280f7d8134006d54145d291409d692ef28a40';TEST_SIZE=384557
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task7-train-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_problem(raw):
    if not isinstance(raw,dict) or 'num_conditions' not in raw or 'conditions' not in raw:raise RuntimeError('invalid official aircraft problem')
    n=int(raw['num_conditions']);conds=raw['conditions']
    if n<1 or not isinstance(conds,list) or len(conds)!=n:raise RuntimeError('invalid condition count')
    required={'condition_id','CDA0','C_Lmax','N_ult','S_wetratio','V_min','W_0','W_W_coeff1','W_W_coeff2','e','k','mu','rho','tau'}
    for c in conds:
        if not isinstance(c,dict) or not required.issubset(c):raise RuntimeError('invalid condition schema')
    return raw
def timed(fn:Callable,p):
    try:
        t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(problem,solution,ref):
    try:
        if not isinstance(solution,dict) or not {'A','S','avg_drag','condition_results'}.issubset(solution):return False,'format'
        if isinstance(solution['A'],list):return False,'empty'
        A=float(solution['A']);S=float(solution['S']);avg=float(solution['avg_drag']);results=solution['condition_results'];n=int(problem['num_conditions'])
        if not (A>0 and S>0 and math.isfinite(A) and math.isfinite(S) and math.isfinite(avg)) or not isinstance(results,list) or len(results)!=n:return False,'shape'
        eps=1e-5;total=0.0;byid={r['condition_id']:r for r in results if isinstance(r,dict) and 'condition_id' in r}
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
            if abs(drag-de)>eps*max(1.0,de):return False,'drag'
            total+=drag
        ae=total/n
        if abs(avg-ae)>eps*max(1.0,ae):return False,'avg'
        if not isinstance(ref,dict) or isinstance(ref.get('A'),list):return False,'reference_invalid'
        if avg>float(ref['avg_drag'])*1.01:return False,'optimality'
        return True,None
    except Exception as e:return False,f'{type(e).__name__}:{e}'
def flat():
    out=[(arm,name,fn) for arm,cs in CANDIDATES_BY_ARM.items() for name,fn in cs]
    if len(out)!=30:raise RuntimeError(f'expected 30 candidates got {len(out)}')
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not 0<=a.shard<SHARDS:raise ValueError('invalid shard')
    raw=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(raw)!=TRAIN_SIZE or blob(raw)!=TRAIN_OID:raise RuntimeError(f'train manifest identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(rows)}')
    candidates=flat();evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(candidates);ordered=candidates[shift:]+candidates[:shift]
        if idx%2==0:
            ref,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:
            cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];ref,rs,re=timed(reference_exact,p);order='candidates_first'
        if ref is None or rs is None or re or isinstance(ref.get('A'),list):raise RuntimeError(f'reference failed record {idx+1}: {re or ref}')
        for arm,name,got,cs,ce in cr:
            if ce is None:valid,reason=verify(p,got,ref)
            else:valid,reason=False,'exception'
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'learned_template':CANDIDATE_META[name]['learned_template'],'transfer_ids':CANDIDATE_META[name]['transfer_ids'],'valid':bool(valid and ce is None),'failure_reason':ce or reason,'candidate_s':cs,'reference_s':rs,'speedup':rs/cs if cs and cs>0 else 0.0,'num_conditions':int(p['num_conditions']),'candidate_avg_drag':got.get('avg_drag') if isinstance(got,dict) else None,'reference_avg_drag':ref.get('avg_drag'),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':hashlib.sha256(raw).hexdigest(),'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_tree_oid':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'execution_order':order,'shard':a.shard,'candidate_executions':1,'reference_executions_for_record':1,'invalid_output_retries':0,'test_manifest_contents_opened':False,'test_payloads_opened':0})
            gc.collect()
        del p,ref,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__':main()
