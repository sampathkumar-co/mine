from __future__ import annotations
import argparse,base64,gc,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np
from candidates import CANDIDATES_BY_ARM,reference_exact,_safe_exp
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='least_squares';SHARDS=10;EXPECTED_RECORDS=100
TEST_NAME='least_squares_T100ms_n102713_size100_test.jsonl';TEST_OID='a2a05f3b2d737d15c4cb46dc327f3954b78ace31';TEST_SIZE=27500
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SELECTED={'v5_full':'v5_full_r2_41510e43e8fafb598496','v5_no_transfer':'v5_no_transfer_r6_66c5848a3c8a4f51b562','random_search':'random_search_r5_e818498ab004d266d2a1','static_template':'static_template_r6_d044a19fd4551034dc11','v4_compatible':'v4_compatible_r3_ec4b9c17aaa3767d4f6d'}
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task9-blind-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_value(v):
    if isinstance(v,list):return [decode_value(x) for x in v]
    if not isinstance(v,dict):return v
    kind=v.get('__type__')
    if kind is None:return {k:decode_value(x) for k,x in v.items()}
    if kind=='ndarray_ref':
        p=str(v.get('npy_path',''))
        if not p or p.startswith('/') or '..' in Path(p).parts:raise RuntimeError(f'unsafe ndarray_ref {p}')
        raw=fetch(f'{BASE}/{p}?download=true');return np.load(__import__('io').BytesIO(raw),allow_pickle=False)
    if kind=='ndarray_b64':
        raw=base64.b64decode(str(v.get('data_b64','')).encode('ascii'));a=np.frombuffer(raw,dtype=np.dtype(v['dtype']));shape=tuple(v.get('shape',[]));return a.reshape(shape) if shape else a
    if kind=='ndarray':return np.array(v['data'],dtype=np.dtype(v.get('dtype')))
    if kind=='tuple':return tuple(decode_value(x) for x in v.get('data',[]))
    return {k:decode_value(x) for k,x in v.items() if k!='__type__'}
def decode_problem(raw):
    p=decode_value(raw)
    if not isinstance(p,dict) or not {'x_data','y_data','model_type'}.issubset(p):raise RuntimeError('invalid official least_squares problem')
    x=np.asarray(p['x_data']);y=np.asarray(p['y_data'])
    if x.ndim!=1 or y.ndim!=1 or len(x)!=len(y) or len(x)==0:raise RuntimeError('invalid data shape')
    if p['model_type']=='polynomial' and 'degree' not in p:raise RuntimeError('missing degree')
    return p
def timed(fn,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def mse(problem,solution):
    x=np.asarray(problem['x_data']);y=np.asarray(problem['y_data']);params=np.asarray(solution['params'],dtype=float);m=problem['model_type']
    if m=='polynomial':yf=np.polyval(params,x)
    elif m=='exponential':a,b,c=params;yf=a*_safe_exp(b*x)+c
    elif m=='logarithmic':a,b,c,d=params;yf=a*np.log(b*x+c)+d
    elif m=='sigmoid':a,b,c,d=params;yf=a/(1+_safe_exp(-b*(x-c)))+d
    elif m=='sinusoidal':a,b,c,d=params;yf=a*np.sin(b*x+c)+d
    else:raise ValueError(f'unknown model {m}')
    return float(np.mean((y-yf)**2))
def verify(problem,got,ref):
    try:
        if not isinstance(got,dict) or 'params' not in got:return False,'format'
        p=np.asarray(got['params'],dtype=float)
        if p.ndim!=1 or p.size==0 or not np.all(np.isfinite(p)):return False,'params'
        gm=mse(problem,got);rm=mse(problem,ref)
        return bool(np.isfinite(gm) and np.isfinite(rm) and gm<=1.05*rm),('mse' if gm>1.05*rm else None)
    except Exception as e:return False,f'{type(e).__name__}:{e}'
def selected_functions():
    lookup={name:fn for cs in CANDIDATES_BY_ARM.values() for name,fn in cs}
    return [(arm,name,lookup[name]) for arm,name in SELECTED.items()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not 0<=a.shard<SHARDS:raise ValueError('invalid shard')
    raw=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(raw)!=TEST_SIZE or blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(rows)}')
    selected=selected_functions();e=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(selected);ordered=selected[shift:]+selected[:shift]
        if idx%2==0:ref,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];ref,rs,re=timed(reference_exact,p);order='candidates_first'
        if ref is None or rs is None or re:raise RuntimeError(f'reference failed test record {idx+1}: {re}')
        for arm,name,got,t,err in cr:
            ok,reason=verify(p,got,ref) if err is None else (False,'exception')
            e.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'valid':bool(ok and err is None),'failure_reason':err or reason,'candidate_s':t,'reference_s':rs,'speedup':rs/t if t and t>0 else 0.0,'model_type':p['model_type'],'n':len(p['x_data']),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':hashlib.sha256(raw).hexdigest(),'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'candidate_executions':1})
        del p,ref,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in e)+'\n')
if __name__=='__main__':main()
