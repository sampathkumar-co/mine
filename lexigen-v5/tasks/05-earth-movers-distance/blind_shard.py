from __future__ import annotations
import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np
from candidates import CANDIDATES_BY_ARM,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8'; TASK='earth_movers_distance'; SHARDS=10
TEST_NAME='earth_movers_distance_T100ms_n1151_size100_test.jsonl'; TEST_OID='dd5b455a8ca5b6af0165be7fbd318f331112ff3f'; TEST_SIZE=2492900
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SELECTED={'v5_full':'v5_full_r1_3304c859d463a501bd86','v5_no_transfer':'v5_no_transfer_r2_b2614109e1a5ccc10c14','random_search':'random_search_r5_7a22ae1a8866ee0b94da','static_template':'static_template_r6_d044a19fd4551034dc11','v4_compatible':'v4_compatible_r4_7c30efb65d2c20ff8cc9'}
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task5-blind-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.URLError,urllib.error.HTTPError) as e:last=e; time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_array(v):
    if isinstance(v,list):return np.asarray(v)
    if not isinstance(v,dict):raise RuntimeError('unsupported array')
    kind=v.get('__type__')
    if kind=='ndarray_ref':
        p=str(v.get('npy_path',''))
        if not p or p.startswith('/') or '..' in Path(p).parts:raise RuntimeError('unsafe ndarray ref')
        return np.load(io.BytesIO(fetch(f'{BASE}/{p}?download=true')),allow_pickle=False)
    if kind=='ndarray_b64':
        raw=base64.b64decode(str(v['data_b64']).encode('ascii'),validate=True); return np.frombuffer(raw,dtype=np.dtype(str(v['dtype']))).reshape(tuple(int(x) for x in v['shape']))
    if kind=='ndarray':return np.asarray(v['data'],dtype=np.dtype(str(v.get('dtype','float64'))))
    raise RuntimeError(f'unsupported wrapper {kind!r}')
def decode_problem(r):return {'source_weights':np.asarray(decode_array(r['source_weights']),dtype=np.float64),'target_weights':np.asarray(decode_array(r['target_weights']),dtype=np.float64),'cost_matrix':np.asarray(decode_array(r['cost_matrix']),dtype=np.float64)}
def timed(fn,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(g,e):
    if not isinstance(g,dict) or 'transport_plan' not in g:return False
    try:a=np.asarray(g['transport_plan'],dtype=np.float64);b=np.asarray(e['transport_plan'],dtype=np.float64)
    except Exception:return False
    return a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=1e-7,atol=1e-7)
def selected_functions():
    lookup={name:fn for cs in CANDIDATES_BY_ARM.values() for name,fn in cs}; return [(arm,name,lookup[name]) for arm,name in SELECTED.items()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(raw)!=TEST_SIZE or blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100:raise RuntimeError(f'expected 100 test records got {len(rows)}')
    selected=selected_functions(); evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']); shift=idx%len(selected); ordered=selected[shift:]+selected[:shift]
        if idx%2==0:expected,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];expected,rs,re=timed(reference_exact,p);order='candidates_first'
        if expected is None or re:raise RuntimeError(f'reference failed test record {idx+1}: {re}')
        for arm,name,got,cs,ce in cr:
            valid=ce is None and verify(got,expected);evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'valid':valid,'candidate_s':cs,'reference_s':rs,'speedup':rs/cs if cs and cs>0 else 0.0,'failure_reason':ce or (None if valid else 'mismatch'),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':hashlib.sha256(raw).hexdigest(),'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'candidate_executions':1})
        del p,expected,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__':main()
