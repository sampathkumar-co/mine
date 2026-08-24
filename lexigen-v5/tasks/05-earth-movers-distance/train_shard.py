from __future__ import annotations
import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
from typing import Callable
import numpy as np
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8'; TASK='earth_movers_distance'; SHARDS=10; EXPECTED_RECORDS=100
TRAIN_NAME='earth_movers_distance_T100ms_n1151_size100_train.jsonl'; TRAIN_OID='19eac35e67713b6a56ccb7ae500519b8c1b57a3f'; TRAIN_SIZE=2492842
TEST_NAME='earth_movers_distance_T100ms_n1151_size100_test.jsonl'; TEST_OID='dd5b455a8ca5b6af0165be7fbd318f331112ff3f'; TEST_SIZE=2492900
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task5-train-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.URLError,urllib.error.HTTPError) as e:
            last=e; time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted: {url}') from last
def blob(b): return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_array(v):
    if isinstance(v,list): return np.asarray(v)
    if not isinstance(v,dict): raise RuntimeError(f'unsupported array encoding {type(v).__name__}')
    kind=v.get('__type__')
    if kind=='ndarray_ref':
        p=str(v.get('npy_path',''))
        if not p or p.startswith('/') or '..' in Path(p).parts: raise RuntimeError(f'unsafe ndarray_ref {p}')
        return np.load(io.BytesIO(fetch(f'{BASE}/{p}?download=true')),allow_pickle=False)
    if kind=='ndarray_b64':
        raw=base64.b64decode(str(v['data_b64']).encode('ascii'),validate=True); return np.frombuffer(raw,dtype=np.dtype(str(v['dtype']))).reshape(tuple(int(x) for x in v['shape']))
    if kind=='ndarray': return np.asarray(v['data'],dtype=np.dtype(str(v.get('dtype','float64'))))
    raise RuntimeError(f'unsupported ndarray wrapper {kind!r}')
def decode_problem(raw):
    if not isinstance(raw,dict): raise RuntimeError('problem is not dict')
    p={'source_weights':np.asarray(decode_array(raw['source_weights']),dtype=np.float64),'target_weights':np.asarray(decode_array(raw['target_weights']),dtype=np.float64),'cost_matrix':np.asarray(decode_array(raw['cost_matrix']),dtype=np.float64)}
    if p['cost_matrix'].shape!=(len(p['source_weights']),len(p['target_weights'])) or not all(np.all(np.isfinite(x)) for x in p.values()): raise RuntimeError('invalid official problem')
    return p
def timed(fn:Callable,p):
    try:
        t=time.perf_counter(); s=fn(p); return s,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(got,expected):
    if not isinstance(got,dict) or 'transport_plan' not in got:return False,'missing_transport_plan',None
    try:g=np.asarray(got['transport_plan'],dtype=np.float64); e=np.asarray(expected['transport_plan'],dtype=np.float64)
    except Exception:return False,'decode_failure',None
    if g.shape!=e.shape:return False,'shape',None
    if not np.all(np.isfinite(g)):return False,'nonfinite',None
    err=float(np.max(np.abs(g-e))) if g.size else 0.0
    return bool(np.allclose(g,e,rtol=1e-7,atol=1e-7)),('mismatch' if not np.allclose(g,e,rtol=1e-7,atol=1e-7) else None),err
def flat():
    out=[(arm,name,fn) for arm,cs in CANDIDATES_BY_ARM.items() for name,fn in cs]
    if len(out)!=30: raise RuntimeError(f'expected 30 candidates got {len(out)}')
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    raw=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(raw)!=TRAIN_SIZE or blob(raw)!=TRAIN_OID: raise RuntimeError(f'train manifest identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100: raise RuntimeError(f'expected 100 records got {len(rows)}')
    candidates=flat(); evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']); shift=idx%len(candidates); ordered=candidates[shift:]+candidates[:shift]
        if idx%2==0:
            expected,ref_s,ref_err=timed(reference_exact,p); cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered]; order='reference_first'
        else:
            cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered]; expected,ref_s,ref_err=timed(reference_exact,p); order='candidates_first'
        if expected is None or ref_s is None or ref_err: raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
        for arm,name,got,cand_s,cerr in cr:
            if cerr is None: valid,reason,maxerr=verify(got,expected)
            else: valid,reason,maxerr=False,'exception',None
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'valid':bool(valid and cerr is None),'failure_reason':cerr or reason,'max_abs_error':maxerr,'candidate_s':cand_s,'reference_s':ref_s,'speedup':ref_s/cand_s if cand_s and cand_s>0 else 0.0,'matrix_shape':list(p['cost_matrix'].shape),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':hashlib.sha256(raw).hexdigest(),'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_tree_oid':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'execution_order':order,'shard':a.shard,'candidate_executions':1,'reference_executions_for_record':1,'invalid_output_retries':0,'test_manifest_contents_opened':False,'test_payloads_opened':0})
        del p,expected,cr; gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__': main()
