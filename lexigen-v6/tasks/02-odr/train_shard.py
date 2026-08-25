from __future__ import annotations

import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np

from candidates import build_candidates,reference_exact

REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='odr';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='odr_T100ms_n31132_size100_train.jsonl';TRAIN_OID='ab4be6c0bc48e7b28442544b1152ce79b6bd7b79';TRAIN_SIZE=41642
TEST_NAME='odr_T100ms_n31132_size100_test.jsonl';TEST_OID='0fd449c0338949c40279fa93e5f5f9c7aa35c7a0';TEST_SIZE=41700
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')
ELIGIBLE_NAMES={
'v6_full_r1_3304c859d463a501bd86','v6_full_r3_a6102573c9f355414229','v6_full_r4_b1ef08a2d68a248c0821','v6_full_r5_4abf2b51384c560522e8','v6_full_r6_c50e493c5549a408f3e5',
'v6_no_transfer_r1_91e027e622f2d9a98240','v6_no_transfer_r2_b2614109e1a5ccc10c14','v6_no_transfer_r4_4a4e1871b7f7b48b9485','v6_no_transfer_r5_d69e86803f54c5a83d06',
'random_search_r1_67668ab6baaece7064f4','random_search_r2_388330cd70b5bbb421fc','random_search_r3_4ff38c49ab25e45cbe27','random_search_r4_66fd154a665dbc8efdb7','random_search_r5_ae3b52160647eaf9707e','random_search_r6_8615e4a35db08222a26b',
'static_template_r1_dbfcd2af539b0b2636e7','static_template_r3_820b1c309b6117eb268d','static_template_r4_8f1dafda0d3fbc099aa9','static_template_r5_357e80313b8b9dc3cf36','static_template_r6_d044a19fd4551034dc11',
'v5_compatible_r1_f9f3239b6866512e4f68','v5_compatible_r2_9f5f55df04a5ad23f542','v5_compatible_r3_ec4b9c17aaa3767d4f6d','v5_compatible_r4_7c30efb65d2c20ff8cc9','v5_compatible_r5_3df5ed91505aea4ed6cb','v5_compatible_r6_0dde88a4a159a3ad0e40',
'strong_baseline_sb_native_numeric_01_lowlevel_odr'}
EXPECTED_BY_ARM={'v6_full':5,'v6_no_transfer':4,'random_search':6,'static_template':5,'v5_compatible':6,'strong_baseline':1}


def fetch(url:str)->bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task2-train-r1'})
            with urllib.request.urlopen(req,timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as exc:
            last=exc;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last

def git_blob(raw:bytes)->str:return hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()

def decode_value(value):
    if isinstance(value,list):return [decode_value(x) for x in value]
    if not isinstance(value,dict):return value
    kind=value.get('__type__')
    if kind is None:return {k:decode_value(v) for k,v in value.items()}
    if kind=='ndarray_ref':
        rel=str(value.get('npy_path',''))
        if not rel or rel.startswith('/') or '..' in Path(rel).parts:raise RuntimeError(f'unsafe ndarray_ref {rel}')
        return np.load(io.BytesIO(fetch(f'{BASE}/{rel}?download=true')),allow_pickle=False)
    if kind=='ndarray_b64':
        raw=base64.b64decode(str(value.get('data_b64','')).encode('ascii'));a=np.frombuffer(raw,dtype=np.dtype(value['dtype']));shape=tuple(value.get('shape',[]));return a.reshape(shape) if shape else a
    if kind=='ndarray':return np.array(value['data'],dtype=np.dtype(value.get('dtype')))
    if kind=='tuple':return tuple(decode_value(x) for x in value.get('data',[]))
    return {k:decode_value(v) for k,v in value.items() if k!='__type__'}

def decode_problem(raw):
    p=decode_value(raw)
    if not isinstance(p,dict) or not {'x','y','sx','sy'}<=set(p):raise RuntimeError('invalid official odr problem')
    arrays=[np.asarray(p[k],dtype=np.float64) for k in ('x','y','sx','sy')]
    if not all(a.ndim==1 for a in arrays) or len({len(a) for a in arrays})!=1 or len(arrays[0])<2:raise RuntimeError('invalid ODR vector shapes')
    if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(arrays[2]<=0) or np.any(arrays[3]<=0):raise RuntimeError('invalid ODR values')
    return p

def timed(fn,p):
    try:t=time.perf_counter_ns();out=fn(p);return out,time.perf_counter_ns()-t,None
    except Exception as exc:return None,None,f'{type(exc).__name__}: {exc}'

def verify(problem,got,ref):
    try:
        obs=np.asarray(got['beta'],dtype=np.float64);exp=np.asarray(ref['beta'],dtype=np.float64)
        if obs.shape!=(2,) or exp.shape!=(2,) or not np.all(np.isfinite(obs)):return False,'format_or_nonfinite',{}
        rtol=float(2*np.finfo(float).eps**(2.0/3.0));atol=float(np.finfo(float).smallest_normal)
        valid=bool(np.allclose(obs,exp,rtol=rtol,atol=atol))
        return valid,(None if valid else 'beta_mismatch'),{'beta0_abs_error':float(abs(obs[0]-exp[0])),'beta1_abs_error':float(abs(obs[1]-exp[1])),'rtol':rtol}
    except Exception as exc:return False,f'verify_exception:{type(exc).__name__}:{exc}',{}

def flat(source_text:str):
    arms=build_candidates(source_text);rows=[];counts={}
    for arm in ARM_ORDER:
        q=[c for c in arms[arm] if c.name in ELIGIBLE_NAMES];counts[arm]=len(q);rows.extend(q)
    if set(c.name for c in rows)!=ELIGIBLE_NAMES:raise RuntimeError('synthetic eligible candidate identity mismatch')
    if counts!=EXPECTED_BY_ARM:raise RuntimeError(f'candidate arm counts mismatch {counts}')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    if not 0<=args.shard<SHARDS:raise ValueError('invalid shard')
    src=args.source.read_bytes();source_sha=hashlib.sha256(src).hexdigest()
    if source_sha!='076efd6697175397912d5d8e3bc1b121ba7461db3fdbf04263fa6d57f81eb68c':raise RuntimeError('source identity mismatch')
    candidates=flat(src.decode())
    manifest=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(manifest)!=TRAIN_SIZE or git_blob(manifest)!=TRAIN_OID:raise RuntimeError(f'train manifest identity mismatch size={len(manifest)} blob={git_blob(manifest)}')
    manifest_sha=hashlib.sha256(manifest).hexdigest();records=[json.loads(x) for x in manifest.decode().splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(records)}')
    ev=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        p=decode_problem(row['problem']);shift=idx%len(candidates);ordered=candidates[shift:]+candidates[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(reference_exact,p);runs=[(c,*timed(c.solve,p)) for c in ordered];execution_order='reference_first'
        else:
            runs=[(c,*timed(c.solve,p)) for c in ordered];ref,ref_ns,ref_err=timed(reference_exact,p);execution_order='candidates_first'
        if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
        for c,got,cand_ns,error in runs:
            valid,reason,metrics=verify(p,got,ref) if error is None else (False,'exception',{})
            ev.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,**metrics,'n':len(np.asarray(p['x'])),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':manifest_sha,'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_tree_oid':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'source_sha256':source_sha,'execution_order':execution_order,'shard':args.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'test_manifest_contents_opened':False,'test_payloads_opened':0})
        del p,ref,runs;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ev)+'\n');print(json.dumps({'shard':args.shard,'rows':len(ev),'manifest_sha256':manifest_sha},indent=2))
if __name__=='__main__':main()
