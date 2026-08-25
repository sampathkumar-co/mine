from __future__ import annotations

import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np

from candidates import build_candidates,reference_exact,relative_l2

REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='dct_type_I_scipy_fftpack';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='dct_type_I_scipy_fftpack_T100ms_n1958_size100_train.jsonl';TRAIN_OID='8f20ec2c4206d84056af61bd738f94cc8f0c7c1c';TRAIN_SIZE=12242
TEST_NAME='dct_type_I_scipy_fftpack_T100ms_n1958_size100_test.jsonl';TEST_OID='5b19ca6d9ca4ae06e6edaab77105e835811ab309';TEST_SIZE=12300
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SOURCE_SHA256='d9667553f833e9966df0d6fde154c473f7b33285fd67a28f216f9c3df25d4e11'
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')
EXPECTED_BY_ARM={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}
HERE=Path(__file__).resolve().parent


def fetch(url:str)->bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task3-train-r1'})
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
    a=np.asarray(p,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or min(a.shape)<2:raise RuntimeError(f'invalid DCT matrix shape {a.shape}')
    if not np.all(np.isfinite(a)):raise RuntimeError('invalid DCT matrix values')
    return np.ascontiguousarray(a,dtype=np.float64)


def timed(fn,p):
    try:t=time.perf_counter_ns();out=fn(p);return out,time.perf_counter_ns()-t,None
    except Exception as exc:return None,None,f'{type(exc).__name__}: {exc}'


def verify(problem,got,ref):
    try:
        obs=np.asarray(got,dtype=np.float64);exp=np.asarray(ref,dtype=np.float64)
        if obs.shape!=exp.shape or not np.all(np.isfinite(obs)):return False,'format_or_nonfinite',{}
        err=relative_l2(obs,exp);valid=bool(err<=1e-6)
        return valid,(None if valid else 'relative_l2_mismatch'),{'relative_l2_error':float(err),'tolerance':1e-6}
    except Exception as exc:return False,f'verify_exception:{type(exc).__name__}:{exc}',{}


def frozen_eligible_names()->set[str]:
    result=json.loads((HERE/'SYNTHETIC_R1_RESULT.json').read_text())
    if result.get('stage')!='synthetic_r1_sealed' or int(result.get('eligible_count',-1))!=31:raise RuntimeError('invalid sealed synthetic result')
    names={str(x) for x in result['eligible_names']}
    if len(names)!=31:raise RuntimeError('synthetic eligible identity count mismatch')
    return names


def flat(source_text:str):
    eligible=frozen_eligible_names();arms=build_candidates(source_text);rows=[];counts={}
    for arm in ARM_ORDER:
        q=[c for c in arms[arm] if c.name in eligible];counts[arm]=len(q);rows.extend(q)
    if {c.name for c in rows}!=eligible:raise RuntimeError('synthetic eligible candidate identity mismatch')
    if counts!=EXPECTED_BY_ARM:raise RuntimeError(f'candidate arm counts mismatch {counts}')
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    if not 0<=args.shard<SHARDS:raise ValueError('invalid shard')
    src=args.source.read_bytes();source_sha=hashlib.sha256(src).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    candidates=flat(src.decode('utf-8'))
    manifest=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(manifest)!=TRAIN_SIZE or git_blob(manifest)!=TRAIN_OID:raise RuntimeError(f'train manifest identity mismatch size={len(manifest)} blob={git_blob(manifest)}')
    manifest_sha=hashlib.sha256(manifest).hexdigest();records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
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
            ev.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,**metrics,'shape':list(p.shape),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':manifest_sha,'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_git_blob_sha1':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'source_sha256':source_sha,'execution_order':execution_order,'shard':args.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'test_manifest_contents_opened':False,'test_payloads_opened':0,'public_task_specific_solvers_opened':False})
        del p,ref,runs;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ev)+'\n');print(json.dumps({'shard':args.shard,'rows':len(ev),'manifest_sha256':manifest_sha},indent=2))
if __name__=='__main__':main()
