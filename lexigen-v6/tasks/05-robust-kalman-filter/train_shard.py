from __future__ import annotations
import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np
from candidates import build_candidates,source_reference,verify_against_reference

REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='robust_kalman_filter';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='robust_kalman_filter_T100ms_n15_size100_train.jsonl';TRAIN_OID='e0d30a3074773642448d753ccc3a753c28f1236f';TRAIN_SIZE=388658
TEST_NAME='robust_kalman_filter_T100ms_n15_size100_test.jsonl';TEST_OID='df3aae40d27127d431ae161461dc163c385d0116';TEST_SIZE=388419
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SOURCE_SHA256='3c589ba1f0d988f1d89db7a21d2f28a6d588334f881a407fbedc1a2c15a5bec2'
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')
EXPECTED_BY_ARM={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}
HERE=Path(__file__).resolve().parent

def fetch(url:str)->bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task5-train-r1'})
            with urllib.request.urlopen(req,timeout=300) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as exc:
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
    if not isinstance(p,dict) or not {'A','B','C','y','x_initial','tau','M'} <= set(p):raise RuntimeError('invalid robust Kalman problem structure')
    A=np.asarray(p['A'],dtype=np.float64);B=np.asarray(p['B'],dtype=np.float64);C=np.asarray(p['C'],dtype=np.float64);y=np.asarray(p['y'],dtype=np.float64);x0=np.asarray(p['x_initial'],dtype=np.float64)
    tau=float(p['tau']);M=float(p['M'])
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or B.ndim!=2 or C.ndim!=2 or y.ndim!=2 or x0.ndim!=1:raise RuntimeError('invalid robust Kalman shapes')
    n=A.shape[0];N,m=y.shape
    if B.shape[0]!=n or C.shape!=(m,n) or x0.shape!=(n,) or N<2 or not all(np.all(np.isfinite(z)) for z in (A,B,C,y,x0)) or not np.isfinite(tau) or not np.isfinite(M):raise RuntimeError('invalid robust Kalman values')
    return {'A':A,'B':B,'C':C,'y':y,'x_initial':x0,'tau':tau,'M':M}

def timed(fn,p):
    try:t=time.perf_counter_ns();out=fn(p);return out,time.perf_counter_ns()-t,None
    except Exception as exc:return None,None,f'{type(exc).__name__}: {exc}'

def frozen_candidates(source_text:str):
    result=json.loads((HERE/'SYNTHETIC_R1_RESULT.json').read_text())
    if result.get('stage')!='synthetic_r1_sealed' or int(result.get('eligible_count',-1))!=31 or not result.get('all_frozen_candidates_eligible'):raise RuntimeError('invalid sealed synthetic result')
    arms=build_candidates(source_text);rows=[];counts={}
    for arm in ARM_ORDER:counts[arm]=len(arms[arm]);rows.extend(arms[arm])
    if counts!=EXPECTED_BY_ARM or len(rows)!=31:raise RuntimeError(f'frozen candidate count mismatch {counts}')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    if not 0<=args.shard<SHARDS:raise ValueError('invalid shard')
    src=args.source.read_bytes();source_sha=hashlib.sha256(src).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    candidates=frozen_candidates(src.decode('utf-8'))
    manifest=fetch(f'{BASE}/{TRAIN_NAME}?download=true');manifest_sha=hashlib.sha256(manifest).hexdigest();blob=git_blob(manifest)
    if len(manifest)!=TRAIN_SIZE or blob!=TRAIN_OID:raise RuntimeError(f'train identity mismatch size={len(manifest)} blob={blob}')
    records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(records)}')
    evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        p=decode_problem(row['problem']);shift=idx%len(candidates);ordered=candidates[shift:]+candidates[:shift]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(source_reference,p);runs=[(c,*timed(c.solve,p)) for c in ordered];execution_order='reference_first'
        else:
            runs=[(c,*timed(c.solve,p)) for c in ordered];ref,ref_ns,ref_err=timed(source_reference,p);execution_order='candidates_first'
        if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed train record {idx+1}: {ref_err}')
        for c,got,cand_ns,error in runs:
            if error is None:valid,reason,metrics=verify_against_reference(p,got,ref,objective_factor=1.01,eps=1e-5)
            else:valid,reason,metrics=False,'exception',{}
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,'objective_ratio':metrics.get('objective_ratio'),'max_dynamics_norm':metrics.get('max_dynamics_norm'),'max_measurement_norm':metrics.get('max_measurement_norm'),'N':int(np.asarray(p['y']).shape[0]),'state_dim':int(np.asarray(p['A']).shape[0]),'process_dim':int(np.asarray(p['B']).shape[1]),'measurement_dim':int(np.asarray(p['C']).shape[0]),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':blob,'train_manifest_sha256':manifest_sha,'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_git_blob_sha1':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'source_sha256':source_sha,'execution_order':execution_order,'shard':args.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'reference_failure_fallback_exploited':False,'test_manifest_contents_opened':False,'test_payloads_opened':0,'public_task_specific_solvers_opened':False})
        del p,ref,runs;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n');print(json.dumps({'shard':args.shard,'rows':len(evidence),'train_sha256':manifest_sha,'git_blob_sha1':blob},indent=2))
if __name__=='__main__':main()
