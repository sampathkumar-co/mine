from __future__ import annotations

import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np

from candidates import ARM_ORDER,build_candidates,independent_source_contract,numpy_source_svd_float64

REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='svd';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='svd_T100ms_n474_size100_train.jsonl';TRAIN_POINTER_OID='79143bfc7daeb9a395640025020aca44dc985560';TRAIN_SIZE=13242
TEST_NAME='svd_T100ms_n474_size100_test.jsonl';TEST_POINTER_OID='f977b820009f4c9672b51c291904a39e5fbfd941';TEST_SIZE=13300
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SOURCE_SHA256='8f7771e5618509c0b8af73390440dd7258a253983cdf8cc03b0208fa4218b018'
EXPECTED_BY_ARM={'v6_full':5,'v6_no_transfer':6,'random_search':5,'static_template':6,'v5_compatible':6,'strong_baseline':1}
HERE=Path(__file__).resolve().parent

def fetch(url:str)->bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task7-svd-train-r1'})
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
    if not isinstance(p,dict) or 'matrix' not in p:raise RuntimeError('invalid SVD problem structure')
    a=np.asarray(p['matrix'],dtype=np.float64)
    if a.ndim!=2 or min(a.shape)<=0 or not np.all(np.isfinite(a)):raise RuntimeError('invalid SVD matrix')
    return {'matrix':np.ascontiguousarray(a,dtype=np.float64)}

def timed(fn,p):
    try:t=time.perf_counter_ns();out=fn(p);return out,time.perf_counter_ns()-t,None
    except Exception as exc:return None,None,f'{type(exc).__name__}: {exc}'

def frozen_candidates(source_text:str):
    sealed=json.loads((HERE/'SYNTHETIC_R1_RESULT.json').read_text())
    if sealed.get('stage')!='synthetic_r1_sealed' or int(sealed.get('eligible_count',-1))!=29:raise RuntimeError('invalid sealed synthetic result')
    excluded={str(x['candidate']) for x in sealed.get('ineligible_candidates',[])}
    if excluded!={'v6_full_r4_b1ef08a2d68a248c0821','random_search_r5_11970ee5ebe67b874464'}:raise RuntimeError(f'unexpected synthetic exclusions {excluded}')
    arms=build_candidates(source_text);rows=[];counts={}
    for arm in ARM_ORDER:
        q=[c for c in arms[arm] if c.name not in excluded];counts[arm]=len(q);rows.extend(q)
    if counts!=EXPECTED_BY_ARM or len(rows)!=29:raise RuntimeError(f'eligible candidate mismatch {counts}/{len(rows)}')
    return rows

def evaluate_candidate(c,p):
    got,cand_ns,error=timed(c.solve,p)
    if error is None:
        valid,reason,metrics=independent_source_contract(p,got)
    else:
        valid,reason,metrics=False,'exception',{}
    del got;gc.collect()
    return {'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,**metrics}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    if not 0<=args.shard<SHARDS:raise ValueError('invalid shard')
    src=args.source.read_bytes();source_sha=hashlib.sha256(src).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    candidates=frozen_candidates(src.decode('utf-8'))
    manifest=fetch(f'{BASE}/{TRAIN_NAME}?download=true');manifest_sha=hashlib.sha256(manifest).hexdigest();resolved_blob=git_blob(manifest)
    if len(manifest)!=TRAIN_SIZE or resolved_blob!=TRAIN_POINTER_OID:raise RuntimeError(f'train payload identity mismatch size={len(manifest)} blob={resolved_blob}')
    records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(records)}')
    ev=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        p=decode_problem(row['problem']);shift=idx%len(candidates);ordered=candidates[shift:]+candidates[:shift]
        compact=[]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(numpy_source_svd_float64,p);execution_order='reference_first'
            if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
            ref_valid,ref_reason,ref_metrics=independent_source_contract(p,ref);del ref;gc.collect()
            if not ref_valid:raise RuntimeError(f'reference contract failure record {idx+1}: {ref_reason} {ref_metrics}')
            for c in ordered:compact.append(evaluate_candidate(c,p))
        else:
            execution_order='candidates_first'
            for c in ordered:compact.append(evaluate_candidate(c,p))
            ref,ref_ns,ref_err=timed(numpy_source_svd_float64,p)
            if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
            ref_valid,ref_reason,ref_metrics=independent_source_contract(p,ref);del ref;gc.collect()
            if not ref_valid:raise RuntimeError(f'reference contract failure record {idx+1}: {ref_reason} {ref_metrics}')
        m,n=np.asarray(p['matrix']).shape
        for r in compact:
            cand_ns=r['candidate_ns'];r.update({'index':idx+1,'seed':int(row.get('seed',idx+1)),'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,'matrix_rows':int(m),'matrix_cols':int(n),'train_manifest_name':TRAIN_NAME,'train_manifest_pointer_oid':TRAIN_POINTER_OID,'train_manifest_payload_sha256':manifest_sha,'train_manifest_resolved_git_blob_sha1':resolved_blob,'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_pointer_oid':TEST_POINTER_OID,'expected_test_manifest_size':TEST_SIZE,'source_sha256':source_sha,'execution_order':execution_order,'shard':args.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'test_manifest_contents_opened':False,'test_payloads_opened':0,'public_task_specific_solvers_opened':False,'training_revision':'R1'})
            ev.append(r)
        del p,compact;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ev)+'\n');print(json.dumps({'shard':args.shard,'rows':len(ev),'manifest_sha256':manifest_sha,'resolved_git_blob_sha1':resolved_blob},indent=2))
if __name__=='__main__':main()
