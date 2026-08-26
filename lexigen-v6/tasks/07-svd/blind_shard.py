from __future__ import annotations

import argparse,base64,gc,hashlib,io,json,time,urllib.error,urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np

from candidates import build_candidates,independent_source_contract,numpy_source_svd_float64

REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='svd';SHARDS=10;EXPECTED_RECORDS=100
TEST_NAME='svd_T100ms_n474_size100_test.jsonl';TEST_POINTER_OID='f977b820009f4c9672b51c291904a39e5fbfd941';TEST_SIZE=13300
TRAIN_NAME='svd_T100ms_n474_size100_train.jsonl';TRAIN_POINTER_OID='79143bfc7daeb9a395640025020aca44dc985560';TRAIN_PAYLOAD_SHA256='dc4fa2ccbb665a70c332c1b72e0ab063f7f54cebd3368bd6a1fba1dd443ac6a3'
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SOURCE_SHA256='8f7771e5618509c0b8af73390440dd7258a253983cdf8cc03b0208fa4218b018'
HERE=Path(__file__).resolve().parent
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline','recipe_removal')

@dataclass(frozen=True)
class BlindEntry:
    arm:str;candidate:str;proposal_id:str|None;operators:tuple[str,...];transfer_ids:tuple[str,...];learned_template:str|None;implementation_class:str;semantic_implementation_key:str;baseline_id:str|None;solve:Callable[[dict],dict]

def fetch(url:str)->bytes:
    last=None
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task7-svd-blind-r1'})
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

def frozen_entries(source_text:str)->list[BlindEntry]:
    train=json.loads((HERE/'TRAIN_R1_RESULT.json').read_text())
    if train.get('stage')!='official_training_r1_sealed' or not train.get('blind_selection_ready'):raise RuntimeError('invalid sealed training result')
    expected={
      'v6_full':'v6_full_r2_41510e43e8fafb598496',
      'v6_no_transfer':'v6_no_transfer_r3_20375ceceffce4d406a4',
      'random_search':'random_search_r6_dc1b3d1c6cee178fb752',
      'static_template':'static_template_r2_8fd871e046faa7e4d37c',
      'v5_compatible':'v5_compatible_r1_f9f3239b6866512e4f68',
      'strong_baseline':'strong_baseline_sb_reduced_linalg_01_scipy_gesdd'}
    selected={a:str(v['candidate']) for a,v in train['selected_by_arm'].items()}
    if selected!=expected:raise RuntimeError(f'blind selection identity mismatch {selected}')
    arms=build_candidates(source_text);by_name={c.name:c for xs in arms.values() for c in xs};entries=[]
    for arm in expected:
        c=by_name.get(expected[arm])
        if c is None or c.arm!=arm:raise RuntimeError(f'missing selected candidate {arm}')
        entries.append(BlindEntry(arm,c.name,c.proposal_id,c.operators,c.transfer_ids,c.learned_template,c.implementation_class,c.semantic_implementation_key,c.baseline_id,c.solve))
    full=entries[0]
    if full.semantic_implementation_key!='guarded_gram_eigh_svd_float64' or tuple(full.transfer_ids)!=('TM-RRR-01',):raise RuntimeError('unexpected full recipe-removal source')
    entries.append(BlindEntry('recipe_removal','recipe_removal_from_'+full.candidate,full.proposal_id,full.operators,(),None,full.implementation_class,full.semantic_implementation_key,None,full.solve))
    if [e.arm for e in entries]!=list(ARM_ORDER):raise RuntimeError('blind arm order mismatch')
    return entries

def evaluate(entry,p):
    got,cand_ns,error=timed(entry.solve,p)
    if error is None:valid,reason,metrics=independent_source_contract(p,got)
    else:valid,reason,metrics=False,'exception',{}
    del got;gc.collect()
    return {'arm':entry.arm,'candidate':entry.candidate,'proposal_id':entry.proposal_id,'operators':list(entry.operators),'transfer_ids':list(entry.transfer_ids),'learned_template':entry.learned_template,'implementation_class':entry.implementation_class,'semantic_implementation_key':entry.semantic_implementation_key,'baseline_id':entry.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns':cand_ns,**metrics}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    if not 0<=args.shard<SHARDS:raise ValueError('invalid shard')
    src=args.source.read_bytes();source_sha=hashlib.sha256(src).hexdigest()
    if source_sha!=SOURCE_SHA256:raise RuntimeError('source identity mismatch')
    entries=frozen_entries(src.decode('utf-8'))
    manifest=fetch(f'{BASE}/{TEST_NAME}?download=true');manifest_sha=hashlib.sha256(manifest).hexdigest();resolved_blob=git_blob(manifest)
    if len(manifest)!=TEST_SIZE or resolved_blob!=TEST_POINTER_OID:raise RuntimeError(f'test payload identity mismatch size={len(manifest)} blob={resolved_blob}')
    records=[json.loads(x) for x in manifest.decode('utf-8').splitlines() if x.strip()]
    if len(records)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 test records got {len(records)}')
    ev=[]
    for idx,row in ((i,r) for i,r in enumerate(records) if i%SHARDS==args.shard):
        p=decode_problem(row['problem']);shift=idx%len(entries);ordered=entries[shift:]+entries[:shift];compact=[]
        if idx%2==0:
            ref,ref_ns,ref_err=timed(numpy_source_svd_float64,p);execution_order='reference_first'
            if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
            ref_valid,ref_reason,ref_metrics=independent_source_contract(p,ref);del ref;gc.collect()
            if not ref_valid:raise RuntimeError(f'reference contract failure record {idx+1}: {ref_reason} {ref_metrics}')
            for e in ordered:compact.append(evaluate(e,p))
        else:
            execution_order='candidates_first'
            for e in ordered:compact.append(evaluate(e,p))
            ref,ref_ns,ref_err=timed(numpy_source_svd_float64,p)
            if ref is None or ref_ns is None or ref_err:raise RuntimeError(f'reference failed record {idx+1}: {ref_err}')
            ref_valid,ref_reason,ref_metrics=independent_source_contract(p,ref);del ref;gc.collect()
            if not ref_valid:raise RuntimeError(f'reference contract failure record {idx+1}: {ref_reason} {ref_metrics}')
        m,n=np.asarray(p['matrix']).shape
        for r in compact:
            cand_ns=r['candidate_ns'];r.update({'index':idx+1,'seed':int(row.get('seed',idx+1)),'reference_ns':ref_ns,'speedup':(ref_ns/cand_ns) if cand_ns and cand_ns>0 else 0.0,'matrix_rows':int(m),'matrix_cols':int(n),'test_manifest_name':TEST_NAME,'test_manifest_pointer_oid':TEST_POINTER_OID,'test_manifest_payload_sha256':manifest_sha,'test_manifest_resolved_git_blob_sha1':resolved_blob,'test_manifest_size':len(manifest),'frozen_train_manifest_name':TRAIN_NAME,'frozen_train_manifest_pointer_oid':TRAIN_POINTER_OID,'frozen_train_manifest_payload_sha256':TRAIN_PAYLOAD_SHA256,'source_sha256':source_sha,'execution_order':execution_order,'shard':args.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1,'public_task_specific_solvers_opened':False,'blind_revision':'R1'})
            ev.append(r)
        del p,compact;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ev)+'\n');print(json.dumps({'shard':args.shard,'rows':len(ev),'test_manifest_sha256':manifest_sha,'resolved_git_blob_sha1':resolved_blob},indent=2))
if __name__=='__main__':main()
