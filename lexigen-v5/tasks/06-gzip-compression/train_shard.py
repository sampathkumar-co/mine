from __future__ import annotations
import argparse,base64,gc,gzip,hashlib,json,math,time,urllib.error,urllib.request
from pathlib import Path
from typing import Callable
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='gzip_compression';SHARDS=10;EXPECTED_RECORDS=100
TRAIN_NAME='gzip_compression_T100ms_n658_size100_train.jsonl';TRAIN_OID='f6c04acbfdacbb589c53024618a45bc2706f337f';TRAIN_SIZE=16142
TEST_NAME='gzip_compression_T100ms_n658_size100_test.jsonl';TEST_OID='fc0b222f5b452a049aaf049df5280421318d1425';TEST_SIZE=16200
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task6-train-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:
            last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_bytes(v):
    if isinstance(v,bytes):return v
    if not isinstance(v,dict):raise RuntimeError(f'unsupported bytes encoding {type(v).__name__}')
    kind=v.get('__type__')
    if kind=='bytes':return base64.b64decode(str(v['data_b64']).encode('ascii'),validate=True)
    if kind=='bytes_ref':
        p=str(v.get('bin_path',''))
        if not p or p.startswith('/') or '..' in Path(p).parts:raise RuntimeError(f'unsafe bytes_ref {p}')
        raw=fetch(f'{BASE}/{p}?download=true');size=v.get('size')
        if size is not None and len(raw)!=int(size):raise RuntimeError(f'bytes_ref size mismatch {p}')
        return raw
    raise RuntimeError(f'unsupported bytes wrapper {kind!r}')
def decode_problem(raw):
    if not isinstance(raw,dict) or 'plaintext' not in raw:raise RuntimeError('invalid official gzip problem')
    p=decode_bytes(raw['plaintext'])
    if not isinstance(p,bytes):raise RuntimeError('plaintext is not bytes')
    return {'plaintext':p}
def timed(fn:Callable,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(problem,got,expected):
    if not isinstance(got,dict) or not isinstance(got.get('compressed_data'),bytes):return False,'format'
    data=bytes(got['compressed_data'])
    try:plain=gzip.decompress(data)
    except Exception as e:return False,f'decompress:{type(e).__name__}'
    if plain!=problem['plaintext']:return False,'plaintext_mismatch'
    maxlen=math.ceil(len(expected['compressed_data'])*1.001)
    if len(data)>maxlen:return False,f'size:{len(data)}>{maxlen}'
    return True,None
def flat():
    out=[(arm,name,fn) for arm,cs in CANDIDATES_BY_ARM.items() for name,fn in cs]
    if len(out)!=30:raise RuntimeError(f'expected 30 candidates got {len(out)}')
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(raw)!=TRAIN_SIZE or blob(raw)!=TRAIN_OID:raise RuntimeError(f'train manifest identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 records got {len(rows)}')
    candidates=flat();evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(candidates);ordered=candidates[shift:]+candidates[:shift]
        if idx%2==0:
            expected,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:
            cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];expected,rs,re=timed(reference_exact,p);order='candidates_first'
        if expected is None or rs is None or re:raise RuntimeError(f'reference failed record {idx+1}: {re}')
        for arm,name,got,cs,ce in cr:
            if ce is None:valid,reason=verify(p,got,expected)
            else:valid,reason=False,'exception'
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'valid':bool(valid and ce is None),'failure_reason':ce or reason,'candidate_s':cs,'reference_s':rs,'speedup':rs/cs if cs and cs>0 else 0.0,'plaintext_size':len(p['plaintext']),'candidate_compressed_size':len(got['compressed_data']) if isinstance(got,dict) and isinstance(got.get('compressed_data'),bytes) else None,'reference_compressed_size':len(expected['compressed_data']),'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':hashlib.sha256(raw).hexdigest(),'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_tree_oid':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'execution_order':order,'shard':a.shard,'candidate_executions':1,'reference_executions_for_record':1,'invalid_output_retries':0,'test_manifest_contents_opened':False,'test_payloads_opened':0})
        del p,expected,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__':main()
