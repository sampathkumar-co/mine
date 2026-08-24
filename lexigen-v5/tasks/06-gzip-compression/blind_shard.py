from __future__ import annotations
import argparse,base64,gc,gzip,hashlib,json,math,time,urllib.error,urllib.request
from pathlib import Path
from candidates import CANDIDATES_BY_ARM,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='gzip_compression';SHARDS=10
TEST_NAME='gzip_compression_T100ms_n658_size100_test.jsonl';TEST_OID='fc0b222f5b452a049aaf049df5280421318d1425';TEST_SIZE=16200
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SELECTED={'v5_full':'v5_full_r6_7653e3865aa7a6def4dc','v5_no_transfer':'v5_no_transfer_r1_66c5848a3c8a4f51b562','random_search':'random_search_r1_bcb2865badc647aa6bc2','static_template':'static_template_r6_d044a19fd4551034dc11','v4_compatible':'v4_compatible_r4_d9863922b850e9717a05'}
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task6-blind-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_bytes(v):
    if isinstance(v,bytes):return v
    if not isinstance(v,dict):raise RuntimeError('unsupported bytes encoding')
    kind=v.get('__type__')
    if kind=='bytes':return base64.b64decode(str(v['data_b64']).encode('ascii'),validate=True)
    if kind=='bytes_ref':
        p=str(v.get('bin_path',''))
        if not p or p.startswith('/') or '..' in Path(p).parts:raise RuntimeError('unsafe bytes_ref')
        raw=fetch(f'{BASE}/{p}?download=true');size=v.get('size')
        if size is not None and len(raw)!=int(size):raise RuntimeError('bytes_ref size mismatch')
        return raw
    raise RuntimeError(f'unsupported bytes wrapper {kind!r}')
def decode_problem(r):return {'plaintext':decode_bytes(r['plaintext'])}
def timed(fn,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(problem,got,expected):
    if not isinstance(got,dict) or not isinstance(got.get('compressed_data'),bytes):return False
    data=bytes(got['compressed_data'])
    try:plain=gzip.decompress(data)
    except Exception:return False
    return plain==problem['plaintext'] and len(data)<=math.ceil(len(expected['compressed_data'])*1.001)
def selected_functions():
    lookup={name:fn for cs in CANDIDATES_BY_ARM.values() for name,fn in cs};return [(arm,name,lookup[name]) for arm,name in SELECTED.items()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(raw)!=TEST_SIZE or blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100:raise RuntimeError(f'expected 100 test records got {len(rows)}')
    selected=selected_functions();evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(selected);ordered=selected[shift:]+selected[:shift]
        if idx%2==0:expected,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];expected,rs,re=timed(reference_exact,p);order='candidates_first'
        if expected is None or re:raise RuntimeError(f'reference failed test record {idx+1}: {re}')
        for arm,name,got,cs,ce in cr:
            valid=ce is None and verify(p,got,expected);evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'valid':valid,'candidate_s':cs,'reference_s':rs,'speedup':rs/cs if cs and cs>0 else 0.0,'failure_reason':ce or (None if valid else 'verifier_reject'),'plaintext_size':len(p['plaintext']),'candidate_compressed_size':len(got['compressed_data']) if isinstance(got,dict) and isinstance(got.get('compressed_data'),bytes) else None,'reference_compressed_size':len(expected['compressed_data']),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':hashlib.sha256(raw).hexdigest(),'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'candidate_executions':1})
        del p,expected,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__':main()
