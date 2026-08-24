from __future__ import annotations
import argparse,gc,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='vertex_cover';SHARDS=10;EXPECTED_RECORDS=100
TEST_NAME='vertex_cover_T100ms_n15_size100_test.jsonl';TEST_OID='a11bc56b01a7ab254843102454d790637b89ec56';TEST_SIZE=1143200
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SELECTED={'v5_full':'v5_full_r1_3304c859d463a501bd86','v5_no_transfer':'v5_no_transfer_r1_66c5848a3c8a4f51b562','random_search':'random_search_r3_2ef21250df83098c75bd','static_template':'static_template_r2_8fd871e046faa7e4d37c','v4_compatible':'v4_compatible_r5_cdae8cbf0d73bd4d047c'}
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task10-blind-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_problem(p):
    if not isinstance(p,list):raise RuntimeError('vertex_cover problem is not list')
    n=len(p)
    if n==0 or any(not isinstance(row,list) or len(row)!=n for row in p):raise RuntimeError('invalid adjacency shape')
    for i,row in enumerate(p):
        if row[i]!=0:raise RuntimeError('nonzero diagonal')
        for j,x in enumerate(row):
            if x not in (0,1) or p[j][i]!=x:raise RuntimeError('invalid adjacency matrix')
    return p
def timed(fn,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(problem,got,ref):
    try:
        n=len(problem)
        if not isinstance(got,list) or len(set(got))!=len(got) or any(not isinstance(x,int) or isinstance(x,bool) or x<0 or x>=n for x in got):return False,'format'
        s=set(got)
        for i in range(n):
            for j in range(i+1,n):
                if problem[i][j] and i not in s and j not in s:return False,'uncovered_edge'
        if len(got)!=len(ref):return False,f'nonoptimal:{len(got)}!={len(ref)}'
        return True,None
    except Exception as e:return False,f'{type(e).__name__}:{e}'
def selected_functions():
    lookup={name:fn for cs in CANDIDATES_BY_ARM.values() for name,fn in cs}
    missing=[name for name in SELECTED.values() if name not in lookup]
    if missing:raise RuntimeError(f'missing frozen selections {missing}')
    return [(arm,name,lookup[name]) for arm,name in SELECTED.items()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not 0<=a.shard<SHARDS:raise ValueError('invalid shard')
    raw=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(raw)!=TEST_SIZE or blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=EXPECTED_RECORDS:raise RuntimeError(f'expected 100 test records got {len(rows)}')
    selected=selected_functions();e=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(selected);ordered=selected[shift:]+selected[:shift]
        if idx%2==0:ref,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];ref,rs,re=timed(reference_exact,p);order='candidates_first'
        if ref is None or rs is None or re:raise RuntimeError(f'reference failed test record {idx+1}: {re}')
        for arm,name,got,t,err in cr:
            ok,reason=verify(p,got,ref) if err is None else (False,'exception')
            meta=CANDIDATE_META[name]
            e.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'implementation_class':meta['implementation_class'],'learned_template':meta['learned_template'],'transfer_ids':meta['transfer_ids'],'valid':bool(ok and err is None),'failure_reason':err or reason,'candidate_s':t,'reference_s':rs,'speedup':rs/t if t and t>0 else 0.0,'nodes':len(p),'edges':sum(p[i][j] for i in range(len(p)) for j in range(i+1,len(p))),'optimal_cover_size':len(ref),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':hashlib.sha256(raw).hexdigest(),'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'candidate_executions':1,'reference_executions_for_record':1})
        del p,ref,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in e)+'\n')
if __name__=='__main__':main()
