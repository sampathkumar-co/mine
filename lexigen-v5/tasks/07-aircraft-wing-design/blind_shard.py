from __future__ import annotations
import argparse,gc,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
from candidates import CANDIDATES_BY_ARM,reference_exact
from train_shard import decode_problem,timed,verify,blob
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='aircraft_wing_design';SHARDS=10
TEST_NAME='aircraft_wing_design_T100ms_n10_size100_test.jsonl';TEST_OID='adf280f7d8134006d54145d291409d692ef28a40';TEST_SIZE=384557
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
SELECTED={'v5_full':'v5_full_r1_3304c859d463a501bd86','v5_no_transfer':'v5_no_transfer_r1_91e027e622f2d9a98240','random_search':'random_search_r1_487d4f738090692a3fa8','static_template':'static_template_r3_820b1c309b6117eb268d','v4_compatible':'v4_compatible_r3_ec4b9c17aaa3767d4f6d'}
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task7-blind-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def selected_functions():
    lookup={name:fn for cs in CANDIDATES_BY_ARM.values() for name,fn in cs};return [(arm,name,lookup[name]) for arm,name in SELECTED.items()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not 0<=a.shard<SHARDS:raise ValueError('invalid shard')
    raw=fetch(f'{BASE}/{TEST_NAME}?download=true')
    if len(raw)!=TEST_SIZE or blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch size={len(raw)} blob={blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100:raise RuntimeError(f'expected 100 test records got {len(rows)}')
    selected=selected_functions();evidence=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(selected);ordered=selected[shift:]+selected[:shift]
        if idx%2==0:ref,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];ref,rs,re=timed(reference_exact,p);order='candidates_first'
        if ref is None or rs is None or re or isinstance(ref.get('A'),list):raise RuntimeError(f'reference failed test record {idx+1}: {re or ref}')
        for arm,name,got,cs,ce in cr:
            if ce is None:valid,reason=verify(p,got,ref)
            else:valid,reason=False,'exception'
            evidence.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'valid':bool(valid and ce is None),'failure_reason':ce or reason,'candidate_s':cs,'reference_s':rs,'speedup':rs/cs if cs and cs>0 else 0.0,'num_conditions':int(p['num_conditions']),'candidate_avg_drag':got.get('avg_drag') if isinstance(got,dict) else None,'reference_avg_drag':ref.get('avg_drag'),'test_manifest_name':TEST_NAME,'test_manifest_git_blob_sha1':TEST_OID,'test_manifest_sha256':hashlib.sha256(raw).hexdigest(),'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'candidate_executions':1})
            gc.collect()
        del p,ref,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in evidence)+'\n')
if __name__=='__main__':main()
