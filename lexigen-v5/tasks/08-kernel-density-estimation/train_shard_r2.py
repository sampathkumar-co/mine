from __future__ import annotations
import argparse,gc,hashlib,json,time,urllib.error,urllib.request
from pathlib import Path
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact
import numpy as np
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='kernel_density_estimation';SHARDS=10
TRAIN_NAME='kernel_density_estimation_T100ms_n300_size100_train.jsonl';TRAIN_OID='a70fb5e71c9da5d1bdf0bb2e0466d5b89b915bc6';TRAIN_SIZE=5085207
TEST_NAME='kernel_density_estimation_T100ms_n300_size100_test.jsonl';TEST_OID='8ba6a13986708aacec3496c7d7cfbc1a778ffc89';TEST_SIZE=5963910
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
ELIGIBLE=set(['v5_full_r1_3304c859d463a501bd86','v5_full_r3_a6102573c9f355414229','v5_full_r5_4abf2b51384c560522e8','v5_full_r6_c50e493c5549a408f3e5','v5_no_transfer_r1_91e027e622f2d9a98240','v5_no_transfer_r2_b2614109e1a5ccc10c14','v5_no_transfer_r3_20375ceceffce4d406a4','v5_no_transfer_r4_4a4e1871b7f7b48b9485','v5_no_transfer_r6_66c5848a3c8a4f51b562','random_search_r4_4904ae971430bf5f6d77','static_template_r1_dbfcd2af539b0b2636e7','static_template_r2_8fd871e046faa7e4d37c','static_template_r3_820b1c309b6117eb268d','v4_compatible_r1_f9f3239b6866512e4f68','v4_compatible_r2_9f5f55df04a5ad23f542','v4_compatible_r3_ec4b9c17aaa3767d4f6d','v4_compatible_r5_3df5ed91505aea4ed6cb','v4_compatible_r6_0dde88a4a159a3ad0e40'])
def fetch(url):
    last=None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task8-train-r2'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**attempt))
    raise RuntimeError(f'fetch exhausted {url}') from last
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def decode_problem(p):
    if not isinstance(p,dict) or not {'data_points','query_points','kernel','bandwidth'}.issubset(p):raise RuntimeError('invalid KDE problem')
    X=np.asarray(p['data_points']);raw_q=p['query_points']
    if X.ndim!=2 or X.shape[0]==0:raise RuntimeError('invalid KDE data shape')
    if not isinstance(raw_q,list):raise RuntimeError('invalid KDE query container')
    if len(raw_q)==0:return p
    Q=np.asarray(raw_q)
    if Q.ndim!=2 or X.shape[1]!=Q.shape[1]:raise RuntimeError('invalid KDE query shape')
    return p
def timed(fn,p):
    try:t=time.perf_counter();x=fn(p);return x,time.perf_counter()-t,None
    except Exception as e:return None,None,f'{type(e).__name__}: {e}'
def verify(problem,got,ref):
    try:
        if not isinstance(got,dict) or 'log_density' not in got:return False,'format'
        a=np.asarray(got['log_density'],dtype=float);b=np.asarray(ref['log_density'],dtype=float);q=len(problem['query_points']);a=np.squeeze(a);b=np.squeeze(b)
        if q==0:
            if a.size!=0 or b.size!=0:return False,'empty_query_shape'
            return True,None
        if a.ndim==0:a=np.expand_dims(a,0)
        if b.ndim==0:b=np.expand_dims(b,0)
        if a.ndim!=1 or a.shape!=(q,) or b.shape!=(q,):return False,'shape'
        am=~np.isfinite(a);bm=~np.isfinite(b)
        if not np.array_equal(am,bm):return False,'nonfinite_mask'
        m=np.isfinite(b)
        if m.any() and not np.allclose(a[m],b[m],rtol=1e-4,atol=1e-6):return False,'tolerance'
        return True,None
    except Exception as e:return False,f'{type(e).__name__}:{e}'
def flat():
    out=[(arm,name,fn) for arm,cs in CANDIDATES_BY_ARM.items() for name,fn in cs if name in ELIGIBLE]
    if len(out)!=18:raise RuntimeError(f'expected 18 eligible candidates got {len(out)}')
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=fetch(f'{BASE}/{TRAIN_NAME}?download=true')
    if len(raw)!=TRAIN_SIZE or blob(raw)!=TRAIN_OID:raise RuntimeError('train manifest identity mismatch')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100:raise RuntimeError(f'expected 100 records got {len(rows)}')
    cs=flat();e=[]
    for idx,row in ((i,r) for i,r in enumerate(rows) if i%SHARDS==a.shard):
        p=decode_problem(row['problem']);shift=idx%len(cs);ordered=cs[shift:]+cs[:shift]
        if idx%2==0:ref,rs,re=timed(reference_exact,p);cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];order='reference_first'
        else:cr=[(arm,name,*timed(fn,p)) for arm,name,fn in ordered];ref,rs,re=timed(reference_exact,p);order='candidates_first'
        if ref is None or rs is None or re:raise RuntimeError(f'reference failed record {idx+1}: {re}')
        for arm,name,got,t,err in cr:
            ok,reason=verify(p,got,ref) if err is None else (False,'exception')
            e.append({'index':idx+1,'seed':int(row.get('seed',idx+1)),'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'learned_template':CANDIDATE_META[name]['learned_template'],'transfer_ids':CANDIDATE_META[name]['transfer_ids'],'valid':bool(ok and err is None),'failure_reason':err or reason,'candidate_s':t,'reference_s':rs,'speedup':rs/t if t and t>0 else 0.0,'dims':len(p['data_points'][0]),'num_points':len(p['data_points']),'num_queries':len(p['query_points']),'kernel':p['kernel'],'train_manifest_name':TRAIN_NAME,'train_manifest_git_blob_sha1':TRAIN_OID,'train_manifest_sha256':hashlib.sha256(raw).hexdigest(),'expected_test_manifest_name':TEST_NAME,'expected_test_manifest_tree_oid':TEST_OID,'expected_test_manifest_size':TEST_SIZE,'execution_order':order,'shard':a.shard,'invalid_output_retries':0,'training_revision':2})
        del p,ref,cr;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in e)+'\n')
if __name__=='__main__':main()
