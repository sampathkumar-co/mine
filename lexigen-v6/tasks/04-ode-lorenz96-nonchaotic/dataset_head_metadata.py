from __future__ import annotations
import hashlib,json,urllib.error,urllib.request
from pathlib import Path
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='ode_lorenz96_nonchaotic'
NAMES=('ode_lorenz96_nonchaotic_T100ms_n7856_size100_train.jsonl','ode_lorenz96_nonchaotic_T100ms_n7856_size100_test.jsonl')
BASE=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None
opener=urllib.request.build_opener(NoRedirect)
keep=('etag','x-linked-etag','x-linked-size','x-repo-commit','x-xet-hash','content-length','content-type','accept-ranges')
rows=[]
for name in NAMES:
    url=f'{BASE}/{name}?download=true';req=urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v6-task4-head-metadata-r1'},method='HEAD')
    try:
        r=opener.open(req,timeout=120);status=getattr(r,'status',200);headers=r.headers
    except urllib.error.HTTPError as e:
        status=e.code;headers=e.headers
    selected={k:headers.get(k) for k in keep if headers.get(k) is not None}
    rows.append({'name':name,'status':status,'headers':selected,'body_bytes_read':0})
canonical=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
out={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':4,'task':TASK,'stage':'dataset_head_metadata_r1','dataset_revision':REVISION,'rows':rows,'canonical_headers_sha256':hashlib.sha256(canonical).hexdigest(),'train_manifest_body_opened_by_this_stage':False,'test_manifest_body_opened_by_this_stage':False,'payloads_opened':0,'public_task_specific_solvers_opened':False}
Path('dataset-head-metadata.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
