from __future__ import annotations
import hashlib,json,urllib.request
from pathlib import Path
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='ode_lorenz96_nonchaotic'
URL=f'https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=true&limit=1000'
req=urllib.request.Request(URL,headers={'User-Agent':'LEXIGEN-v6-task4-storage-metadata-r1'})
with urllib.request.urlopen(req,timeout=120) as r:
    raw=r.read();etag=r.headers.get('ETag')
rows=json.loads(raw.decode('utf-8'))
files=[]
for x in rows:
    if x.get('type')!='file':continue
    files.append({k:x.get(k) for k in ('path','size','oid','lfs','xetHash') if k in x})
files=sorted(files,key=lambda x:x['path'])
out={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':4,'task':TASK,'stage':'dataset_storage_metadata_r1','dataset_revision':REVISION,'metadata_url':URL,'response_sha256':hashlib.sha256(raw).hexdigest(),'etag':etag,'files':files,'train_manifest_body_opened':False,'test_manifest_body_opened':False,'payloads_opened':0,'public_task_specific_solvers_opened':False}
Path('dataset-storage-metadata.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
