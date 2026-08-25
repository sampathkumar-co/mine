import hashlib,json,urllib.request
from pathlib import Path
REV='bb02811fa47ca1c833baaa344949bcd8fb307ac8'
TASK='robust_kalman_filter'
URL=f'https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REV}/data/{TASK}?recursive=false&expand=false&limit=1000'
req=urllib.request.Request(URL,headers={'User-Agent':'LEXIGEN-v6-task5-metadata-r1'})
with urllib.request.urlopen(req,timeout=240) as r:
    raw=r.read(); etag=r.headers.get('etag')
rows=json.loads(raw)
files=[]
for row in rows:
    if row.get('type')=='file':
        path=str(row['path']); files.append({'path':path,'name':path.rsplit('/',1)[-1],'size':int(row.get('size',0)),'oid':str(row.get('oid',''))})
tr=[x for x in files if '_T100ms_' in x['name'] and x['name'].endswith('_size100_train.jsonl')]
te=[x for x in files if '_T100ms_' in x['name'] and x['name'].endswith('_size100_test.jsonl')]
pairs=[(a,b) for a in tr for b in te if a['name'].replace('_train.jsonl','')==b['name'].replace('_test.jsonl','')]
if len(pairs)!=1: raise RuntimeError(f'expected one dataset pair, got {len(pairs)}')
train,test=pairs[0]
out={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':5,'task':TASK,'stage':'dataset_metadata_r1','dataset_revision':REV,'directory_response_sha256':hashlib.sha256(raw).hexdigest(),'directory_etag':etag,'file_count':len(files),'train':train,'test':test,'train_manifest_opened':False,'test_manifest_opened':False,'payloads_opened':0}
Path('dataset-metadata.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
