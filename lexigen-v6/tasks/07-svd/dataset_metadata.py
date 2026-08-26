from __future__ import annotations
import hashlib,json,urllib.request
from pathlib import Path
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8';TASK='svd'
URL=f'https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false&limit=1000'

def main():
    req=urllib.request.Request(URL,headers={'User-Agent':'LEXIGEN-v6-task7-svd-metadata-r1','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r: raw=r.read();etag=r.headers.get('etag')
    payload=json.loads(raw)
    if not isinstance(payload,list):raise RuntimeError('dataset directory metadata is not a list')
    files=[]
    for row in payload:
        if isinstance(row,dict) and row.get('type')=='file':
            path=str(row.get('path',''));name=path.rsplit('/',1)[-1]
            files.append({'path':path,'name':name,'size':int(row.get('size',0)),'oid':str(row.get('oid',''))})
    trains=[r for r in files if '_T100ms_' in r['name'] and '_size100_train.jsonl' in r['name']]
    tests=[r for r in files if '_T100ms_' in r['name'] and '_size100_test.jsonl' in r['name']]
    pairs=[]
    for tr in trains:
        stem=tr['name'].replace('_train.jsonl','')
        for te in tests:
            if te['name'].replace('_test.jsonl','')==stem:pairs.append((tr,te))
    if len(pairs)!=1:raise RuntimeError(f'expected one T100ms size100 pair, got {len(pairs)}')
    train,test=pairs[0]
    out={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':7,'task':TASK,'stage':'dataset_metadata_r1','dataset_revision':REVISION,'directory_metadata_url':URL,'directory_response_sha256':hashlib.sha256(raw).hexdigest(),'directory_etag':etag,'file_count':len(files),'train':train,'test':test,'train_manifest_opened':False,'test_manifest_opened':False,'payloads_opened':0,'reports_opened':False,'public_task_specific_solvers_opened':False}
    Path('dataset-metadata.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
