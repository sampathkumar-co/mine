from __future__ import annotations
import argparse,json,urllib.parse,urllib.request
from pathlib import Path
REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8'; TASK='aircraft_wing_design'
API=f'https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    url=API+'?'+urllib.parse.urlencode({'recursive':'false','expand':'true'})
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LEXIGEN-v5-task7-metadata-r1'}),timeout=180) as r: payload=json.loads(r.read().decode())
    rows=[]
    for x in payload:
        p=str(x.get('path',''))
        if not p.startswith(f'data/{TASK}/'): raise RuntimeError(f'unexpected path {p}')
        rows.append({'path':p,'type':x.get('type'),'size':x.get('size'),'oid':x.get('oid'),'lfs':x.get('lfs'),'lastCommit':x.get('lastCommit')})
    js=sorted((x for x in rows if str(x['path']).endswith('.jsonl')),key=lambda x:str(x['path']))
    tr=[x for x in js if str(x['path']).endswith('_train.jsonl')]; te=[x for x in js if str(x['path']).endswith('_test.jsonl')]
    if len(tr)!=1 or len(te)!=1: raise RuntimeError('expected one train/test manifest')
    if str(tr[0]['path']).replace('_train.jsonl','_test.jsonl')!=str(te[0]['path']): raise RuntimeError('train/test stems differ')
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':7,'task':TASK,'stage':'dataset_directory_metadata_r1','dataset_revision':REVISION,'training_manifest_metadata':tr[0],'test_manifest_metadata':te[0],'manifest_contents_opened':False,'payload_contents_opened':False,'resolve_or_download_urls_followed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
