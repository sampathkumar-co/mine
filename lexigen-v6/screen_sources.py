from __future__ import annotations
import hashlib,json,sys,urllib.error,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lexigen-v5'))
from engine import generate_proposals

SOURCE_COMMIT='dff9914c10800c7a031c9e8c3d4d1c8cd1b38906'
RAW=f'https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}'

def fetch(path,optional=False):
    try:
        q=urllib.request.Request(f'{RAW}/{path}',headers={'User-Agent':'LEXIGEN-v6-source-screen-r1'})
        with urllib.request.urlopen(q,timeout=180) as r:return r.read()
    except urllib.error.HTTPError as e:
        if optional and e.code==404:return None
        raise

def git_blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()

def baseline_ids(features,catalog):
    f=set(features);out=[]
    for row in catalog['entries']:
        any_need=set(row.get('preconditions_any',[]));all_need=set(row.get('preconditions_all',[]))
        if all_need and not all_need<=f:continue
        if any_need and not any_need.intersection(f):continue
        if not any_need and not all_need:continue
        out.append(row['baseline_id'])
    return sorted(out)

def main():
    here=Path(__file__).resolve().parent
    selection=json.loads((here/'SCREENING_POOL_SELECTION_R1_RESULT.json').read_text())
    memory=json.loads((here/'TRANSFER_MEMORY.json').read_text())
    catalog=json.loads((here/'STRONG_BASELINE_CATALOG.json').read_text())
    rows=selection['selected']
    if len(rows)!=24 or selection['task_contents_opened_during_selection'] or selection['data_manifests_opened_during_selection']:raise RuntimeError('invalid sealed screening denominator')
    source_family={v['causal_id']:v['learned_from_family'] for v in memory['learned_templates'].values()}
    out=[]
    for row in rows:
        task=row['task'];family=row['family']
        sp=f'AlgoTuneTasks/{task}/{task}.py';dp=f'AlgoTuneTasks/{task}/description.txt'
        src=fetch(sp);desc=fetch(dp,True)
        proposals=generate_proposals(src.decode('utf-8'))
        fp=proposals['fingerprint'];raw_templates=proposals['applicable_transfer_templates']
        eligible=[];same=[]
        for x in raw_templates:
            cid=x['causal_id'];origin=source_family[cid]
            rec={'causal_id':cid,'template':x['template'],'learned_from_family':origin,'operators':x['operators']}
            if origin==family:same.append(rec)
            else:eligible.append(rec)
        out.append({
          'task':task,'family':family,'frozen_selection_score':row['score'],
          'source_path':sp,'source_git_blob_sha1':git_blob(src),'source_sha256':hashlib.sha256(src).hexdigest(),
          'description_path':dp,'description_present':desc is not None,'description_git_blob_sha1':git_blob(desc) if desc is not None else None,'description_sha256':hashlib.sha256(desc).hexdigest() if desc is not None else None,
          'fingerprint_features':fp['features'],'dependency_calls':fp['dependency_calls'],'raw_applicable_templates':raw_templates,
          'same_family_templates_excluded':same,'applicable_templates':eligible,'applicable_causal_ids':sorted({x['causal_id'] for x in eligible}),
          'strong_baseline_ids':baseline_ids(fp['features'],catalog),
          'training_manifest_opened':False,'training_payloads_opened':0,'test_manifest_opened':False,'test_payloads_opened':0,'reports_opened':False,'public_solvers_opened':False
        })
    payload='\n'.join(json.dumps(x,separators=(',',':')) for x in out)+'\n'
    Path('screening-source-evidence.jsonl').write_text(payload)
    ids=sorted({cid for r in out for cid in r['applicable_causal_ids']});apps=[r for r in out if r['applicable_causal_ids']]
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','stage':'source_fingerprint_screen_r1','screening_pool_count':24,'source_opened_count':24,'applicable_count':len(apps),'applicable_families':sorted({r['family'] for r in apps}),'applicable_causal_ids':ids,'source_commit':SOURCE_COMMIT,'official_train_manifests_opened':False,'official_test_manifests_opened':False,'public_solvers_opened':False,'screening_evidence_sha256':hashlib.sha256(payload.encode()).hexdigest()}
    Path('source-screen-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
