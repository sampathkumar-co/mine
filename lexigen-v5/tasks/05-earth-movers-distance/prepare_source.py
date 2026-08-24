from __future__ import annotations
import argparse, hashlib, json, urllib.error, urllib.request
from pathlib import Path
from engine import LEARNED_SIGNATURES, generate_proposals, verify_transfer_memory
SOURCE_COMMIT='dff9914c10800c7a031c9e8c3d4d1c8cd1b38906'; TASK='earth_movers_distance'; TASK_INDEX=5
RAW_BASE=f'https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}'
def fetch(p,optional=False):
    req=urllib.request.Request(f'{RAW_BASE}/{p}',headers={'User-Agent':'LEXIGEN-v5-task5-source-r1'})
    try:
        with urllib.request.urlopen(req,timeout=180) as r:return r.read()
    except urllib.error.HTTPError as e:
        if optional and e.code==404:return None
        raise
def blob(b): return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task-start',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    s=json.loads(a.task_start.read_text())
    if s['task']!=TASK or int(s['task_index'])!=TASK_INDEX: raise RuntimeError('Task 5 identity mismatch')
    if any(s[k] for k in ['task_source_opened_before_task_start','task_description_opened_before_task_start','training_manifest_opened_before_task_start','training_payloads_opened_before_task_start','test_manifest_opened_before_task_start','test_payloads_opened_before_task_start','reports_opened','public_solvers_opened']): raise RuntimeError('unclean boundary')
    memory=verify_transfer_memory(); src=fetch(s['expected_source_path']); desc=fetch(s['expected_description_path'],True)
    if blob(src)!=s['expected_source_git_blob_sha1']: raise RuntimeError('source blob mismatch')
    if desc is not None and blob(desc)!=s['expected_description_git_blob_sha1']: raise RuntimeError('description blob mismatch')
    proposals=generate_proposals(src.decode()); arms=proposals['arms']; expected={'v5_full','v5_no_transfer','random_search','static_template','v4_compatible'}
    if set(arms)!=expected or any(len(v)>6 for v in arms.values()): raise RuntimeError('proposal budget mismatch')
    learned={tuple(v) for v in LEARNED_SIGNATURES.values()}
    for row in arms['v5_no_transfer']:
        if row['transfer_ids'] or row['learned_template'] is not None or tuple(row['operators']) in learned: raise RuntimeError('causal separation violation')
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':5,'task':TASK,'family':s['family'],'stage':'source_analysis_r1','source_commit':SOURCE_COMMIT,'source_git_blob_sha1':blob(src),'source_sha256':hashlib.sha256(src).hexdigest(),'description_git_blob_sha1':blob(desc) if desc else None,'description_sha256':hashlib.sha256(desc).hexdigest() if desc else None,'transfer_memory':memory,'engine_output':proposals,'official_training_manifest_opened':False,'official_training_payloads_opened':0,'official_test_manifest_opened':False,'official_test_payloads_opened':0,'reports_opened':False,'public_solvers_opened':False,'human_task_specific_solver_design':False}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'source-analysis.json').write_text(json.dumps(report,indent=2)+'\n'); (a.output/'task-source.py').write_bytes(src)
    if desc:(a.output/'description.txt').write_bytes(desc)
    print(json.dumps({'features':proposals['fingerprint']['features'],'applicable_transfer_templates':proposals['applicable_transfer_templates'],'proposal_counts':{k:len(v) for k,v in arms.items()}},indent=2))
if __name__=='__main__': main()
