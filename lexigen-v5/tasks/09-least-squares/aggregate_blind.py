from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
VALID_REQUIRED=100;HARMONIC_REQUIRED=1.5;MINIMUM_REQUIRED=1.05
SELECTED={'v5_full':'v5_full_r2_41510e43e8fafb598496','v5_no_transfer':'v5_no_transfer_r6_66c5848a3c8a4f51b562','random_search':'random_search_r5_e818498ab004d266d2a1','static_template':'static_template_r6_d044a19fd4551034dc11','v4_compatible':'v4_compatible_r3_ec4b9c17aaa3767d4f6d'}
def harmonic(v):return len(v)/sum(1/x for x in v) if v and all(x>0 for x in v) else 0.0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fs=sorted(a.input.rglob('blind-shard-*.jsonl'))
    if len(fs)!=10:raise RuntimeError(f'expected 10 shards got {len(fs)}')
    rows=[json.loads(line) for f in fs for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=500:raise RuntimeError(f'expected 500 rows got {len(rows)}')
    arms={}
    for arm,name in SELECTED.items():
        s=sorted((r for r in rows if r['arm']==arm and r['candidate']==name),key=lambda r:int(r['index']))
        if len(s)!=100 or len({int(r['index']) for r in s})!=100:raise RuntimeError(f'{arm} coverage mismatch')
        speeds=[float(r['speedup']) for r in s];valid=sum(bool(r['valid']) for r in s);retries=sum(int(r['invalid_output_retries']) for r in s)
        arms[arm]={'candidate':name,'valid':valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'passes_blind_gate':bool(valid==100 and retries==0 and harmonic(speeds)>=HARMONIC_REQUIRED and min(speeds)>=MINIMUM_REQUIRED)}
    ids={(r['test_manifest_name'],r['test_manifest_git_blob_sha1'],r['test_manifest_sha256']) for r in rows}
    if len(ids)!=1:raise RuntimeError('test identity mismatch')
    name,oid,sha=next(iter(ids));full=arms['v5_full'];nt=arms['v5_no_transfer'];v4=arms['v4_compatible']
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':9,'task':'least_squares','stage':'blind_r1','test_manifest_name':name,'test_manifest_git_blob_sha1':oid,'test_manifest_sha256':sha,'records':100,'arms':arms,'clean_unseen_win':bool(full['passes_blind_gate']),'v5_over_v4_task_win':bool(full['passes_blind_gate'] and not v4['passes_blind_gate']),'causal_transfer_win':False,'causal_transfer_reason':'Pre-blind selection already precludes causal credit: full and no-transfer use the same polynomial_linear_reduction_plus_reference implementation class, and TM-RRR-01 was learned from the same linear_algebra family as this task.','full_over_no_transfer_ratio':float(full['harmonic_speedup'])/max(float(nt['harmonic_speedup']),1e-12),'blind_reruns':0,'threshold_changes':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'blind-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'blind-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda r:(int(r['index']),r['arm'])))+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
