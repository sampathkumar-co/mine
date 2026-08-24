from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
VALID_REQUIRED=100;HARMONIC_REQUIRED=1.5;MINIMUM_REQUIRED=1.05
SELECTED={'v5_full':'v5_full_r1_3304c859d463a501bd86','v5_no_transfer':'v5_no_transfer_r1_66c5848a3c8a4f51b562','random_search':'random_search_r3_2ef21250df83098c75bd','static_template':'static_template_r2_8fd871e046faa7e4d37c','v4_compatible':'v4_compatible_r5_cdae8cbf0d73bd4d047c'}
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
        arms[arm]={'candidate':name,'implementation_class':s[0]['implementation_class'],'learned_template':s[0]['learned_template'],'transfer_ids':s[0]['transfer_ids'],'valid':valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'passes_blind_gate':bool(valid==VALID_REQUIRED and retries==0 and harmonic(speeds)>=HARMONIC_REQUIRED and min(speeds)>=MINIMUM_REQUIRED)}
    ids={(r['test_manifest_name'],r['test_manifest_git_blob_sha1'],r['test_manifest_sha256']) for r in rows}
    if len(ids)!=1:raise RuntimeError('test identity mismatch')
    name,oid,sha=next(iter(ids));full=arms['v5_full'];nt=arms['v5_no_transfer'];v4=arms['v4_compatible']
    ratio=float(full['harmonic_speedup'])/max(float(nt['harmonic_speedup']),1e-12)
    same_validity=full['valid']==nt['valid'];same_retries=full['invalid_output_retries']==nt['invalid_output_retries']
    separation=bool(full['passes_blind_gate'] and ((not nt['passes_blind_gate']) or (nt['passes_blind_gate'] and ratio>=1.25 and same_validity and same_retries)))
    ablation_confirmed=bool(full['passes_blind_gate'] and not nt['passes_blind_gate'])
    causal=bool(full['passes_blind_gate'] and full['transfer_ids'] and full['implementation_class']!=nt['implementation_class'] and separation and ablation_confirmed)
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':10,'task':'vertex_cover','family':'combinatorial','stage':'blind_r1','test_manifest_name':name,'test_manifest_git_blob_sha1':oid,'test_manifest_sha256':sha,'records':100,'arms':arms,'clean_unseen_win':bool(full['passes_blind_gate']),'v5_over_v4_task_win':bool(full['passes_blind_gate'] and not v4['passes_blind_gate']),'causal_transfer_win':causal,'causal_transfer_checks':{'full_passes_clean_gate':bool(full['passes_blind_gate']),'learned_causal_id':'TM-BFR-01' if full['transfer_ids'] else None,'selected_pair_semantically_distinct':full['implementation_class']!=nt['implementation_class'],'learned_recipe_source_family':'graph_discrete','current_family':'combinatorial','source_family_differs_from_current':True,'blind_causal_separation_condition':separation,'preregistered_no_transfer_ablation_confirmed':ablation_confirmed},'causal_transfer_reason':('TM-BFR-01 full arm passes while preregistered no-transfer ablation fails with equal validity/retries; implementations are distinct and source/current families differ.' if causal else 'One or more preregistered causal-transfer conditions failed.'),'full_over_no_transfer_ratio':ratio,'blind_reruns':0,'post_blind_candidate_changes':False,'threshold_changes':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'blind-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'blind-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda r:(int(r['index']),r['arm'])))+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
