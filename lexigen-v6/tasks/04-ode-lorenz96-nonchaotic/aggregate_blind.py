from __future__ import annotations

import argparse,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
EXPECTED_SHARDS=10;EXPECTED_RECORDS=100;EXPECTED_ENTRIES=7
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline','recipe_removal_ablation')

def harmonic(v):return len(v)/sum(1.0/x for x in v) if v and all(math.isfinite(x) and x>0 for x in v) else 0.0

def summarize(rows):
    speeds=[float(r['speedup']) for r in rows];valid=sum(bool(r['valid']) for r in rows);retries=sum(int(r['invalid_output_retries']) for r in rows)
    out={'arm':rows[0]['arm'],'candidate':rows[0]['candidate'],'implementation_class':rows[0]['implementation_class'],'operators':rows[0]['operators'],'transfer_ids':rows[0]['transfer_ids'],'learned_template':rows[0]['learned_template'],'baseline_id':rows[0]['baseline_id'],'recipe_removal_ablation':bool(rows[0].get('recipe_removal_ablation')),'records':len(rows),'valid':valid,'invalid_outputs':len(rows)-valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds)}
    out['passes_clean_blind_gate']=out['records']==100 and valid==100 and retries==0 and out['harmonic_speedup']>=1.50 and out['minimum_speedup']>=1.05;return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    files=sorted(args.input.glob('blind-shard-*.jsonl'))
    if len(files)!=EXPECTED_SHARDS:raise RuntimeError(f'expected 10 shards got {len(files)}')
    rows=[json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=700:raise RuntimeError(f'expected 700 rows got {len(rows)}')
    if any(int(r['invalid_output_retries'])!=0 for r in rows):raise RuntimeError('blind retries forbidden')
    if any(bool(r.get('verifier_capacity_loophole_exploited')) for r in rows):raise RuntimeError('verifier loophole detected')
    identities={(r['test_manifest_name'],r['test_manifest_pointer_oid'],r['test_manifest_payload_sha256'],int(r['test_manifest_size']),r['source_sha256']) for r in rows}
    if len(identities)!=1:raise RuntimeError('blind identity disagreement')
    test_name,test_pointer,test_sha,test_size,source_sha=next(iter(identities));g=defaultdict(list)
    for r in rows:g[r['candidate']].append(r)
    if len(g)!=7 or any(len(x)!=100 for x in g.values()):raise RuntimeError('blind coverage failure')
    summaries={n:summarize(x) for n,x in g.items()};by_arm={}
    for arm in ARM_ORDER:
        m=[v for v in summaries.values() if v['arm']==arm]
        if len(m)!=1:raise RuntimeError(f'expected one {arm}, got {len(m)}')
        by_arm[arm]=m[0]
    full=by_arm['v6_full'];nt=by_arm['v6_no_transfer'];strong=by_arm['strong_baseline'];abl=by_arm['recipe_removal_ablation']
    full_nt=full['harmonic_speedup']/nt['harmonic_speedup'];full_abl=full['harmonic_speedup']/abl['harmonic_speedup'];baseline_comp=full['harmonic_speedup']/strong['harmonic_speedup']
    lock=json.loads((Path(__file__).resolve().parent/'BLIND_R1_LOCK.json').read_text())
    equal_nt=full['valid']==nt['valid'] and full['invalid_output_retries']==nt['invalid_output_retries'];equal_abl=full['valid']==abl['valid'] and full['invalid_output_retries']==abl['invalid_output_retries']
    causal_sep=(not nt['passes_clean_blind_gate']) or (full_nt>=1.25 and equal_nt);recipe_elim=(not abl['passes_clean_blind_gate']) or (full_abl>=1.25 and equal_abl)
    diag=lock['causal_preblind_diagnostics'];baseline_valid=strong['records']==100 and strong['valid']==100 and strong['invalid_output_retries']==0;baseline_pass=baseline_comp>=0.80
    causal_conditions={'full_passes_clean_blind_gate':bool(full['passes_clean_blind_gate']),'selected_full_uses_learned_transfer':bool(full['transfer_ids']),'selected_full_uses_different_family_transfer':bool(diag['selected_different_family_transfer_ids']),'selected_full_no_transfer_semantically_distinct':bool(diag['selected_full_no_transfer_semantically_distinct']),'source_family_differs_from_current':bool(diag['source_family_differs_from_current']),'causal_separation':bool(causal_sep),'recipe_removal_eliminates_qualifying_advantage':bool(recipe_elim),'strong_baseline_valid_same_denominator':bool(baseline_valid),'strong_baseline_competitiveness_passes':bool(baseline_pass)}
    causal_win=all(causal_conditions.values());clean_wins={a:bool(by_arm[a]['passes_clean_blind_gate']) for a in ARM_ORDER if a!='recipe_removal_ablation'}
    payload='\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda x:(int(x['index']),x['arm'],x['candidate'])))+'\n'
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':4,'task':'ode_lorenz96_nonchaotic','family':'scientific_computing','stage':'official_blind_r1','blind_records':100,'blind_entries':7,'candidate_evaluations':700,'test_manifest_name':test_name,'test_manifest_pointer_oid':test_pointer,'test_manifest_payload_sha256':test_sha,'test_manifest_size':test_size,'source_sha256':source_sha,'frozen_clean_gate':{'valid_required':100,'harmonic_speedup_minimum':1.50,'minimum_speedup':1.05,'invalid_output_retries':0},'by_arm':by_arm,'clean_wins':clean_wins,'full_no_transfer_harmonic_ratio':full_nt,'full_recipe_removal_harmonic_ratio':full_abl,'strong_baseline_time_over_full_time_harmonic':baseline_comp,'causal_conditions':causal_conditions,'baseline_qualified_causal_transfer_win':causal_win,'preblind_causal_win_impossible':bool(diag['task4_causal_win_impossible_preblind']),'invalid_output_retries_total':0,'verifier_capacity_loophole_exploited':False,'results_sha256':hashlib.sha256(payload.encode()).hexdigest(),'blind_run_complete':True,'post_blind_candidate_revision_allowed':False,'post_blind_timing_rerun_allowed':False}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'blind-results.jsonl').write_text(payload);(args.output/'blind-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'clean_wins':clean_wins,'full_harmonic':full['harmonic_speedup'],'full_minimum':full['minimum_speedup'],'full_no_transfer_ratio':full_nt,'full_recipe_removal_ratio':full_abl,'strong_baseline_time_over_full_time_harmonic':baseline_comp,'baseline_qualified_causal_transfer_win':causal_win,'results_sha256':summary['results_sha256']},indent=2))
if __name__=='__main__':main()
