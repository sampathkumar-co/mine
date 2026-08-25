from __future__ import annotations
import argparse,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_SHARDS=10;EXPECTED_RECORDS=100;EXPECTED_ENTRIES=7
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline','recipe_removal_ablation')

def harmonic(values):
    if not values or any((not math.isfinite(x) or x<=0) for x in values):return 0.0
    return len(values)/sum(1.0/x for x in values)

def summarize(rows):
    speeds=[float(r['speedup']) for r in rows];valid=sum(bool(r['valid']) for r in rows);retries=sum(int(r['invalid_output_retries']) for r in rows)
    out={'arm':rows[0]['arm'],'candidate':rows[0]['candidate'],'implementation_class':rows[0]['implementation_class'],'operators':rows[0]['operators'],'transfer_ids':rows[0]['transfer_ids'],'learned_template':rows[0]['learned_template'],'baseline_id':rows[0]['baseline_id'],'recipe_removal_ablation':bool(rows[0].get('recipe_removal_ablation')),'records':len(rows),'valid':valid,'invalid_outputs':len(rows)-valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds)}
    out['passes_clean_blind_gate']=out['records']==100 and out['valid']==100 and out['invalid_output_retries']==0 and out['harmonic_speedup']>=1.50 and out['minimum_speedup']>=1.05
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    files=sorted(args.input.glob('blind-shard-*.jsonl'))
    if len(files)!=EXPECTED_SHARDS:raise RuntimeError(f'expected 10 blind shards got {len(files)}')
    rows=[json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=EXPECTED_RECORDS*EXPECTED_ENTRIES:raise RuntimeError(f'expected 700 blind rows got {len(rows)}')
    if any(int(r['invalid_output_retries'])!=0 for r in rows):raise RuntimeError('blind retries forbidden')
    if any(bool(r.get('reference_failure_fallback_exploited')) for r in rows):raise RuntimeError('reference-failure fallback exploitation detected')
    if any(bool(r.get('verifier_capacity_loophole_exploited')) for r in rows):raise RuntimeError('verifier loophole exploitation detected')
    identities={(r['test_manifest_name'],r['test_manifest_git_blob_sha1'],r['test_manifest_sha256'],r['source_sha256']) for r in rows}
    if len(identities)!=1:raise RuntimeError('blind identity disagreement')
    test_name,test_oid,test_sha,source_sha=next(iter(identities))
    grouped=defaultdict(list)
    for row in rows:grouped[row['candidate']].append(row)
    if len(grouped)!=EXPECTED_ENTRIES:raise RuntimeError(f'expected 7 blind candidates got {len(grouped)}')
    if any(len(v)!=100 for v in grouped.values()):raise RuntimeError('every blind candidate must cover 100 records')
    summaries={name:summarize(group) for name,group in grouped.items()};by_arm={}
    for arm in ARM_ORDER:
        matches=[v for v in summaries.values() if v['arm']==arm]
        if len(matches)!=1:raise RuntimeError(f'expected one candidate for {arm}, got {len(matches)}')
        by_arm[arm]=matches[0]
    full=by_arm['v6_full'];nt=by_arm['v6_no_transfer'];strong=by_arm['strong_baseline'];abl=by_arm['recipe_removal_ablation']
    full_nt=full['harmonic_speedup']/nt['harmonic_speedup'] if nt['harmonic_speedup']>0 else math.inf
    full_abl=full['harmonic_speedup']/abl['harmonic_speedup'] if abl['harmonic_speedup']>0 else math.inf
    baseline_comp=full['harmonic_speedup']/strong['harmonic_speedup'] if strong['harmonic_speedup']>0 else math.inf
    lock=json.loads((Path(__file__).resolve().parent/'BLIND_R1_LOCK.json').read_text())
    equal_nt=full['valid']==nt['valid'] and full['invalid_output_retries']==nt['invalid_output_retries']
    equal_abl=full['valid']==abl['valid'] and full['invalid_output_retries']==abl['invalid_output_retries']
    causal_separation=(not nt['passes_clean_blind_gate']) or (full_nt>=1.25 and equal_nt)
    recipe_removal_eliminates=(not abl['passes_clean_blind_gate']) or (full_abl>=1.25 and equal_abl)
    selected_diff_ids=list(lock['causal_preblind_diagnostics']['selected_different_family_transfer_ids'])
    selected_full_uses_learned=len(full['transfer_ids'])>0
    selected_pair_distinct=bool(lock['causal_preblind_diagnostics']['selected_full_no_transfer_semantically_distinct'])
    baseline_valid=strong['records']==100 and strong['valid']==100 and strong['invalid_output_retries']==0
    baseline_pass=baseline_comp>=0.80
    causal_conditions={'full_passes_clean_blind_gate':bool(full['passes_clean_blind_gate']),'selected_full_uses_learned_transfer':selected_full_uses_learned,'selected_full_uses_different_family_transfer':bool(selected_diff_ids),'selected_full_no_transfer_semantically_distinct':selected_pair_distinct,'source_family_differs_from_current':bool(selected_diff_ids),'causal_separation':bool(causal_separation),'recipe_removal_eliminates_qualifying_advantage':bool(recipe_removal_eliminates),'strong_baseline_valid_same_denominator':bool(baseline_valid),'strong_baseline_competitiveness_passes':bool(baseline_pass)}
    causal_win=all(causal_conditions.values())
    if causal_win:raise RuntimeError('preblind causal-impossibility invariant violated')
    clean_wins={arm:bool(by_arm[arm]['passes_clean_blind_gate']) for arm in ARM_ORDER if arm!='recipe_removal_ablation'}
    payload='\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda x:(int(x['index']),x['arm'],x['candidate'])))+'\n'
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':5,'task':'robust_kalman_filter','family':'signal_processing','stage':'official_blind_r1','blind_records':100,'blind_entries':7,'candidate_evaluations':len(rows),'test_manifest_name':test_name,'test_manifest_git_blob_sha1':test_oid,'test_manifest_sha256':test_sha,'source_sha256':source_sha,'frozen_clean_gate':{'valid_required':100,'harmonic_speedup_minimum':1.50,'minimum_speedup':1.05,'invalid_output_retries':0},'by_arm':by_arm,'clean_wins':clean_wins,'full_no_transfer_harmonic_ratio':full_nt,'full_recipe_removal_harmonic_ratio':full_abl,'strong_baseline_time_over_full_time_harmonic':baseline_comp,'causal_conditions':causal_conditions,'baseline_qualified_causal_transfer_win':False,'preblind_recipe_removal_equivalence_flag':bool(lock['recipe_removal_ablation']['preblind_semantic_equivalence_to_full']),'task5_causal_win_impossible_preblind':True,'invalid_output_retries_total':sum(int(r['invalid_output_retries']) for r in rows),'reference_failure_fallback_exploited':False,'verifier_capacity_loophole_exploited':False,'results_sha256':hashlib.sha256(payload.encode()).hexdigest(),'blind_run_complete':True,'post_blind_candidate_revision_allowed':False,'post_blind_timing_rerun_allowed':False}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'blind-results.jsonl').write_text(payload);(args.output/'blind-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'clean_wins':clean_wins,'full_harmonic':full['harmonic_speedup'],'full_minimum':full['minimum_speedup'],'full_no_transfer_ratio':full_nt,'full_recipe_removal_ratio':full_abl,'strong_baseline_time_over_full_time_harmonic':baseline_comp,'baseline_qualified_causal_transfer_win':False,'results_sha256':summary['results_sha256']},indent=2))
if __name__=='__main__':main()
