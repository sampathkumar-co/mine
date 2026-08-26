from __future__ import annotations

import argparse,json,math,statistics
from pathlib import Path

VALID_REQUIRED=100;HARMONIC_REQUIRED=1.50;MINIMUM_REQUIRED=1.05;CURRENT_FAMILY='linear_algebra'
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline','recipe_removal')
EXPECTED_NAMES={
 'v6_full':'v6_full_r2_41510e43e8fafb598496',
 'v6_no_transfer':'v6_no_transfer_r3_20375ceceffce4d406a4',
 'random_search':'random_search_r6_dc1b3d1c6cee178fb752',
 'static_template':'static_template_r2_8fd871e046faa7e4d37c',
 'v5_compatible':'v5_compatible_r1_f9f3239b6866512e4f68',
 'strong_baseline':'strong_baseline_sb_reduced_linalg_01_scipy_gesdd',
 'recipe_removal':'recipe_removal_from_v6_full_r2_41510e43e8fafb598496'}

def harmonic(v):return len(v)/sum(1.0/x for x in v) if v and all(math.isfinite(x) and x>0 for x in v) else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    files=sorted(args.input.rglob('blind-shard-*.jsonl'))
    if len(files)!=10:raise RuntimeError(f'expected 10 shards got {len(files)}')
    rows=[json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=700:raise RuntimeError(f'expected 700 rows got {len(rows)}')
    ids={(r['test_manifest_name'],r['test_manifest_pointer_oid'],r['test_manifest_payload_sha256'],r['test_manifest_resolved_git_blob_sha1'],int(r['test_manifest_size']),r['frozen_train_manifest_name'],r['frozen_train_manifest_pointer_oid'],r['frozen_train_manifest_payload_sha256'],r['source_sha256']) for r in rows}
    if len(ids)!=1:raise RuntimeError('blind manifest/source identity mismatch across shards')
    test_name,test_pointer,test_sha,test_blob,test_size,train_name,train_pointer,train_sha,source_sha=next(iter(ids))
    summaries={}
    for arm in ARM_ORDER:
        name=EXPECTED_NAMES[arm];sub=sorted((r for r in rows if r['arm']==arm and r['candidate']==name),key=lambda r:int(r['index']))
        if len(sub)!=100 or len({int(r['index']) for r in sub})!=100:raise RuntimeError(f'blind coverage failure {arm}/{name}')
        if any(r['candidate']!=name for r in sub):raise RuntimeError('candidate identity drift')
        stable=('proposal_id','implementation_class','semantic_implementation_key','learned_template','baseline_id')
        for field in stable:
            if len({json.dumps(r.get(field),sort_keys=True) for r in sub})!=1:raise RuntimeError(f'metadata mismatch {arm}/{field}')
        tids={tuple(r.get('transfer_ids',[])) for r in sub};ops={tuple(r.get('operators',[])) for r in sub}
        if len(tids)!=1 or len(ops)!=1:raise RuntimeError(f'proposal metadata mismatch {arm}')
        speeds=[float(r['speedup']) for r in sub];valid=sum(bool(r['valid']) for r in sub);retries=sum(int(r['invalid_output_retries']) for r in sub)
        x={'arm':arm,'candidate':name,'proposal_id':sub[0].get('proposal_id'),'valid':valid,'invalid_outputs':100-valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'implementation_class':sub[0]['implementation_class'],'semantic_implementation_key':sub[0]['semantic_implementation_key'],'operators':list(next(iter(ops))),'transfer_ids':list(next(iter(tids))),'learned_template':sub[0].get('learned_template'),'baseline_id':sub[0].get('baseline_id')}
        x['passes_correctness']=valid==VALID_REQUIRED and retries==0;x['clean_win']=x['passes_correctness'] and x['harmonic_speedup']>=HARMONIC_REQUIRED and x['minimum_speedup']>=MINIMUM_REQUIRED;summaries[arm]=x
    full=summaries['v6_full'];nt=summaries['v6_no_transfer'];strong=summaries['strong_baseline'];recipe=summaries['recipe_removal']
    full_nt=full['harmonic_speedup']/max(nt['harmonic_speedup'],1e-12);runtime_comp=full['harmonic_speedup']/max(strong['harmonic_speedup'],1e-12);full_recipe=full['harmonic_speedup']/max(recipe['harmonic_speedup'],1e-12)
    selected_uses_transfer=bool(full['transfer_ids']);different_family_ids=[];same_family_ids=['TM-RRR-01'] if 'TM-RRR-01' in full['transfer_ids'] else []
    selected_pair_distinct=full['semantic_implementation_key']!=nt['semantic_implementation_key']
    causal_separation=bool(selected_pair_distinct and ((not nt['clean_win']) or (full['valid']==nt['valid'] and full_nt>=1.25)))
    recipe_semantically_removed=full['semantic_implementation_key']!=recipe['semantic_implementation_key']
    recipe_eliminates=bool(recipe_semantically_removed and full['clean_win'] and not recipe['clean_win'])
    strong_valid=strong['passes_correctness'];baseline_competitive=bool(strong_valid and runtime_comp>=0.80)
    causal_conditions={'full_clean_win':full['clean_win'],'selected_full_uses_learned_transfer':selected_uses_transfer,'selected_full_no_transfer_semantically_distinct':selected_pair_distinct,'learned_source_family_differs_from_holdout':bool(different_family_ids),'causal_separation':causal_separation,'recipe_removal_eliminates_advantage':recipe_eliminates,'strong_baseline_valid_same_denominator':strong_valid,'strong_baseline_competitiveness_passes':baseline_competitive}
    causal_win=all(causal_conditions.values())
    report={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':7,'task':'svd','family':CURRENT_FAMILY,'stage':'blind_r1','blind_records':100,'blind_entries':7,'blind_evaluations':700,'test_manifest_name':test_name,'test_manifest_pointer_oid':test_pointer,'test_manifest_payload_sha256':test_sha,'test_manifest_resolved_git_blob_sha1':test_blob,'test_manifest_size':test_size,'frozen_train_manifest_name':train_name,'frozen_train_manifest_pointer_oid':train_pointer,'frozen_train_manifest_payload_sha256':train_sha,'source_sha256':source_sha,'frozen_default_gate':{'valid_required':100,'harmonic_speedup_minimum':1.5,'minimum_speedup':1.05,'invalid_output_retries':0},'arms':summaries,'architecture_comparison':{'v6_full_over_no_transfer_ratio':full_nt,'strong_baseline_time_over_full_time_harmonic':runtime_comp,'v6_full_over_recipe_removal_ratio':full_recipe,'selected_full_uses_learned_transfer':selected_uses_transfer,'selected_causal_ids':list(full['transfer_ids']),'selected_different_family_transfer_ids':different_family_ids,'selected_same_family_transfer_ids':same_family_ids,'selected_pair_semantically_distinct':selected_pair_distinct,'recipe_removal_semantically_distinct':recipe_semantically_removed,'causal_conditions':causal_conditions},'v6_full_clean_win':full['clean_win'],'baseline_qualified_causal_transfer_win':causal_win,'causal_transfer_credit':causal_win,'causal_failure_reasons':[k for k,v in causal_conditions.items() if not v],'post_blind_candidate_revisions':0,'invalid_output_retries_total':sum(int(r['invalid_output_retries']) for r in rows),'public_task_specific_solvers_opened':False,'threshold_changes':False}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'blind-summary.json').write_text(json.dumps(report,indent=2)+'\n');ordered=sorted(rows,key=lambda r:(int(r['index']),ARM_ORDER.index(r['arm'])));(args.output/'blind-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ordered)+'\n');print(json.dumps({'arms':summaries,'v6_full_clean_win':report['v6_full_clean_win'],'baseline_qualified_causal_transfer_win':causal_win,'causal_failure_reasons':report['causal_failure_reasons']},indent=2))
if __name__=='__main__':main()
