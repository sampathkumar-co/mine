from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path

VALID_REQUIRED=100;HARMONIC_REQUIRED=1.50;MINIMUM_REQUIRED=1.05;CURRENT_FAMILY='signal_processing'
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')
EXPECTED_BY_ARM={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}
CAMPAIGN_ELIGIBLE={'TM-BFR-01','TM-CAC-01','TM-RRR-01'}

def harmonic(v):return len(v)/sum(1.0/x for x in v) if v and all(math.isfinite(x) and x>0 for x in v) else 0.0

def semantic_key(impl:str)->str:
    if impl in {'cvxpy_source_reference','source_equivalent_bfr_false_positive_fallback','source_equivalent_cac_uncertified_fallback','source_equivalent_campaign_ineligible_pbeb_fallback','generic_clarabel_source_equivalent','cvxpy_clarabel_source_equivalent'}:return 'source_formulation'
    if impl=='generic_shallow_v_elimination':return 'shallow_v_elimination'
    if impl=='rrr_deep_w_only_exact':return 'rrr_deep_w_only_exact'
    return impl

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    files=sorted(args.input.rglob('train-shard-*.jsonl'))
    if len(files)!=10:raise RuntimeError(f'expected 10 shards got {len(files)}')
    rows=[json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=3100:raise RuntimeError(f'expected 3100 rows got {len(rows)}')
    identities={(r['train_manifest_name'],r['train_manifest_git_blob_sha1'],r['train_manifest_sha256'],r['expected_test_manifest_name'],r['expected_test_manifest_git_blob_sha1'],int(r['expected_test_manifest_size']),r['source_sha256']) for r in rows}
    if len(identities)!=1:raise RuntimeError('manifest/source identity mismatch across shards')
    train_name,train_blob,train_sha,test_name,test_blob,test_size,source_sha=next(iter(identities))
    keys=sorted({(r['arm'],r['candidate']) for r in rows},key=lambda x:(ARM_ORDER.index(x[0]),x[1]))
    if len(keys)!=31:raise RuntimeError(f'expected 31 candidates got {len(keys)}')
    summaries=[]
    for arm,name in keys:
        sub=sorted((r for r in rows if r['arm']==arm and r['candidate']==name),key=lambda r:int(r['index']))
        if len(sub)!=100 or len({int(r['index']) for r in sub})!=100:raise RuntimeError(f'coverage failure {arm}/{name}')
        tids={tuple(r.get('transfer_ids',[])) for r in sub};ops={tuple(r.get('operators',[])) for r in sub}
        if len(tids)!=1 or len(ops)!=1:raise RuntimeError(f'proposal metadata mismatch {arm}/{name}')
        speeds=[float(r['speedup']) for r in sub];valid=sum(bool(r['valid']) for r in sub);retries=sum(int(r['invalid_output_retries']) for r in sub)
        x={'arm':arm,'candidate':name,'valid':valid,'invalid_outputs':100-valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'implementation_class':sub[0]['implementation_class'],'semantic_implementation_key':semantic_key(sub[0]['implementation_class']),'operators':list(next(iter(ops))),'transfer_ids':list(next(iter(tids))),'learned_template':sub[0].get('learned_template'),'baseline_id':sub[0].get('baseline_id')}
        x['passes_training_correctness']=valid==VALID_REQUIRED and retries==0;x['passes_gate']=x['passes_training_correctness'] and x['harmonic_speedup']>=HARMONIC_REQUIRED and x['minimum_speedup']>=MINIMUM_REQUIRED;summaries.append(x)
    arms={}
    for arm in ARM_ORDER:
        pool=[x for x in summaries if x['arm']==arm]
        if len(pool)!=EXPECTED_BY_ARM[arm]:raise RuntimeError(f'expected {EXPECTED_BY_ARM[arm]} in {arm}, got {len(pool)}')
        correct=[x for x in pool if x['passes_training_correctness']];selpool=correct or pool
        selected=min(selpool,key=lambda x:(-int(x['passes_gate']),-x['valid'],-x['harmonic_speedup'],-x['minimum_speedup'],-x['median_speedup'],x['candidate']))
        arms[arm]={'selected':selected,'candidate_count':len(pool),'correct_candidate_count':len(correct),'performance_gate_candidate_count':sum(bool(x['passes_gate']) for x in pool)}
    memory=json.loads((Path(__file__).resolve().parents[2]/'TRANSFER_MEMORY.json').read_text());source_by_id={str(v['causal_id']):str(v['learned_from_family']) for v in memory['learned_templates'].values()}
    full=arms['v6_full']['selected'];nt=arms['v6_no_transfer']['selected'];strong=arms['strong_baseline']['selected']
    full_nt=full['harmonic_speedup']/max(nt['harmonic_speedup'],1e-12);baseline_comp=full['harmonic_speedup']/max(strong['harmonic_speedup'],1e-12)
    selected_ids=list(full.get('transfer_ids',[]));eligible_ids=[x for x in selected_ids if x in CAMPAIGN_ELIGIBLE];different=[x for x in eligible_ids if source_by_id.get(x) and source_by_id[x]!=CURRENT_FAMILY];same=[x for x in eligible_ids if source_by_id.get(x)==CURRENT_FAMILY]
    origins={x:source_by_id.get(x) for x in selected_ids};nt_keys={x['semantic_implementation_key'] for x in summaries if x['arm']=='v6_no_transfer' and x['passes_training_correctness']};equiv=full['semantic_implementation_key'] in nt_keys
    pair_distinct=full['semantic_implementation_key']!=nt['semantic_implementation_key'];equal=full['valid']==nt['valid'] and full['invalid_output_retries']==nt['invalid_output_retries']
    sep=bool(different and pair_distinct and not equiv and full['passes_training_correctness'] and ((not nt['passes_gate']) or (equal and full_nt>=1.25)))
    comparison={'v6_full_harmonic':full['harmonic_speedup'],'v6_no_transfer_harmonic':nt['harmonic_speedup'],'strong_baseline_harmonic':strong['harmonic_speedup'],'v6_full_over_no_transfer_ratio':full_nt,'strong_baseline_time_over_full_time_harmonic':baseline_comp,'strong_baseline_competitiveness_threshold':0.80,'training_strong_baseline_competitiveness_passes':bool(strong['passes_training_correctness'] and baseline_comp>=0.80),'selected_v6_uses_learned_transfer':bool(selected_ids),'selected_transfer_ids':selected_ids,'selected_campaign_eligible_transfer_ids':eligible_ids,'selected_different_family_transfer_ids':different,'selected_same_family_transfer_ids':same,'learned_recipe_source_families':origins,'current_family':CURRENT_FAMILY,'selected_v6_implementation_class':full['implementation_class'],'selected_v6_semantic_key':full['semantic_implementation_key'],'selected_no_transfer_implementation_class':nt['implementation_class'],'selected_no_transfer_semantic_key':nt['semantic_implementation_key'],'selected_pair_semantically_distinct':pair_distinct,'equivalent_implementation_available_in_no_transfer':equiv,'training_causal_separation_condition':sep,'training_baseline_qualified_causal_diagnostic':bool(full['passes_gate'] and different and sep and strong['passes_training_correctness'] and baseline_comp>=0.80),'causal_transfer_credit':False,'reason_no_credit':'Training evidence only; causal credit is blind-only.'}
    report={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':5,'task':'robust_kalman_filter','family':CURRENT_FAMILY,'stage':'official_training_r1','training_records':100,'candidate_count':31,'candidate_evaluations':3100,'train_manifest_name':train_name,'train_manifest_git_blob_sha1':train_blob,'train_manifest_sha256':train_sha,'expected_test_manifest_name':test_name,'expected_test_manifest_git_blob_sha1':test_blob,'expected_test_manifest_size':test_size,'source_sha256':source_sha,'frozen_default_gate':{'valid_required':100,'harmonic_speedup_minimum':1.5,'minimum_speedup':1.05,'invalid_output_retries':0},'arms':arms,'all_candidates':summaries,'architecture_comparison':comparison,'blind_selection_ready':all(arms[k]['correct_candidate_count']>0 for k in ARM_ORDER),'official_test_manifest_contents_opened':False,'official_test_payloads_opened':0,'public_task_specific_solvers_opened':False,'threshold_changes':False,'scientific_candidate_changes':False,'verifier_changes':False}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n');ordered=sorted(rows,key=lambda r:(int(r['index']),ARM_ORDER.index(r['arm']),r['candidate']));(args.output/'training-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in ordered)+'\n');print(json.dumps({'selected_by_arm':{k:v['selected'] for k,v in arms.items()},'comparison':comparison},indent=2))
if __name__=='__main__':main()
