from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from candidates import CANDIDATE_META,CANDIDATES_BY_ARM
VALID_REQUIRED=100;HARMONIC_REQUIRED=1.50;MINIMUM_REQUIRED=1.05
def harmonic(v):return len(v)/sum(1/x for x in v) if v and all(x>0 for x in v) else 0.0
def summarise(rows,arm,candidate):
    s=sorted((r for r in rows if r['arm']==arm and r['candidate']==candidate),key=lambda r:int(r['index']))
    if len(s)!=100 or len({int(r['index']) for r in s})!=100:raise RuntimeError(f'{arm}/{candidate} lacks 100 records')
    speeds=[float(r['speedup']) for r in s];valid=sum(bool(r['valid']) for r in s);retries=sum(int(r['invalid_output_retries']) for r in s);times=[float(r['candidate_s']) for r in s if r['candidate_s'] is not None];meta=CANDIDATE_META[candidate]
    out={'arm':arm,'candidate':candidate,'valid':valid,'invalid_outputs':100-valid,'invalid_output_retries':retries,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'median_candidate_s':statistics.median(times),'total_candidate_s':sum(times),'implementation_class':meta['implementation_class'],'learned_transfer':({'causal_id':meta['transfer_ids'][0],'learned_template':meta['learned_template']} if meta['transfer_ids'] else None)}
    out['passes_training_correctness']=valid==100 and retries==0;out['passes_default_performance_gate_on_training']=bool(out['passes_training_correctness'] and out['harmonic_speedup']>=HARMONIC_REQUIRED and out['minimum_speedup']>=MINIMUM_REQUIRED);return out
def select(ss,arm):
    a=[x for x in ss if x['arm']==arm];correct=[x for x in a if x['passes_training_correctness']];pool=correct or a
    chosen=min(pool,key=lambda x:(-int(bool(x['passes_default_performance_gate_on_training'])),-int(x['valid']),-float(x['harmonic_speedup']),-float(x['minimum_speedup']),-float(x['median_speedup']),str(x['candidate'])))
    return {'arm':arm,'selected':chosen,'candidate_count':len(a),'correct_candidate_count':len(correct),'performance_gate_candidate_count':sum(bool(x['passes_default_performance_gate_on_training']) for x in a)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fs=sorted(a.input.rglob('train-shard-*.jsonl'))
    if len(fs)!=10:raise RuntimeError(f'expected 10 shards got {len(fs)}')
    rows=[json.loads(line) for f in fs for line in f.read_text().splitlines() if line.strip()]
    if len(rows)!=3000:raise RuntimeError(f'expected 3000 rows got {len(rows)}')
    names={arm:[name for name,_ in cs] for arm,cs in CANDIDATES_BY_ARM.items()};pairs={(arm,n) for arm,ns in names.items() for n in ns}
    if {(r['arm'],r['candidate']) for r in rows}!=pairs:raise RuntimeError('candidate identity mismatch')
    if any(sum(1 for r in rows if int(r['index'])==i)!=30 for i in range(1,101)):raise RuntimeError('record coverage mismatch')
    ss=[summarise(rows,arm,n) for arm,ns in names.items() for n in ns];arms={arm:select(ss,arm) for arm in names}
    full=arms['v5_full']['selected'];nt=arms['v5_no_transfer']['selected'];ratio=float(full['harmonic_speedup'])/max(float(nt['harmonic_speedup']),1e-12)
    eq=any(CANDIDATE_META[n]['implementation_class']==full['implementation_class'] for n in names['v5_no_transfer'])
    comp={'v5_full_harmonic':full['harmonic_speedup'],'v5_no_transfer_harmonic':nt['harmonic_speedup'],'v5_over_no_transfer_ratio':ratio,'selected_v5_uses_learned_transfer':full['learned_transfer'] is not None,'selected_v5_implementation_class':full['implementation_class'],'selected_no_transfer_implementation_class':nt['implementation_class'],'selected_pair_semantically_distinct':full['implementation_class']!=nt['implementation_class'],'equivalent_implementation_available_in_no_transfer':eq,'learned_recipe_source_family':'linear_algebra' if full.get('learned_transfer',{} ) and full['learned_transfer'].get('causal_id')=='TM-RRR-01' else None,'current_family':'miscellaneous','source_family_differs_from_current':bool(full.get('learned_transfer') and full['learned_transfer'].get('causal_id')=='TM-RRR-01'),'training_causal_separation_threshold_crossed':bool(full['learned_transfer'] is not None and full['implementation_class']!=nt['implementation_class'] and full['passes_training_correctness'] and (not nt['passes_training_correctness'] or ratio>=1.25)),'causal_transfer_credit':False}
    ids={(r['train_manifest_name'],r['train_manifest_git_blob_sha1'],r['train_manifest_sha256'],r['expected_test_manifest_name'],r['expected_test_manifest_tree_oid'],r['expected_test_manifest_size']) for r in rows}
    if len(ids)!=1:raise RuntimeError('manifest identity mismatch')
    tr_name,tr_oid,tr_sha,te_name,te_oid,te_size=next(iter(ids))
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':7,'task':'aircraft_wing_design','revision':1,'stage':'official_training','train_manifest_name':tr_name,'train_manifest_git_blob_sha1':tr_oid,'train_manifest_sha256':tr_sha,'expected_test_manifest_name':te_name,'expected_test_manifest_tree_oid':te_oid,'expected_test_manifest_size':te_size,'training_records':100,'candidate_count':30,'frozen_default_gate':{'valid_required':100,'harmonic_speedup_minimum':1.5,'minimum_speedup':1.05,'invalid_output_retries':0},'all_candidates':ss,'arms':arms,'architecture_comparison':comp,'blind_selection_ready':bool(arms['v5_full']['correct_candidate_count']),'official_test_manifest_contents_opened':False,'official_test_payloads_opened':0}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'training-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda r:(int(r['index']),r['arm'],r['candidate'])))+'\n');print(json.dumps({'selected_by_arm':{k:v['selected'] for k,v in arms.items()},'architecture_comparison':comp},indent=2))
if __name__=='__main__':main()
