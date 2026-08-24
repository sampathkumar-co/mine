from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from candidates import CANDIDATE_META,CANDIDATES_BY_ARM
VALID_REQUIRED=100;HARMONIC_REQUIRED=1.5;MINIMUM_REQUIRED=1.05
def harmonic(v):return len(v)/sum(1/x for x in v) if v and all(x>0 for x in v) else 0.0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();fs=sorted(a.input.rglob('train-shard-*.jsonl'))
    if len(fs)!=10:raise RuntimeError(f'expected 10 shards got {len(fs)}')
    rows=[json.loads(x) for f in fs for x in f.read_text().splitlines() if x.strip()]
    if len(rows)!=3000:raise RuntimeError(f'expected 3000 rows got {len(rows)}')
    names={arm:[n for n,_ in cs] for arm,cs in CANDIDATES_BY_ARM.items()};summ=[]
    for arm,ns in names.items():
      for name in ns:
        s=sorted((r for r in rows if r['arm']==arm and r['candidate']==name),key=lambda r:int(r['index']))
        if len(s)!=100 or len({int(r['index']) for r in s})!=100:raise RuntimeError(f'coverage {arm}/{name}')
        speeds=[float(r['speedup']) for r in s];valid=sum(bool(r['valid']) for r in s);retry=sum(int(r['invalid_output_retries']) for r in s);meta=CANDIDATE_META[name]
        x={'arm':arm,'candidate':name,'valid':valid,'invalid_outputs':100-valid,'invalid_output_retries':retry,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'implementation_class':meta['implementation_class'],'learned_transfer':({'causal_id':meta['transfer_ids'][0],'learned_template':meta['learned_template']} if meta['transfer_ids'] else None)};x['passes_training_correctness']=valid==100 and retry==0;x['passes_gate']=x['passes_training_correctness'] and x['harmonic_speedup']>=HARMONIC_REQUIRED and x['minimum_speedup']>=MINIMUM_REQUIRED;summ.append(x)
    arms={}
    for arm in names:
        p=[x for x in summ if x['arm']==arm];correct=[x for x in p if x['passes_training_correctness']];pool=correct or p;sel=min(pool,key=lambda x:(-int(x['passes_gate']),-x['valid'],-x['harmonic_speedup'],-x['minimum_speedup'],-x['median_speedup'],x['candidate']));arms[arm]={'selected':sel,'candidate_count':len(p),'correct_candidate_count':len(correct),'performance_gate_candidate_count':sum(bool(x['passes_gate']) for x in p)}
    full=arms['v5_full']['selected'];nt=arms['v5_no_transfer']['selected'];ratio=full['harmonic_speedup']/max(nt['harmonic_speedup'],1e-12);eq=any(CANDIDATE_META[n]['implementation_class']==full['implementation_class'] for n in names['v5_no_transfer'])
    comp={'v5_full_harmonic':full['harmonic_speedup'],'v5_no_transfer_harmonic':nt['harmonic_speedup'],'v5_over_no_transfer_ratio':ratio,'selected_v5_uses_learned_transfer':full['learned_transfer'] is not None,'selected_v5_implementation_class':full['implementation_class'],'selected_no_transfer_implementation_class':nt['implementation_class'],'selected_pair_semantically_distinct':full['implementation_class']!=nt['implementation_class'],'equivalent_implementation_available_in_no_transfer':eq,'current_family':'combinatorial','learned_recipe_source_family':'graph_discrete' if full.get('learned_transfer',{}).get('causal_id')=='TM-BFR-01' else None,'source_family_differs_from_current':bool(full.get('learned_transfer') and full.get('learned_transfer',{}).get('causal_id')=='TM-BFR-01'),'training_causal_separation_threshold_crossed':bool(full['learned_transfer'] is not None and full['implementation_class']!=nt['implementation_class'] and full['passes_training_correctness'] and (not nt['passes_training_correctness'] or ratio>=1.25)),'causal_transfer_credit':False}
    ids={(r['train_manifest_name'],r['train_manifest_git_blob_sha1'],r['train_manifest_sha256'],r['expected_test_manifest_name'],r['expected_test_manifest_tree_oid'],r['expected_test_manifest_size']) for r in rows}
    if len(ids)!=1:raise RuntimeError('manifest identity mismatch')
    trn,tro,trsha,ten,teo,tes=next(iter(ids));report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':10,'task':'vertex_cover','stage':'official_training','training_records':100,'candidate_count':30,'candidate_evaluations':3000,'train_manifest_name':trn,'train_manifest_git_blob_sha1':tro,'train_manifest_sha256':trsha,'expected_test_manifest_name':ten,'expected_test_manifest_tree_oid':teo,'expected_test_manifest_size':tes,'frozen_default_gate':{'valid_required':100,'harmonic_speedup_minimum':1.5,'minimum_speedup':1.05,'invalid_output_retries':0},'arms':arms,'all_candidates':summ,'architecture_comparison':comp,'blind_selection_ready':all(arms[k]['correct_candidate_count']>0 for k in arms),'official_test_manifest_contents_opened':False,'official_test_payloads_opened':0}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'training-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda r:(int(r['index']),r['arm'],r['candidate'])))+'\n');print(json.dumps({'selected_by_arm':{k:v['selected'] for k,v in arms.items()},'comparison':comp},indent=2))
if __name__=='__main__':main()
