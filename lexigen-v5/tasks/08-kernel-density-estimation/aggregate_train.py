from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from candidates import CANDIDATE_META
VALID_REQUIRED=100;HARMONIC_REQUIRED=1.5;MINIMUM_REQUIRED=1.05
def harmonic(v):return len(v)/sum(1/x for x in v) if v and all(x>0 for x in v) else 0.0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();fs=sorted(a.input.rglob('train-shard-*.jsonl'))
    if len(fs)!=10:raise RuntimeError(f'expected 10 shards got {len(fs)}')
    rows=[json.loads(x) for f in fs for x in f.read_text().splitlines() if x.strip()]
    if len(rows)!=1800:raise RuntimeError(f'expected 1800 rows got {len(rows)}')
    pairs=sorted({(r['arm'],r['candidate']) for r in rows});summ=[]
    for arm,name in pairs:
        s=sorted((r for r in rows if r['arm']==arm and r['candidate']==name),key=lambda r:int(r['index']))
        if len(s)!=100:raise RuntimeError(f'coverage {arm}/{name}')
        speeds=[float(r['speedup']) for r in s];valid=sum(bool(r['valid']) for r in s);retry=sum(int(r['invalid_output_retries']) for r in s);meta=CANDIDATE_META[name]
        x={'arm':arm,'candidate':name,'valid':valid,'invalid_output_retries':retry,'harmonic_speedup':harmonic(speeds),'minimum_speedup':min(speeds),'median_speedup':statistics.median(speeds),'maximum_speedup':max(speeds),'implementation_class':meta['implementation_class'],'learned_transfer':({'causal_id':meta['transfer_ids'][0],'learned_template':meta['learned_template']} if meta['transfer_ids'] else None)};x['passes_training_correctness']=valid==100 and retry==0;x['passes_gate']=x['passes_training_correctness'] and x['harmonic_speedup']>=1.5 and x['minimum_speedup']>=1.05;summ.append(x)
    arms={}
    for arm in ['v5_full','v5_no_transfer','random_search','static_template','v4_compatible']:
        p=[x for x in summ if x['arm']==arm];correct=[x for x in p if x['passes_training_correctness']];pool=correct or p;sel=min(pool,key=lambda x:(-int(x['passes_gate']),-x['valid'],-x['harmonic_speedup'],-x['minimum_speedup'],x['candidate']));arms[arm]={'selected':sel,'candidate_count':len(p),'correct_candidate_count':len(correct)}
    full=arms['v5_full']['selected'];nt=arms['v5_no_transfer']['selected'];ratio=full['harmonic_speedup']/max(nt['harmonic_speedup'],1e-12)
    report={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':8,'task':'kernel_density_estimation','stage':'official_training','training_records':100,'eligible_candidate_count':18,'candidate_evaluations':1800,'arms':arms,'all_candidates':summ,'architecture_comparison':{'v5_over_no_transfer_ratio':ratio,'selected_v5_implementation_class':full['implementation_class'],'selected_no_transfer_implementation_class':nt['implementation_class'],'selected_pair_semantically_distinct':full['implementation_class']!=nt['implementation_class'],'causal_transfer_credit_possible':False,'reason':'Synthetic gate rejected distinct learned RRR/PBEB candidates; surviving learned BFR/CAC mappings are sklearn reference-equivalent.'},'blind_selection_ready':all(arms[a]['correct_candidate_count']>0 for a in arms),'official_test_manifest_contents_opened':False,'official_test_payloads_opened':0}
    ids={(r['train_manifest_sha256'],r['expected_test_manifest_tree_oid']) for r in rows};
    if len(ids)!=1:raise RuntimeError('manifest identity mismatch')
    report['train_manifest_sha256'],report['expected_test_manifest_oid']=next(iter(ids));a.output.mkdir(parents=True,exist_ok=True);(a.output/'training-summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'training-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in sorted(rows,key=lambda r:(int(r['index']),r['arm'],r['candidate'])))+'\n');print(json.dumps({'selected_by_arm':{k:v['selected'] for k,v in arms.items()},'comparison':report['architecture_comparison']},indent=2))
if __name__=='__main__':main()
