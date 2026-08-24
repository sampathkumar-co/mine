from __future__ import annotations
import importlib.util,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def blob(path):return subprocess.check_output(['git','hash-object',str(path)],text=True).strip()

def main():
    lock=json.loads((HERE/'ENGINE_LOCK.json').read_text());mem=json.loads((HERE/'TRANSFER_MEMORY.json').read_text());exc=json.loads((HERE/'CONTAMINATION_EXCLUSIONS.json').read_text());cat=json.loads((HERE/'STRONG_BASELINE_CATALOG.json').read_text())
    for name,sha in lock['locked_files'].items():
        if blob(HERE/name)!=sha:raise RuntimeError(f'lock mismatch {name}')
    if blob(ROOT/'lexigen-v5'/'engine.py')!=lock['proposal_engine']['git_blob_sha1']:raise RuntimeError('v5 engine mismatch')
    if len(set(exc['combined_exclusions']))!=34 or exc['combined_exclusion_count']!=34:raise RuntimeError('exclusion denominator mismatch')
    v5set={'clustering_outliers','aes_gcm_encryption','spectral_clustering','quantile_regression','earth_movers_distance','gzip_compression','aircraft_wing_design','kernel_density_estimation','least_squares','vertex_cover'}
    if not v5set<=set(exc['combined_exclusions']):raise RuntimeError('v5 holdout missing from exclusions')
    expected={
      'TM-BFR-01':['encode candidate state in word-parallel form','restrict work to a provably relevant frontier','preserve an exact validity check'],
      'TM-CAC-01':['identify a candidate active subset','solve only the active core','verify the full solution with an independent certificate','fall back only on certificate failure'],
      'TM-RRR-01':['solve a structurally reduced representation first','lift the result to the required representation','spend bounded refinement only where the official certificate requires it'],
      'TM-PBEB-01':['evaluate a lower-cost precision/backend representation','bound or measure numerical error against the frozen verifier budget','use the lower-cost path only when the error budget is structurally safe']}
    seen={v['causal_id']:v['recipe'] for v in mem['learned_templates'].values()}
    if seen!=expected:raise RuntimeError('v6 changed learned recipe content')
    if 'weak_reference_speedup_is_not_discovery' not in mem['negative_lessons']:raise RuntimeError('weak-reference lesson absent')
    if len(cat['entries'])<6 or not cat['rules']['no_public_task_specific_solver_access']:raise RuntimeError('strong baseline catalog invalid')
    sel=load('v6_selector',HERE/'selector.py');fin=load('v6_final_selector',HERE/'final_selector.py')
    synthetic=[]
    fams=['graph_discrete','combinatorial','statistics','linear_algebra','signal_processing','machine_learning','geometry','miscellaneous']
    for i in range(32):synthetic.append({'task':f't{i:02d}','family':fams[i%len(fams)],'score':f'{i:064x}'})
    pool=sel.select_pool(synthetic)
    if len(pool)!=24 or len({x['family'] for x in pool})<8:raise RuntimeError('screen selector selftest failed')
    rows=[]
    for i,r in enumerate(pool):
        ids=['TM-BFR-01'] if i%2==0 else ['TM-CAC-01']
        rows.append({'task':r['task'],'family':r['family'],'frozen_selection_score':r['score'],'source_sha256':f'{1000+i:064x}','applicable_causal_ids':ids if i<16 else [],'training_manifest_opened':False,'test_manifest_opened':False,'public_solvers_opened':False})
    cov,a,ok=fin.select(rows);cov2,b,ok2=fin.select(rows)
    if not ok or not ok2 or a!=b or len(a)!=8 or len({x['family'] for x in a})<5:raise RuntimeError('final selector deterministic selftest failed')
    if len(cov['applicable_causal_ids'])<2:raise RuntimeError('causal-id coverage selftest failed')
    print(json.dumps({'status':'passed','exclusions':34,'baseline_entries':len(cat['entries']),'screen_pool':len(pool),'final_selected':len(a),'final_families':len({x['family'] for x in a})},indent=2))
if __name__=='__main__':main()
