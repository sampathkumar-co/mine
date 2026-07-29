from __future__ import annotations
import argparse, importlib, json, random, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V19R4=HERE; V19R3=HERE.parent/'v19r3'; V19R2=HERE.parent/'v19r2'; V17=HERE.parent/'v17'
for folder in (HERE,V19R3,V19R2,V17):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from constructive_dsl_v17 import synthesize as synthesize_v17
from portable_runtime_v19r2 import execute_portable
from runtime_v19r2 import as_grid, canonical, execute, sha256_json
from run_cegis_v19r4 import expand_production

def load(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8'))
def generate(task:str,root:Path,seed:int):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    random.seed(seed); pair=importlib.import_module(f'tasks.task_{task}').generate(); return as_grid(pair['input']),as_grid(pair['output'])
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--package-root',type=Path,required=True); parser.add_argument('--arcgen-root',type=Path,required=True); parser.add_argument('--cases',type=int,required=True); parser.add_argument('--output',type=Path,default=HERE/'V19R4_VALIDATION_REPORT.json'); args=parser.parse_args()
    pre=load(HERE/'V19R4_PRECOMMIT.json'); cegis=load(HERE/'V19R4_CEGIS_REPORT.json'); production=load(HERE/'production'/'v19r4-production.json'); arguments=load(HERE/'production'/'v19r4-arguments.json'); concrete=load(HERE/'production'/'v19r4-concrete-program.json')
    if cegis['status']!='unique_survivor' or cegis['final_survivors']!=1: raise RuntimeError('CEGIS selection not unique')
    if sha256_json(production)!=cegis['production_sha256'] or sha256_json(concrete)!=cegis['selected_concrete_program_sha256']: raise RuntimeError('frozen production identity changed')
    expanded=expand_production(production,arguments)
    if sha256_json(expanded)!=sha256_json(concrete): raise RuntimeError('production expansion mismatch')
    task='2bcee788'; gate=10; package=load(args.package_root/'v13-campaign-10'/'redacted-task.json'); demos=[(as_grid(x['input']),as_grid(x['output'])) for x in package['train']]
    v17_failure=None
    try: synthesize_v17(demos)
    except RuntimeError as error: v17_failure=str(error)
    if v17_failure is None: raise RuntimeError('v17 unexpectedly solved gate 10')
    demo_primary=sum(execute(concrete,s)==t for s,t in demos); demo_portable=sum(execute_portable(concrete,s)==t for s,t in demos)
    accepted=attempts=rejections=primary=portable=agreement=0; rejection_examples=[]
    while accepted<args.cases:
        seed=9_500_000+gate*100_000+attempts; attempts+=1
        try: source,target=generate(task,args.arcgen_root,seed)
        except (ValueError,IndexError,TypeError,RuntimeError) as error:
            rejections+=1
            if len(rejection_examples)<20: rejection_examples.append({'seed':seed,'type':type(error).__name__,'message':str(error)})
            if attempts>args.cases*5+1000: raise RuntimeError('too many generator rejections')
            continue
        left=execute(concrete,source); right=execute_portable(concrete,source); primary+=left==target; portable+=right==target; agreement+=left==right; accepted+=1
    passed=primary==portable==agreement==accepted
    report={'schema':'lexigen-v19r4-disjoint-validation-report-v1','gate':gate,'task_id':task,'production_sha256':sha256_json(production),'concrete_program_sha256':sha256_json(concrete),'arguments_sha256':sha256_json(arguments),'cegis_initial_survivors':cegis['initial_survivors'],'cegis_accepted_cases':cegis['accepted_refinement_cases'],'cegis_final_survivors':cegis['final_survivors'],'selected_descriptor_removal_survivors':cegis['selected_descriptor_removal_survivors'],'demonstrations':len(demos),'demonstration_primary_exact':demo_primary,'demonstration_portable_exact':demo_portable,'accepted_fresh_cases':accepted,'generator_attempts':attempts,'generator_rejections':rejections,'generator_rejection_examples':rejection_examples,'fresh_primary_exact':primary,'fresh_portable_exact':portable,'fresh_runtime_agreement':agreement,'fresh_gate_passed':passed,'frozen_v17_ablation_failed':True,'frozen_v17_failure':v17_failure,'task_id_hits':[] if task not in canonical(production) and task not in canonical(concrete) else [task],'fixed_pool_negative_gates':[7,13],'hidden_outputs_opened':False,'sealed_external_success':False,'transfer_demonstrated':False,'world_level_breakthrough':False}
    write(args.output,report); print('SUMMARY',json.dumps({k:report[k] for k in ('accepted_fresh_cases','fresh_primary_exact','fresh_portable_exact','fresh_runtime_agreement','generator_rejections','fresh_gate_passed')},sort_keys=True))
    if not passed: raise RuntimeError('disjoint validation failed; report preserved')
if __name__=='__main__': main()
