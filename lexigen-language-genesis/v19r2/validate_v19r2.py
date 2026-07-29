from __future__ import annotations
import argparse, hashlib, importlib, json, random, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V17=HERE.parent/'v17'
for folder in (HERE,V17):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from constructive_dsl_v17 import synthesize as synthesize_v17
from invent_v19r2 import expand_production, invent
from portable_runtime_v19r2 import execute_portable
from runtime_v19r2 import FORBIDDEN_NAMED_OPS, as_grid, canonical, execute, sha256_json, walk_ops

def fsha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8'))
def generate(task:str,root:Path,seed:int):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    random.seed(seed); pair=importlib.import_module(f'tasks.task_{task}').generate()
    return as_grid(pair['input']),as_grid(pair['output'])
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--package-root',type=Path,required=True); parser.add_argument('--arcgen-root',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=0); parser.add_argument('--output',type=Path,default=HERE/'V19R2_REPORT.json'); args=parser.parse_args()
    pre=load(HERE/'V19R2_PRECOMMIT.json'); outcomes=[]; any_fresh_failure=False
    for item in pre['fixed_development_gates']:
        gate=int(item['gate']); task=str(item['task_id']); path=args.package_root/f'v13-campaign-{gate:02d}'/'redacted-task.json'
        if fsha(path)!=item['redacted_task_sha256']: raise RuntimeError(f'package hash changed for gate {gate}')
        package=load(path); examples=[(as_grid(x['input']),as_grid(x['output'])) for x in package['train']]
        v17_failure=None
        try: synthesize_v17(examples)
        except RuntimeError as error: v17_failure=str(error)
        if v17_failure is None: raise RuntimeError(f'frozen v17 unexpectedly solved gate {gate}')
        base={'gate':gate,'task_id':task,'demonstrations':len(examples),'frozen_v17_ablation_failed':True,'frozen_v17_failure':v17_failure,'hidden_outputs_opened':False}
        try: production,arguments,concrete,invention=invent(examples)
        except RuntimeError as error:
            outcomes.append({**base,'status':'no_program','reason':str(error),'fresh_cases_requested':args.cases,'fresh_cases_run':0})
            print(json.dumps(outcomes[-1],sort_keys=True),flush=True); continue
        expanded=expand_production(production,arguments)
        forbidden=sorted((set(walk_ops(production))|set(walk_ops(expanded)))&FORBIDDEN_NAMED_OPS)
        if forbidden: raise RuntimeError(f'forbidden operators gate {gate}: {forbidden}')
        if task in canonical(production) or task in canonical(expanded): raise RuntimeError(f'task id leak gate {gate}')
        demo_primary=sum(execute(expanded,s)==t for s,t in examples); demo_portable=sum(execute_portable(expanded,s)==t for s,t in examples)
        if demo_primary!=len(examples) or demo_portable!=len(examples): raise RuntimeError(f'demo replay gate {gate}')
        accepted=attempts=rejections=primary=portable=agreement=0; rejection_examples=[]
        while accepted<args.cases:
            seed=7_400_000+gate*100_000+attempts; attempts+=1
            try: source,target=generate(task,args.arcgen_root,seed)
            except (ValueError,IndexError,TypeError,RuntimeError) as error:
                rejections+=1
                if len(rejection_examples)<20: rejection_examples.append({'seed':seed,'type':type(error).__name__,'message':str(error)})
                if attempts>args.cases*5+1000: raise RuntimeError(f'too many generator rejections gate {gate}')
                continue
            left=execute(expanded,source); right=execute_portable(expanded,source)
            primary+=left==target; portable+=right==target; agreement+=left==right; accepted+=1
        fresh_pass=primary==portable==agreement==accepted
        any_fresh_failure|=not fresh_pass
        stem=f'gate-{gate:02d}'; write(HERE/'productions'/f'{stem}-production.json',production); write(HERE/'productions'/f'{stem}-arguments.json',arguments); write(HERE/'productions'/f'{stem}-concrete.json',concrete)
        outcome={**base,'status':'production_found','production_sha256':sha256_json(production),'arguments_sha256':sha256_json(arguments),'concrete_program_sha256':sha256_json(concrete),'invention':invention,'demonstration_primary_exact':demo_primary,'demonstration_portable_exact':demo_portable,'forbidden_opcode_hits':forbidden,'task_id_hits':[],'fresh_cases_requested':args.cases,'accepted_fresh_cases':accepted,'generator_attempts':attempts,'generator_rejections':rejections,'generator_rejection_examples':rejection_examples,'fresh_primary_exact':primary,'fresh_portable_exact':portable,'fresh_runtime_agreement':agreement,'fresh_gate_passed':fresh_pass}
        outcomes.append(outcome); print(json.dumps(outcome,sort_keys=True),flush=True)
    report={'schema':'lexigen-v19r2-fixed-pool-report-v1','base_rejected_v19_commit':pre['base_rejected_v19_commit'],'gates_fixed':len(outcomes),'cases_per_success':args.cases,'productions_found':sum(x['status']=='production_found' for x in outcomes),'no_program_failures':sum(x['status']=='no_program' for x in outcomes),'hidden_outputs_opened':False,'world_level_breakthrough':False,'claim_boundary':{'autonomous_grammar_production_candidate':not any_fresh_failure and any(x['status']=='production_found' for x in outcomes),'sealed_external_success':False,'transfer_demonstrated':False,'human_supplied_bias':'primitive inventory and bounded affine composition template'},'outcomes':outcomes}
    write(args.output,report); print('SUMMARY',json.dumps({'gates':report['gates_fixed'],'productions':report['productions_found'],'no_program':report['no_program_failures'],'fresh_failure':any_fresh_failure},sort_keys=True))
    if any_fresh_failure: raise RuntimeError('one or more fresh gates failed; report preserved')
if __name__=='__main__': main()
