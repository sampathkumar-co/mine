from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V19R2=HERE.parent/'v19r2'; V17=HERE.parent/'v17'
for folder in (HERE,V19R2,V17):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from constructive_dsl_v17 import synthesize as synthesize_v17
from enumerate_v19r3 import TOTAL_CANDIDATES, enumerate_compositions
from portable_runtime_v19r2 import execute_portable
from runtime_v19r2 import as_grid, canonical, execute, sha256_json

def load(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8'))
def fsha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--package-root',type=Path,required=True); parser.add_argument('--output',type=Path,default=HERE/'V19R3_ATTEMPT_REPORT.json'); args=parser.parse_args()
    pre=load(HERE/'V19R3_PRECOMMIT.json'); outcomes=[]
    for item in pre['fixed_development_gates']:
        gate=int(item['gate']); task=str(item['task_id']); path=args.package_root/f'v13-campaign-{gate:02d}'/'redacted-task.json'
        if fsha(path)!=item['redacted_task_sha256']: raise RuntimeError(f'package hash changed gate {gate}')
        package=load(path); examples=[(as_grid(x['input']),as_grid(x['output'])) for x in package['train']]
        v17_failure=None
        try: synthesize_v17(examples)
        except RuntimeError as error: v17_failure=str(error)
        if v17_failure is None: raise RuntimeError(f'v17 unexpectedly solved gate {gate}')
        base={'gate':gate,'task_id':task,'demonstrations':len(examples),'candidate_denominator':TOTAL_CANDIDATES,'frozen_v17_ablation_failed':True,'frozen_v17_failure':v17_failure,'hidden_outputs_opened':False}
        try: production,arguments,program,descriptor,search=enumerate_compositions(examples)
        except RuntimeError as error:
            outcome={**base,'status':'no_program','reason':str(error),'fresh_scoring_performed':False}; outcomes.append(outcome); print(json.dumps(outcome,sort_keys=True),flush=True); continue
        primary=sum(execute(program,s)==t for s,t in examples); portable=sum(execute_portable(program,s)==t for s,t in examples)
        if primary!=len(examples) or portable!=len(examples): raise RuntimeError(f'runtime replay gate {gate}')
        task_hits=[]
        if task in canonical(production) or task in canonical(program): task_hits=[task]
        status='unique_production' if search['exact_survivors']==1 and search['selected_descriptor_removal_survivors']==0 else 'ambiguous_program_set'
        outcome={**base,'status':status,'demonstration_primary_exact':primary,'demonstration_portable_exact':portable,'production_sha256':sha256_json(production),'concrete_program_sha256':sha256_json(program),'arguments_sha256':sha256_json(arguments),'selected_descriptor':descriptor,'search':search,'task_id_hits':task_hits,'fresh_scoring_performed':False,'eligible_for_v19_freeze':status=='unique_production'}
        outcomes.append(outcome); write(HERE/'candidates'/f'gate-{gate:02d}-selected-production.json',production); write(HERE/'candidates'/f'gate-{gate:02d}-selected-arguments.json',arguments); write(HERE/'candidates'/f'gate-{gate:02d}-selected-program.json',program); print(json.dumps(outcome,sort_keys=True),flush=True)
    report={'schema':'lexigen-v19r3-full-composition-attempt-report-v1','fixed_gates':len(outcomes),'candidate_denominator_per_gate':TOTAL_CANDIDATES,'unique_productions':sum(x['status']=='unique_production' for x in outcomes),'ambiguous_program_sets':sum(x['status']=='ambiguous_program_set' for x in outcomes),'no_program_failures':sum(x['status']=='no_program' for x in outcomes),'fresh_scoring_performed':False,'hidden_outputs_opened':False,'v19_pass':False,'world_level_breakthrough':False,'claim_boundary':{'reason':'The only demonstration-exact gate retained multiple full-composition survivors and failed selected-descriptor removal ablation. No survivor was promoted or fresh-scored.'},'outcomes':outcomes}
    write(args.output,report); print('SUMMARY',json.dumps({k:report[k] for k in ('fixed_gates','unique_productions','ambiguous_program_sets','no_program_failures','fresh_scoring_performed')},sort_keys=True))
if __name__=='__main__': main()
