from __future__ import annotations
import argparse, importlib, json, random, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V19R3=HERE.parent/'v19r3'; V19R2=HERE.parent/'v19r2'; V17=HERE.parent/'v17'
for folder in (HERE,V19R3,V19R2,V17):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from constructive_dsl_v17 import synthesize as synthesize_v17
from enumerate_v19r3 import PRODUCTION_SCHEMA as OLD_PRODUCTION_SCHEMA, build_program, param
from portable_runtime_v19r2 import execute_portable
from runtime_v19r2 import as_grid, canonical, execute, sha256_json
PRODUCTION_SCHEMA='lexigen-v19r4-invented-production-v1'
def load(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8'))
def generate(task:str,root:Path,seed:int):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    random.seed(seed); pair=importlib.import_module(f'tasks.task_{task}').generate(); return as_grid(pair['input']),as_grid(pair['output'])
def descriptor_program(d): return build_program(int(d['marker_colour']),int(d['output_background']),tuple(d['actions']),str(d['set_mode']),str(d['base_mode']))
def descriptor_hash(d): return sha256_json(descriptor_program(d))
def _substitute(value:Any,args:dict[str,Any]):
    if isinstance(value,dict):
        if value.get('op')=='param': return args[str(value['name'])]
        return {k:_substitute(v,args) for k,v in value.items()}
    if isinstance(value,list): return [_substitute(v,args) for v in value]
    return value
def expand_production(production,args):
    if production.get('schema')!=PRODUCTION_SCHEMA: raise RuntimeError('production schema')
    if sorted(x['name'] for x in production['parameters'])!=sorted(args): raise RuntimeError('argument mismatch')
    return _substitute(production['body'],args)
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--package-root',type=Path,required=True); parser.add_argument('--arcgen-root',type=Path,required=True); parser.add_argument('--output',type=Path,default=HERE/'V19R4_CEGIS_REPORT.json'); args=parser.parse_args()
    pre=load(HERE/'V19R4_PRECOMMIT.json'); previous=load(V19R3/'V19R3_ATTEMPT_REPORT.json'); gate=10; task='2bcee788'; old=next(x for x in previous['outcomes'] if x['gate']==gate); survivors=[dict(x) for x in old['search']['exact_survivor_descriptors']]
    hashes=[descriptor_hash(x) for x in survivors]
    if hashes!=pre['frozen_gate10_survivor_hashes']: raise RuntimeError(f'frozen survivor identities changed: {hashes}')
    package=load(args.package_root/'v13-campaign-10'/'redacted-task.json'); demos=[(as_grid(x['input']),as_grid(x['output'])) for x in package['train']]
    v17_failure=None
    try: synthesize_v17(demos)
    except RuntimeError as error: v17_failure=str(error)
    if v17_failure is None: raise RuntimeError('v17 unexpectedly solved gate 10')
    revisions=[]; attempts=accepted=rejections=0; rejection_examples=[]
    while len(survivors)>1 and accepted<int(pre['cegis']['maximum_accepted_refinement_cases']):
        seed=8_500_000+gate*100_000+attempts; attempts+=1
        try: source,target=generate(task,args.arcgen_root,seed)
        except (ValueError,IndexError,TypeError,RuntimeError) as error:
            rejections+=1; rejection={'attempt_index':attempts-1,'seed':seed,'type':type(error).__name__,'message':str(error)}; rejection_examples.append(rejection); revisions.append({'kind':'generator_rejection',**rejection,'survivor_hashes':[descriptor_hash(x) for x in survivors]}); continue
        accepted+=1; before=[descriptor_hash(x) for x in survivors]; kept=[]; evaluations=[]
        for descriptor in survivors:
            program=descriptor_program(descriptor); primary=execute(program,source); portable=execute_portable(program,source)
            if primary!=portable: raise RuntimeError('runtime disagreement during CEGIS')
            exact=primary==target; evaluations.append({'program_sha256':sha256_json(program),'exact':exact,'prediction_sha256':sha256_json(primary)})
            if exact: kept.append(descriptor)
        if not kept:
            revisions.append({'kind':'accepted_case','accepted_index':accepted-1,'attempt_index':attempts-1,'seed':seed,'input':source,'output':target,'input_sha256':sha256_json(source),'output_sha256':sha256_json(target),'survivors_before':before,'survivors_after':[],'evaluations':evaluations}); write(args.output,{'schema':'lexigen-v19r4-cegis-report-v1','status':'all_survivors_eliminated','revisions':revisions,'hidden_outputs_opened':False,'world_level_breakthrough':False}); raise RuntimeError('all frozen survivors eliminated')
        after=[descriptor_hash(x) for x in kept]; revisions.append({'kind':'accepted_case','accepted_index':accepted-1,'attempt_index':attempts-1,'seed':seed,'input':source,'output':target,'input_sha256':sha256_json(source),'output_sha256':sha256_json(target),'survivors_before':before,'survivors_after':after,'eliminated':sorted(set(before)-set(after)),'evaluations':evaluations}); survivors=kept
    status='unique_survivor' if len(survivors)==1 else 'budget_exhausted_ambiguous'
    report={'schema':'lexigen-v19r4-cegis-report-v1','status':status,'gate':gate,'task_id':task,'initial_survivors':len(pre['frozen_gate10_survivor_hashes']),'final_survivors':len(survivors),'attempted_seeds':attempts,'accepted_refinement_cases':accepted,'generator_rejections':rejections,'generator_rejection_examples':rejection_examples,'frozen_v17_ablation_failed':True,'frozen_v17_failure':v17_failure,'revisions':revisions,'hidden_outputs_opened':False,'world_level_breakthrough':False}
    if len(survivors)==1:
        selected=survivors[0]; concrete=descriptor_program(selected); production={'schema':PRODUCTION_SCHEMA,'parameters':[{'name':'marker_colour','type':'colour'},{'name':'output_background','type':'colour'}],'body':build_program(param('marker_colour'),param('output_background'),tuple(selected['actions']),selected['set_mode'],selected['base_mode']),'origin':{'method':'full_composition_enumeration_then_preregistered_public_cegis','initial_survivor_hashes':pre['frozen_gate10_survivor_hashes'],'cegis_revision_count':accepted,'selected_concrete_program_sha256':sha256_json(concrete)}}; production['name']='generated_'+sha256_json(production)[:16]; arguments={'marker_colour':selected['marker_colour'],'output_background':selected['output_background']}; expanded=expand_production(production,arguments)
        if sha256_json(expanded)!=sha256_json(concrete): raise RuntimeError('production expansion mismatch')
        if task in canonical(production) or task in canonical(concrete): raise RuntimeError('task id leaked')
        report.update({'selected_descriptor':selected,'selected_concrete_program_sha256':sha256_json(concrete),'production_sha256':sha256_json(production),'arguments_sha256':sha256_json(arguments),'selected_descriptor_removal_survivors':0,'complete_composition_invention_candidate':True,'task_id_hits':[]})
        write(HERE/'production'/'v19r4-production.json',production); write(HERE/'production'/'v19r4-arguments.json',arguments); write(HERE/'production'/'v19r4-concrete-program.json',concrete)
    else: report.update({'complete_composition_invention_candidate':False})
    write(args.output,report); print('SUMMARY',json.dumps({k:report[k] for k in ('status','initial_survivors','final_survivors','attempted_seeds','accepted_refinement_cases','generator_rejections')},sort_keys=True))
    if status!='unique_survivor': raise RuntimeError('CEGIS did not identify a unique composition')
if __name__=='__main__': main()
