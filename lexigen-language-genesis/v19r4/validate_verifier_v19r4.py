from __future__ import annotations
import argparse, hashlib, importlib, json, random, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V19R2=HERE.parent/'v19r2'
for folder in (HERE,V19R2):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from cosynthesize_verifier_v19r2 import synthesize_contract
from mutations_v19r2 import apply_output_mutation
from portable_runtime_v19r2 import execute_portable
from portable_verifier_v19r2 import screening_holds_portable, verify_against_reference_portable
from runtime_v19r2 import as_grid, execute, sha256_json
from verifier_grammar_v19r2 import screening_holds, verify_against_reference

def load(path:Path)->Any: return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\n').encode('utf-8'))
def fsha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def generate(task:str,root:Path,seed:int):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    random.seed(seed); pair=importlib.import_module(f'tasks.task_{task}').generate(); return as_grid(pair['input']),as_grid(pair['output'])
def fresh_cases(task:str,gate:int,root:Path,count:int):
    cases=[]; attempts=rejections=0; examples=[]
    while len(cases)<count:
        seed=11_500_000+gate*100_000+attempts; attempts+=1
        try: cases.append(generate(task,root,seed))
        except (ValueError,IndexError,TypeError,RuntimeError) as error:
            rejections+=1
            if len(examples)<20: examples.append({'seed':seed,'type':type(error).__name__,'message':str(error)})
            if attempts>count*5+1000: raise RuntimeError('too many generator rejections')
    return cases,attempts,rejections,examples
def evaluate_contract(contract,production,program,cases,manifest,mutations_per_case:int):
    correct_primary=correct_portable=mutants=screen_primary=screen_portable=full_primary=full_portable=runtime_invalid=0; false_accepts=[]
    for case_index,(source,target) in enumerate(cases):
        reference=execute(program,source); portable_reference=execute_portable(program,source)
        if reference!=target or portable_reference!=reference: raise RuntimeError('frozen production failed verifier fresh target')
        correct_primary+=verify_against_reference(contract,production,program,source,reference,reference)
        correct_portable+=verify_against_reference_portable(contract,production,program,source,reference,portable_reference)
        used=0
        for mutation in manifest:
            if used>=mutations_per_case: break
            try:
                candidate=execute(mutation['ast'],source) if mutation['kind']=='ast' else apply_output_mutation(reference,str(mutation['operator']))
            except Exception:
                runtime_invalid+=1; continue
            if candidate==reference: continue
            used+=1; mutants+=1
            ps=screening_holds(contract,source,candidate,reference); qs=screening_holds_portable(contract,source,candidate,portable_reference)
            pf=verify_against_reference(contract,production,program,source,candidate,reference)
            qf=verify_against_reference_portable(contract,production,program,source,candidate,portable_reference)
            screen_primary+=not ps; screen_portable+=not qs; full_primary+=not pf; full_portable+=not qf
            if ps or qs:
                false_accepts.append({'example_index':100000+case_index,'mutation_sha256':mutation['mutation_sha256'],'source':source,'candidate':candidate,'reference':reference,'primary_screen_accept':bool(ps),'portable_screen_accept':bool(qs)})
        if used!=mutations_per_case: raise RuntimeError(f'insufficient valid mutations for case {case_index}: {used}')
    return {'correct_primary':correct_primary,'correct_portable':correct_portable,'mutant_cases':mutants,'screening_rejected_primary':screen_primary,'screening_rejected_portable':screen_portable,'soundness_rejected_primary':full_primary,'soundness_rejected_portable':full_portable,'fresh_runtime_invalid_mutations':runtime_invalid,'false_accepts':false_accepts}
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--package-root',type=Path,required=True); parser.add_argument('--arcgen-root',type=Path,required=True); parser.add_argument('--cases',type=int,default=100); parser.add_argument('--mutations-per-case',type=int,default=8); parser.add_argument('--max-revisions',type=int,default=3); parser.add_argument('--output',type=Path,default=HERE/'V19R4_VERIFIER_REPORT.json'); args=parser.parse_args()
    gate=10; task='2bcee788'; package=load(args.package_root/'v13-campaign-10'/'redacted-task.json'); examples=[(as_grid(x['input']),as_grid(x['output'])) for x in package['train']]
    production=load(HERE/'production'/'v19r4-production.json'); program=load(HERE/'production'/'v19r4-concrete-program.json')
    primary_hash=fsha(V19R2/'runtime_v19r2.py'); portable_hash=fsha(V19R2/'portable_runtime_v19r2.py'); demo_hash=sha256_json([{'input':s,'output':t} for s,t in examples])
    fresh,attempts,rejections,rejection_examples=fresh_cases(task,gate,args.arcgen_root,args.cases)
    counterexamples=[]; revisions=[]; final_contract=final_eval=manifest=None; revision_dir=HERE/'contracts'/'revisions'
    for revision in range(args.max_revisions+1):
        contract,training,manifest=synthesize_contract(production,program,examples,primary_runtime_sha256=primary_hash,portable_runtime_sha256=portable_hash,demonstration_sha256=demo_hash,additional_cases=counterexamples,revision=revision)
        write(revision_dir/f'gate-10-contract-r{revision}.json',contract)
        evaluation=evaluate_contract(contract,production,program,fresh,manifest,args.mutations_per_case)
        write(revision_dir/f'gate-10-false-accepts-r{revision}.json',evaluation['false_accepts'])
        revisions.append({'revision':revision,'contract_sha256':contract['contract_sha256'],'predicates':contract['predicates'],'predicate_cost':contract['predicate_cost'],'training_mutation_cases':len(training),'training_runtime_invalid_mutations':contract['training_runtime_invalid_mutations'],'fresh_mutant_cases':evaluation['mutant_cases'],'fresh_runtime_invalid_mutations':evaluation['fresh_runtime_invalid_mutations'],'fresh_false_accepts':len(evaluation['false_accepts']),'exact_digest_used':contract['exact_digest_used']})
        final_contract,final_eval=contract,evaluation
        if not evaluation['false_accepts']: break
        counterexamples.extend(evaluation['false_accepts'])
    else: raise RuntimeError('verifier CEGIS revision budget exhausted')
    assert final_contract is not None and final_eval is not None and manifest is not None
    if final_eval['correct_primary']!=len(fresh) or final_eval['correct_portable']!=len(fresh): raise RuntimeError('correct output rejected')
    for key in ('screening_rejected_primary','screening_rejected_portable','soundness_rejected_primary','soundness_rejected_portable'):
        if final_eval[key]!=final_eval['mutant_cases']: raise RuntimeError(key+' missed mutations')
    if final_contract['exact_digest_used']: raise RuntimeError('learned exact digest used')
    write(HERE/'contracts'/'gate-10-contract.json',final_contract)
    report={'schema':'lexigen-v19r4-verifier-cosynthesis-report-v1','gate':gate,'task_id':task,'accepted_cases':len(fresh),'generator_attempts':attempts,'generator_rejections':rejections,'generator_rejection_examples':rejection_examples,'mutations_per_case':args.mutations_per_case,'mutation_manifest_size':len(manifest),'correct_primary':final_eval['correct_primary'],'correct_portable':final_eval['correct_portable'],'mutant_cases':final_eval['mutant_cases'],'screening_rejected_primary':final_eval['screening_rejected_primary'],'screening_rejected_portable':final_eval['screening_rejected_portable'],'soundness_rejected_primary':final_eval['soundness_rejected_primary'],'soundness_rejected_portable':final_eval['soundness_rejected_portable'],'fresh_runtime_invalid_mutations':final_eval['fresh_runtime_invalid_mutations'],'contracts_requiring_revision':len(revisions)>1,'exact_digest_used_by_learned_screen':False,'production_sha256':sha256_json(production),'concrete_program_sha256':sha256_json(program),'contract_sha256':final_contract['contract_sha256'],'primary_runtime_sha256':primary_hash,'portable_runtime_sha256':portable_hash,'demonstration_sha256':demo_hash,'revisions':revisions,'hidden_outputs_opened':False,'world_level_breakthrough':False}
    write(args.output,report); print('SUMMARY',json.dumps({k:report[k] for k in ('accepted_cases','mutant_cases','screening_rejected_primary','screening_rejected_portable','contracts_requiring_revision','exact_digest_used_by_learned_screen')},sort_keys=True))
if __name__=='__main__': main()
