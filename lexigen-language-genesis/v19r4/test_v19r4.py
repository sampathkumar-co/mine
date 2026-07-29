from __future__ import annotations
import copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; V19R2=HERE.parent/'v19r2'
if str(V19R2) not in sys.path: sys.path.insert(0,str(V19R2))
from mutations_v19r2 import apply_output_mutation
from portable_runtime_v19r2 import execute_portable
from portable_verifier_v19r2 import screening_holds_portable, verify_against_reference_portable
from runtime_v19r2 import canonical, execute, sha256_json
from run_cegis_v19r4 import expand_production
from verifier_grammar_v19r2 import screening_holds, verify_against_reference, verify_contract_integrity

def load(path): return json.loads((HERE/path).read_text(encoding='utf-8'))
def check(v,m):
    if not v: raise AssertionError(m)
def objects(): return load('production/v19r4-production.json'),load('production/v19r4-arguments.json'),load('production/v19r4-concrete-program.json'),load('contracts/gate-10-contract.json')
def fixture_top():
    source=((0,0,0,0,0,0),(0,0,2,2,0,0),(0,0,5,5,0,0),(0,0,5,5,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0))
    target=((3,3,5,5,3,3),(3,3,5,5,3,3),(3,3,5,5,3,3),(3,3,5,5,3,3),(3,3,3,3,3,3),(3,3,3,3,3,3))
    return source,target
def test_cegis_unique_and_preserved():
    report=load('V19R4_CEGIS_REPORT.json'); check(report['initial_survivors']==5,'initial'); check(report['final_survivors']==1,'final'); check(report['accepted_refinement_cases']==7,'refinement count'); check(report['selected_descriptor_removal_survivors']==0,'ablation'); check(report['selected_descriptor']['actions']==['reflect_left','reflect_right','reflect_top','reflect_bottom'],'actions')
def test_production_expansion_and_dual_runtime():
    production,args,program,_=objects(); expanded=expand_production(production,args); check(sha256_json(expanded)==sha256_json(program),'expansion'); source,target=fixture_top(); check(execute(program,source)==target,'primary top'); check(execute_portable(program,source)==target,'portable top')
def test_task_id_absent():
    production,_,program,_=objects(); check('2bcee788' not in canonical(production),'production task id'); check('2bcee788' not in canonical(program),'program task id')
def test_contract_integrity_and_binding():
    production,_,program,contract=objects(); check(verify_contract_integrity(contract),'contract integrity'); check(contract['production_sha256']==sha256_json(production),'production binding'); check(contract['concrete_program_sha256']==sha256_json(program),'program binding'); check(not contract['exact_digest_used'],'exact digest')
def test_verifier_rejects_mutation_both():
    production,_,program,contract=objects(); source,_=fixture_top(); reference=execute(program,source); candidate=apply_output_mutation(reference,'flip_first_cell'); check(not screening_holds(contract,source,candidate,reference),'primary screen'); check(not screening_holds_portable(contract,source,candidate,reference),'portable screen'); check(not verify_against_reference(contract,production,program,source,candidate,reference),'primary full'); check(not verify_against_reference_portable(contract,production,program,source,candidate,reference),'portable full')
def test_contract_tamper_rejected():
    production,_,program,contract=objects(); source,_=fixture_top(); reference=execute(program,source); bad=copy.deepcopy(contract); bad['predicates']=[]; check(not verify_against_reference(bad,production,program,source,reference,reference),'tampered primary'); check(not verify_against_reference_portable(bad,production,program,source,reference,reference),'tampered portable')
def main():
    tests=[v for n,v in sorted(globals().items()) if n.startswith('test_') and callable(v)]
    for test in tests: test(); print('PASS',test.__name__)
    print(f'SUMMARY {len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
