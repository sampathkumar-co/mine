from __future__ import annotations
import copy, json
from pathlib import Path
from mutations_v19r2 import apply_output_mutation
from portable_runtime_v19r2 import execute_portable
from portable_verifier_v19r2 import screening_holds_portable, verify_against_reference_portable
from runtime_v19r2 import execute
from verifier_grammar_v19r2 import screening_holds, verify_against_reference, verify_contract_integrity
HERE=Path(__file__).resolve().parent
def load(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def check(value,message):
    if not value: raise AssertionError(message)
def fixture():
    source=((0,0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,2,5,5,0),(0,0,0,2,5,5,0),(0,0,0,0,0,0,0))
    return source
def objects():
    return load('productions/gate-10-production.json'),load('productions/gate-10-concrete.json'),load('contracts/gate-10-contract.json')
def test_contract_integrity_and_no_exact_screen():
    _,_,contract=objects(); check(verify_contract_integrity(contract),'primary integrity'); check(not contract['exact_digest_used'],'exact digest flag'); check(all(x['name']!='exact_digest' for x in contract['predicates']),'exact predicate')
def test_correct_output_accepted_both():
    production,program,contract=objects(); source=fixture(); reference=execute(program,source); portable=execute_portable(program,source); check(reference==portable,'runtime mismatch')
    check(verify_against_reference(contract,production,program,source,reference,reference),'primary rejected correct')
    check(verify_against_reference_portable(contract,production,program,source,reference,portable),'portable rejected correct')
def test_mutation_rejected_by_screens_and_anchors():
    production,program,contract=objects(); source=fixture(); reference=execute(program,source); candidate=apply_output_mutation(reference,'flip_first_cell')
    check(not screening_holds(contract,source,candidate,reference),'primary screen accepted')
    check(not screening_holds_portable(contract,source,candidate,reference),'portable screen accepted')
    check(not verify_against_reference(contract,production,program,source,candidate,reference),'primary anchor accepted')
    check(not verify_against_reference_portable(contract,production,program,source,candidate,reference),'portable anchor accepted')
def test_contract_tamper_rejected():
    production,program,contract=objects(); source=fixture(); reference=execute(program,source); changed=copy.deepcopy(contract); changed['predicates']=[]
    check(not verify_against_reference(changed,production,program,source,reference,reference),'tampered primary accepted')
    check(not verify_against_reference_portable(changed,production,program,source,reference,reference),'tampered portable accepted')
def test_wrong_production_and_program_rejected():
    production,program,contract=objects(); source=fixture(); reference=execute(program,source); wrong_production=copy.deepcopy(production); wrong_production['name']='wrong'; wrong_program=copy.deepcopy(program); wrong_program['bindings'][0]['expr']['items']['predicate']['items'][1]['right']=7
    check(not verify_against_reference(contract,wrong_production,program,source,reference,reference),'wrong production accepted')
    check(not verify_against_reference(contract,production,wrong_program,source,reference,reference),'wrong program accepted')
def main():
    tests=[v for n,v in sorted(globals().items()) if n.startswith('test_') and callable(v)]
    for test in tests: test(); print('PASS',test.__name__)
    print(f'SUMMARY {len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
