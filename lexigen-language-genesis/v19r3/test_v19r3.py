from __future__ import annotations
import json
from pathlib import Path
from enumerate_v19r3 import TOTAL_CANDIDATES, ACTIONS, BASE_MODES, SET_MODES
HERE=Path(__file__).resolve().parent
def check(v,m):
    if not v: raise AssertionError(m)
def test_candidate_denominator():
    check(TOTAL_CANDIDATES==250000,'candidate denominator')
def test_fixed_action_and_structure_catalogues():
    check(ACTIONS==('identity','reflect_left','reflect_right','reflect_top','reflect_bottom'),'actions')
    check(SET_MODES==('mapped_only','union_source_and_mapped'),'sets')
    check(BASE_MODES==('input','new_canvas'),'bases')
def test_preserved_attempt_verdict():
    report=json.loads((HERE/'V19R3_ATTEMPT_REPORT.json').read_text()); check(report['fixed_gates']==3,'gates'); check(report['unique_productions']==0,'unique'); check(report['ambiguous_program_sets']==1,'ambiguity'); check(report['no_program_failures']==2,'failures'); check(not report['fresh_scoring_performed'],'fresh scoring')
def test_ambiguity_is_only_top_action():
    report=json.loads((HERE/'V19R3_ATTEMPT_REPORT.json').read_text()); gate=next(x for x in report['outcomes'] if x['gate']==10); survivors=gate['search']['exact_survivor_descriptors']; check(len(survivors)==5,'survivor count')
    fixed={(tuple(x['actions'][:2]),x['actions'][3],x['marker_colour'],x['output_background'],x['set_mode'],x['base_mode']) for x in survivors}; check(len(fixed)==1,'non-top fields differ'); check({x['actions'][2] for x in survivors}==set(ACTIONS),'top actions incomplete')
def main():
    tests=[v for n,v in sorted(globals().items()) if n.startswith('test_') and callable(v)]
    for test in tests: test(); print('PASS',test.__name__)
    print(f'SUMMARY {len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
