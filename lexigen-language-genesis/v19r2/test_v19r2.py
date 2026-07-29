from __future__ import annotations
import copy
from invent_v19r2 import build_program, expand_production, invent
from portable_runtime_v19r2 import PortableV19R2Error, execute_portable
from runtime_v19r2 import FORBIDDEN_NAMED_OPS, RuntimeV19R2Error, canonical, execute, sha256_json, walk_ops

def check(value:bool,message:str):
    if not value: raise AssertionError(message)
def make(side:str):
    h,w=7,9; grid=[[0]*w for _ in range(h)]; payload={(2,4),(2,5),(3,4),(3,5)}
    if side=='left': marker={(2,3),(3,3)}; reflected={(2,2),(2,3),(3,2),(3,3)}
    elif side=='right': marker={(2,6),(3,6)}; reflected={(2,6),(2,7),(3,6),(3,7)}
    elif side=='top': marker={(1,4),(1,5)}; reflected={(0,4),(0,5),(1,4),(1,5)}
    elif side=='bottom': marker={(4,4),(4,5)}; reflected={(4,4),(4,5),(5,4),(5,5)}
    else: raise ValueError(side)
    for r,c in payload: grid[r][c]=5
    for r,c in marker: grid[r][c]=2
    target=[[3]*w for _ in range(h)]
    for r,c in payload|reflected: target[r][c]=5
    return tuple(map(tuple,grid)),tuple(map(tuple,target))
def test_four_sides_primary_portable():
    program=build_program(2,3)
    for side in ('left','right','top','bottom'):
        source,target=make(side)
        check(execute(program,source)==target,'primary '+side)
        check(execute_portable(program,source)==target,'portable '+side)
def test_invention_and_abstraction():
    examples=[make(side) for side in ('left','right','top','bottom')]
    production,args,program,report=invent(examples)
    check(args=={'marker_colour':2,'output_background':3},'wrong arguments')
    check(report['exact_survivors']==1,'not unique')
    expanded=expand_production(production,args)
    check(sha256_json(expanded)==sha256_json(program),'expansion differs')
    for source,target in examples: check(execute(expanded,source)==target,'expanded failure')
def test_determinism():
    examples=[make(side) for side in ('left','right','top','bottom')]
    a=invent(examples); b=invent(examples)
    check(sha256_json(a[0])==sha256_json(b[0]),'production nondeterministic')
    check(a[1]==b[1],'arguments nondeterministic')
def test_forbidden_ops_absent_and_rejected():
    production,args,program,_=invent([make(side) for side in ('left','right','top','bottom')])
    check(not ((set(walk_ops(production))|set(walk_ops(program)))&FORBIDDEN_NAMED_OPS),'forbidden present')
    bad={'schema':'lexigen-v19r2-executable-production-v1','body':{'op':'symmetry_completion'}}
    for runner,error in ((execute,RuntimeV19R2Error),(execute_portable,PortableV19R2Error)):
        try: runner(bad,((0,),))
        except error: pass
        else: raise AssertionError('forbidden accepted')
def test_ambiguous_payload_rejected():
    source,_=make('left'); values=[list(row) for row in source]; values[0][0]=7; source=tuple(map(tuple,values)); program=build_program(2,3)
    for runner,error in ((execute,RuntimeV19R2Error),(execute_portable,PortableV19R2Error)):
        try: runner(program,source)
        except error: pass
        else: raise AssertionError('ambiguous palette accepted')
def test_tamper_hash_and_canonical():
    production,_,_,_=invent([make(side) for side in ('left','right','top','bottom')]); changed=copy.deepcopy(production); changed['body']['body']['grid']['fill']=9
    check(sha256_json(changed)!=sha256_json(production),'tamper hash unchanged')
    check(canonical({'b':1,'a':2})==canonical({'a':2,'b':1}),'canonical order')
def test_task_id_absent():
    production,_,program,_=invent([make(side) for side in ('left','right','top','bottom')])
    check('2bcee788' not in canonical(production),'task id production')
    check('2bcee788' not in canonical(program),'task id program')
def main():
    tests=[v for n,v in sorted(globals().items()) if n.startswith('test_') and callable(v)]
    for test in tests: test(); print('PASS',test.__name__)
    print(f'SUMMARY {len(tests)}/{len(tests)} tests passed')
if __name__=='__main__': main()
