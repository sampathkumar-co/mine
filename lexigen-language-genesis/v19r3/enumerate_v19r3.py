from __future__ import annotations
import itertools, sys
from collections import Counter
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent; V19R2=HERE.parent/'v19r2'
if str(V19R2) not in sys.path: sys.path.insert(0,str(V19R2))
from runtime_v19r2 import SCHEMA, Grid, RuntimeV19R2Error, canonical, execute, node_count, sha256_json
PRODUCTION_SCHEMA='lexigen-v19r3-invented-production-v1'
ACTIONS=('identity','reflect_left','reflect_right','reflect_top','reflect_bottom')
SET_MODES=('mapped_only','union_source_and_mapped')
BASE_MODES=('input','new_canvas')
TOTAL_CANDIDATES=10*10*(len(ACTIONS)**4)*len(SET_MODES)*len(BASE_MODES)
def v(name): return {'op':'var','name':name}
def b(op,left,right): return {'op':op,'left':left,'right':right}
def field(name): return {'op':'bbox_field','name':name,'bbox':v('payload_box')}
def marker_field(name): return {'op':'bbox_field','name':name,'bbox':v('marker_box')}
def pair(row,col): return {'op':'pair','row':row,'col':col}
def param(name): return {'op':'param','name':name}
def action_expr(name:str):
    point=v('point'); row={'op':'first','value':point}; col={'op':'second','value':point}
    if name=='identity': return pair(row,col)
    if name=='reflect_left': return pair(row,b('sub',b('sub',b('mul',2,field('min_col')),1),col))
    if name=='reflect_right': return pair(row,b('sub',b('add',b('mul',2,field('max_col')),1),col))
    if name=='reflect_top': return pair(b('sub',b('sub',b('mul',2,field('min_row')),1),row),col)
    if name=='reflect_bottom': return pair(b('sub',b('add',b('mul',2,field('max_row')),1),row),col)
    raise ValueError(name)
def build_program(marker_colour:Any,output_background:Any,actions:tuple[str,str,str,str],set_mode:str,base_mode:str):
    colour=v('colour'); payload=v('payload_coords')
    payload_colour={'op':'unique','items':{'op':'filter','items':{'op':'palette'},'var':'colour','predicate':{'op':'and','items':[b('ne',colour,{'op':'mode'}),b('ne',colour,marker_colour)]}}}
    left=b('lt',marker_field('max_col'),field('min_col')); right=b('gt',marker_field('min_col'),field('max_col')); top=b('lt',marker_field('max_row'),field('min_row'))
    mapped_body={'op':'if','condition':left,'then':action_expr(actions[0]),'else':{'op':'if','condition':right,'then':action_expr(actions[1]),'else':{'op':'if','condition':top,'then':action_expr(actions[2]),'else':action_expr(actions[3])}}}
    mapped={'op':'map','items':payload,'var':'point','body':mapped_body}
    coords=mapped if set_mode=='mapped_only' else {'op':'set_union','items':[payload,mapped]}
    base={'op':'input'} if base_mode=='input' else {'op':'canvas','rows':{'op':'height'},'cols':{'op':'width'},'fill':output_background}
    return {'schema':SCHEMA,'bindings':[{'name':'payload_colour','expr':payload_colour},{'name':'payload_coords','expr':{'op':'coords_colour','colour':v('payload_colour')}},{'name':'marker_coords','expr':{'op':'coords_colour','colour':marker_colour}},{'name':'payload_box','expr':{'op':'bbox','coords':payload}},{'name':'marker_box','expr':{'op':'bbox','coords':v('marker_coords')}}],'body':{'op':'paint','grid':base,'coords':coords,'colour':v('payload_colour')}}
def _mode(grid:Grid)->int:
    counts=Counter(x for row in grid for x in row); return min(counts,key=lambda x:(-counts[x],x))
def _bbox(coords:set[tuple[int,int]]):
    rs=[r for r,_ in coords]; cs=[c for _,c in coords]
    return min(rs),max(rs),min(cs),max(cs)
def _prepare(source:Grid,target:Grid,marker_colour:int):
    mode=_mode(source); palette=sorted({x for row in source for x in row}); remaining=[x for x in palette if x!=mode and x!=marker_colour]
    marker={(r,c) for r,row in enumerate(source) for c,x in enumerate(row) if x==marker_colour}
    if len(remaining)!=1 or not marker: return None
    payload_colour=remaining[0]; payload={(r,c) for r,row in enumerate(source) for c,x in enumerate(row) if x==payload_colour}
    if not payload: return None
    p0,p1,q0,q1=_bbox(payload); m0,m1,n0,n1=_bbox(marker)
    side=0 if n1<q0 else 1 if n0>q1 else 2 if m1<p0 else 3
    def transform(name):
        if name=='identity': return set(payload)
        if name=='reflect_left': return {(r,2*q0-1-c) for r,c in payload}
        if name=='reflect_right': return {(r,2*q1+1-c) for r,c in payload}
        if name=='reflect_top': return {(2*p0-1-r,c) for r,c in payload}
        if name=='reflect_bottom': return {(2*p1+1-r,c) for r,c in payload}
        raise ValueError(name)
    maps=tuple(transform(name) for name in ACTIONS)
    return {'source':source,'target':target,'payload_colour':payload_colour,'payload':payload,'side':side,'maps':maps}
def _render(prepared,action_index:int,set_mode:str,base_mode:str,output_background:int):
    source=prepared['source']; coords=set(prepared['maps'][action_index])
    if set_mode=='union_source_and_mapped': coords.update(prepared['payload'])
    if base_mode=='input': out=[list(row) for row in source]
    else: out=[[output_background for _ in row] for row in source]
    colour=prepared['payload_colour']; h=len(out); w=len(out[0])
    for r,c in coords:
        if 0<=r<h and 0<=c<w: out[r][c]=colour
    return tuple(tuple(row) for row in out)
def descriptors():
    for marker in range(10):
        for background in range(10):
            for actions in itertools.product(ACTIONS,repeat=4):
                for set_mode in SET_MODES:
                    for base_mode in BASE_MODES:
                        yield marker,background,actions,set_mode,base_mode
def _substitute(value:Any,args:dict[str,Any]):
    if isinstance(value,dict):
        if value.get('op')=='param': return args[str(value['name'])]
        return {k:_substitute(v,args) for k,v in value.items()}
    if isinstance(value,list): return [_substitute(v,args) for v in value]
    return value
def expand_production(production,args):
    if production.get('schema')!=PRODUCTION_SCHEMA: raise RuntimeV19R2Error('production schema')
    expected=sorted(x['name'] for x in production['parameters'])
    if expected!=sorted(args): raise RuntimeV19R2Error('argument mismatch')
    return _substitute(production['body'],args)
def enumerate_compositions(examples:list[tuple[Grid,Grid]]):
    prepared_by_marker={m:[_prepare(s,t,m) for s,t in examples] for m in range(10)}
    survivors=[]; evaluated=invalid=0
    for marker,background,actions,set_mode,base_mode in descriptors():
        evaluated+=1; prepared=prepared_by_marker[marker]
        if any(item is None for item in prepared): invalid+=1; continue
        action_indexes=tuple(ACTIONS.index(name) for name in actions)
        exact=True
        for item in prepared:
            assert item is not None
            if _render(item,action_indexes[item['side']],set_mode,base_mode,background)!=item['target']:
                exact=False; break
        if exact:
            program=build_program(marker,background,actions,set_mode,base_mode)
            if not all(execute(program,s)==t for s,t in examples): raise RuntimeError('semantic prefilter disagreed with executable runtime')
            survivors.append((node_count(program),canonical(program),marker,background,actions,set_mode,base_mode,program))
    if evaluated!=TOTAL_CANDIDATES: raise RuntimeError(f'candidate denominator changed: {evaluated}')
    if not survivors: raise RuntimeError('full composition meta-grammar found no exact production')
    selected=min(survivors); size,_,marker,background,actions,set_mode,base_mode,program=selected
    production={'schema':PRODUCTION_SCHEMA,'parameters':[{'name':'marker_colour','type':'colour'},{'name':'output_background','type':'colour'}],'body':build_program(param('marker_colour'),param('output_background'),actions,set_mode,base_mode),'origin':{'method':'complete_descriptor_enumeration','selected_actions':list(actions),'selected_set_mode':set_mode,'selected_base_mode':base_mode,'concrete_program_sha256':sha256_json(program)}}
    production['name']='generated_'+sha256_json(production)[:16]; args={'marker_colour':marker,'output_background':background}
    descriptor={'marker_colour':marker,'output_background':background,'actions':list(actions),'set_mode':set_mode,'base_mode':base_mode}
    return production,args,program,descriptor,{'candidate_compositions_evaluated':evaluated,'runtime_invalid_candidates':invalid,'exact_survivors':len(survivors),'selected_node_count':size,'selected_descriptor':descriptor,'concrete_program_sha256':sha256_json(program),'production_sha256':sha256_json(production),'selected_descriptor_removal_survivors':len(survivors)-1,'exact_survivor_descriptors':[{'marker_colour':item[2],'output_background':item[3],'actions':list(item[4]),'set_mode':item[5],'base_mode':item[6],'program_sha256':sha256_json(item[7])} for item in survivors],'complete_composition_enumerated':True}
