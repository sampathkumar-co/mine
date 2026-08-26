from __future__ import annotations

import json, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'lexigen-v5'))
from engine import generate_proposals

HERE=Path(__file__).resolve().parent
SOURCE_SHA256='24d194fbf8f604d318b9f330e61ad084ff4ea498de2c0a299835ad7ecce55d9a'
ARM_NAME_MAP={'v5_full':'v6_full','v5_no_transfer':'v6_no_transfer','random_search':'random_search','static_template':'static_template','v4_compatible':'v5_compatible'}
ARM_ORDER=('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')

@dataclass(frozen=True)
class Candidate:
    arm:str
    name:str
    rank:int
    proposal_id:str|None
    operators:tuple[str,...]
    transfer_ids:tuple[str,...]
    learned_template:str|None
    implementation_class:str
    semantic_implementation_key:str
    baseline_id:str|None
    solve:Callable[[dict],dict[str,float]]


def _parts(problem:dict)->tuple[list[list[int]],set[int],int]:
    if not isinstance(problem,dict) or 'adjacency_list' not in problem or 'nodes_S' not in problem:
        raise ValueError('invalid problem structure')
    adj=problem['adjacency_list']; raw_s=problem['nodes_S']
    if not isinstance(adj,list) or not isinstance(raw_s,list): raise ValueError('invalid problem container types')
    out=[]
    for row in adj:
        if not isinstance(row,list): raise ValueError('adjacency row is not a list')
        rr=[]
        for v in row:
            if isinstance(v,bool) or not isinstance(v,int): raise ValueError('neighbor must be int')
            rr.append(int(v))
        out.append(rr)
    s=set()
    for u in raw_s:
        if isinstance(u,bool) or not isinstance(u,int): raise ValueError('S node must be int')
        s.add(int(u))
    return out,s,len(out)


def networkx_source_reference(problem:dict)->dict[str,float]:
    adj,s,n=_parts(problem)
    if n==0 or not s or len(s)==n:return {'edge_expansion':0.0}
    g=nx.DiGraph();g.add_nodes_from(range(n))
    for u,row in enumerate(adj):
        for v in row:g.add_edge(u,v)
    return {'edge_expansion':float(nx.edge_expansion(g,s))}


def _duplicate_safe_boundary(adj:list[list[int]],s:set[int],n:int)->float:
    if n==0 or not s or len(s)==n:return 0.0
    if any(u<0 or u>=n for u in s):return float(networkx_source_reference({'adjacency_list':adj,'nodes_S':sorted(s)})['edge_expansion'])
    cut=0
    for u in s:
        for v in set(adj[u]):
            if v not in s:cut+=1
    return float(cut)/float(len(s))


def guarded_direct_sparse_frontier_boundary_exact(problem:dict)->dict[str,float]:
    adj,s,n=_parts(problem)
    if n==0 or not s or len(s)==n:return {'edge_expansion':0.0}
    if any(u<0 or u>=n for u in s):return networkx_source_reference(problem)
    cut=0
    for u in s:
        row=adj[u]
        prev=None
        for j,v in enumerate(row):
            if j and v<=prev:
                return {'edge_expansion':_duplicate_safe_boundary(adj,s,n)}
            prev=v
            if v not in s:cut+=1
    return {'edge_expansion':float(cut)/float(len(s))}


def python_int_bitset_boundary_exact(problem:dict)->dict[str,float]:
    adj,s,n=_parts(problem)
    if n==0 or not s or len(s)==n:return {'edge_expansion':0.0}
    if any(u<0 or u>=n for u in s):return {'edge_expansion':_duplicate_safe_boundary(adj,s,n)}
    smask=0
    for u in s:smask|=1<<u
    cut=0
    for u in s:
        mask=0
        for v in adj[u]:
            if v<0 or v>=n:return {'edge_expansion':_duplicate_safe_boundary(adj,s,n)}
            mask|=1<<v
        cut+=(mask & ~smask).bit_count()
    return {'edge_expansion':float(cut)/float(len(s))}


def independent_definition_certificate(problem:dict)->dict[str,float]:
    adj,s,n=_parts(problem)
    if n==0 or not s or len(s)==n:return {'edge_expansion':0.0}
    if any(u<0 or u>=n for u in s):
        raise ValueError('certificate expects source-generator-valid S node ids')
    edges={(u,v) for u,row in enumerate(adj) for v in row}
    cut=sum(1 for u,v in edges if u in s and v not in s)
    return {'edge_expansion':float(cut)/float(len(s))}


def verify_value(problem:dict,solution:dict,reference:dict|None=None)->tuple[bool,str|None,dict[str,float]]:
    try:
        if not isinstance(solution,dict) or 'edge_expansion' not in solution:return False,'format',{}
        got=float(solution['edge_expansion'])
        if not math.isfinite(got) or got<0:return False,'nonfinite_or_negative',{}
        cert=float(independent_definition_certificate(problem)['edge_expansion'])
        ref=float((reference or networkx_source_reference(problem))['edge_expansion'])
        cert_ok=math.isclose(got,cert,rel_tol=1e-12,abs_tol=1e-12)
        official_ok=math.isclose(got,ref,rel_tol=1e-5,abs_tol=1e-8)
        return bool(cert_ok and official_ok),(None if cert_ok and official_ok else 'certificate_or_reference_mismatch'),{'value':got,'certificate':cert,'reference':ref,'abs_error_certificate':abs(got-cert),'abs_error_reference':abs(got-ref)}
    except Exception as exc:
        return False,f'verify_exception:{type(exc).__name__}:{exc}',{}


def _implementation(transfer_ids:tuple[str,...],operators:tuple[str,...]):
    if 'TM-BFR-01' in transfer_ids:
        return guarded_direct_sparse_frontier_boundary_exact,'guarded_direct_sparse_frontier_boundary_exact','direct_sparse_frontier_boundary'
    if 'TM-CAC-01' in transfer_ids or 'TM-RRR-01' in transfer_ids:
        return networkx_source_reference,'networkx_source_reference','networkx_source_formulation'
    if 'sparse_frontier_search' in operators:
        return guarded_direct_sparse_frontier_boundary_exact,'guarded_direct_sparse_frontier_boundary_exact','direct_sparse_frontier_boundary'
    if 'bit_parallel_representation' in operators:
        return python_int_bitset_boundary_exact,'python_int_bitset_boundary_exact','python_int_bitset_boundary'
    return networkx_source_reference,'networkx_source_reference','networkx_source_formulation'


def build_candidates(source_text:str)->dict[str,list[Candidate]]:
    generated=generate_proposals(source_text)
    sealed=json.loads((HERE/'SOURCE_SCREEN_R2_RESULT.json').read_text())
    expected=sealed['proposal_ids_by_arm']
    arms={arm:[] for arm in ARM_ORDER}
    for old_arm,rows in generated['arms'].items():
        arm=ARM_NAME_MAP[old_arm]
        ids=[str(r['proposal_id']) for r in rows]
        if ids!=expected[arm]:raise RuntimeError(f'proposal identity/order mismatch for {arm}: {ids} != {expected[arm]}')
        for r in rows:
            rank=int(r['rank']);pid=str(r['proposal_id']);ops=tuple(str(x) for x in r['operators']);tids=tuple(str(x) for x in r['transfer_ids']);template=r.get('learned_template')
            fn,impl,sem=_implementation(tids,ops)
            arms[arm].append(Candidate(arm=arm,name=f'{arm}_r{rank}_{pid}',rank=rank,proposal_id=pid,operators=ops,transfer_ids=tids,learned_template=template,implementation_class=impl,semantic_implementation_key=sem,baseline_id=None,solve=fn))
    arms['strong_baseline']=[Candidate(arm='strong_baseline',name='strong_baseline_sb_graph_bitset_01_word_parallel_boundary',rank=1,proposal_id=None,operators=('word_parallel_adjacency_frontier','python_int_bitset'),transfer_ids=(),learned_template=None,implementation_class='python_int_bitset_boundary_exact',semantic_implementation_key='python_int_bitset_boundary',baseline_id='SB-GRAPH-BITSET-01',solve=python_int_bitset_boundary_exact)]
    counts={k:len(v) for k,v in arms.items()}
    if counts!={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}:raise RuntimeError(f'candidate count mismatch {counts}')
    return arms
