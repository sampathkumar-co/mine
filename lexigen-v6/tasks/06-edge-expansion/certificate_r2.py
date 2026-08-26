from __future__ import annotations

import math


def _parts(problem:dict)->tuple[list[list[int]],set[int],int]:
    if not isinstance(problem,dict) or 'adjacency_list' not in problem or 'nodes_S' not in problem:raise ValueError('invalid problem structure')
    adj=problem['adjacency_list'];raw_s=problem['nodes_S']
    if not isinstance(adj,list) or not isinstance(raw_s,list):raise ValueError('invalid problem container types')
    out=[]
    for row in adj:
        if not isinstance(row,list):raise ValueError('adjacency row is not a list')
        rr=[]
        for v in row:
            if isinstance(v,bool) or not isinstance(v,int):raise ValueError('neighbor must be int')
            rr.append(int(v))
        out.append(rr)
    s=set()
    for u in raw_s:
        if isinstance(u,bool) or not isinstance(u,int):raise ValueError('S node must be int')
        s.add(int(u))
    return out,s,len(out)


def networkx35_executable_semantics_certificate(problem:dict)->dict[str,float]:
    adj,s,n=_parts(problem)
    # Preserve explicit edge cases in the AlgoTune task before it constructs nx.DiGraph.
    if n==0 or not s or len(s)==n:return {'edge_expansion':0.0}
    # Synthetic cases are source-generator-valid: S is a subset of 0..n-1.
    if any(u<0 or u>=n for u in s):raise ValueError('certificate R2 expects source-generator-valid S nodes')
    nodes=set(range(n))
    edges=set()
    for u,row in enumerate(adj):
        for v in row:
            nodes.add(v)  # nx.DiGraph.add_edge creates a target node if needed.
            edges.add((u,v))  # DiGraph collapses repeated adjacency entries.
    t=nodes-s
    denominator=min(len(s),len(t))
    if denominator<=0:raise ValueError('unexpected zero denominator outside explicit source edge cases')
    cut=sum(1 for u,v in edges if (u in s and v in t) or (u in t and v in s))
    return {'edge_expansion':float(cut)/float(denominator)}


def verify_r2(problem:dict,solution:dict,reference:dict)->tuple[bool,str|None,dict[str,float]]:
    try:
        if not isinstance(solution,dict) or 'edge_expansion' not in solution:return False,'format',{}
        got=float(solution['edge_expansion'])
        if not math.isfinite(got) or got<0:return False,'nonfinite_or_negative',{}
        cert=float(networkx35_executable_semantics_certificate(problem)['edge_expansion'])
        ref=float(reference['edge_expansion'])
        certificate_matches_reference=math.isclose(cert,ref,rel_tol=1e-12,abs_tol=1e-12)
        if not certificate_matches_reference:return False,'certificate_reference_disagreement',{'value':got,'certificate':cert,'reference':ref,'abs_error_certificate_reference':abs(cert-ref)}
        official_ok=math.isclose(got,ref,rel_tol=1e-5,abs_tol=1e-8)
        exact_cert_ok=math.isclose(got,cert,rel_tol=1e-12,abs_tol=1e-12)
        return bool(official_ok and exact_cert_ok),(None if official_ok and exact_cert_ok else 'candidate_semantic_mismatch'),{'value':got,'certificate':cert,'reference':ref,'abs_error_certificate':abs(got-cert),'abs_error_reference':abs(got-ref)}
    except Exception as exc:
        return False,f'verify_exception:{type(exc).__name__}:{exc}',{}
