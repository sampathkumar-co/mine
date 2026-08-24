from __future__ import annotations

import parallel_candidates as pc
import parallel_train as pt


def edge_direct_exact(problem):
    adj=problem["adjacency_list"]
    S=set(int(x) for x in problem["nodes_S"])
    n=len(adj)
    if n==0 or not S or len(S)==n:
        return {"edge_expansion":0.0}
    crossing=0
    for u,neighbors in enumerate(adj):
        u_in=u in S
        for v in neighbors:
            if u_in != (int(v) in S):
                crossing += 1
    return {"edge_expansion":float(crossing)/len(S)}


def edge_bitset_exact(problem):
    adj=problem["adjacency_list"]
    S=set(int(x) for x in problem["nodes_S"])
    n=len(adj)
    if n==0 or not S or len(S)==n:
        return {"edge_expansion":0.0}
    smask=0
    for v in S:
        smask |= 1 << v
    allmask=(1 << n)-1
    outside=allmask & ~smask
    crossing=0
    for u,neighbors in enumerate(adj):
        mask=0
        for v in neighbors:
            mask |= 1 << int(v)
        if u in S:
            crossing += (mask & outside).bit_count()
        else:
            crossing += (mask & smask).bit_count()
    return {"edge_expansion":float(crossing)/len(S)}


def repaired_map_impl(task: str, operators: tuple[str,...], transfer_ids: tuple[str,...]):
    ops=set(operators)
    if task=="ode_lorenz96_nonchaotic":
        # DOP853 was rejected by the frozen pre-training equivalence certificate.
        # Every optimized generic proposal now remains on the source-equivalent RK45
        # method and can only optimize the RHS representation itself.
        if ops.intersection({"native_backend_substitution","vectorized_batch_kernel","zero_copy_representation","contiguous_layout"}):
            return "solve_ivp_fast_rhs_rk45",pc.ode_fast_rk45
        if not transfer_ids and ops.intersection({"risk_aware_staging","early_certificate_exit"}):
            return "solve_ivp_fast_rhs_rk45",pc.ode_fast_rk45
        return "solve_ivp_reference_rhs_rk45",pc.ode_reference
    if task=="edge_expansion":
        if transfer_ids==("TM-BFR-01",) or "bit_parallel_representation" in ops:
            return "python_bitset_boundary_count_bidir",edge_bitset_exact
        if ops.intersection({"sparse_frontier_search","reduced_representation","zero_copy_representation","vectorized_batch_kernel","contiguous_layout"}):
            return "direct_boundary_count_bidir",edge_direct_exact
        return "networkx_edge_expansion",pc.edge_reference
    return _ORIGINAL_MAP(task,operators,transfer_ids)


_ORIGINAL_MAP=pc._map_impl


def activate() -> None:
    pc.edge_direct=edge_direct_exact
    pc.edge_bitset=edge_bitset_exact
    pc._map_impl=repaired_map_impl
    # parallel_train.verify holds a module-local reference to the direct exact
    # edge certificate, so update only that certificate binding for Task 6.
    pt.edge_direct=edge_direct_exact
    pt.flat_candidates=pc.flat_candidates
