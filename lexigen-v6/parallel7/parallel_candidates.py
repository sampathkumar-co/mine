from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import networkx as nx
import numpy as np
import scipy.fft
import scipy.fftpack
import scipy.linalg
import scipy.odr
from scipy.integrate import solve_ivp
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Candidate:
    name: str
    arm: str
    implementation_class: str
    operators: tuple[str, ...]
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    baseline_id: str | None
    solve: Callable[[object], object]


ARM_MAP = {"v5_full":"v6_full", "v5_no_transfer":"v6_no_transfer", "v4_compatible":"v5_compatible"}
ARM_ORDER = ("v6_full", "v6_no_transfer", "random_search", "static_template", "v5_compatible", "strong_baseline")


def _odr_model_fn(B, x):
    return B[0] * x + B[1]


_ODR_MODEL = scipy.odr.Model(_odr_model_fn)


def odr_reference(problem):
    x=np.asarray(problem["x"],dtype=float); y=np.asarray(problem["y"],dtype=float)
    sx=np.asarray(problem["sx"],dtype=float); sy=np.asarray(problem["sy"],dtype=float)
    model=scipy.odr.Model(_odr_model_fn)
    out=scipy.odr.ODR(scipy.odr.RealData(x,y=y,sx=sx,sy=sy),model,beta0=[0.0,1.0]).run()
    return {"beta":out.beta.tolist()}


def odr_cached_model(problem):
    x=np.asarray(problem["x"],dtype=float); y=np.asarray(problem["y"],dtype=float)
    sx=np.asarray(problem["sx"],dtype=float); sy=np.asarray(problem["sy"],dtype=float)
    out=scipy.odr.ODR(scipy.odr.RealData(x,y=y,sx=sx,sy=sy),_ODR_MODEL,beta0=[0.0,1.0]).run()
    return {"beta":out.beta.tolist()}


def dct_reference(problem):
    return scipy.fftpack.dctn(np.asarray(problem,dtype=float),type=1)


def dct_native(problem):
    return scipy.fft.dctn(np.asarray(problem,dtype=float),type=1,workers=1)


def _lorenz_rhs_reference(F):
    def rhs(t,x):
        N=len(x)
        ip1=np.roll(np.arange(N),-1); im1=np.roll(np.arange(N),1); im2=np.roll(np.arange(N),2)
        return (x[ip1]-x[im2])*x[im1]-x+F
    return rhs


def _lorenz_rhs_fast(F):
    def rhs(t,x):
        return (np.roll(x,-1)-np.roll(x,2))*np.roll(x,1)-x+F
    return rhs


def _ode(problem, fast: bool, method: str):
    y0=np.asarray(problem["y0"],dtype=float); t0=float(problem["t0"]); t1=float(problem["t1"]); F=float(problem["F"])
    rhs=_lorenz_rhs_fast(F) if fast else _lorenz_rhs_reference(F)
    sol=solve_ivp(rhs,[t0,t1],y0,method=method,rtol=1e-8,atol=1e-8,t_eval=None,dense_output=False)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y[:,-1].tolist()


def ode_reference(problem): return _ode(problem,False,"RK45")
def ode_fast_rk45(problem): return _ode(problem,True,"RK45")
def ode_fast_dop853(problem): return _ode(problem,True,"DOP853")


def _kalman_problem(problem):
    A=np.asarray(problem["A"],dtype=float); B=np.asarray(problem["B"],dtype=float); C=np.asarray(problem["C"],dtype=float)
    y=np.asarray(problem["y"],dtype=float); x0=np.asarray(problem["x_initial"],dtype=float)
    tau=float(problem["tau"]); M=float(problem["M"])
    N,m=y.shape; n=A.shape[1]; p=B.shape[1]
    x=cp.Variable((N+1,n)); w=cp.Variable((N,p)); v=cp.Variable((N,m))
    obj=cp.Minimize(cp.sum_squares(w)+tau*cp.sum([cp.huber(cp.norm(v[t,:]),M) for t in range(N)]))
    constraints=[x[0]==x0]
    for t in range(N):
        constraints.append(x[t+1]==A@x[t]+B@w[t]); constraints.append(y[t]==C@x[t]+v[t])
    return cp.Problem(obj,constraints),x,w,v


def _kalman_solve(problem, solver=None):
    prob,x,w,v=_kalman_problem(problem)
    if solver is None: prob.solve()
    else: prob.solve(solver=solver)
    if prob.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE} or x.value is None: raise RuntimeError(f"cvxpy status {prob.status}")
    return {"x_hat":x.value.tolist(),"w_hat":w.value.tolist(),"v_hat":v.value.tolist()}


def kalman_reference(problem): return _kalman_solve(problem,None)
def kalman_clarabel(problem): return _kalman_solve(problem,cp.CLARABEL)


def edge_reference(problem):
    adj=problem["adjacency_list"]; S=set(int(x) for x in problem["nodes_S"]); n=len(adj)
    if n==0 or not S or len(S)==n: return {"edge_expansion":0.0}
    g=nx.DiGraph(); g.add_nodes_from(range(n))
    for u,neigh in enumerate(adj):
        for v in neigh: g.add_edge(u,int(v))
    return {"edge_expansion":float(nx.edge_expansion(g,S))}


def edge_direct(problem):
    adj=problem["adjacency_list"]; S=set(int(x) for x in problem["nodes_S"]); n=len(adj)
    if n==0 or not S or len(S)==n: return {"edge_expansion":0.0}
    crossing=sum(1 for u in S for v in adj[u] if int(v) not in S)
    return {"edge_expansion":float(crossing)/len(S)}


def edge_bitset(problem):
    adj=problem["adjacency_list"]; nodes=[int(x) for x in problem["nodes_S"]]; n=len(adj)
    if n==0 or not nodes or len(set(nodes))==n: return {"edge_expansion":0.0}
    smask=0
    for v in nodes: smask |= 1<<v
    crossing=0
    for u in nodes:
        mask=0
        for v in adj[u]: mask |= 1<<int(v)
        crossing += (mask & ~smask).bit_count()
    return {"edge_expansion":float(crossing)/len(set(nodes))}


def svd_reference(problem):
    A=np.asarray(problem["matrix"] if isinstance(problem,dict) else problem,dtype=float)
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    return {"U":U,"S":s,"V":Vh.T}


def svd_gesdd(problem):
    A=np.asarray(problem["matrix"] if isinstance(problem,dict) else problem,dtype=float)
    U,s,Vh=scipy.linalg.svd(A,full_matrices=False,check_finite=False,lapack_driver="gesdd")
    return {"U":U,"S":s,"V":Vh.T}


def svd_gesvd(problem):
    A=np.asarray(problem["matrix"] if isinstance(problem,dict) else problem,dtype=float)
    U,s,Vh=scipy.linalg.svd(A,full_matrices=False,check_finite=False,lapack_driver="gesvd")
    return {"U":U,"S":s,"V":Vh.T}


def mis_reference(problem):
    n=len(problem); model=cp_model.CpModel(); nodes=[model.new_bool_var(f"x_{i}") for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if int(problem[i][j])==1: model.add(nodes[i]+nodes[j]<=1)
    model.maximize(sum(nodes)); solver=cp_model.CpSolver(); status=solver.solve(model)
    if status!=cp_model.OPTIMAL: raise RuntimeError(f"CP-SAT status {status}")
    return [i for i in range(n) if solver.value(nodes[i])==1]


def mis_cpsat_tuned(problem):
    n=len(problem); model=cp_model.CpModel(); nodes=[model.new_bool_var(f"x_{i}") for i in range(n)]
    for i,row in enumerate(problem):
        for j in range(i+1,n):
            if int(row[j])==1: model.add(nodes[i]+nodes[j]<=1)
    model.maximize(sum(nodes)); solver=cp_model.CpSolver(); solver.parameters.random_seed=0; solver.parameters.num_search_workers=2
    status=solver.solve(model)
    if status!=cp_model.OPTIMAL: raise RuntimeError(f"CP-SAT status {status}")
    return [i for i in range(n) if solver.value(nodes[i])==1]


def _mis_complement_masks(problem):
    n=len(problem); full=(1<<n)-1; masks=[]
    for i,row in enumerate(problem):
        blocked=1<<i
        for j,val in enumerate(row):
            if int(val): blocked |= 1<<j
        masks.append(full & ~blocked)
    return masks


def mis_bitset_exact(problem):
    n=len(problem)
    if n==0: return []
    adj=_mis_complement_masks(problem); best_mask=0; best_size=0
    def color_sort(P):
        order=[]; bounds=[]; U=P; color=0
        while U:
            color+=1; Q=U
            while Q:
                bit=Q & -Q; v=bit.bit_length()-1
                order.append(v); bounds.append(color)
                U &= ~bit; Q &= ~bit; Q &= ~adj[v]
        return order,bounds
    def expand(P,chosen_mask,chosen_size):
        nonlocal best_mask,best_size
        if not P:
            if chosen_size>best_size: best_size=chosen_size; best_mask=chosen_mask
            return
        order,bounds=color_sort(P)
        for idx in range(len(order)-1,-1,-1):
            if chosen_size+bounds[idx] <= best_size: return
            v=order[idx]; bit=1<<v
            expand(P & adj[v],chosen_mask|bit,chosen_size+1)
            P &= ~bit
    expand((1<<n)-1,0,0)
    return [i for i in range(n) if (best_mask>>i)&1]


REFERENCE_SOLVERS={
    "odr":odr_reference,
    "dct_type_I_scipy_fftpack":dct_reference,
    "ode_lorenz96_nonchaotic":ode_reference,
    "robust_kalman_filter":kalman_reference,
    "edge_expansion":edge_reference,
    "svd":svd_reference,
    "max_independent_set_cpsat":mis_reference,
}


def _map_impl(task: str, operators: tuple[str,...], transfer_ids: tuple[str,...]):
    ops=set(operators)
    if task=="odr":
        if transfer_ids or ops.intersection({"zero_copy_representation","contiguous_layout","native_backend_substitution","vectorized_batch_kernel","reduced_representation","dtype_specialization"}):
            return "scipy_odr_cached_model",odr_cached_model
        return "scipy_odr_reference",odr_reference
    if task=="dct_type_I_scipy_fftpack":
        if transfer_ids or ops.intersection({"native_backend_substitution","vectorized_batch_kernel","zero_copy_representation","contiguous_layout","reduced_representation","dtype_specialization"}):
            return "scipy_fft_dctn_type1",dct_native
        return "scipy_fftpack_dctn_type1",dct_reference
    if task=="ode_lorenz96_nonchaotic":
        if ops.intersection({"native_backend_substitution","vectorized_batch_kernel","zero_copy_representation","contiguous_layout"}):
            return "solve_ivp_fast_rhs_rk45",ode_fast_rk45
        if not transfer_ids and ops.intersection({"risk_aware_staging","early_certificate_exit"}):
            return "solve_ivp_fast_rhs_dop853",ode_fast_dop853
        return "solve_ivp_reference_rhs_rk45",ode_reference
    if task=="robust_kalman_filter":
        if transfer_ids or ops.intersection({"native_backend_substitution","active_set_decomposition","risk_aware_staging","early_certificate_exit"}):
            return "cvxpy_clarabel",kalman_clarabel
        return "cvxpy_default",kalman_reference
    if task=="edge_expansion":
        if transfer_ids==("TM-BFR-01",) or "bit_parallel_representation" in ops:
            return "python_bitset_boundary_count",edge_bitset
        if ops.intersection({"sparse_frontier_search","reduced_representation","zero_copy_representation","vectorized_batch_kernel","contiguous_layout"}):
            return "direct_set_boundary_count",edge_direct
        return "networkx_edge_expansion",edge_reference
    if task=="svd":
        if transfer_ids==("TM-PBEB-01",) or ops.intersection({"native_backend_substitution","dtype_specialization","contiguous_layout","zero_copy_representation","vectorized_batch_kernel","reduced_representation"}):
            return "scipy_linalg_gesdd",svd_gesdd
        if not transfer_ids and "risk_aware_staging" in ops:
            return "scipy_linalg_gesvd",svd_gesvd
        return "numpy_linalg_svd",svd_reference
    if task=="max_independent_set_cpsat":
        if transfer_ids==("TM-BFR-01",) or ops.intersection({"bit_parallel_representation","sparse_frontier_search"}):
            return "exact_complement_bitset_clique",mis_bitset_exact
        if ops.intersection({"native_backend_substitution","risk_aware_staging","early_certificate_exit"}):
            return "ortools_cpsat_tuned",mis_cpsat_tuned
        return "ortools_cpsat_reference",mis_reference
    raise KeyError(task)


def _strong_baseline(task: str) -> Candidate:
    spec={
        "odr":("strong_baseline_sb_native_numeric_01_odr","scipy_odr_cached_model",("native_numeric_backend",),"SB-NATIVE-NUMERIC-01",odr_cached_model),
        "dct_type_I_scipy_fftpack":("strong_baseline_sb_native_numeric_01_fft","scipy_fft_dctn_type1",("native_numeric_backend",),"SB-NATIVE-NUMERIC-01",dct_native),
        "ode_lorenz96_nonchaotic":("strong_baseline_sb_native_numeric_01_ode","solve_ivp_fast_rhs_rk45",("native_numeric_backend",),"SB-NATIVE-NUMERIC-01",ode_fast_rk45),
        "robust_kalman_filter":("strong_baseline_sb_convex_01_clarabel","cvxpy_clarabel",("independent_convex_backend",),"SB-CONVEX-01",kalman_clarabel),
        "edge_expansion":("strong_baseline_sb_graph_bitset_01_edge","python_bitset_boundary_count",("certified_bitset_graph_search",),"SB-GRAPH-BITSET-01",edge_bitset),
        "svd":("strong_baseline_sb_reduced_linalg_01_gesdd","scipy_linalg_gesdd",("specialized_linear_algebra_backend",),"SB-REDUCED-LINALG-01",svd_gesdd),
        "max_independent_set_cpsat":("strong_baseline_sb_graph_bitset_01_mis","exact_complement_bitset_clique",("certified_bitset_graph_search",),"SB-GRAPH-BITSET-01",mis_bitset_exact),
    }[task]
    name,impl,ops,bid,fn=spec
    return Candidate(name,"strong_baseline",impl,tuple(ops),(),None,bid,fn)


def build_candidates(task: str, task_source_text: str) -> dict[str,list[Candidate]]:
    from engine import generate_proposals
    generated=generate_proposals(task_source_text)
    arms={arm:[] for arm in ARM_ORDER}
    for engine_arm,proposals in generated["arms"].items():
        public=ARM_MAP.get(engine_arm,engine_arm)
        if public not in arms: continue
        for proposal in proposals:
            ops=tuple(str(x) for x in proposal["operators"]); tids=tuple(str(x) for x in proposal["transfer_ids"])
            impl,fn=_map_impl(task,ops,tids)
            arms[public].append(Candidate(
                f"{public}_r{proposal['rank']}_{proposal['proposal_id']}",public,impl,ops,tids,proposal.get("learned_template"),None,fn
            ))
    arms["strong_baseline"].append(_strong_baseline(task))
    for arm in ARM_ORDER[:-1]:
        if len(arms[arm])!=6: raise RuntimeError(f"{task}: expected 6 proposals for {arm}, got {len(arms[arm])}")
    if len(arms["strong_baseline"])!=1: raise RuntimeError("strong baseline count")
    return arms


def flat_candidates(task: str, source_text: str) -> list[Candidate]:
    arms=build_candidates(task,source_text); out=[]
    for arm in ARM_ORDER: out.extend(arms[arm])
    if len(out)!=31: raise RuntimeError(f"{task}: expected 31 candidates got {len(out)}")
    return out
