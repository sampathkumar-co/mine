from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class Candidate:
    name: str
    arm: str
    implementation_class: str
    operators: tuple[str, ...]
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    baseline_id: str | None
    solve: Callable[[dict], dict]


CAMPAIGN_ELIGIBLE_TRANSFER_IDS = frozenset({"TM-BFR-01", "TM-CAC-01", "TM-RRR-01"})


def _problem(problem: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float]:
    if not isinstance(problem, dict) or not {"A","B","C","y","x_initial","tau","M"} <= set(problem):
        raise ValueError("invalid robust Kalman problem")
    A=np.asarray(problem["A"],dtype=np.float64)
    B=np.asarray(problem["B"],dtype=np.float64)
    C=np.asarray(problem["C"],dtype=np.float64)
    y=np.asarray(problem["y"],dtype=np.float64)
    x0=np.asarray(problem["x_initial"],dtype=np.float64)
    tau=float(problem["tau"]);M=float(problem["M"])
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or B.ndim!=2 or C.ndim!=2 or y.ndim!=2 or x0.ndim!=1:
        raise ValueError("invalid robust Kalman shapes")
    n=A.shape[0];N,m=y.shape
    if N<2 or B.shape[0]!=n or C.shape!=(m,n) or x0.shape!=(n,) or B.shape[1]<1:
        raise ValueError("incompatible robust Kalman shapes")
    if not all(np.all(np.isfinite(z)) for z in (A,B,C,y,x0)) or not np.isfinite(tau) or not np.isfinite(M) or tau<=0 or M<=0:
        raise ValueError("invalid robust Kalman values")
    return A,B,C,y,x0,tau,M


def _measurement_term(residuals:list, tau:float, M:float):
    terms=[cp.huber(cp.norm(r,2),M) for r in residuals]
    return tau*cp.sum(cp.hstack(terms))


def _solve_problem(prob:cp.Problem, *, tight:bool=False) -> None:
    if tight:
        prob.solve(solver=cp.CLARABEL,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
    else:
        prob.solve(solver=cp.CLARABEL)
    if prob.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY solver status {prob.status}")


def source_reference(problem:dict) -> dict:
    """Faithful source-equivalent CVXPY formulation; default solver selection as in task source."""
    A,B,C,y,x0,tau,M=_problem(problem);N,m=y.shape;n=A.shape[1];p=B.shape[1]
    x=cp.Variable((N+1,n),name="x");w=cp.Variable((N,p),name="w");v=cp.Variable((N,m),name="v")
    obj=cp.Minimize(cp.sum_squares(w)+_measurement_term([v[t,:] for t in range(N)],tau,M))
    constraints=[x[0]==x0]
    for t in range(N):
        constraints.append(x[t+1]==A@x[t]+B@w[t])
        constraints.append(y[t]==C@x[t]+v[t])
    prob=cp.Problem(obj,constraints);prob.solve()
    if prob.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE} or x.value is None or w.value is None or v.value is None:
        raise RuntimeError(f"source-equivalent solver status {prob.status}")
    return {"x_hat":np.asarray(x.value).tolist(),"w_hat":np.asarray(w.value).tolist(),"v_hat":np.asarray(v.value).tolist()}


def source_clarabel(problem:dict) -> dict:
    """Independent strong-baseline backend choice with the original source formulation unchanged."""
    A,B,C,y,x0,tau,M=_problem(problem);N,m=y.shape;n=A.shape[1];p=B.shape[1]
    x=cp.Variable((N+1,n),name="x");w=cp.Variable((N,p),name="w");v=cp.Variable((N,m),name="v")
    obj=cp.Minimize(cp.sum_squares(w)+_measurement_term([v[t,:] for t in range(N)],tau,M))
    constraints=[x[0]==x0]
    for t in range(N):
        constraints.append(x[t+1]==A@x[t]+B@w[t]);constraints.append(y[t]==C@x[t]+v[t])
    prob=cp.Problem(obj,constraints);_solve_problem(prob)
    if x.value is None or w.value is None or v.value is None:raise RuntimeError("CLARABEL returned no primal solution")
    return {"x_hat":np.asarray(x.value).tolist(),"w_hat":np.asarray(w.value).tolist(),"v_hat":np.asarray(v.value).tolist()}


def shallow_reduced(problem:dict) -> dict:
    """Generic reduction: eliminate measurement-noise variables v, retain x/w and all dynamics constraints."""
    A,B,C,y,x0,tau,M=_problem(problem);N,_=y.shape;n=A.shape[1];p=B.shape[1]
    x=cp.Variable((N+1,n),name="x");w=cp.Variable((N,p),name="w")
    residuals=[y[t]-C@x[t] for t in range(N)]
    obj=cp.Minimize(cp.sum_squares(w)+_measurement_term(residuals,tau,M))
    constraints=[x[0]==x0]+[x[t+1]==A@x[t]+B@w[t] for t in range(N)]
    prob=cp.Problem(obj,constraints);_solve_problem(prob)
    if x.value is None or w.value is None:raise RuntimeError("shallow reduction returned no primal solution")
    xv=np.asarray(x.value,dtype=np.float64);wv=np.asarray(w.value,dtype=np.float64);vv=y-xv[:N]@C.T
    return {"x_hat":xv.tolist(),"w_hat":wv.tolist(),"v_hat":vv.tolist()}


def rrr_deep_reduced(problem:dict) -> dict:
    """Learned RRR instantiation: eliminate v and x and prove w[N-1]=0, then lift the full solution."""
    A,B,C,y,x0,tau,M=_problem(problem);N,_=y.shape;p=B.shape[1]
    # y[t] observes x[t] only through t=N-1.  w[N-1] affects only x[N], while
    # the objective contains +||w[N-1]||^2, so the unique optimal terminal
    # process-noise choice is exactly zero.  Solve only w[0:N-2].
    wcore=cp.Variable((N-1,p),name="w_core")
    xexpr=[cp.Constant(x0)]
    for t in range(N-1):xexpr.append(A@xexpr[-1]+B@wcore[t])
    residuals=[y[t]-C@xexpr[t] for t in range(N)]
    obj=cp.Minimize(cp.sum_squares(wcore)+_measurement_term(residuals,tau,M))
    prob=cp.Problem(obj,[]);_solve_problem(prob)
    if wcore.value is None:raise RuntimeError("RRR reduction returned no primal solution")
    wv=np.zeros((N,p),dtype=np.float64);wv[:N-1]=np.asarray(wcore.value,dtype=np.float64)
    xv=np.empty((N+1,len(x0)),dtype=np.float64);xv[0]=x0
    for t in range(N):xv[t+1]=A@xv[t]+B@wv[t]
    vv=y-xv[:N]@C.T
    return {"x_hat":xv.tolist(),"w_hat":wv.tolist(),"v_hat":vv.tolist()}


def high_accuracy_independent(problem:dict) -> dict:
    """Independent tighter solve of the shallow equivalent formulation for synthetic certification only."""
    A,B,C,y,x0,tau,M=_problem(problem);N,_=y.shape;n=A.shape[1];p=B.shape[1]
    x=cp.Variable((N+1,n),name="x_cert");w=cp.Variable((N,p),name="w_cert")
    obj=cp.Minimize(cp.sum_squares(w)+_measurement_term([y[t]-C@x[t] for t in range(N)],tau,M))
    constraints=[x[0]==x0]+[x[t+1]==A@x[t]+B@w[t] for t in range(N)]
    prob=cp.Problem(obj,constraints);_solve_problem(prob,tight=True)
    if x.value is None or w.value is None:raise RuntimeError("independent certificate solve returned no primal solution")
    xv=np.asarray(x.value,dtype=np.float64);wv=np.asarray(w.value,dtype=np.float64);vv=y-xv[:N]@C.T
    return {"x_hat":xv.tolist(),"w_hat":wv.tolist(),"v_hat":vv.tolist()}


def objective_value(problem:dict,solution:dict) -> float:
    _,_,_,_,_,tau,M=_problem(problem)
    w=np.asarray(solution["w_hat"],dtype=np.float64);v=np.asarray(solution["v_hat"],dtype=np.float64)
    J=float(np.sum(w*w))
    for row in v:
        val=float(np.linalg.norm(row))
        J+=tau*(val*val if val<=M else 2*M*val-M*M)
    return J


def feasibility_metrics(problem:dict,solution:dict) -> dict:
    A,B,C,y,x0,_,_=_problem(problem);N,m=y.shape;n=A.shape[1];p=B.shape[1]
    try:
        x=np.asarray(solution["x_hat"],dtype=np.float64);w=np.asarray(solution["w_hat"],dtype=np.float64);v=np.asarray(solution["v_hat"],dtype=np.float64)
    except Exception:
        return {"shape_ok":False,"finite":False,"initial_norm":float("inf"),"max_dynamics_norm":float("inf"),"max_measurement_norm":float("inf")}
    shape_ok=x.shape==(N+1,n) and w.shape==(N,p) and v.shape==(N,m)
    finite=bool(shape_ok and np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all())
    if not finite:return {"shape_ok":shape_ok,"finite":False,"initial_norm":float("inf"),"max_dynamics_norm":float("inf"),"max_measurement_norm":float("inf")}
    initial=float(np.linalg.norm(x[0]-x0))
    dyn=max(float(np.linalg.norm(x[t+1]-(A@x[t]+B@w[t]))) for t in range(N))
    meas=max(float(np.linalg.norm(y[t]-(C@x[t]+v[t]))) for t in range(N))
    return {"shape_ok":True,"finite":True,"initial_norm":initial,"max_dynamics_norm":dyn,"max_measurement_norm":meas}


def verify_against_reference(problem:dict,solution:dict,reference_solution:dict,*,objective_factor:float=1.01,eps:float=1e-5) -> tuple[bool,str,dict]:
    metrics=feasibility_metrics(problem,solution)
    if not metrics["shape_ok"]:return False,"shape",metrics
    if not metrics["finite"]:return False,"nonfinite",metrics
    if metrics["initial_norm"]>eps:return False,"initial",metrics
    if metrics["max_dynamics_norm"]>eps:return False,"dynamics",metrics
    if metrics["max_measurement_norm"]>eps:return False,"measurement",metrics
    try:J=objective_value(problem,solution);Jref=objective_value(problem,reference_solution)
    except Exception as exc:return False,f"objective:{type(exc).__name__}",metrics
    metrics={**metrics,"objective":J,"reference_objective":Jref,"objective_ratio":J/Jref if Jref>0 else (1.0 if J<=1e-12 else float("inf"))}
    if J>Jref*objective_factor+1e-10:return False,"suboptimal",metrics
    return True,"ok",metrics


def independent_semantic_certificate(problem:dict,solution:dict,independent_solution:dict) -> tuple[bool,dict]:
    # Stricter feasibility plus a tighter independent objective comparison.  This
    # deliberately refuses the official verifier's feasibility-only fallback.
    metrics=feasibility_metrics(problem,solution)
    if not metrics["shape_ok"] or not metrics["finite"] or metrics["initial_norm"]>2e-6 or metrics["max_dynamics_norm"]>2e-6 or metrics["max_measurement_norm"]>2e-6:
        return False,metrics
    J=objective_value(problem,solution);Jind=objective_value(problem,independent_solution)
    metrics={**metrics,"objective":J,"independent_objective":Jind,"independent_objective_ratio":J/Jind if Jind>0 else (1.0 if J<=1e-12 else float("inf"))}
    return bool(J<=Jind*1.005+1e-10),metrics


def _map_engine_candidate(engine_arm:str,proposal:dict) -> Candidate:
    ops=tuple(str(x) for x in proposal["operators"]);tids=tuple(str(x) for x in proposal["transfer_ids"]);s=set(ops)
    template=proposal.get("learned_template")
    if tids==("TM-RRR-01",):
        impl,fn="rrr_deep_w_only_exact",rrr_deep_reduced
    elif tids==("TM-BFR-01",):
        impl,fn="source_equivalent_bfr_false_positive_fallback",source_reference
    elif tids==("TM-CAC-01",):
        impl,fn="source_equivalent_cac_uncertified_fallback",source_reference
    elif tids==("TM-PBEB-01",):
        impl,fn="source_equivalent_campaign_ineligible_pbeb_fallback",source_reference
    elif "reduced_representation" in s:
        impl,fn="generic_shallow_v_elimination",shallow_reduced
    elif "native_backend_substitution" in s:
        impl,fn="generic_clarabel_source_equivalent",source_clarabel
    else:
        impl,fn="cvxpy_source_reference",source_reference
    public={"v5_full":"v6_full","v5_no_transfer":"v6_no_transfer","v4_compatible":"v5_compatible"}.get(engine_arm,engine_arm)
    return Candidate(f"{public}_r{proposal['rank']}_{proposal['proposal_id']}",public,impl,ops,tids,template,None,fn)


def build_candidates(task_source_text:str) -> dict[str,list[Candidate]]:
    from engine import generate_proposals
    generated=generate_proposals(task_source_text)
    arms={k:[] for k in ("v6_full","v6_no_transfer","random_search","static_template","v5_compatible","strong_baseline")}
    for engine_arm,proposals in generated["arms"].items():
        for proposal in proposals:
            c=_map_engine_candidate(engine_arm,proposal);arms[c.arm].append(c)
    arms["strong_baseline"].append(Candidate(
        "strong_baseline_sb_convex_01_clarabel_source","strong_baseline","cvxpy_clarabel_source_equivalent",
        ("independent_convex_solver_backend","clarabel"),(),None,"SB-CONVEX-01",source_clarabel))
    return arms
