from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Candidate:
    name: str
    arm: str
    implementation_class: str
    operators: tuple[str, ...]
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    baseline_id: str | None
    solve: Callable[[dict], list[float]]


def _problem(problem: dict) -> tuple[np.ndarray,float,float,float]:
    if not isinstance(problem,dict) or not {'F','t0','t1','y0'} <= set(problem):
        raise ValueError('invalid Lorenz96 problem')
    y0=np.asarray(problem['y0'],dtype=np.float64)
    F=float(problem['F']);t0=float(problem['t0']);t1=float(problem['t1'])
    if y0.ndim!=1 or len(y0)<4 or not np.all(np.isfinite(y0)) or not all(np.isfinite(v) for v in (F,t0,t1)) or t1<=t0:
        raise ValueError('invalid Lorenz96 values')
    return y0,F,t0,t1


def reference_exact(problem: dict) -> list[float]:
    y0,F,t0,t1=_problem(problem)
    def rhs(t,x):
        N=len(x)
        ip1=np.roll(np.arange(N),-1);im1=np.roll(np.arange(N),1);im2=np.roll(np.arange(N),2)
        return (x[ip1]-x[im2])*x[im1]-x+F
    sol=solve_ivp(rhs,[t0,t1],y0,method='RK45',rtol=1e-8,atol=1e-8)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1].tolist()


def preindexed_rk45_exact(problem: dict) -> list[float]:
    y0,F,t0,t1=_problem(problem);N=len(y0);idx=np.arange(N);ip1=np.roll(idx,-1);im1=np.roll(idx,1);im2=np.roll(idx,2)
    def rhs(t,x):return (x[ip1]-x[im2])*x[im1]-x+F
    sol=solve_ivp(rhs,[t0,t1],y0,method='RK45',rtol=1e-8,atol=1e-8)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1].tolist()


def sliced_cyclic_rk45_exact(problem: dict) -> list[float]:
    y0,F,t0,t1=_problem(problem)
    def rhs(t,x):
        out=np.empty_like(x)
        out[2:-1]=(x[3:]-x[:-3])*x[1:-2]-x[2:-1]+F
        out[0]=(x[1]-x[-2])*x[-1]-x[0]+F
        out[1]=(x[2]-x[-1])*x[0]-x[1]+F
        out[-1]=(x[0]-x[-3])*x[-2]-x[-1]+F
        return out
    sol=solve_ivp(rhs,[t0,t1],y0,method='RK45',rtol=1e-8,atol=1e-8)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1].tolist()


def high_accuracy_independent(problem: dict) -> np.ndarray:
    y0,F,t0,t1=_problem(problem);N=len(y0);idx=np.arange(N);ip1=np.roll(idx,-1);im1=np.roll(idx,1);im2=np.roll(idx,2)
    def rhs(t,x):return (x[ip1]-x[im2])*x[im1]-x+F
    sol=solve_ivp(rhs,[t0,t1],y0,method='DOP853',rtol=1e-11,atol=1e-11)
    if not sol.success:raise RuntimeError(sol.message)
    return np.asarray(sol.y[:,-1],dtype=np.float64)


def independent_semantic_certificate(problem: dict, solution) -> bool:
    try:
        obs=np.asarray(solution,dtype=np.float64);y0,_,_,_=_problem(problem)
        if obs.shape!=y0.shape or not np.all(np.isfinite(obs)):return False
        independent=high_accuracy_independent(problem)
        scale=np.maximum(1.0,np.abs(independent))
        return bool(np.max(np.abs(obs-independent)/scale) <= 2.5e-4)
    except Exception:
        return False


def official_verifier_accepts(problem: dict, solution) -> bool:
    try:
        obs=np.asarray(solution,dtype=np.float64);ref=np.asarray(reference_exact(problem),dtype=np.float64);y0,_,_,_=_problem(problem)
        return bool(obs.shape==y0.shape and np.all(np.isfinite(obs)) and np.allclose(obs,ref,rtol=1e-5,atol=1e-8))
    except Exception:
        return False


def _map_engine_candidate(arm: str, proposal: dict) -> Candidate:
    ops=tuple(str(x) for x in proposal['operators']);tids=tuple(str(x) for x in proposal['transfer_ids']);s=set(ops)
    # Frozen applicability stress controls: neither discrete frontier restriction nor
    # active-constraint extraction has a lawful semantic instantiation in this ODE.
    if tids in (("TM-BFR-01",),("TM-CAC-01",)):
        impl,fn='source_equivalent_transfer_fallback',reference_exact
    elif 'vectorized_batch_kernel' in s:
        impl,fn='sliced_cyclic_rk45_exact',sliced_cyclic_rk45_exact
    elif s.intersection({'zero_copy_representation','contiguous_layout'}):
        impl,fn='preindexed_rk45_exact',preindexed_rk45_exact
    else:
        # dtype changes, alternate integrators, early exits, active-set, bitset,
        # sparse-frontier and tolerance changes are not assumed safe for a long-horizon
        # ODE merely because the lexical fingerprint emitted those operators.
        impl,fn='scipy_rk45_reference',reference_exact
    public={'v5_full':'v6_full','v5_no_transfer':'v6_no_transfer','v4_compatible':'v5_compatible'}.get(arm,arm)
    return Candidate(f"{public}_r{proposal['rank']}_{proposal['proposal_id']}",public,impl,ops,tids,proposal.get('learned_template'),None,fn)


def build_candidates(task_source_text: str) -> dict[str,list[Candidate]]:
    from engine import generate_proposals
    generated=generate_proposals(task_source_text)
    arms={k:[] for k in ('v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline')}
    for engine_arm,proposals in generated['arms'].items():
        for proposal in proposals:
            c=_map_engine_candidate(engine_arm,proposal);arms[c.arm].append(c)
    arms['strong_baseline'].append(Candidate(
        'strong_baseline_sb_native_numeric_01_vectorized_rk45','strong_baseline','sliced_cyclic_rk45_exact',
        ('vectorized_native_kernel','static_index_hoisting'),(),None,'SB-NATIVE-NUMERIC-01',sliced_cyclic_rk45_exact))
    return arms
