from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.odr as odr
import scipy.optimize as optimize


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


def _arrays(problem: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    x=np.asarray(problem['x'],dtype=np.float64)
    y=np.asarray(problem['y'],dtype=np.float64)
    sx=np.asarray(problem['sx'],dtype=np.float64)
    sy=np.asarray(problem['sy'],dtype=np.float64)
    if not (x.ndim==y.ndim==sx.ndim==sy.ndim==1 and len(x)==len(y)==len(sx)==len(sy) and len(x)>=2):
        raise ValueError('x/y/sx/sy must be equal nontrivial vectors')
    if not all(np.all(np.isfinite(a)) for a in (x,y,sx,sy)) or np.any(sx<=0) or np.any(sy<=0):
        raise ValueError('nonfinite values or nonpositive uncertainties')
    return x,y,sx,sy


def reference_exact(problem: dict) -> dict:
    x,y,sx,sy=_arrays(problem)
    data=odr.RealData(x,y=y,sx=sx,sy=sy)
    model=odr.Model(lambda B,z:B[0]*z+B[1])
    out=odr.ODR(data,model,beta0=[0.0,1.0]).run()
    return {'beta':[float(v) for v in out.beta]}


def lowlevel_odr_exact(problem: dict) -> dict:
    x,y,sx,sy=_arrays(problem)
    # RealData converts standard deviations to inverse-variance weights.
    out=odr.odr(lambda B,z:B[0]*z+B[1],[0.0,1.0],y,x,we=1.0/(sy*sy),wd=1.0/(sx*sx),full_output=0)
    return {'beta':[float(v) for v in out[0]]}


def lowlevel_contiguous_exact(problem: dict) -> dict:
    x,y,sx,sy=_arrays(problem)
    x=np.ascontiguousarray(x);y=np.ascontiguousarray(y);sx=np.ascontiguousarray(sx);sy=np.ascontiguousarray(sy)
    out=odr.odr(lambda B,z:B[0]*z+B[1],[0.0,1.0],y,x,we=1.0/(sy*sy),wd=1.0/(sx*sx),full_output=0)
    return {'beta':[float(v) for v in out[0]]}


def jacobian_reference(problem: dict) -> dict:
    x,y,sx,sy=_arrays(problem)
    def f(B,z): return B[0]*z+B[1]
    def fjacb(B,z): return np.vstack((z,np.ones_like(z)))
    def fjacd(B,z): return np.full_like(z,B[0])
    data=odr.RealData(x,y=y,sx=sx,sy=sy)
    model=odr.Model(f,fjacb=fjacb,fjacd=fjacd)
    out=odr.ODR(data,model,beta0=[0.0,1.0]).run()
    return {'beta':[float(v) for v in out.beta]}


def reduced_scalar_objective(problem: dict) -> dict:
    # Exact reduction of the mathematical weighted straight-line ODR objective
    # to one scalar slope parameter. It is intentionally NOT assumed equivalent
    # to ODRPACK's stopped numerical iterate; synthetic screening decides.
    x,y,sx,sy=_arrays(problem)
    sx2=sx*sx;sy2=sy*sy
    m0=float(np.polyfit(x,y,1)[0])
    def intercept(m: float) -> float:
        den=sy2+(m*m)*sx2
        w=1.0/den
        return float(np.sum(w*(y-m*x))/np.sum(w))
    def objective(m: float) -> float:
        b=intercept(m);den=sy2+(m*m)*sx2;r=y-m*x-b
        return float(np.sum((r*r)/den))
    # Wide deterministic bracket, then high-accuracy Brent minimization.
    scale=max(1.0,abs(m0))
    res=optimize.minimize_scalar(objective,method='bounded',bounds=(m0-8.0*scale,m0+8.0*scale),options={'xatol':1e-14,'maxiter':500})
    if not res.success: raise RuntimeError('reduced scalar ODR optimization failed')
    m=float(res.x);b=intercept(m)
    return {'beta':[m,b]}


def precision_budgeted_backend(problem: dict) -> dict:
    # Frozen verifier rtol is about 7e-11. float32 epsilon is >1e-7, so the
    # PBEB rule must conservatively retain float64 and use the cheaper direct backend.
    verifier_rtol=float(2*np.finfo(float).eps**(2.0/3.0))
    if float(np.finfo(np.float32).eps) > verifier_rtol/8.0:
        return lowlevel_odr_exact(problem)
    # This branch is structurally unreachable on IEEE-754 float32/float64 hosts.
    x,y,sx,sy=_arrays(problem)
    q={'x':x.astype(np.float32),'y':y.astype(np.float32),'sx':sx.astype(np.float32),'sy':sy.astype(np.float32)}
    return lowlevel_odr_exact(q)


def _objective(problem: dict,beta: list[float]) -> float:
    x,y,sx,sy=_arrays(problem);m,b=(float(beta[0]),float(beta[1]))
    den=sy*sy+(m*m)*sx*sx;r=y-m*x-b
    return float(np.sum((r*r)/den))


def independent_semantic_certificate(problem: dict,solution: dict) -> bool:
    try:
        beta=np.asarray(solution['beta'],dtype=np.float64)
        if beta.shape!=(2,) or not np.all(np.isfinite(beta)): return False
        # Independently check near-local optimality of the weighted orthogonal
        # objective without comparing coefficients to source output.
        base=_objective(problem,beta.tolist())
        m,b=float(beta[0]),float(beta[1])
        dm=max(1e-7,abs(m)*1e-7);db=max(1e-7,abs(b)*1e-7)
        probes=[[m+dm,b],[m-dm,b],[m,b+db],[m,b-db]]
        return all(base <= _objective(problem,p)*(1.0+2e-6)+1e-10 for p in probes)
    except Exception:
        return False


def official_verifier_accepts(problem: dict,solution: dict) -> bool:
    try:
        expected=np.asarray(reference_exact(problem)['beta'],dtype=np.float64)
        observed=np.asarray(solution['beta'],dtype=np.float64)
        if observed.shape!=(2,) or not np.all(np.isfinite(observed)): return False
        rtol=float(2*np.finfo(float).eps**(2.0/3.0));atol=float(np.finfo(float).smallest_normal)
        return bool(np.allclose(observed,expected,rtol=rtol,atol=atol))
    except Exception:
        return False


def _map_engine_candidate(arm: str,proposal: dict) -> Candidate:
    ops=tuple(str(x) for x in proposal['operators']);tids=tuple(str(x) for x in proposal['transfer_ids']);s=set(ops)
    # BFR/CAC are fingerprint false-positive stress cases here: no lawful graph/frontier
    # or active-constraint reduction is mechanically present in weighted linear ODR.
    if tids in (("TM-BFR-01",),("TM-CAC-01",)):
        impl,fn='source_equivalent_fallback',reference_exact
    elif tids==("TM-RRR-01",):
        impl,fn='reduced_scalar_weighted_odr',reduced_scalar_objective
    elif tids==("TM-PBEB-01",):
        impl,fn='precision_budgeted_lowlevel_odr',precision_budgeted_backend
    elif 'native_backend_substitution' in s or s.intersection({'contiguous_layout','zero_copy_representation','vectorized_batch_kernel'}):
        impl,fn='lowlevel_odrpack_direct_weights',lowlevel_odr_exact
    elif 'reduced_representation' in s:
        impl,fn='reduced_scalar_weighted_odr',reduced_scalar_objective
    elif 'dtype_specialization' in s:
        impl,fn='precision_budgeted_lowlevel_odr',precision_budgeted_backend
    elif 'bounded_exact_refinement' in s:
        impl,fn='analytic_jacobian_source_path',jacobian_reference
    else:
        impl,fn='scipy_odr_reference',reference_exact
    public_arm={'v5_full':'v6_full','v5_no_transfer':'v6_no_transfer','v4_compatible':'v5_compatible'}.get(arm,arm)
    return Candidate(f"{public_arm}_r{proposal['rank']}_{proposal['proposal_id']}",public_arm,impl,ops,tids,proposal.get('learned_template'),None,fn)


def build_candidates(task_source_text: str) -> dict[str,list[Candidate]]:
    from engine import generate_proposals
    generated=generate_proposals(task_source_text)
    arms={k:[] for k in ['v6_full','v6_no_transfer','random_search','static_template','v5_compatible','strong_baseline']}
    for engine_arm,proposals in generated['arms'].items():
        for proposal in proposals:
            c=_map_engine_candidate(engine_arm,proposal);arms[c.arm].append(c)
    arms['strong_baseline'].append(Candidate(
        'strong_baseline_sb_native_numeric_01_lowlevel_odr','strong_baseline','lowlevel_odrpack_direct_weights',
        ('direct_general_purpose_numeric_backend',),(),None,'SB-NATIVE-NUMERIC-01',lowlevel_odr_exact))
    return arms
