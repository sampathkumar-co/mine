from __future__ import annotations

import json,math,sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg as sla

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'lexigen-v5'))
from engine import generate_proposals

HERE=Path(__file__).resolve().parent
SOURCE_SHA256='8f7771e5618509c0b8af73390440dd7258a253983cdf8cc03b0208fa4218b018'
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
    solve:Callable[[dict],dict[str,np.ndarray]]


def _matrix(problem:dict,dtype=np.float64)->np.ndarray:
    if not isinstance(problem,dict) or 'matrix' not in problem:raise ValueError('invalid problem structure')
    a=np.asarray(problem['matrix'],dtype=dtype)
    if a.ndim!=2 or min(a.shape)<=0 or not np.all(np.isfinite(a)):raise ValueError('invalid matrix')
    return a


def numpy_source_svd_float64(problem:dict)->dict[str,np.ndarray]:
    a=_matrix(problem,np.float64);u,s,vh=np.linalg.svd(a,full_matrices=False)
    return {'U':u,'S':s,'V':vh.T}


def scipy_gesdd_svd_float64(problem:dict)->dict[str,np.ndarray]:
    a=_matrix(problem,np.float64);u,s,vh=sla.svd(a,full_matrices=False,lapack_driver='gesdd',check_finite=False,overwrite_a=False)
    return {'U':u,'S':s,'V':vh.T}


def scipy_gesdd_svd_float32(problem:dict)->dict[str,np.ndarray]:
    a=_matrix(problem,np.float32);u,s,vh=sla.svd(a,full_matrices=False,lapack_driver='gesdd',check_finite=False,overwrite_a=False)
    return {'U':u,'S':s,'V':vh.T}


def numpy_svd_float32(problem:dict)->dict[str,np.ndarray]:
    a=_matrix(problem,np.float32);u,s,vh=np.linalg.svd(a,full_matrices=False)
    return {'U':u,'S':s,'V':vh.T}


def guarded_gram_eigh_svd_float64(problem:dict)->dict[str,np.ndarray]:
    a=_matrix(problem,np.float64);m,n=a.shape;eps=np.finfo(np.float64).eps
    try:
        if m>=n:
            gram=a.T@a;w,v=np.linalg.eigh(gram);order=np.argsort(w)[::-1];w=w[order];v=v[:,order]
            wmax=float(w[0]);tol=eps*max(m,n)*max(wmax,1.0)
            if not np.all(np.isfinite(w)) or float(w[-1])<=tol:return numpy_source_svd_float64(problem)
            uraw=a@v;s=np.linalg.norm(uraw,axis=0)
            if not np.all(np.isfinite(s)) or np.any(s<=0):return numpy_source_svd_float64(problem)
            u=uraw/s
            return {'U':u,'S':s,'V':v}
        gram=a@a.T;w,u=np.linalg.eigh(gram);order=np.argsort(w)[::-1];w=w[order];u=u[:,order]
        wmax=float(w[0]);tol=eps*max(m,n)*max(wmax,1.0)
        if not np.all(np.isfinite(w)) or float(w[-1])<=tol:return numpy_source_svd_float64(problem)
        vraw=a.T@u;s=np.linalg.norm(vraw,axis=0)
        if not np.all(np.isfinite(s)) or np.any(s<=0):return numpy_source_svd_float64(problem)
        v=vraw/s
        return {'U':u,'S':s,'V':v}
    except np.linalg.LinAlgError:
        return numpy_source_svd_float64(problem)


def independent_source_contract(problem:dict,solution:dict)->tuple[bool,str|None,dict[str,float]]:
    try:
        a=_matrix(problem,np.float64);m,n=a.shape;k=min(m,n)
        if not isinstance(solution,dict) or any(x not in solution for x in ('U','S','V')):return False,'format',{}
        u=np.asarray(solution['U']);s=np.asarray(solution['S']);v=np.asarray(solution['V'])
        if u.shape!=(m,k) or s.ndim!=1 or s.shape[0]!=k or v.shape!=(n,k):return False,'shape',{}
        if not all(np.all(np.isfinite(x)) for x in (u,s,v)):return False,'nonfinite',{}
        if np.any(s<0):return False,'negative_singular_value',{}
        ui=u.T@u;vi=v.T@v;eye=np.eye(k)
        u_ok=bool(np.allclose(ui,eye,atol=1e-6));v_ok=bool(np.allclose(vi,eye,atol=1e-6))
        reconstructed=(u*s)@v.T
        rec_ok=bool(np.allclose(a,reconstructed,atol=1e-6))
        max_u=float(np.max(np.abs(ui-eye))) if k else 0.0;max_v=float(np.max(np.abs(vi-eye))) if k else 0.0;max_rec=float(np.max(np.abs(a-reconstructed)))
        valid=u_ok and v_ok and rec_ok
        return valid,(None if valid else 'orthonormality_or_reconstruction'),{'max_u_orth_error':max_u,'max_v_orth_error':max_v,'max_reconstruction_abs_error':max_rec}
    except Exception as exc:return False,f'certificate_exception:{type(exc).__name__}:{exc}',{}


def _implementation(tids:tuple[str,...],ops:tuple[str,...]):
    if 'TM-RRR-01' in tids:return guarded_gram_eigh_svd_float64,'guarded_gram_eigh_svd_float64','guarded_gram_eigh_svd_float64'
    if 'TM-PBEB-01' in tids:return scipy_gesdd_svd_float32,'scipy_gesdd_svd_float32','scipy_gesdd_svd_float32'
    if 'TM-BFR-01' in tids or 'TM-CAC-01' in tids:return numpy_source_svd_float64,'numpy_source_svd_float64','numpy_source_svd_float64'
    if 'reduced_representation' in ops:return guarded_gram_eigh_svd_float64,'guarded_gram_eigh_svd_float64','guarded_gram_eigh_svd_float64'
    if 'dtype_specialization' in ops and 'native_backend_substitution' in ops:return scipy_gesdd_svd_float32,'scipy_gesdd_svd_float32','scipy_gesdd_svd_float32'
    if 'native_backend_substitution' in ops:return scipy_gesdd_svd_float64,'scipy_gesdd_svd_float64','scipy_gesdd_svd_float64'
    if 'dtype_specialization' in ops:return numpy_svd_float32,'numpy_svd_float32','numpy_svd_float32'
    return numpy_source_svd_float64,'numpy_source_svd_float64','numpy_source_svd_float64'


def build_candidates(source_text:str)->dict[str,list[Candidate]]:
    generated=generate_proposals(source_text);sealed=json.loads((HERE/'SOURCE_SCREEN_R1_RESULT.json').read_text());expected=sealed['proposal_ids_by_arm']
    arms={a:[] for a in ARM_ORDER}
    for old_arm,rows in generated['arms'].items():
        arm=ARM_NAME_MAP[old_arm];ids=[str(r['proposal_id']) for r in rows]
        if ids!=expected[arm]:raise RuntimeError(f'proposal identity/order mismatch {arm}')
        for r in rows:
            rank=int(r['rank']);pid=str(r['proposal_id']);ops=tuple(str(x) for x in r['operators']);tids=tuple(str(x) for x in r['transfer_ids']);template=r.get('learned_template');fn,impl,sem=_implementation(tids,ops)
            arms[arm].append(Candidate(arm,f'{arm}_r{rank}_{pid}',rank,pid,ops,tids,template,impl,sem,None,fn))
    arms['strong_baseline']=[Candidate('strong_baseline','strong_baseline_sb_reduced_linalg_01_scipy_gesdd',1,None,('specialized_decomposition_driver','lapack_gesdd'),(),None,'scipy_gesdd_svd_float64','scipy_gesdd_svd_float64','SB-REDUCED-LINALG-01',scipy_gesdd_svd_float64)]
    counts={k:len(v) for k,v in arms.items()}
    if counts!={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}:raise RuntimeError(f'candidate count mismatch {counts}')
    return arms
