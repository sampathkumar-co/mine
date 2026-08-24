from __future__ import annotations
import cvxpy as cp
import numpy as np
from typing import Callable


def solve_gp(problem: dict, mode: str = 'reference') -> dict:
    n=int(problem['num_conditions']); conditions=problem['conditions']
    A=cp.Variable(pos=True,name='A');S=cp.Variable(pos=True,name='S')
    V=[cp.Variable(pos=True,name=f'V_{i}') for i in range(n)]
    W=[cp.Variable(pos=True,name=f'W_{i}') for i in range(n)]
    Re=[cp.Variable(pos=True,name=f'Re_{i}') for i in range(n)]
    CD=[cp.Variable(pos=True,name=f'C_D_{i}') for i in range(n)]
    CL=[cp.Variable(pos=True,name=f'C_L_{i}') for i in range(n)]
    Cf=[cp.Variable(pos=True,name=f'C_f_{i}') for i in range(n)]
    Ww=[cp.Variable(pos=True,name=f'W_w_{i}') for i in range(n)]
    constraints=[]; total_drag=0
    for i,c in enumerate(conditions):
        CDA0=float(c['CDA0']);CLmax=float(c['C_Lmax']);Nult=float(c['N_ult']);Swet=float(c['S_wetratio']);Vmin=float(c['V_min']);W0=float(c['W_0']);w1=float(c['W_W_coeff1']);w2=float(c['W_W_coeff2']);e=float(c['e']);k=float(c['k']);mu=float(c['mu']);rho=float(c['rho']);tau=float(c['tau'])
        total_drag += 0.5*rho*V[i]**2*CD[i]*S
        constraints += [CD[i]>=CDA0/S+k*Cf[i]*Swet+CL[i]**2/(np.pi*A*e),Cf[i]>=0.074/Re[i]**0.2,Re[i]*mu>=rho*V[i]*cp.sqrt(S/A),Ww[i]>=w2*S+w1*Nult*(A**(3/2))*cp.sqrt(W0*W[i])/tau,W[i]>=W0+Ww[i],W[i]<=0.5*rho*V[i]**2*CL[i]*S,2*W[i]/(rho*Vmin**2*S)<=CLmax]
    prob=cp.Problem(cp.Minimize(total_drag/n),constraints)
    try:
        if mode=='relaxed':
            prob.solve(gp=True,solver=cp.CLARABEL,tol_gap_abs=1e-6,tol_gap_rel=1e-6,tol_feas=1e-6,max_iter=100)
        elif mode=='explicit_clarabel':
            prob.solve(gp=True,solver=cp.CLARABEL)
        else:
            prob.solve(gp=True)
        if prob.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE} or A.value is None:return {'A':[],'S':[],'avg_drag':0.0,'condition_results':[]}
        results=[]
        for i,c in enumerate(conditions):
            drag=float(0.5*float(c['rho'])*V[i].value**2*CD[i].value*S.value)
            results.append({'condition_id':c['condition_id'],'V':float(V[i].value),'W':float(W[i].value),'W_w':float(Ww[i].value),'C_L':float(CL[i].value),'C_D':float(CD[i].value),'C_f':float(Cf[i].value),'Re':float(Re[i].value),'drag':drag})
        return {'A':float(A.value),'S':float(S.value),'avg_drag':float(prob.value),'condition_results':results}
    except Exception:
        return {'A':[],'S':[],'avg_drag':0.0,'condition_results':[]}


def reference_exact(problem:dict)->dict:return solve_gp(problem,'reference')
def relaxed_rrr(problem:dict)->dict:return solve_gp(problem,'relaxed')
def explicit_clarabel(problem:dict)->dict:return solve_gp(problem,'explicit_clarabel')

PROPOSALS={
'v5_full':[
('3304c859d463a501bd86',['bit_parallel_representation','sparse_frontier_search','early_certificate_exit'],['TM-BFR-01'],'bit_frontier_restriction'),
('41510e43e8fafb598496',['reduced_representation','bounded_exact_refinement','risk_aware_staging'],['TM-RRR-01'],'reduced_representation_refinement'),
('a6102573c9f355414229',['active_set_decomposition','early_certificate_exit','risk_aware_staging'],['TM-CAC-01'],'certified_active_core'),
('4abf2b51384c560522e8',['active_set_decomposition','bit_parallel_representation','bounded_exact_refinement'],[],None),
('c50e493c5549a408f3e5',['active_set_decomposition','bit_parallel_representation','sparse_frontier_search'],[],None),
('514b3e8a41ba1f8b73a1',['active_set_decomposition','bit_parallel_representation','reduced_representation'],[],None)],
'v5_no_transfer':[
('91e027e622f2d9a98240',['active_set_decomposition','bit_parallel_representation','bounded_exact_refinement'],[],None),
('b2614109e1a5ccc10c14',['active_set_decomposition','bit_parallel_representation','sparse_frontier_search'],[],None),
('20375ceceffce4d406a4',['active_set_decomposition','bit_parallel_representation','reduced_representation'],[],None),
('4a4e1871b7f7b48b9485',['active_set_decomposition','bit_parallel_representation','sort_partition_reduction'],[],None),
('d69e86803f54c5a83d06',['vectorized_batch_kernel','active_set_decomposition','bit_parallel_representation'],[],None),
('66c5848a3c8a4f51b562',['bit_parallel_representation','reduced_representation','bounded_exact_refinement'],[],None)],
'random_search':[
('487d4f738090692a3fa8',['early_certificate_exit','sparse_frontier_search','bounded_exact_refinement'],[],None),
('3af6a706b42274e3e8bc',['contiguous_layout','sparse_frontier_search','reduced_representation'],[],None),
('4fba670e406eaf5628e8',['dtype_specialization','early_certificate_exit','active_set_decomposition'],[],None),
('867431373f686fe07b65',['risk_aware_staging','early_certificate_exit','reduced_representation'],[],None),
('1ec748b05be008194d8f',['native_backend_substitution','vectorized_batch_kernel','early_certificate_exit'],[],None),
('817b76aaea3723c02800',['zero_copy_representation','dtype_specialization','native_backend_substitution'],[],None)],
'static_template':[
('dbfcd2af539b0b2636e7',['bit_parallel_representation','sparse_frontier_search'],[],None),
('8fd871e046faa7e4d37c',['reduced_representation','bounded_exact_refinement'],[],None),
('820b1c309b6117eb268d',['active_set_decomposition','early_certificate_exit'],[],None),
('8f1dafda0d3fbc099aa9',['zero_copy_representation','vectorized_batch_kernel'],[],None),
('357e80313b8b9dc3cf36',['contiguous_layout','vectorized_batch_kernel'],[],None),
('d044a19fd4551034dc11',['dtype_specialization','risk_aware_staging'],[],None)],
'v4_compatible':[
('f9f3239b6866512e4f68',['active_set_decomposition','bit_parallel_representation','bounded_exact_refinement'],[],None),
('9f5f55df04a5ad23f542',['active_set_decomposition','bit_parallel_representation','sparse_frontier_search'],[],None),
('ec4b9c17aaa3767d4f6d',['active_set_decomposition','bit_parallel_representation','sort_partition_reduction'],[],None),
('7c30efb65d2c20ff8cc9',['vectorized_batch_kernel','active_set_decomposition','bit_parallel_representation'],[],None),
('3df5ed91505aea4ed6cb',['active_set_decomposition','sparse_frontier_search','bounded_exact_refinement'],[],None),
('0dde88a4a159a3ad0e40',['bit_parallel_representation','bounded_exact_refinement','sort_partition_reduction'],[],None)]}

CANDIDATES_BY_ARM={};CANDIDATE_META={}
for arm,rows in PROPOSALS.items():
    built=[]
    for rank,(pid,ops,tids,learned) in enumerate(rows,1):
        name=f'{arm}_r{rank}_{pid}'
        if arm=='v5_full' and learned=='reduced_representation_refinement':fn=relaxed_rrr;impl='clarabel_relaxed_1e-6'
        elif 'native_backend_substitution' in ops:fn=explicit_clarabel;impl='explicit_clarabel_default'
        else:fn=reference_exact;impl='cvxpy_default_gp'
        built.append((name,fn));CANDIDATE_META[name]={'arm':arm,'rank':rank,'proposal_id':pid,'operators':ops,'transfer_ids':tids,'learned_template':learned,'implementation_class':impl,'mapping_policy':'RRR learned recipe -> bounded 1e-6 CLARABEL tolerances; native backend -> explicit CLARABEL; all other proposals -> exact source-equivalent fallback because no safe source-level reduction is certified'}
    CANDIDATES_BY_ARM[arm]=built
if sum(len(v) for v in CANDIDATES_BY_ARM.values())!=30:raise RuntimeError('candidate budget mismatch')
