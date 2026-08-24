from __future__ import annotations
import numpy as np
from scipy.optimize import leastsq,least_squares

def _safe_exp(z):return np.exp(np.clip(z,-50.0,50.0))
def residual_guess(problem:dict):
    x=np.asarray(problem['x_data']);y=np.asarray(problem['y_data']);m=str(problem['model_type'])
    if m=='polynomial':
        deg=int(problem['degree'])
        def r(p):return y-np.polyval(p,x)
        g=np.ones(deg+1)
    elif m=='exponential':
        def r(p):
            a,b,c=p;return y-(a*_safe_exp(b*x)+c)
        g=np.array([1.0,0.05,0.0])
    elif m=='logarithmic':
        def r(p):
            a,b,c,d=p;return y-(a*np.log(b*x+c)+d)
        g=np.array([1.0,1.0,1.0,0.0])
    elif m=='sigmoid':
        def r(p):
            a,b,c,d=p;return y-(a/(1+_safe_exp(-b*(x-c)))+d)
        g=np.array([3.0,0.5,np.median(x),0.0])
    elif m=='sinusoidal':
        def r(p):
            a,b,c,d=p;return y-(a*np.sin(b*x+c)+d)
        g=np.array([2.0,1.0,0.0,0.0])
    else:raise ValueError(f'unknown model {m}')
    return r,g

def reference_exact(problem:dict)->dict:
    r,g=residual_guess(problem);p=leastsq(r,g,full_output=True,maxfev=10000)[0];return {'params':p.tolist()}
def polynomial_reduction(problem:dict,dtype=np.float64)->dict:
    if str(problem['model_type'])!='polynomial':return reference_exact(problem)
    x=np.asarray(problem['x_data'],dtype=dtype);y=np.asarray(problem['y_data'],dtype=dtype);deg=int(problem['degree']);v=np.vander(x,deg+1);p=np.linalg.lstsq(v,y,rcond=None)[0];return {'params':np.asarray(p,dtype=float).tolist()}
def reduced64(problem:dict)->dict:return polynomial_reduction(problem,np.float64)
def reduced32(problem:dict)->dict:return polynomial_reduction(problem,np.float32)
def native_lm(problem:dict)->dict:
    r,g=residual_guess(problem)
    try:p=least_squares(r,g,method='lm',max_nfev=10000).x;return {'params':p.tolist()}
    except Exception:return reference_exact(problem)
def reduced_native(problem:dict)->dict:
    if str(problem['model_type'])=='polynomial':return reduced64(problem)
    return native_lm(problem)

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
('67668ab6baaece7064f4',['native_backend_substitution','bit_parallel_representation','reduced_representation'],[],None),
('7cb58654becb58e60790',['native_backend_substitution','risk_aware_staging','bit_parallel_representation'],[],None),
('9b5cccfc4ff73a930808',['contiguous_layout','native_backend_substitution','active_set_decomposition'],[],None),
('f26db723331fcd5d2652',['vectorized_batch_kernel','risk_aware_staging','early_certificate_exit'],[],None),
('e818498ab004d266d2a1',['contiguous_layout','dtype_specialization','sparse_frontier_search'],[],None),
('0df835c1573eed74eae2',['risk_aware_staging','active_set_decomposition'],[],None)],
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
        if learned=='reduced_representation_refinement':fn=reduced64;impl='polynomial_linear_reduction_plus_reference'
        elif learned in {'bit_frontier_restriction','certified_active_core'}:fn=reference_exact;impl='scipy_leastsq_reference'
        elif 'native_backend_substitution' in ops and 'reduced_representation' in ops:fn=reduced_native;impl='polynomial_reduction_plus_scipy_least_squares_lm'
        elif 'native_backend_substitution' in ops:fn=native_lm;impl='scipy_least_squares_lm'
        elif 'reduced_representation' in ops:fn=reduced64;impl='polynomial_linear_reduction_plus_reference'
        elif 'dtype_specialization' in ops:fn=reduced32;impl='polynomial_float32_reduction_plus_reference'
        else:fn=reference_exact;impl='scipy_leastsq_reference'
        built.append((name,fn));CANDIDATE_META[name]={'arm':arm,'rank':rank,'proposal_id':pid,'operators':ops,'transfer_ids':tids,'learned_template':learned,'implementation_class':impl}
    CANDIDATES_BY_ARM[arm]=built
if sum(len(v) for v in CANDIDATES_BY_ARM.values())!=30:raise RuntimeError('candidate budget mismatch')
