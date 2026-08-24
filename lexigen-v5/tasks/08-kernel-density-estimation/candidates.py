from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from sklearn.neighbors._ball_tree import kernel_norm


def reference_exact(problem:dict)->dict:
    X=np.asarray(problem['data_points'],dtype=float);Q=np.asarray(problem['query_points'],dtype=float);k=str(problem['kernel']);h=float(problem['bandwidth'])
    if Q.shape[0]==0:return {'log_density':[]}
    return {'log_density':KernelDensity(kernel=k,bandwidth=h).fit(X).score_samples(Q).tolist()}

def direct_pairwise(problem:dict,dtype)->dict:
    X=np.asarray(problem['data_points'],dtype=dtype);Q=np.asarray(problem['query_points'],dtype=dtype);k=str(problem['kernel']);h=float(problem['bandwidth'])
    if Q.shape[0]==0:return {'log_density':[]}
    if X.ndim!=2 or Q.ndim!=2 or X.shape[0]==0 or X.shape[1]!=Q.shape[1] or h<=0:raise ValueError('invalid KDE problem')
    D=cdist(Q,X,metric='euclidean')/h
    with np.errstate(divide='ignore',invalid='ignore'):
        if k=='gaussian':lw=-0.5*D*D
        elif k=='exponential':lw=-D
        elif k=='tophat':lw=np.where(D<=1.0,0.0,-np.inf)
        elif k=='epanechnikov':lw=np.where(D<=1.0,np.log(np.maximum(1.0-D*D,0.0)),-np.inf)
        elif k=='linear':lw=np.where(D<=1.0,np.log(np.maximum(1.0-D,0.0)),-np.inf)
        elif k=='cosine':lw=np.where(D<=1.0,np.log(np.maximum(np.cos(0.5*np.pi*D),0.0)),-np.inf)
        else:raise ValueError(f'unknown kernel {k}')
    out=logsumexp(lw,axis=1)+float(np.log(kernel_norm(h,X.shape[1],k)))-float(np.log(X.shape[0]))
    return {'log_density':out.tolist()}
def direct64(problem:dict)->dict:return direct_pairwise(problem,np.float64)
def direct32(problem:dict)->dict:return direct_pairwise(problem,np.float32)

PROPOSALS={
'v5_full':[
('3304c859d463a501bd86',['bit_parallel_representation','sparse_frontier_search','early_certificate_exit'],['TM-BFR-01'],'bit_frontier_restriction'),
('41510e43e8fafb598496',['reduced_representation','bounded_exact_refinement','risk_aware_staging'],['TM-RRR-01'],'reduced_representation_refinement'),
('a6102573c9f355414229',['active_set_decomposition','early_certificate_exit','risk_aware_staging'],['TM-CAC-01'],'certified_active_core'),
('b1ef08a2d68a248c0821',['dtype_specialization','native_backend_substitution','risk_aware_staging'],['TM-PBEB-01'],'precision_backend_error_budget'),
('4abf2b51384c560522e8',['active_set_decomposition','bit_parallel_representation','bounded_exact_refinement'],[],None),
('c50e493c5549a408f3e5',['active_set_decomposition','bit_parallel_representation','sparse_frontier_search'],[],None)],
'v5_no_transfer':[
('91e027e622f2d9a98240',['active_set_decomposition','bit_parallel_representation','bounded_exact_refinement'],[],None),
('b2614109e1a5ccc10c14',['active_set_decomposition','bit_parallel_representation','sparse_frontier_search'],[],None),
('20375ceceffce4d406a4',['active_set_decomposition','bit_parallel_representation','reduced_representation'],[],None),
('4a4e1871b7f7b48b9485',['active_set_decomposition','bit_parallel_representation','sort_partition_reduction'],[],None),
('d69e86803f54c5a83d06',['vectorized_batch_kernel','active_set_decomposition','bit_parallel_representation'],[],None),
('66c5848a3c8a4f51b562',['bit_parallel_representation','reduced_representation','bounded_exact_refinement'],[],None)],
'random_search':[
('07576bade1090738c52c',['dtype_specialization','vectorized_batch_kernel','sort_partition_reduction'],[],None),
('06b4e91e3e12c3594872',['native_backend_substitution','risk_aware_staging','sort_partition_reduction'],[],None),
('3f2d81a2ab49d84ac4d2',['dtype_specialization','early_certificate_exit','sparse_frontier_search'],[],None),
('4904ae971430bf5f6d77',['zero_copy_representation','early_certificate_exit','sort_partition_reduction'],[],None),
('0caec4dfee3c41a273d6',['native_backend_substitution','active_set_decomposition'],[],None),
('7a6b971d8a9c4d14a5dc',['zero_copy_representation','dtype_specialization'],[],None)],
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
        if learned=='precision_backend_error_budget':fn=direct32;impl='scipy_pairwise_float32_input'
        elif learned=='reduced_representation_refinement':fn=direct64;impl='scipy_pairwise_float64'
        elif 'dtype_specialization' in ops:fn=direct32;impl='scipy_pairwise_float32_input'
        elif 'native_backend_substitution' in ops or 'vectorized_batch_kernel' in ops:fn=direct64;impl='scipy_pairwise_float64'
        else:fn=reference_exact;impl='sklearn_kde_reference'
        built.append((name,fn));CANDIDATE_META[name]={'arm':arm,'rank':rank,'proposal_id':pid,'operators':ops,'transfer_ids':tids,'learned_template':learned,'implementation_class':impl,'mapping_policy':'PBEB/dtype specialization -> float32 inputs with native scipy cdist; RRR/native/vectorized -> float64 scipy pairwise kernel reduction; other operators -> conservative sklearn reference'}
    CANDIDATES_BY_ARM[arm]=built
if sum(len(v) for v in CANDIDATES_BY_ARM.values())!=30:raise RuntimeError('candidate budget mismatch')
