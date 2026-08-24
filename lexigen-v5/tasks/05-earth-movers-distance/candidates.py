from __future__ import annotations
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

def reference_exact(problem):
    a=np.asarray(problem['source_weights'],dtype=np.float64); b=np.asarray(problem['target_weights'],dtype=np.float64); M=np.ascontiguousarray(problem['cost_matrix'],dtype=np.float64)
    return {'transport_plan': ot.lp.emd(a,b,M,check_marginals=False)}

def assignment_exact(problem):
    a=np.asarray(problem['source_weights'],dtype=np.float64); b=np.asarray(problem['target_weights'],dtype=np.float64); M=np.asarray(problem['cost_matrix'],dtype=np.float64)
    n,m=M.shape
    if n!=m or len(a)!=n or len(b)!=m or n==0:
        return reference_exact(problem)
    if not (np.allclose(a,a[0],rtol=0,atol=1e-14) and np.allclose(b,b[0],rtol=0,atol=1e-14) and abs(float(a[0])-float(b[0]))<=1e-14):
        return reference_exact(problem)
    rows,cols=linear_sum_assignment(M)
    G=np.zeros((n,n),dtype=np.float64); G[rows,cols]=a[rows]
    return {'transport_plan': G}

PROPOSALS = {
    'v5_full': [
        (1, '3304c859d463a501bd86', ['bit_parallel_representation', 'sparse_frontier_search', 'early_certificate_exit'], ['TM-BFR-01'], 'bit_frontier_restriction'),
        (2, '41510e43e8fafb598496', ['reduced_representation', 'bounded_exact_refinement', 'risk_aware_staging'], ['TM-RRR-01'], 'reduced_representation_refinement'),
        (3, 'a6102573c9f355414229', ['active_set_decomposition', 'early_certificate_exit', 'risk_aware_staging'], ['TM-CAC-01'], 'certified_active_core'),
        (4, 'b1ef08a2d68a248c0821', ['dtype_specialization', 'native_backend_substitution', 'risk_aware_staging'], ['TM-PBEB-01'], 'precision_backend_error_budget'),
        (5, '4abf2b51384c560522e8', ['active_set_decomposition', 'bit_parallel_representation', 'bounded_exact_refinement'], [], None),
        (6, 'c50e493c5549a408f3e5', ['active_set_decomposition', 'bit_parallel_representation', 'sparse_frontier_search'], [], None),
    ],
    'v5_no_transfer': [
        (1, '91e027e622f2d9a98240', ['active_set_decomposition', 'bit_parallel_representation', 'bounded_exact_refinement'], [], None),
        (2, 'b2614109e1a5ccc10c14', ['active_set_decomposition', 'bit_parallel_representation', 'sparse_frontier_search'], [], None),
        (3, '20375ceceffce4d406a4', ['active_set_decomposition', 'bit_parallel_representation', 'reduced_representation'], [], None),
        (4, '4a4e1871b7f7b48b9485', ['active_set_decomposition', 'bit_parallel_representation', 'sort_partition_reduction'], [], None),
        (5, 'd69e86803f54c5a83d06', ['vectorized_batch_kernel', 'active_set_decomposition', 'bit_parallel_representation'], [], None),
        (6, '66c5848a3c8a4f51b562', ['bit_parallel_representation', 'reduced_representation', 'bounded_exact_refinement'], [], None),
    ],
    'random_search': [
        (1, '73cd4b0781edb40877c2', ['zero_copy_representation', 'bit_parallel_representation', 'bounded_exact_refinement'], [], None),
        (2, '17e71dc0e369bc902ae5', ['dtype_specialization', 'bounded_exact_refinement', 'sort_partition_reduction'], [], None),
        (3, '3168a87c49e98c886412', ['dtype_specialization', 'native_backend_substitution', 'active_set_decomposition'], [], None),
        (4, '614f3090e8375b79582f', ['vectorized_batch_kernel', 'early_certificate_exit', 'sparse_frontier_search'], [], None),
        (5, '7a22ae1a8866ee0b94da', ['zero_copy_representation', 'dtype_specialization', 'sparse_frontier_search'], [], None),
        (6, 'aba8f4e5dd68d9066783', ['contiguous_layout', 'dtype_specialization', 'reduced_representation'], [], None),
    ],
    'static_template': [
        (1, 'dbfcd2af539b0b2636e7', ['bit_parallel_representation', 'sparse_frontier_search'], [], None),
        (2, '8fd871e046faa7e4d37c', ['reduced_representation', 'bounded_exact_refinement'], [], None),
        (3, '820b1c309b6117eb268d', ['active_set_decomposition', 'early_certificate_exit'], [], None),
        (4, '8f1dafda0d3fbc099aa9', ['zero_copy_representation', 'vectorized_batch_kernel'], [], None),
        (5, '357e80313b8b9dc3cf36', ['contiguous_layout', 'vectorized_batch_kernel'], [], None),
        (6, 'd044a19fd4551034dc11', ['dtype_specialization', 'risk_aware_staging'], [], None),
    ],
    'v4_compatible': [
        (1, 'f9f3239b6866512e4f68', ['active_set_decomposition', 'bit_parallel_representation', 'bounded_exact_refinement'], [], None),
        (2, '9f5f55df04a5ad23f542', ['active_set_decomposition', 'bit_parallel_representation', 'sparse_frontier_search'], [], None),
        (3, 'ec4b9c17aaa3767d4f6d', ['active_set_decomposition', 'bit_parallel_representation', 'sort_partition_reduction'], [], None),
        (4, '7c30efb65d2c20ff8cc9', ['vectorized_batch_kernel', 'active_set_decomposition', 'bit_parallel_representation'], [], None),
        (5, '3df5ed91505aea4ed6cb', ['active_set_decomposition', 'sparse_frontier_search', 'bounded_exact_refinement'], [], None),
        (6, '0dde88a4a159a3ad0e40', ['bit_parallel_representation', 'bounded_exact_refinement', 'sort_partition_reduction'], [], None),
    ],
}

def _impl(operators):
    ops=set(operators)
    if 'reduced_representation' in ops or 'native_backend_substitution' in ops:
        return assignment_exact, 'uniform_assignment_exact'
    return reference_exact, 'pot_reference_exact'

CANDIDATES_BY_ARM={}
CANDIDATE_META={}
for arm,rows in PROPOSALS.items():
    out=[]
    for rank,pid,ops,tids,template in rows:
        fn,impl=_impl(ops); name=f'{arm}_r{rank}_{pid}'
        out.append((name,fn)); CANDIDATE_META[name]={'arm':arm,'rank':rank,'proposal_id':pid,'operators':ops,'transfer_ids':tids,'learned_template':template,'implementation_class':impl}
    CANDIDATES_BY_ARM[arm]=out
