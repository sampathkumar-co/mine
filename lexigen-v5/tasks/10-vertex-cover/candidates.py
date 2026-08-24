from __future__ import annotations
from pysat.card import CardEnc,EncType
from pysat.formula import CNF,WCNF
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2

def _validate_problem(problem):
    if not isinstance(problem,list):raise ValueError('problem must be adjacency list')
    n=len(problem)
    if any(not isinstance(r,list) or len(r)!=n for r in problem):raise ValueError('adjacency must be square')
    return n

def reference_exact(problem):
    n=_validate_problem(problem)
    def sat_for(k):
        cnf=CNF()
        for i in range(n):
            for j in range(i+1,n):
                if problem[i][j]==1:cnf.append([i+1,j+1])
        atmost=CardEnc.atmost(lits=[i+1 for i in range(n)],bound=k,encoding=EncType.seqcounter)
        cnf.extend(atmost.clauses)
        with Solver(name='Minicard') as solver:
            solver.append_formula(cnf);sat=solver.solve();model=solver.get_model() if sat else None
        if not model:return None
        return [i for i in range(n) if model[i]>0]
    z=sat_for(0)
    if z is not None:return z
    left=0;selected=list(range(n));right=n
    while right>left:
        if right==left+1:return selected
        mid=(right+left)//2;s=sat_for(mid)
        if s is not None:selected=s;right=len(s)
        else:left=mid
    return selected

def _adj_masks(problem):
    n=_validate_problem(problem);out=[0]*n
    for i,row in enumerate(problem):
        m=0
        for j,x in enumerate(row):
            if x:m|=1<<j
        out[i]=m & ~(1<<i)
    return out

def _greedy_independent(cand,adj):
    chosen=0
    while cand:
        tmp=cand;v=None;bestdeg=10**18
        while tmp:
            b=tmp&-tmp;i=b.bit_length()-1;d=(adj[i]&cand).bit_count()
            if d<bestdeg:v=i;bestdeg=d
            tmp^=b
        b=1<<v;chosen|=b;cand &= ~b & ~adj[v]
    return chosen

def learned_bit_frontier_exact(problem):
    n=_validate_problem(problem)
    if n==0:return []
    adj=_adj_masks(problem);all_bits=(1<<n)-1
    best=_greedy_independent(all_bits,adj);best_n=best.bit_count()
    def dfs(cand,chosen,chosen_n):
        nonlocal best,best_n
        if not cand:
            if chosen_n>best_n:best,best_n=chosen,chosen_n
            return
        if chosen_n+cand.bit_count()<=best_n:return
        tmp=cand;v=None;maxdeg=-1
        while tmp:
            b=tmp&-tmp;i=b.bit_length()-1;d=(adj[i]&cand).bit_count()
            if d>maxdeg:v=i;maxdeg=d
            tmp^=b
        b=1<<v
        dfs(cand & ~b & ~adj[v],chosen|b,chosen_n+1)
        dfs(cand & ~b,chosen,chosen_n)
    dfs(all_bits,0,0)
    return [i for i in range(n) if not ((best>>i)&1)]

def _frontier_probe(problem):
    n=_validate_problem(problem)
    if n==0:return []
    adj=_adj_masks(problem);ind=_greedy_independent((1<<n)-1,adj)
    return [i for i in range(n) if not ((ind>>i)&1)]

def frontier_then_reference(problem):
    _frontier_probe(problem)
    return reference_exact(problem)

def isolated_core_reference(problem):
    n=_validate_problem(problem)
    active=[i for i in range(n) if any(problem[i][j] for j in range(n) if j!=i)]
    if len(active)==n:return reference_exact(problem)
    if not active:return []
    core=[[problem[i][j] for j in active] for i in active]
    sol=reference_exact(core)
    return [active[i] for i in sol]

def rc2_exact(problem):
    n=_validate_problem(problem);w=WCNF()
    for i in range(n):
        for j in range(i+1,n):
            if problem[i][j]==1:w.append([i+1,j+1])
    for i in range(n):w.append([-(i+1)],weight=1)
    with RC2(w) as rc2:model=rc2.compute()
    pos={lit for lit in model if 1<=lit<=n}
    return [i for i in range(n) if i+1 in pos]

PROPOSALS={
'v5_full':[
(1,'3304c859d463a501bd86',['bit_parallel_representation','sparse_frontier_search','early_certificate_exit'],['TM-BFR-01'],'bit_frontier_restriction'),
(2,'41510e43e8fafb598496',['reduced_representation','bounded_exact_refinement','risk_aware_staging'],['TM-RRR-01'],'reduced_representation_refinement'),
(3,'a6102573c9f355414229',['active_set_decomposition','early_certificate_exit','risk_aware_staging'],['TM-CAC-01'],'certified_active_core'),
(4,'e909b567bac8aa01b86e',['bit_parallel_representation','reduced_representation','bounded_exact_refinement'],[],None),
(5,'2c6961f9ca6711ce3a3f',['bit_parallel_representation','sparse_frontier_search','reduced_representation'],[],None),
(6,'7653e3865aa7a6def4dc',['bit_parallel_representation','sparse_frontier_search','bounded_exact_refinement'],[],None)],
'v5_no_transfer':[
(1,'66c5848a3c8a4f51b562',['bit_parallel_representation','reduced_representation','bounded_exact_refinement'],[],None),
(2,'b93eda021fe3bc5d89cb',['bit_parallel_representation','sparse_frontier_search','reduced_representation'],[],None),
(3,'d14c06bd6ae45a8dd009',['bit_parallel_representation','sparse_frontier_search','bounded_exact_refinement'],[],None),
(4,'2c6f67fc6c6a0adc20f9',['vectorized_batch_kernel','bit_parallel_representation','reduced_representation'],[],None),
(5,'14a8ffbc5159ff111ac9',['vectorized_batch_kernel','bit_parallel_representation','bounded_exact_refinement'],[],None),
(6,'477905d60634240ebda9',['vectorized_batch_kernel','bit_parallel_representation','sparse_frontier_search'],[],None)],
'random_search':[
(1,'7cb58654becb58e60790',['native_backend_substitution','risk_aware_staging','bit_parallel_representation'],[],None),
(2,'1d3035e256f80c907182',['contiguous_layout','sparse_frontier_search','bounded_exact_refinement'],[],None),
(3,'2ef21250df83098c75bd',['dtype_specialization','vectorized_batch_kernel','early_certificate_exit'],[],None),
(4,'87482658f5cb465531e7',['contiguous_layout','early_certificate_exit','reduced_representation'],[],None),
(5,'3bc5a8394ff6f6de8859',['dtype_specialization','native_backend_substitution','early_certificate_exit'],[],None),
(6,'2ce87a129edc8c085c79',['zero_copy_representation','contiguous_layout','dtype_specialization'],[],None)],
'static_template':[
(1,'dbfcd2af539b0b2636e7',['bit_parallel_representation','sparse_frontier_search'],[],None),
(2,'8fd871e046faa7e4d37c',['reduced_representation','bounded_exact_refinement'],[],None),
(3,'820b1c309b6117eb268d',['active_set_decomposition','early_certificate_exit'],[],None),
(4,'8f1dafda0d3fbc099aa9',['zero_copy_representation','vectorized_batch_kernel'],[],None),
(5,'357e80313b8b9dc3cf36',['contiguous_layout','vectorized_batch_kernel'],[],None),
(6,'d044a19fd4551034dc11',['dtype_specialization','risk_aware_staging'],[],None)],
'v4_compatible':[
(1,'bd9a928b0a959b433de2',['bit_parallel_representation','sparse_frontier_search','bounded_exact_refinement'],[],None),
(2,'885bf4f21e819b330732',['vectorized_batch_kernel','bit_parallel_representation','bounded_exact_refinement'],[],None),
(3,'695b907772d8a69a1186',['vectorized_batch_kernel','bit_parallel_representation','sparse_frontier_search'],[],None),
(4,'d9863922b850e9717a05',['risk_aware_staging','bit_parallel_representation','bounded_exact_refinement'],[],None),
(5,'cdae8cbf0d73bd4d047c',['vectorized_batch_kernel','sparse_frontier_search','bounded_exact_refinement'],[],None),
(6,'af7d36f83a386b7726b9',['risk_aware_staging','bit_parallel_representation','sparse_frontier_search'],[],None)]}

def _implementation(arm,ops,transfer_ids):
    s=set(ops)
    if transfer_ids==['TM-BFR-01'] and {'bit_parallel_representation','sparse_frontier_search','early_certificate_exit'}<=s:return 'learned_bit_frontier_exact',learned_bit_frontier_exact
    if 'native_backend_substitution' in s:return 'rc2_maxsat_exact',rc2_exact
    if 'active_set_decomposition' in s or {'reduced_representation','bounded_exact_refinement'}<=s:return 'isolated_core_plus_reference',isolated_core_reference
    if {'bit_parallel_representation','sparse_frontier_search'}<=s:return 'frontier_probe_plus_reference',frontier_then_reference
    return 'pysat_minicard_reference',reference_exact

CANDIDATES_BY_ARM={};CANDIDATE_META={}
for arm,rows in PROPOSALS.items():
    out=[]
    for rank,pid,ops,tids,tmpl in rows:
        name=f'{arm}_r{rank}_{pid}';impl,fn=_implementation(arm,ops,tids);out.append((name,fn));CANDIDATE_META[name]={'arm':arm,'rank':rank,'operators':ops,'transfer_ids':tids,'learned_template':tmpl,'implementation_class':impl}
    CANDIDATES_BY_ARM[arm]=out
