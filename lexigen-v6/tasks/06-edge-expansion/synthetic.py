from __future__ import annotations

import hashlib,json,math,sys,time
from collections import defaultdict
from pathlib import Path

from candidates import ARM_ORDER,build_candidates,independent_definition_certificate,networkx_source_reference,verify_value

SOURCE_SHA256='24d194fbf8f604d318b9f330e61ad084ff4ea498de2c0a299835ad7ecce55d9a'
SIZES=(0,1,2,7,31,127,511,1023)


def _problem(n:int,mode:int)->dict:
    if n==0:return {'adjacency_list':[],'nodes_S':[]}
    adj=[]
    for u in range(n):
        if mode==0:
            vals={u,(u+1)%n,(u+3)%n,(u*7+5)%n}
            row=sorted(vals)
        elif mode==1:
            vals={(u+d*d+3*d)%n for d in range(1,min(9,n+1))}
            row=sorted(vals)
        else:
            base=[u,(u+1)%n,(u+1)%n,(u+5)%n,(u*11+3)%n]
            row=list(reversed(base))
        adj.append(row)
    if n==1:s=[0]
    elif mode==0:s=list(range(0,n,2)) or [0]
    elif mode==1:s=list(range(max(1,n//3)))
    else:s=sorted({0,n-1,*range(1,n,3)})
    if mode==2 and n==2:s=[0,1]
    return {'adjacency_list':adj,'nodes_S':s}


def cases()->list[dict]:
    out=[]
    for n in SIZES:
        for mode in range(3):out.append({'case_id':f'n{n}_m{mode}','n':n,'mode':mode,'problem':_problem(n,mode)})
    if len(out)!=24:raise RuntimeError('synthetic case count mismatch')
    return out


def main():
    source_path=Path(sys.argv[1] if len(sys.argv)>1 else 'task-source.py');source=source_path.read_bytes();sha=hashlib.sha256(source).hexdigest()
    if sha!=SOURCE_SHA256:raise RuntimeError(f'source identity mismatch {sha}')
    arms=build_candidates(source.decode('utf-8'));flat=[c for arm in ARM_ORDER for c in arms[arm]]
    if len(flat)!=31:raise RuntimeError(f'expected 31 candidates got {len(flat)}')
    plan=[]
    for c in flat:
        plan.append({'arm':c.arm,'candidate':c.name,'rank':c.rank,'proposal_id':c.proposal_id,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'learned_template':c.learned_template,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'baseline_id':c.baseline_id})
    rows=[];by_candidate=defaultdict(list)
    for item in cases():
        p=item['problem'];ref=networkx_source_reference(p);cert=independent_definition_certificate(p)
        rv=float(ref['edge_expansion']);cv=float(cert['edge_expansion'])
        if not math.isclose(rv,cv,rel_tol=1e-12,abs_tol=1e-12):raise RuntimeError(f'independent certificate disagrees with source reference on {item["case_id"]}: {rv} vs {cv}')
        for c in flat:
            t0=time.perf_counter_ns();error=None
            try:sol=c.solve(p)
            except Exception as exc:sol={};error=f'{type(exc).__name__}:{exc}'
            elapsed=time.perf_counter_ns()-t0
            valid,reason,metrics=verify_value(p,sol,reference=ref) if error is None else (False,'exception',{})
            row={'case_id':item['case_id'],'n':item['n'],'mode':item['mode'],'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'proposal_id':c.proposal_id,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns_diagnostic':elapsed,'reference_value':rv,'invalid_output_retries':0,'official_data_opened':False,**metrics}
            rows.append(row);by_candidate[c.name].append(row)
    eligible=[];candidate_summary=[]
    for c in flat:
        sub=by_candidate[c.name];valid=sum(bool(r['valid']) for r in sub)
        x={'arm':c.arm,'candidate':c.name,'valid':valid,'required':24,'eligible':valid==24,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'proposal_id':c.proposal_id,'transfer_ids':list(c.transfer_ids),'baseline_id':c.baseline_id}
        candidate_summary.append(x)
        if x['eligible']:eligible.append(c.name)
    per_arm={arm:sum(1 for x in candidate_summary if x['arm']==arm and x['eligible']) for arm in ARM_ORDER}
    if per_arm!={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}:raise RuntimeError(f'eligibility failure {per_arm}')
    Path('synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    Path('synthetic-candidate-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':6,'task':'edge_expansion','stage':'synthetic_r1','source_sha256':sha,'case_count':24,'candidate_count':31,'evaluation_count':len(rows),'eligible_count':len(eligible),'eligible_names':eligible,'eligible_by_arm':per_arm,'candidate_summary':candidate_summary,'timing_is_diagnostic_only':True,'official_train_manifest_opened':False,'official_test_manifest_opened':False,'official_payloads_opened':0,'public_task_specific_solvers_opened':False,'threshold_changes':False}
    Path('synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'case_count':24,'candidate_count':31,'evaluation_count':len(rows),'eligible_count':len(eligible),'eligible_by_arm':per_arm},indent=2))
if __name__=='__main__':main()
