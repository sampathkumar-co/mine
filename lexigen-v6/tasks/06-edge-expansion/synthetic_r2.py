from __future__ import annotations

import hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path

from candidates import ARM_ORDER,build_candidates,networkx_source_reference
from certificate_r2 import networkx35_executable_semantics_certificate,verify_r2
from synthetic import cases

SOURCE_SHA256='24d194fbf8f604d318b9f330e61ad084ff4ea498de2c0a299835ad7ecce55d9a'
EXPECTED_BY_ARM={'v6_full':6,'v6_no_transfer':6,'random_search':6,'static_template':6,'v5_compatible':6,'strong_baseline':1}


def main():
    source_path=Path(sys.argv[1] if len(sys.argv)>1 else 'task-source.py');source=source_path.read_bytes();sha=hashlib.sha256(source).hexdigest()
    if sha!=SOURCE_SHA256:raise RuntimeError(f'source identity mismatch {sha}')
    arms=build_candidates(source.decode('utf-8'))
    if {k:len(v) for k,v in arms.items()}!=EXPECTED_BY_ARM:raise RuntimeError('candidate identities/counts changed from R1')
    flat=[c for arm in ARM_ORDER for c in arms[arm]]
    rows=[];by_candidate=defaultdict(list);certificate_checks=[]
    for item in cases():
        p=item['problem'];ref=networkx_source_reference(p);cert=networkx35_executable_semantics_certificate(p)
        rv=float(ref['edge_expansion']);cv=float(cert['edge_expansion'])
        certificate_checks.append({'case_id':item['case_id'],'reference':rv,'certificate':cv,'match':abs(rv-cv)<=1e-12})
        if abs(rv-cv)>1e-12:raise RuntimeError(f'R2 certificate still disagrees with frozen NetworkX reference on {item["case_id"]}: {rv} vs {cv}')
        for c in flat:
            error=None;t0=time.perf_counter_ns()
            try:sol=c.solve(p)
            except Exception as exc:sol={};error=f'{type(exc).__name__}:{exc}'
            elapsed=time.perf_counter_ns()-t0
            valid,reason,metrics=verify_r2(p,sol,ref) if error is None else (False,'exception',{})
            row={'case_id':item['case_id'],'n':item['n'],'mode':item['mode'],'arm':c.arm,'candidate':c.name,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'proposal_id':c.proposal_id,'operators':list(c.operators),'transfer_ids':list(c.transfer_ids),'baseline_id':c.baseline_id,'valid':bool(valid and error is None),'failure_reason':error or reason,'candidate_ns_diagnostic':elapsed,'invalid_output_retries':0,'candidate_solver_revision':'R1_UNCHANGED','certificate_revision':'R2_NETWORKX35_SEMANTICS','official_data_opened':False,**metrics}
            rows.append(row);by_candidate[c.name].append(row)
    candidate_summary=[];eligible=[]
    for c in flat:
        sub=by_candidate[c.name];valid=sum(bool(r['valid']) for r in sub)
        x={'arm':c.arm,'candidate':c.name,'valid':valid,'required':24,'eligible':valid==24,'implementation_class':c.implementation_class,'semantic_implementation_key':c.semantic_implementation_key,'proposal_id':c.proposal_id,'transfer_ids':list(c.transfer_ids),'baseline_id':c.baseline_id,'first_failure':next((r['case_id'] for r in sub if not r['valid']),None)}
        candidate_summary.append(x)
        if x['eligible']:eligible.append(c.name)
    eligible_by_arm={arm:sum(1 for x in candidate_summary if x['arm']==arm and x['eligible']) for arm in ARM_ORDER}
    training_ready=all(eligible_by_arm[a]>=1 for a in ARM_ORDER)
    Path('synthetic-r2-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    summary={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':6,'task':'edge_expansion','stage':'synthetic_r2_non_rescuing_semantic_audit','source_sha256':sha,'r1_failure_run_id':32935225520,'candidate_solver_blob_sha1':'67d5fe2269ce48c7ada57eda6b042ac2f174f2f5','candidate_solvers_changed_from_r1':False,'certificate_revision_only':True,'case_count':24,'candidate_count':31,'evaluation_count':len(rows),'certificate_reference_checks':certificate_checks,'eligible_count':len(eligible),'eligible_names':eligible,'eligible_by_arm':eligible_by_arm,'candidate_summary':candidate_summary,'official_training_ready':training_ready,'arm_collapse':[a for a,n in eligible_by_arm.items() if n==0],'timing_is_diagnostic_only':True,'official_train_manifest_opened':False,'official_test_manifest_opened':False,'official_payloads_opened':0,'public_task_specific_solvers_opened':False,'threshold_changes':False}
    Path('synthetic-r2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'evaluation_count':len(rows),'eligible_count':len(eligible),'eligible_by_arm':eligible_by_arm,'official_training_ready':training_ready,'arm_collapse':summary['arm_collapse']},indent=2))
if __name__=='__main__':main()
