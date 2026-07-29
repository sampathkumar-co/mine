from __future__ import annotations
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=('README.md','V19R3_PRECOMMIT.json','V19R3_ATTEMPT_REPORT.json','V19R3_INTEGRITY_AUDIT.json','enumerate_v19r3.py','validate_v19r3.py','test_v19r3.py','candidates/gate-10-selected-production.json','candidates/gate-10-selected-arguments.json','candidates/gate-10-selected-program.json')
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(v,m):
    if not v: raise RuntimeError(m)
def main():
    report=load(HERE/'V19R3_ATTEMPT_REPORT.json'); audit=load(HERE/'V19R3_INTEGRITY_AUDIT.json'); gate=next(x for x in report['outcomes'] if x['gate']==10)
    req(report['fixed_gates']==3,'gates'); req(report['candidate_denominator_per_gate']==250000,'denominator'); req(report['unique_productions']==0,'unique'); req(report['ambiguous_program_sets']==1,'ambiguous'); req(report['no_program_failures']==2,'failures'); req(not report['fresh_scoring_performed'],'fresh'); req(gate['search']['exact_survivors']==5,'survivors'); req(gate['search']['selected_descriptor_removal_survivors']==4,'ablation'); req(audit['audit_verdict']=='failed_unique_production_gate','audit')
    evidence={'schema':'lexigen-v19r3-frozen-negative-evidence-v1','base_v19r2_commit':'7a662e0f2bcfc07fcd955e865b253ff6da9cc749','fixed_gates':[7,10,13],'candidate_compositions_per_gate':250000,'total_candidate_compositions':750000,'no_program_gates':[7,13],'ambiguous_gate':10,'exact_survivors':5,'selected_descriptor_removal_survivors':4,'fresh_scoring_performed':False,'hidden_outputs_opened':False,'v19_pass':False,'world_level_breakthrough':False,'files':{name:sha(HERE/name) for name in FILES},'claim_boundary':{'reason':audit['ambiguity_source']}}
    out=HERE/'V19R3_EVIDENCE.json'; out.write_bytes((json.dumps(evidence,indent=2,sort_keys=True)+'\n').encode()); digest=sha(out); (HERE/'EVIDENCE.md').write_text(f'# v19r3 frozen negative evidence\n\n- Fixed gates: **3**\n- Complete compositions evaluated: **750,000**\n- No-program gates: **2**\n- Ambiguous gate: **1**, with **5** exact survivors\n- Fresh scoring: **not performed**\n\nThe selected descriptor failed removal ablation because four other complete compositions remained exact. v19r3 is not a v19 pass or a breakthrough.\n\nEvidence JSON SHA-256: `{digest}`\n',encoding='utf-8',newline='\n'); print(json.dumps({'evidence_sha256':digest},sort_keys=True))
if __name__=='__main__': main()
