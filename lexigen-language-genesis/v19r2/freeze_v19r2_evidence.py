from __future__ import annotations
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE_REJECTED_V19="eb15da3373817ffa69633fb6cd423b2fb06d0f60"
NUMERICAL_CHECKPOINT="5e3d82b89078f6ee3c7b4e5d364b2071571cf726"
FROZEN_V17="cd89382e38b45d12916e662af052a7aa1a374896"
ARCGEN="a15cbdb44c776610aeeb9f487a06af875d3d0878"
VISIBLE_EVIDENCE="13cd271a2d4813001563842a51a1e72dd100aa1a"
PRODUCTION_SHA="39a2528b370b899e4909e22698d47aa7414aa3db6092e75e64cff384d710f8df"
CODE_FILES=("runtime_v19r2.py","portable_runtime_v19r2.py","invent_v19r2.py","validate_v19r2.py","test_v19r2.py","mutations_v19r2.py","verifier_grammar_v19r2.py","portable_verifier_v19r2.py","cosynthesize_verifier_v19r2.py","validate_verifier_v19r2.py","test_verifier_v19r2.py","freeze_v19r2_evidence.py")
ARTIFACTS=("README.md","V19R2_PRECOMMIT.json","V19R2_SMOKE_REPORT.json","V19R2_INTERMEDIATE_REPORT.json","V19R2_REPORT.json","V19R2_VERIFIER_SMOKE_REPORT.json","V19R2_VERIFIER_REPORT.json","V19R2_INTEGRITY_AUDIT.json","productions/gate-10-production.json","productions/gate-10-arguments.json","productions/gate-10-concrete.json","contracts/gate-10-contract.json","contracts/revisions/gate-10-contract-r0.json","contracts/revisions/gate-10-false-accepts-r0.json")
def load(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value:bool,message:str):
    if not value: raise RuntimeError(message)
def main()->None:
    report=load(HERE/'V19R2_REPORT.json'); verifier=load(HERE/'V19R2_VERIFIER_REPORT.json'); audit=load(HERE/'V19R2_INTEGRITY_AUDIT.json'); contract=load(HERE/'contracts'/'gate-10-contract.json')
    require(report['gates_fixed']==3,'gate denominator changed'); require(report['productions_found']==1,'production count changed'); require(report['no_program_failures']==2,'no-program count changed'); require(not report['hidden_outputs_opened'],'hidden outputs flag')
    outcomes={int(x['gate']):x for x in report['outcomes']}; require(outcomes[7]['status']==outcomes[13]['status']=='no_program','negative gates changed'); gate=outcomes[10]
    require(gate['production_sha256']==PRODUCTION_SHA,'production identity changed'); require(gate['accepted_fresh_cases']==10000,'fresh denominator changed'); require(gate['fresh_primary_exact']==gate['fresh_portable_exact']==gate['fresh_runtime_agreement']==10000,'fresh gate mismatch'); require(gate['generator_rejections']==0,'generator rejections changed'); require(gate['frozen_v17_ablation_failed'],'v17 ablation changed')
    require(verifier['production_sha256']==PRODUCTION_SHA,'verifier production mismatch'); require(verifier['accepted_cases']==1000,'verifier correct denominator'); require(verifier['mutant_cases']==8000,'verifier mutant denominator')
    for key in ('screening_rejected_primary','screening_rejected_portable','soundness_rejected_primary','soundness_rejected_portable'): require(verifier[key]==8000,key+' changed')
    require(not verifier['exact_digest_used_by_learned_screen'],'learned exact digest'); require(not contract['exact_digest_used'],'contract exact digest'); require(all(x['name']!='exact_digest' for x in contract['predicates']),'exact predicate present')
    require(audit['audit_verdict']=='partial_mechanism_not_v19_pass','integrity verdict weakened'); require(not audit['autonomous_grammar_production_invention'],'autonomy claim weakened'); require(not audit['world_level_breakthrough'],'world claim weakened')
    evidence={'schema':'lexigen-v19r2-frozen-evidence-v1','version':'19r2','base_rejected_v19_commit':BASE_REJECTED_V19,'numerical_checkpoint_commit':NUMERICAL_CHECKPOINT,'frozen_v17_commit':FROZEN_V17,'arcgen_commit':ARCGEN,'visible_evidence_commit':VISIBLE_EVIDENCE,'fixed_development_gates':[7,10,13],'production_found_gates':[10],'no_program_gates':[7,13],'production_sha256':PRODUCTION_SHA,'production_file_sha256':sha(HERE/'productions'/'gate-10-production.json'),'concrete_program_sha256':gate['concrete_program_sha256'],'arguments_sha256':gate['arguments_sha256'],'fresh_cases':10000,'fresh_primary_exact':10000,'fresh_portable_exact':10000,'fresh_runtime_agreement':10000,'generator_rejections':0,'frozen_v17_ablation_failed':True,'verifier':{'correct_outputs':1000,'mutant_outputs':8000,'primary_screening_failures':0,'portable_screening_failures':0,'primary_soundness_failures':0,'portable_soundness_failures':0,'learned_exact_digest':False,'contract_sha256':verifier['contract_sha256'],'contract_revision':contract['revision'],'predicates':contract['predicates'],'training_runtime_invalid_mutations':contract['training_runtime_invalid_mutations'],'fresh_runtime_invalid_mutations':verifier['fresh_runtime_invalid_mutations']},'files':{name:sha(HERE/name) for name in (*CODE_FILES,*ARTIFACTS)},'claim_boundary':{'automated_argument_selection_and_abstraction':True,'autonomous_full_production_invention':False,'sealed_external_success':False,'transfer_demonstrated':False,'world_level_breakthrough':False,'reason':audit['reason']}}
    out=HERE/'V19R2_EVIDENCE.json'; out.write_bytes((json.dumps(evidence,indent=2,sort_keys=True)+'\n').encode('utf-8')); digest=sha(out)
    markdown=("# v19r2 frozen evidence\n\n"
      "- Fixed development gates: **3**\n"
      "- Executable production found: **1** (gate 10)\n"
      "- Preserved no-program failures: **2** (gates 7 and 13)\n"
      "- Fresh dual-runtime cases: **10,000 / 10,000 exact**\n"
      "- Verifier correct outputs: **1,000 / 1,000 accepted in both implementations**\n"
      "- Mutant outputs: **8,000 / 8,000 rejected by both learned screens and soundness anchors**\n"
      "- Learned exact-digest predicate: **no**\n\n"
      "## Integrity verdict\n\n"
      "This is a strong automated argument-selection, executable-abstraction, dual-runtime, and verifier-co-synthesis result. It is not autonomous full production invention because the exact affine composition skeleton was human-authored after demonstration analysis. It is not transfer evidence, sealed external success, or a world-level breakthrough.\n\n"
      f"Evidence JSON SHA-256: `{digest}`\n")
    (HERE/'EVIDENCE.md').write_bytes(markdown.encode('utf-8')); print(json.dumps({'evidence_sha256':digest},sort_keys=True))
if __name__=='__main__': main()
