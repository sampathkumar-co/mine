from __future__ import annotations
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=("V20_PRECOMMIT.json","V20_TRANSFER_SCAN.json","V20_EVIDENCE.json","EVIDENCE.md","scan_transfer_v20.py","verify_v20_rejection.py","V20_REPRODUCTION.json","V20_INTEGRITY_AUDIT.json","freeze_v20_integrity.py")
def load(name): return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def req(value,message):
    if not value: raise RuntimeError(message)
def main():
    pre=load('V20_PRECOMMIT.json'); scan=load('V20_TRANSFER_SCAN.json'); repro=load('V20_REPRODUCTION.json'); audit=load('V20_INTEGRITY_AUDIT.json')
    req(pre['fixed_gates']==list(range(14,24)),'fixed gates changed'); req(pre['production_sha256']=='751e258d9dbdcdf2641503bacdc32f927fb2872fddaadbbbe348d668f5f25523','production changed')
    req(scan['gates_checked']==10 and scan['total_exact_survivors']==0,'scan result changed'); req(not scan['hidden_outputs_opened'],'hidden outputs flag')
    req(repro['package_hashes_verified']==10 and repro['byte_equivalent_reproduction'],'reproduction failed'); req(repro['argument_pairs_checked']==274,'argument denominator changed'); req(repro['runtime_invalid_pairs']==254,'invalid denominator changed')
    req(audit['audit_verdict']=='reproduced_negative_transfer_result','audit weakened'); req(not audit['transfer_demonstrated'] and not audit['v20_pass'] and not audit['world_level_breakthrough'],'claim weakened')
    evidence={'schema':'lexigen-v20-final-integrity-evidence-v1','base_v19r4_commit':pre['base_v19r4_commit'],'precommit_commit':'17b96af07f726c202cede3a54fba70ecd6da3cbc','negative_result_commit':'6b24fd0f74c02448eefb22456d3b8fc5c7549f24','production_sha256':pre['production_sha256'],'fixed_gates':pre['fixed_gates'],'package_hashes_verified':10,'gates_checked':10,'argument_pairs_checked':274,'runtime_invalid_pairs':254,'total_exact_survivors':0,'hidden_outputs_opened':False,'transfer_demonstrated':False,'v20_pass':False,'world_level_breakthrough':False,'files':{name:sha(HERE/name) for name in FILES},'architectural_verdict':audit['architectural_verdict']}
    out=HERE/'V20_FINAL_EVIDENCE.json'; out.write_bytes((json.dumps(evidence,indent=2,sort_keys=True)+'\n').encode()); digest=sha(out)
    (HERE/'INTEGRITY.md').write_text('# Lexigen v20 audited negative transfer evidence\n\n- Fixed gates: **10**\n- Package hashes verified: **10/10**\n- Argument pairs reproduced: **274**\n- Exact survivors: **0**\n- Hidden outputs opened: **no**\n\nThe exact v19r4 production did not transfer. v20 failed, and the architecture must be redesigned rather than repeatedly resampling tasks.\n\nEvidence JSON SHA-256: `'+digest+'`\n',encoding='utf-8',newline='\n')
    print(json.dumps({'evidence_sha256':digest},sort_keys=True))
if __name__=='__main__': main()
