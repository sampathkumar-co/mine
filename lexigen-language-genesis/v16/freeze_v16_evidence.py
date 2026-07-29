from __future__ import annotations
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
FILES=('mutations_v16.py','verifier_grammar_v16.py','portable_verifier_v16.py','cosynthesize_verifier_v16.py','validate_v16.py','V16_REPORT.json')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'V16_REPORT.json').read_text(encoding='utf-8'))
assert r['families']==9 and r['correct_outputs_checked']==900
assert r['mutant_outputs_checked']==6969
for k in ('screening_rejections','portable_screening_rejections','soundness_rejections','portable_soundness_rejections'): assert r[k]==6969
assert r['contracts_using_exact_digest']==0
manifest={'schema':'lexigen-v16-frozen-evidence-v1','version':16,'base_v15_commit':'4a8d1bf303951b1c4de704eada76bb95ffc89b77','families':9,'correct_outputs_checked':900,'mutant_outputs_checked':6969,'contracts_requiring_revision':r['contracts_requiring_revision'],'screening_contracts_using_exact_digest':0,'mandatory_soundness_anchor':'exact_digest','world_level_breakthrough':False,'claim_boundary':'Verifier grammar and scene IR atoms remain human supplied; this is verifier co-synthesis, not autonomous primitive invention.','files':{n:sha(HERE/n) for n in FILES}}
p=HERE/'V16_EVIDENCE.json'; p.write_bytes((json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode())
h=sha(p)
(HERE/'EVIDENCE.md').write_bytes((f'# v16 frozen evidence\n\n- Families: **9**\n- Correct outputs verified in both runtimes: **900**\n- Mutant outputs rejected by learned screening and mandatory soundness anchor: **6,969**\n- Contracts requiring CEGIS revision: **{r["contracts_requiring_revision"]}**\n- Learned screening contracts using exact digest: **0**\n\nThe exact digest is a mandatory final soundness anchor, not a learned screening predicate. v16 is not autonomous primitive invention and not a world-level breakthrough.\n\nEvidence JSON SHA-256: `{h}`\n').encode())
print(h)
