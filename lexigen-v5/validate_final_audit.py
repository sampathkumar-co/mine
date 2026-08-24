from __future__ import annotations
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
AUDIT=ROOT/'CAMPAIGN_FINAL_AUDIT_R1.json'
TASK_ROOT=ROOT/'tasks'
ARMS=('v5_full','v5_no_transfer','random_search','static_template','v4_compatible')
def clean(r):return bool(r.get('clean_unseen_win',r.get('clean_unseen_task_win',r.get('v5_full_passed_blind_gate',False))))
def passed(node):
    if not isinstance(node,dict):return False
    return bool(node.get('passes_blind_gate',node.get('passes_gate',False)))
def git_blob(path):return subprocess.check_output(['git','hash-object',str(path)],text=True).strip()
def main():
    audit=json.loads(AUDIT.read_text())
    prov=audit['task_result_provenance']
    if len(prov)!=10 or [int(x['index']) for x in prov]!=list(range(1,11)):raise RuntimeError('task provenance coverage mismatch')
    task_files=sorted(TASK_ROOT.glob('*/TASK_RESULT.json'))
    if len(task_files)!=10:raise RuntimeError(f'expected 10 task results, got {len(task_files)}')
    by_index={int(json.loads(p.read_text())['task_index']):(p,json.loads(p.read_text())) for p in task_files}
    full=0;v5v4=0;causal=0;families=set();causal_families=set();causal_ids=set()
    for row in prov:
        i=int(row['index']);p,r=by_index[i]
        if git_blob(p)!=row['task_result_git_blob_sha1']:raise RuntimeError(f'task {i} blob mismatch')
        c=clean(r);v=bool(r.get('v5_over_v4_task_win',False));cw=bool(r.get('causal_transfer_win',False))
        if c!=bool(row['clean_win']) or v!=bool(row['v5_over_v4']) or cw!=bool(row['causal_win']):raise RuntimeError(f'task {i} outcome mismatch')
        full+=int(c);v5v4+=int(v);causal+=int(cw)
        if c:families.add(str(r['family']))
        if cw:
            causal_families.add(str(r['family']))
            cid=r.get('learned_causal_id') or row.get('causal_id')
            if cid:causal_ids.add(str(cid))
    wins={a:0 for a in ARMS}
    # Prefer blind evidence for per-arm task wins. Denominator-negative tasks have no blind evidence and therefore no arm win.
    for p in sorted(TASK_ROOT.glob('*/BLIND_R1_RESULT.json')):
        r=json.loads(p.read_text())
        arms=r.get('arms')
        if isinstance(arms,dict):
            for a in ARMS:wins[a]+=int(passed(arms.get(a)))
        else:
            for a in ARMS:wins[a]+=int(passed(r.get(a)))
    # Some task result branches use BLIND_R1_RESULT while others seal the same fields in TASK_RESULT; ensure full count is denominator-authoritative.
    wins['v5_full']=full
    expected={'v5_full':2,'v5_no_transfer':1,'random_search':1,'static_template':1,'v4_compatible':0}
    if wins!=expected:raise RuntimeError(f'control win count mismatch: {wins}')
    if audit['task_win_counts']!=expected:raise RuntimeError('audit task_win_counts mismatch')
    if full!=2 or v5v4!=2 or causal!=1:raise RuntimeError((full,v5v4,causal))
    if families!={'machine_learning','combinatorial'}:raise RuntimeError(f'winning families mismatch {families}')
    if causal_families!={'combinatorial'} or causal_ids!={'TM-BFR-01'}:raise RuntimeError('causal diversity mismatch')
    checks={x['requirement']:bool(x['pass']) for x in audit['gate_checks']}
    if audit['campaign_gate_passed'] is not False:raise RuntimeError('campaign gate must fail')
    if checks['v5_full beats v4_compatible by >=2 task wins'] is not True:raise RuntimeError('v4 margin gate expected pass')
    if any(checks[k] for k in ['v5_full clean wins >= 6/10','clean wins span >= 6 families','v5_full beats v5_no_transfer by >=2 task wins','v5_full beats v5_no_transfer by >=20 percentage points','v5_full beats random_search by >=2 task wins','v5_full beats static_template by >=2 task wins','causal transfer wins >=2','causal wins span >=2 current holdout families','causal wins use >=2 distinct learned-template causal IDs']):raise RuntimeError('one or more frozen failure gates were incorrectly marked passed')
    out={'status':'validated','task_count':10,'task_win_counts':wins,'clean_wins':full,'winning_families':sorted(families),'v5_over_v4_task_wins':v5v4,'causal_transfer_wins':causal,'causal_families':sorted(causal_families),'causal_ids':sorted(causal_ids),'campaign_gate_passed':False,'task10_causal_result_preserved':True}
    Path('final-audit-validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
