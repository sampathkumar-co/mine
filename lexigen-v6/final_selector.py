from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path

SEED='LEXIGEN-V6-FINAL-APPLICABLE-2026-08-24-A'
FINAL_COUNT=8;MIN_FAMILIES=5;MAX_PER_FAMILY=2

def applicability_score(row):
    ids=','.join(sorted(str(x) for x in row['applicable_causal_ids']))
    payload=f"{SEED}\0{row['task']}\0{row['source_sha256']}\0{ids}"
    return hashlib.sha256(payload.encode()).hexdigest()

def validate(rows):
    names=[r['task'] for r in rows]
    if len(names)!=24 or len(set(names))!=24:raise RuntimeError('expected frozen 24-task screening denominator')
    for r in rows:
        needed={'task','family','frozen_selection_score','source_sha256','applicable_causal_ids','training_manifest_opened','test_manifest_opened','public_solvers_opened'}
        if not needed<=set(r):raise RuntimeError(f'missing screening fields {r.get("task")}')
        if r['training_manifest_opened'] or r['test_manifest_opened'] or r['public_solvers_opened']:raise RuntimeError('screening boundary crossed')

def select(rows):
    validate(rows)
    applicable=[]
    all_ids=set();families=set()
    for r in rows:
        ids=sorted(set(str(x) for x in r['applicable_causal_ids']))
        # Different-family applicability must already have been enforced by the frozen screening workflow.
        if ids:
            q=dict(r);q['applicable_causal_ids']=ids;q['applicability_score']=applicability_score(q);applicable.append(q);all_ids.update(ids);families.add(q['family'])
    coverage={'screening_pool_count':len(rows),'applicable_count':len(applicable),'applicable_families':sorted(families),'applicable_causal_ids':sorted(all_ids),'count_gate':len(applicable)>=8,'family_gate':len(families)>=5,'causal_id_gate':len(all_ids)>=2}
    if not all((coverage['count_gate'],coverage['family_gate'],coverage['causal_id_gate'])):return coverage,[],False
    ordered=sorted(applicable,key=lambda r:(r['applicability_score'],r['frozen_selection_score'],r['task']))
    def possible(i,c):return len(set(c)|{r['family'] for r in ordered[i:] if c[r['family']]<MAX_PER_FAMILY})
    def rec(i,chosen,c):
        if len(chosen)==FINAL_COUNT:return chosen[:] if len(c)>=MIN_FAMILIES else None
        if len(ordered)-i<FINAL_COUNT-len(chosen) or possible(i,c)<MIN_FAMILIES:return None
        for pos in range(i,len(ordered)):
            r=ordered[pos];f=r['family']
            if c[f]>=MAX_PER_FAMILY:continue
            chosen.append(r);c[f]+=1
            z=rec(pos+1,chosen,c)
            if z is not None:return z
            c[f]-=1
            if c[f]==0:del c[f]
            chosen.pop()
        return None
    chosen=rec(0,[],Counter())
    ok=chosen is not None
    if not ok:chosen=[]
    coverage['final_selection_diversity_gate']=ok
    return coverage,chosen,ok

def main():
    p=Path('screening-source-evidence.jsonl')
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    coverage,chosen,ok=select(rows)
    report={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','stage':'applicability_conditioned_final_selection','selection_seed':SEED,'coverage':coverage,'final_count_required':FINAL_COUNT,'minimum_final_families':MIN_FAMILIES,'maximum_per_final_family':MAX_PER_FAMILY,'selected':chosen,'gate_passed':ok,'training_manifests_opened':False,'test_manifests_opened':False,'public_solvers_opened':False,'screened_out_tasks_preserved':True}
    Path('final-selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not ok:raise SystemExit(3)
if __name__=='__main__':main()
