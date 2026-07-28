from __future__ import annotations
import argparse, hashlib, json
from dataclasses import asdict
from pathlib import Path
from common import SEED_MATERIAL,SNAPSHOT_MD5,file_sha256,load_snapshot_json,select_targets
from solver import solve_target

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lineage,targets=select_targets(load_snapshot_json())
    selection={'snapshot_md5':SNAPSHOT_MD5,'seed_material_sha256':hashlib.sha256(SEED_MATERIAL.encode()).hexdigest(),'lineage':{k:[asdict(t) for t in v] for k,v in lineage.items()},'v4_targets':[asdict(t) for t in targets]}
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2));print('FROZEN_SELECTED_TARGETS_V4');print(json.dumps(selection,indent=2),flush=True)
    results=[]
    for target in targets:
        print(f'START {target.name} upper={target.upper} lower={target.lower}',flush=True);r=solve_target(target,a.output);results.append(r);print(f"FINISH {target.name} valid={r['valid']} blocks={r['result_blocks']} record={r['record_candidate']}",flush=True)
    root=Path(__file__).resolve().parent;names=['common.py','solver.py','selector_solver.py','verify_results.py','PROTOCOL.md','requirements.txt']
    summary={'protocol':'LEXIGEN World Covering Record v4','snapshot_md5':SNAPSHOT_MD5,'selected_count':len(targets),'record_candidates':sum(bool(r['record_candidate']) for r in results),'results':results,'code_hashes':{n:file_sha256(root/n) for n in names}}
    (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
