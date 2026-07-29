from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; V19R4=HERE.parent/'v19r4'; V19R2=HERE.parent/'v19r2'
for p in (V19R4,V19R2):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from run_cegis_v19r4 import expand_production
from runtime_v19r2 import as_grid,execute,sha256_json
from portable_runtime_v19r2 import execute_portable

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_bytes((json.dumps(v,indent=2,sort_keys=True)+'\n').encode())
def colours(examples): return sorted({x for s,t in examples for g in (s,t) for row in g for x in row})
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--package-root',type=Path,required=True); ap.add_argument('--output',type=Path,default=HERE/'V20_TRANSFER_SCAN.json'); a=ap.parse_args()
 pre=load(HERE/'V20_PRECOMMIT.json'); production=load(V19R4/'production'/'v19r4-production.json')
 if sha256_json(production)!=pre['production_sha256']: raise RuntimeError('production changed')
 reports=[]
 for item in pre['items']:
  gate=item['gate']; pkg=load(a.package_root/f'v13-campaign-{gate:02d}'/'redacted-task.json'); demos=[(as_grid(x['input']),as_grid(x['output'])) for x in pkg['train']]
  palette=colours(demos); survivors=[]; checked=0; invalid=0
  for marker in palette:
   for bg in palette:
    if checked>=100: break
    checked+=1; args={'marker_colour':marker,'output_background':bg}; program=expand_production(production,args)
    try:
     left=[execute(program,s) for s,_ in demos]; right=[execute_portable(program,s) for s,_ in demos]
    except Exception:
     invalid+=1; continue
    exact=sum(x==t for x,(_,t) in zip(left,demos)); pexact=sum(x==t for x,(_,t) in zip(right,demos))
    if exact==pexact==len(demos) and left==right: survivors.append({'arguments':args,'arguments_sha256':sha256_json(args),'concrete_program_sha256':sha256_json(program)})
  reports.append({'gate':gate,'demonstrations':len(demos),'palette':palette,'argument_pairs_checked':checked,'runtime_invalid_pairs':invalid,'exact_survivors':len(survivors),'survivors':survivors})
  print(json.dumps(reports[-1],sort_keys=True),flush=True)
 out={'schema':'lexigen-v20-fixed-production-transfer-scan-v1','precommit_sha256':sha256_json(pre),'production_sha256':sha256_json(production),'fixed_gates':pre['fixed_gates'],'gates_checked':len(reports),'gates_with_exact_survivor':sum(r['exact_survivors']>0 for r in reports),'total_exact_survivors':sum(r['exact_survivors'] for r in reports),'hidden_outputs_opened':False,'sealed_external_success':False,'transfer_demonstrated':False,'world_level_breakthrough':False,'reports':reports}
 write(a.output,out); print('SUMMARY',json.dumps({k:out[k] for k in ('gates_checked','gates_with_exact_survivor','total_exact_survivors')},sort_keys=True))
if __name__=='__main__': main()
