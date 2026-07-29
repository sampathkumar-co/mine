import json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
V14=Path(__file__).resolve().parent
ARC=Path(r'C:\Users\SAMPATH\AppData\Local\Temp\arcgen')
EVID=Path(r'C:\Users\SAMPATH\AppData\Local\Temp\lexigen-v13-campaign-engine\lexigen-language-genesis\external\evidence\v13-campaign')
sys.path[:0]=[str(V14),str(ARC)]
from scene_synthesizer_v14 import synthesize_scene
from scene_runtime_v14 import execute_pipeline
import task_list
TASKS={1:'dc433765',2:'49d1d64f',3:'c8f0f002',4:'9f236235',5:'ea9794b1'}

def grid(x): return tuple(tuple(r) for r in x)

def main():
 registry=task_list.task_list()
 total=0
 rows=[]
 for gate,task in TASKS.items():
  package=json.loads((EVID/f'v13-campaign-{gate:02d}'/'redacted-task.json').read_text())
  examples=[(grid(p['input']),grid(p['output'])) for p in package['train']]
  result=synthesize_scene(examples,max_depth=2)
  assert result.pipeline is not None
  generator,_=registry[task]
  failures=0
  accepted=0
  for seed in range(91000,91100):
   random.seed(seed)
   pair=generator()
   source,target=grid(pair['input']),grid(pair['output'])
   accepted+=1
   if execute_pipeline(result.pipeline,source)!=target:
    failures+=1
  total+=accepted
  rows.append({'gate':gate,'task':task,'operator_sequence':[s['op'] for s in result.pipeline],'cases':accepted,'failures':failures})
  print(json.dumps(rows[-1],sort_keys=True),flush=True)
 assert all(r['failures']==0 for r in rows)
 report={'tasks':len(rows),'fresh_cases':total,'failures':sum(r['failures'] for r in rows),'rows':rows}
 out=V14/'v14-first5-validation.json'
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'summary':report,'report':str(out)},sort_keys=True))
if __name__=='__main__': main()
