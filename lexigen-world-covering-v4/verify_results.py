import argparse,itertools,json,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text()); probs=[]; verified=0
for r in d['results']:
 t=r['target']; bs=[tuple(x) for x in r.get('blocks_zero_based',[])]; claim=bool(r['record_candidate'])
 ok=False
 if claim:
  covered=set()
  for b in bs:covered.update(itertools.combinations(b,t['t']))
  ok=len(bs)==len(set(bs)) and len(bs)<t['upper'] and len(covered)==math.comb(t['v'],t['t'])
  if ok:verified+=1
  else:probs.append(t['name'])
out={'protocol':'LEXIGEN v4 independent verification','summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'claimed':d['record_candidates'],'verified':verified,'problems':probs,'verification_passes':not probs and verified==d['record_candidates']};a.output.write_text(json.dumps(out,indent=2));raise SystemExit(0 if out['verification_passes'] else 1)
