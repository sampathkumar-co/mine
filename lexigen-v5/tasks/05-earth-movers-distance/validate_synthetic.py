from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import ot
from candidates import CANDIDATES_BY_ARM, reference_exact

def cases():
    out=[]
    for n,seed in [(4,11),(8,17),(12,23),(20,31),(32,43),(48,59)]:
        rng=np.random.default_rng(seed)
        a=ot.utils.unif(n); b=ot.utils.unif(n)
        x=rng.random((n,2)); y=rng.random((n,2)); M=ot.dist(x,y,metric='euclidean')
        out.append((n,seed,{'source_weights':a,'target_weights':b,'cost_matrix':M}))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[]; failures=[]
    total=sum(len(v) for v in CANDIDATES_BY_ARM.values())
    if total!=30: raise RuntimeError(f'expected 30 candidates got {total}')
    for n,seed,p in cases():
        t=time.perf_counter(); expected=reference_exact(p)['transport_plan']; ref_s=time.perf_counter()-t
        for arm,cs in CANDIDATES_BY_ARM.items():
            for name,fn in cs:
                t=time.perf_counter(); err=None
                try: got=np.asarray(fn(p)['transport_plan'],dtype=np.float64); cand_s=time.perf_counter()-t; maxerr=float(np.max(np.abs(got-expected))); valid=bool(got.shape==expected.shape and np.all(np.isfinite(got)) and np.allclose(got,expected,rtol=1e-7,atol=1e-7))
                except Exception as e: cand_s=time.perf_counter()-t; maxerr=None; valid=False; err=f'{type(e).__name__}: {e}'
                row={'n':n,'seed':seed,'arm':arm,'candidate':name,'valid':valid,'max_abs_error':maxerr,'candidate_s':cand_s,'reference_s':ref_s,'diagnostic_speedup':ref_s/cand_s if cand_s>0 else 0.0,'official_training_opened':False,'official_test_opened':False}
                rows.append(row)
                if not valid: failures.append({**row,'error':err})
    summary={'task':'earth_movers_distance','stage':'synthetic_r1','cases':6,'candidate_count':30,'checks':len(rows),'passed':len(rows)-len(failures),'failed':len(failures),'failures':failures,'synthetic_timings_not_used_for_selection':True,'official_training_manifest_opened':False,'official_training_payloads_opened':0,'official_test_manifest_opened':False,'official_test_payloads_opened':0}
    args.output.mkdir(parents=True,exist_ok=True); (args.output/'synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (args.output/'synthetic-results.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    print(json.dumps(summary,indent=2))
    if failures: raise SystemExit(2)
if __name__=='__main__': main()
