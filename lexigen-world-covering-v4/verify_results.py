from __future__ import annotations
import argparse, hashlib, itertools, json, math
from pathlib import Path
EXPECTED_SNAPSHOT_MD5 = 'b2c626b07f216aac830d344eff5ad523'
def verify_record(result):
    target=result['target']; v=int(target['v']); k=int(target['k']); t=int(target['t']); upper=int(target['upper'])
    raw=result.get('blocks_zero_based') or []; blocks=[tuple(int(x) for x in block) for block in raw]; problems=[]
    if len(blocks)!=len(set(blocks)): problems.append('duplicate blocks')
    for index,block in enumerate(blocks):
        if len(block)!=k or tuple(sorted(block))!=block or len(set(block))!=len(block) or any(x<0 or x>=v for x in block): problems.append(f'malformed block {index}')
    covered=set()
    for block in blocks: covered.update(itertools.combinations(block,t))
    required=math.comb(v,t)
    if len(covered)!=required: problems.append(f'covers {len(covered)} of {required}')
    if len(blocks)>=upper: problems.append(f'block count {len(blocks)} not below {upper}')
    return {'target':target['name'],'valid':not problems,'block_count':len(blocks),'prior_upper_bound':upper,'covered_t_subsets':len(covered),'required_t_subsets':required,'blocks_sha256':hashlib.sha256(json.dumps(raw,separators=(',',':')).encode()).hexdigest(),'problems':problems}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    raw=args.summary.read_bytes(); summary=json.loads(raw); problems=[]
    if summary.get('snapshot_md5')!=EXPECTED_SNAPSHOT_MD5: problems.append('snapshot mismatch')
    results=summary.get('results')
    if not isinstance(results,list) or len(results)!=3: problems.append('must contain three results'); results=[]
    candidates=[row for row in results if bool(row.get('record_candidate'))]; checks=[verify_record(row) for row in candidates]
    if any(not check['valid'] for check in checks): problems.append('candidate failed independent verification')
    report={'protocol':'LEXIGEN World Covering Record v4 independent verification','summary_sha256':hashlib.sha256(raw).hexdigest(),'snapshot_md5':summary.get('snapshot_md5'),'selected_results':len(results),'claimed_record_candidates':len(candidates),'verified_record_candidates':sum(bool(check['valid']) for check in checks),'candidate_verifications':checks,'problems':problems,'verification_passes':not problems}
    args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    if problems: raise SystemExit(2)
if __name__=='__main__': main()
