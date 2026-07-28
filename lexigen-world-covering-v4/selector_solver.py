from pathlib import Path
import argparse,json
from dataclasses import asdict
from common import *
from solver import solve
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
groups,targets=lineage(download_snapshot()); sel={'snapshot_md5':SNAPSHOT_MD5,'excluded_lineages':{k:[asdict(x) for x in v] for k,v in groups.items()},'v4_targets':[asdict(x) for x in targets]};(a.output/'selection.json').write_text(json.dumps(sel,indent=2))
rs=[solve(t,a.output) for t in targets]; summary={'protocol':'LEXIGEN World Covering Record v4','snapshot_md5':SNAPSHOT_MD5,'selected_count':3,'record_candidates':sum(bool(r['record_candidate']) for r in rs),'results':rs};(a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
