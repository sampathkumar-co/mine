from __future__ import annotations
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'lexigen-v5'))
from engine import applicable_transfer_templates,fingerprint,generate_proposals

SOURCE_SHA256='24d194fbf8f604d318b9f330e61ad084ff4ea498de2c0a299835ad7ecce55d9a'
CAMPAIGN_IDS=('TM-BFR-01','TM-CAC-01','TM-RRR-01')

def main():
    source=Path(sys.argv[1] if len(sys.argv)>1 else 'task-source.py').read_bytes();sha=hashlib.sha256(source).hexdigest()
    if sha!=SOURCE_SHA256:raise RuntimeError(f'source sha mismatch {sha}')
    text=source.decode('utf-8');fp=fingerprint(text);generated=generate_proposals(text)
    app=[{'template':t,'causal_id':cid,'operators':list(ops),'rationale':list(rationale)} for t,cid,ops,rationale in applicable_transfer_templates(fp)]
    arms={}
    for arm,props in generated['arms'].items():
        arms[arm]=[{'rank':p.rank,'proposal_id':p.proposal_id,'operators':list(p.operators),'transfer_ids':list(p.transfer_ids),'learned_template':p.learned_template,'score':p.score,'rationale':list(p.rationale)} for p in props]
    body={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','task_index':6,'task':'edge_expansion','family':'miscellaneous','stage':'source_screen_r1','source_sha256':sha,'features':list(fp.features),'dependency_calls':list(fp.dependency_calls),'applicable_transfer_templates':app,'campaign_preregistered_applicable_transfer_ids':list(CAMPAIGN_IDS),'arms':arms,'arm_counts':{k:len(v) for k,v in arms.items()},'official_train_manifest_opened':False,'official_test_manifest_opened':False,'official_payloads_opened':0,'public_task_specific_solvers_opened':False}
    canonical=json.dumps(body,sort_keys=True,separators=(',',':'));body['screen_sha256']=hashlib.sha256(canonical.encode()).hexdigest()
    Path('source-screen.json').write_text(json.dumps(body,indent=2)+'\n');print(json.dumps({'source_sha256':sha,'features':body['features'],'applicable_transfer_templates':app,'arm_counts':body['arm_counts'],'screen_sha256':body['screen_sha256']},indent=2))
if __name__=='__main__':main()
