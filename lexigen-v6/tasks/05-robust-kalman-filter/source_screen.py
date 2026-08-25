from __future__ import annotations

import hashlib,json,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'lexigen-v5'))
from engine import ENGINE_VERSION,fingerprint,generate_proposals,applicable_transfer_templates

SOURCE_REVISION='dff9914c10800c7a031c9e8c3d4d1c8cd1b38906'
SOURCE_PATH='AlgoTuneTasks/robust_kalman_filter/robust_kalman_filter.py'
SOURCE_URL=f'https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_REVISION}/{SOURCE_PATH}'

req=urllib.request.Request(SOURCE_URL,headers={'User-Agent':'LEXIGEN-v6-task5-source-screen-r1'})
with urllib.request.urlopen(req,timeout=120) as r:
    raw=r.read()
text=raw.decode('utf-8')
fp=fingerprint(text)
generated=generate_proposals(text)
transfers=[{'template':name,'causal_id':cid,'operators':list(ops),'rationale':list(rationale)} for name,cid,ops,rationale in applicable_transfer_templates(fp)]
arms={}
for arm,rows in generated['arms'].items():
    arms[arm]=[{
        'rank':int(p['rank']),
        'proposal_id':str(p['proposal_id']),
        'operators':[str(x) for x in p['operators']],
        'transfer_ids':[str(x) for x in p['transfer_ids']],
        'learned_template':p.get('learned_template'),
        'score':float(p['score']),
        'rationale':[str(x) for x in p.get('rationale',[])],
    } for p in rows]
out={
    'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication',
    'task_index':5,
    'task':'robust_kalman_filter',
    'stage':'source_screen_r1',
    'engine_version':ENGINE_VERSION,
    'source_revision':SOURCE_REVISION,
    'source_path':SOURCE_PATH,
    'source_url':SOURCE_URL,
    'source_size_bytes':len(raw),
    'source_sha256':hashlib.sha256(raw).hexdigest(),
    'fingerprint':{
        'source_sha256':fp.source_sha256,
        'verifier_sha256':fp.verifier_sha256,
        'features':list(fp.features),
        'dependency_calls':list(fp.dependency_calls),
        'ast_counts':[[k,v] for k,v in fp.ast_counts],
    },
    'applicable_transfer_templates':transfers,
    'arms':arms,
    'official_train_manifest_opened':False,
    'official_test_manifest_opened':False,
    'official_payloads_opened':0,
    'public_task_specific_solvers_opened':False,
}
canonical=json.dumps(out,sort_keys=True,separators=(',',':')).encode()
out['screen_sha256']=hashlib.sha256(canonical).hexdigest()
Path('source-screen.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'source_sha256':out['source_sha256'],'features':out['fingerprint']['features'],'applicable_transfer_templates':transfers,'arm_counts':{k:len(v) for k,v in arms.items()},'screen_sha256':out['screen_sha256']},indent=2))
