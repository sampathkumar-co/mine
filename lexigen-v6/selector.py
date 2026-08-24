from __future__ import annotations
import hashlib,json,re,urllib.request
from collections import Counter
from pathlib import Path

SOURCE_COMMIT='dff9914c10800c7a031c9e8c3d4d1c8cd1b38906'
DATASET_REVISION='bb02811fa47ca1c833baaa344949bcd8fb307ac8'
SEED='LEXIGEN-V6-TRANSFER-REPLICATION-2026-08-24-A'
POOL_COUNT=24;MIN_FAMILIES=8;MAX_PER_FAMILY=4
SOURCE_TREE_URL=f'https://api.github.com/repos/oripress/AlgoTune/git/trees/{SOURCE_COMMIT}?recursive=1'
DATASET_TREE_URL=f'https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{DATASET_REVISION}/data?recursive=false&expand=false&limit=1000'
FAMILY_RULES=(
('cryptography_encoding',(r'cipher',r'encrypt',r'decrypt',r'hash',r'sha',r'base\d+',r'codec',r'encoding',r'compression')),
('linear_algebra',(r'matrix',r'svd',r'eigen',r'cholesky',r'qr',r'least_squares',r'linear_system',r'tensor',r'product')),
('numerical_optimization',(r'optimi',r'projection',r'simplex',r'portfolio',r'resource',r'flow',r'assignment',r'transport',r'cvar')),
('graph_discrete',(r'graph',r'shortest',r'path',r'tree',r'clique',r'color',r'matching',r'articulation',r'network',r'mst',r'cycle')),
('combinatorial',(r'cover',r'subset',r'set_',r'knapsack',r'permutation',r'combination',r'partition',r'scheduling',r'integer',r'factor')),
('signal_processing',(r'fft',r'fourier',r'filter',r'signal',r'convolution',r'correlation',r'wavelet',r'spectral',r'audio',r'image',r'dst',r'dct')),
('statistics',(r'stat',r'mean',r'median',r'quantile',r'regression',r'probab',r'distribution',r'sampling',r'variance',r'entropy')),
('scientific_computing',(r'integrat',r'differential',r'ode',r'pde',r'physics',r'simulation',r'scientific',r'nbody',r'monte_carlo')),
('string_sequence',(r'string',r'sequence',r'edit_distance',r'alignment',r'substring',r'prefix',r'suffix',r'token')),
('geometry',(r'geometry',r'convex_hull',r'distance',r'point',r'polygon',r'mesh',r'spatial')),
('machine_learning',(r'cluster',r'classification',r'neural',r'learning',r'pca',r'embedding',r'nearest',r'kmeans')))

def fetch_json(url,ua):
    q=urllib.request.Request(url,headers={'User-Agent':ua,'Accept':'application/vnd.github+json, application/json'})
    with urllib.request.urlopen(q,timeout=240) as r:raw=r.read();headers={k.lower():v for k,v in r.headers.items()}
    return json.loads(raw),headers,raw

def classify(name):
    x=name.lower()
    for family,patterns in FAMILY_RULES:
        if any(re.search(p,x) for p in patterns):return family
    return 'miscellaneous'

def score(name):return hashlib.sha256(f'{SEED}\0{name}'.encode()).hexdigest()
def exclusions():
    p=json.loads(Path(__file__).with_name('CONTAMINATION_EXCLUSIONS.json').read_text())
    s={str(x) for x in p['combined_exclusions']}
    if len(s)!=p['combined_exclusion_count'] or not p['frozen_before_screening_pool_selection']:raise RuntimeError('bad exclusions')
    return s

def source_inventory():
    p,h,raw=fetch_json(SOURCE_TREE_URL,'LEXIGEN-v6-name-metadata-selector')
    if not isinstance(p,dict) or p.get('truncated'):raise RuntimeError('source tree invalid')
    tasks=set();paths=[]
    for e in p.get('tree',[]):
        if not isinstance(e,dict) or e.get('type')!='blob':continue
        path=str(e.get('path',''));parts=path.split('/')
        if len(parts)==3 and parts[0]=='AlgoTuneTasks' and parts[2]==f'{parts[1]}.py':tasks.add(parts[1]);paths.append(path)
    return tasks,{'commit':SOURCE_COMMIT,'response_sha256':hashlib.sha256(raw).hexdigest(),'etag':h.get('etag'),'matched_paths_sha256':hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest(),'task_contents_opened':False}

def dataset_inventory():
    p,h,raw=fetch_json(DATASET_TREE_URL,'LEXIGEN-v6-name-metadata-selector')
    if not isinstance(p,list):raise RuntimeError('dataset tree invalid')
    tasks=set();ids=[]
    for e in p:
        if not isinstance(e,dict):continue
        path=str(e.get('path',''))
        if e.get('type')=='directory' and path.startswith('data/'):
            n=path.split('/',1)[1]
            if n and '/' not in n:tasks.add(n);ids.append((n,str(e.get('oid',''))))
    return tasks,{'revision':DATASET_REVISION,'response_sha256':hashlib.sha256(raw).hexdigest(),'etag':h.get('etag'),'directory_identity_sha256':hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest(),'manifests_opened':False,'payloads_opened':False}

def select_pool(rows):
    ordered=sorted(rows,key=lambda r:(r['score'],r['task']))
    def possible(i,c):return len(set(c)|{r['family'] for r in ordered[i:] if c[r['family']]<MAX_PER_FAMILY})
    def rec(i,chosen,c):
        if len(chosen)==POOL_COUNT:return chosen[:] if len(c)>=MIN_FAMILIES else None
        if len(ordered)-i<POOL_COUNT-len(chosen) or possible(i,c)<MIN_FAMILIES:return None
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
    z=rec(0,[],Counter())
    if z is None:raise RuntimeError('no admissible v6 screening pool')
    return z

def main():
    src,se=source_inventory();data,de=dataset_inventory();ex=exclusions();common=sorted((src&data)-ex)
    rows=[{'task':t,'family':classify(t),'score':score(t)} for t in common];selected=select_pool(rows);fc=Counter(x['family'] for x in selected)
    inv={'source_commit':SOURCE_COMMIT,'dataset_revision':DATASET_REVISION,'eligible':sorted((r['task'],r['family'],r['score']) for r in rows)}
    report={'campaign':'LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication','stage':'screening_pool_selection','selection_seed':SEED,'source_commit':SOURCE_COMMIT,'dataset_revision':DATASET_REVISION,'source_metadata':se,'dataset_metadata':de,'eligible_common_task_count':len(common),'inventory_sha256':hashlib.sha256(json.dumps(inv,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'excluded_tasks':sorted(ex),'screening_pool_count':POOL_COUNT,'minimum_families':MIN_FAMILIES,'maximum_per_family':MAX_PER_FAMILY,'selected':selected,'selected_family_counts':dict(sorted(fc.items())),'task_contents_opened':False,'task_descriptions_opened':False,'data_manifests_opened':False,'data_payloads_opened':False,'reports_opened':False,'public_solvers_opened':False}
    if len(selected)!=POOL_COUNT or len(fc)<MIN_FAMILIES or any(x>MAX_PER_FAMILY for x in fc.values()):raise RuntimeError('selection constraint violation')
    out=Path('selection-evidence');out.mkdir(exist_ok=True);(out/'screening-pool-selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'eligible':len(common),'selected':selected},indent=2))
if __name__=='__main__':main()
