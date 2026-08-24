from __future__ import annotations
import argparse,gc,hashlib,json,random,statistics,sys,time,urllib.error,urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
LEXIGEN=HERE.parents[1]
TASK=LEXIGEN/'tasks'/'10-vertex-cover'
sys.path.insert(0,str(TASK));sys.path.insert(0,str(HERE))
from candidates import learned_bit_frontier_exact,reference_exact,rc2_exact
from audit_algorithms import reproduced_bfr,color_bound_clique_cover

REV='bb02811fa47ca1c833baaa344949bcd8fb307ac8'
TEST='vertex_cover_T100ms_n15_size100_test.jsonl'
TEST_SIZE=1143200
TEST_OID='a11bc56b01a7ab254843102454d790637b89ec56'
URL=f'https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REV}/data/vertex_cover/{TEST}?download=true'
REPS=5
ALGOS=[
    ('sealed_bfr',learned_bit_frontier_exact),
    ('reproduced_bfr',reproduced_bfr),
    ('color_bound_clique',color_bound_clique_cover),
    ('pysat_reference',reference_exact),
    ('rc2_exact',rc2_exact),
]

def git_blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def fetch():
    last=None
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'LEXIGEN-v5-task10-causal-audit-r1'}),timeout=240) as r:return r.read()
        except (urllib.error.HTTPError,urllib.error.URLError) as e:last=e;time.sleep(min(60,2**k))
    raise RuntimeError('test fetch exhausted') from last

def decode(p):
    if not isinstance(p,list):raise RuntimeError('problem not list')
    n=len(p)
    if n==0 or any(not isinstance(r,list) or len(r)!=n for r in p):raise RuntimeError('bad adjacency shape')
    for i,r in enumerate(p):
        if r[i]!=0:raise RuntimeError('nonzero diagonal')
        for j,x in enumerate(r):
            if x not in (0,1) or p[j][i]!=x:raise RuntimeError('bad adjacency')
    return p

def valid(p,c,opt):
    n=len(p)
    if not isinstance(c,list) or len(set(c))!=len(c) or any(type(x) is not int or x<0 or x>=n for x in c):return False
    s=set(c)
    if any(p[i][j] and i not in s and j not in s for i in range(n) for j in range(i+1,n)):return False
    return len(c)==opt

def timed(fn,p):
    gc.collect();t=time.perf_counter_ns();x=fn(p);dt=(time.perf_counter_ns()-t)/1e9
    return x,dt

def hmean(xs):return statistics.harmonic_mean(xs) if xs else 0.0

def er(n,prob,seed):
    r=random.Random(seed);a=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if r.random()<prob:a[i][j]=a[j][i]=1
    return a

def stress_cases():
    out=[]
    for n in (24,32,40):
        for pi,prob in enumerate((0.10,0.30,0.50,0.70,0.90)):
            for rep in range(2):out.append((f'er_n{n}_p{prob}_r{rep}',er(n,prob,900000+n*100+pi*10+rep)))
    n=40
    out.append(('empty40',[[0]*n for _ in range(n)]))
    complete=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):complete[i][j]=complete[j][i]=1
    out.append(('complete40',complete))
    star=[[0]*n for _ in range(n)]
    for j in range(1,n):star[0][j]=star[j][0]=1
    out.append(('star40',star))
    matching=[[0]*n for _ in range(n)]
    for i in range(0,n,2):matching[i][i+1]=matching[i+1][i]=1
    out.append(('matching40',matching))
    cycle=[[0]*n for _ in range(n)]
    for i in range(n):cycle[i][(i+1)%n]=cycle[(i+1)%n][i]=1
    out.append(('cycle40',cycle))
    kb=[[0]*n for _ in range(n)]
    for i in range(20):
        for j in range(20,40):kb[i][j]=kb[j][i]=1
    out.append(('k20_20',kb))
    disc=[[0]*n for _ in range(n)]
    for i in range(10):
        for j in range(i+1,10):disc[i][j]=disc[j][i]=1
    for i in range(10,25):
        j=10+(i-9)%15;disc[i][j]=disc[j][i]=1
    out.append(('disconnected_isolates40',disc))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raw=fetch()
    if len(raw)!=TEST_SIZE or git_blob(raw)!=TEST_OID:raise RuntimeError(f'test identity mismatch {len(raw)} {git_blob(raw)}')
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=100:raise RuntimeError(f'expected 100 records got {len(rows)}')
    details=[];per={name:[] for name,_ in ALGOS};nodes=[];edges=[]
    for idx,row in enumerate(rows):
        p=decode(row['problem']);nodes.append(len(p));edges.append(sum(p[i][j] for i in range(len(p)) for j in range(i+1,len(p))))
        oracle=reference_exact(p);opt=len(oracle)
        # Correctness pass before any timing aggregation.
        for name,fn in ALGOS:
            got=fn(p)
            if not valid(p,got,opt):raise RuntimeError(f'official exactness failure record {idx+1} alg={name} got={len(got) if isinstance(got,list) else None} opt={opt}')
        times={name:[] for name,_ in ALGOS}
        for rep in range(REPS):
            ordered=list(ALGOS);shift=(idx+rep)%len(ordered);ordered=ordered[shift:]+ordered[:shift]
            if rep%2:ordered=list(reversed(ordered))
            for name,fn in ordered:
                got,dt=timed(fn,p)
                if not valid(p,got,opt):raise RuntimeError(f'timed exactness failure record {idx+1} alg={name}')
                times[name].append(dt)
        med={k:statistics.median(v) for k,v in times.items()}
        ref=med['pysat_reference']
        ratios={name:ref/t for name,t in med.items()}
        for name in per:per[name].append(ratios[name])
        details.append({'kind':'official','index':idx+1,'nodes':len(p),'edges':edges[-1],'opt_cover_size':opt,'median_seconds':med,'speedup_vs_pysat':ratios})
    official={}
    for name in per:
        official[name]={'valid':100,'harmonic_speedup_vs_pysat':hmean(per[name]),'minimum_speedup_vs_pysat':min(per[name]),'median_speedup_vs_pysat':statistics.median(per[name])}
    pair={
        'sealed_over_color_bound_harmonic':hmean([details[i]['median_seconds']['color_bound_clique']/details[i]['median_seconds']['sealed_bfr'] for i in range(100)]),
        'reproduced_over_color_bound_harmonic':hmean([details[i]['median_seconds']['color_bound_clique']/details[i]['median_seconds']['reproduced_bfr'] for i in range(100)]),
        'sealed_over_reproduced_harmonic':hmean([details[i]['median_seconds']['reproduced_bfr']/details[i]['median_seconds']['sealed_bfr'] for i in range(100)]),
    }
    stress=[]
    for name,p in stress_cases():
        ref,rt=timed(reference_exact,p);opt=len(ref)
        row={'kind':'stress','name':name,'nodes':len(p),'edges':sum(p[i][j] for i in range(len(p)) for j in range(i+1,len(p))),'opt_cover_size':opt,'seconds':{'pysat_reference':rt}}
        for an,fn in ALGOS:
            if an=='pysat_reference':continue
            got,dt=timed(fn,p)
            if not valid(p,got,opt):raise RuntimeError(f'stress exactness failure {name} alg={an}')
            row['seconds'][an]=dt
        stress.append(row);details.append(row)
    summary={
        'audit':'LEXIGEN v5 Task 10 causal audit R1',
        'sealed_task10_checkpoint':'4ae98e50a9e4393f5f0a1498ea99423255f56552',
        'official_test':{'name':TEST,'size':TEST_SIZE,'git_blob_sha1':TEST_OID,'sha256':hashlib.sha256(raw).hexdigest(),'records':100,'repetitions_per_algorithm':REPS,'node_range':[min(nodes),max(nodes)],'node_median':statistics.median(nodes),'edge_range':[min(edges),max(edges)]},
        'official_results':official,
        'pairwise':pair,
        'stress':{'cases':len(stress),'all_exact':True},
        'interpretation':{
            'original_timing_effect_reproduced':official['sealed_bfr']['harmonic_speedup_vs_pysat']>=5.0 and official['sealed_bfr']['minimum_speedup_vs_pysat']>=2.0,
            'separate_tm_bfr_reimplementation_exact':official['reproduced_bfr']['valid']==100,
            'separate_tm_bfr_reimplementation_useful':official['reproduced_bfr']['harmonic_speedup_vs_pysat']>=3.0,
            'known_style_strong_baseline_exposes_reference_weakness':official['color_bound_clique']['harmonic_speedup_vs_pysat']>=1.5,
            'algorithmic_novelty_supported':False,
            'v5_campaign_score_changed':False
        },
        'claim_boundary':'This post-hoc audit can strengthen or weaken confidence that TM-BFR selected a useful reusable mechanism. It cannot convert the failed v5 campaign into a pass and cannot establish algorithmic novelty because exact bitset branch-and-bound for clique/MIS is established prior art.'
    }
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'audit-results.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in details)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
