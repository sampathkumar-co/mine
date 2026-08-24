from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
from candidates import CANDIDATES_BY_ARM,CANDIDATE_META,reference_exact

def graph(n,p,seed):
    r=random.Random(seed);a=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if r.random()<p:a[i][j]=a[j][i]=1
    return a

def cycle(n):
    a=[[0]*n for _ in range(n)]
    for i in range(n):a[i][(i+1)%n]=a[(i+1)%n][i]=1
    return a

def complete(n):return [[0 if i==j else 1 for j in range(n)] for i in range(n)]
def path(n):
    a=[[0]*n for _ in range(n)]
    for i in range(n-1):a[i][i+1]=a[i+1][i]=1
    return a

def valid(problem,sol,opt):
    if not isinstance(sol,list) or len(set(sol))!=len(sol) or any(not isinstance(x,int) or x<0 or x>=len(problem) for x in sol):return False
    s=set(sol)
    for i in range(len(problem)):
        for j in range(i+1,len(problem)):
            if problem[i][j] and i not in s and j not in s:return False
    return len(sol)==len(opt)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cases=[('path11',path(11)),('cycle13',cycle(13)),('complete12',complete(12)),('random18',graph(18,0.5,101)),('random24',graph(24,0.5,202)),('random30',graph(30,0.5,303))]
    rows=[]
    for cname,p in cases:
        t=time.perf_counter();opt=reference_exact(p);ref_s=time.perf_counter()-t
        for arm,cs in CANDIDATES_BY_ARM.items():
            for name,fn in cs:
                try:t=time.perf_counter();got=fn(p);dt=time.perf_counter()-t;ok=valid(p,got,opt);err=None
                except Exception as e:got=None;dt=None;ok=False;err=f'{type(e).__name__}: {e}'
                rows.append({'case':cname,'arm':arm,'candidate':name,'implementation_class':CANDIDATE_META[name]['implementation_class'],'learned_template':CANDIDATE_META[name]['learned_template'],'transfer_ids':CANDIDATE_META[name]['transfer_ids'],'valid':ok,'error':err,'candidate_s':dt,'reference_s':ref_s,'speedup':ref_s/dt if dt and dt>0 else 0.0,'optimal_cover_size':len(opt)})
    passed=sum(bool(x['valid']) for x in rows);summary={'campaign':'LEXIGEN v5 Causal Transfer Generalization Experiment','task_index':10,'task':'vertex_cover','stage':'synthetic_r1','cases':len(cases),'candidate_count':30,'evaluations':len(rows),'passed':passed,'failed':len(rows)-passed,'all_passed':passed==len(rows),'official_training_manifest_opened':False,'official_test_manifest_opened':False}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'synthetic-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'synthetic-results.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in rows)+'\n');print(json.dumps(summary,indent=2))
    if not summary['all_passed']:raise SystemExit('synthetic correctness gate failed')
if __name__=='__main__':main()
