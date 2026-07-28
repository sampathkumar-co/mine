from __future__ import annotations
import hashlib,itertools,json,math,re,time,urllib.request
from array import array
from dataclasses import dataclass
from datetime import datetime,timezone
SNAPSHOT_URL="https://zenodo.org/records/19735294/files/coverdata.json?download=1"
SNAPSHOT_MD5="b2c626b07f216aac830d344eff5ad523"
REFERENCE_DATE=datetime(2026,4,24,tzinfo=timezone.utc)
KEY_RE=re.compile(r"^C\((\d+),(\d+),(\d+)\)$")
BASE="32c897005c91865319f1b7da264b6162fc1ff4de|"+SNAPSHOT_MD5
SEEDS={x:f"{BASE}|LEXIGEN_WORLD_COVERING_{x.upper()}" for x in ('v1','v2','v3','v4')}
EXPECTED={
'v1':['C(15,8,5)','C(11,6,5)','C(14,5,3)'],
'v2':['C(12,7,5)','C(14,8,5)','C(16,7,4)'],
'v3':['C(15,6,4)','C(17,9,5)','C(16,8,5)']}
@dataclass(frozen=True)
class Target:
 name:str; v:int; k:int; t:int; upper:int; lower:int; last_update:str; gap:int; candidate_blocks:int; t_subsets:int; incidence_edges:int; opportunity_score:float; tie_break:str
@dataclass
class Incidence:
 blocks:list[tuple[int,...]]; tsets:list[tuple[int,...]]; cover_by_block:list[array]; blocks_by_t:list[array]
def download_snapshot():
 req=urllib.request.Request(SNAPSHOT_URL,headers={'User-Agent':'LEXIGEN-world-covering-v4'})
 with urllib.request.urlopen(req,timeout=180) as r:data=r.read()
 if hashlib.md5(data).hexdigest()!=SNAPSHOT_MD5:raise RuntimeError('snapshot MD5 mismatch')
 return json.loads(data)
def date(s):
 try:d=datetime.fromisoformat((s or '').strip())
 except ValueError:return datetime(1996,1,1,tzinfo=timezone.utc)
 return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
def updated(e):
 x=e.get('imps') or []
 return str(x[0][3]) if isinstance(x,list) and x and isinstance(x[0],list) and len(x[0])>=4 else ''
def make(name,e,profile):
 m=KEY_RE.match(name)
 if not m:return None
 v,k,t=map(int,m.groups()); u=int(e['size']); l=int(e['low_bd']); gap=u-l
 if not(10<=v<=24 and 4<=k<=min(11,v-2) and 3<=t<=min(5,k-1) and gap>=2 and u-1>=l):return None
 c=math.comb(v,k); ts=math.comb(v,t); edges=c*math.comb(k,t)
 limits={'v1':(100,50000,5000,3000000,1.35,18,.15,.35),'v2':(100,60000,5000,3500000,1.45,16,.12,.32),'v3':(120,75000,6500,4500000,1.55,15,.10,.30),'v4':(140,100000,8000,6000000,1.62,14,.08,.285)}[profile]
 umax,cmax,tmax,emax,gp,ad,up,ep=limits
 if u>umax or c>cmax or ts>tmax or edges>emax:return None
 age=max(0.,(REFERENCE_DATE-date(updated(e))).days/365.25)
 score=(gap**gp)*(1+min(age,25)/ad)*((umax/u)**up)/(edges**ep)
 tie=hashlib.sha256(f"{SEEDS[profile]}|{name}".encode()).hexdigest()
 return Target(name,v,k,t,u,l,updated(e),gap,c,ts,edges,score,tie)
def select(items,n):
 items.sort(key=lambda x:(-x.opportunity_score,x.tie_break,x.name)); out=[]; pc={}
 for x in items:
  q=(x.k,x.t)
  if pc.get(q,0)>=2:continue
  out.append(x); pc[q]=pc.get(q,0)+1
  if len(out)==n:return out
 raise RuntimeError('insufficient targets')
def lineage(data):
 used=set(); groups={}
 for p in ('v1','v2','v3'):
  xs=select([z for n,e in data.items() if n not in used and isinstance(e,dict) and (z:=make(n,e,p))],3)
  names=[x.name for x in xs]
  if names!=EXPECTED[p]:raise RuntimeError(f'{p} lineage mismatch {names}')
  groups[p]=xs; used.update(names)
 v4=select([z for n,e in data.items() if n not in used and isinstance(e,dict) and (z:=make(n,e,'v4'))],3)
 if used.intersection(x.name for x in v4):raise RuntimeError('lineage overlap')
 return groups,v4
def incidence(t):
 ts=list(itertools.combinations(range(t.v),t.t)); idx={s:i for i,s in enumerate(ts)}; blocks=[]; cb=[]; bt=[array('I') for _ in ts]
 for bi,b in enumerate(itertools.combinations(range(t.v),t.k)):
  cov=array('I',(idx[s] for s in itertools.combinations(b,t.t))); blocks.append(b); cb.append(cov)
  for s in cov:bt[s].append(bi)
 return Incidence(blocks,ts,cb,bt)
def verify(t,blocks):
 if len(blocks)!=len(set(blocks)):return False,'duplicate blocks'
 covered=set()
 for b in blocks:
  if len(b)!=t.k or tuple(sorted(b))!=b or not set(b)<=set(range(t.v)):return False,'malformed block'
  covered.update(itertools.combinations(b,t.t))
 if len(covered)!=math.comb(t.v,t.t):return False,f'covered {len(covered)}'
 if len(blocks)>=t.upper:return False,'not smaller'
 return True,'verified'
