from __future__ import annotations
import hashlib,json,random
from dataclasses import asdict
from pathlib import Path
from ortools.sat.python import cp_model
from common import *
def greedy(inc,seed):
 rng=random.Random(seed); uncovered=set(range(len(inc.tsets))); chosen=[]
 while uncovered:
  sample=list(range(len(inc.blocks))); rng.shuffle(sample)
  best=max(sample,key=lambda b:(sum(s in uncovered for s in inc.cover_by_block[b]),-b))
  gain=[s for s in inc.cover_by_block[best] if s in uncovered]
  if not gain:break
  chosen.append(best); uncovered.difference_update(gain)
 return chosen
def orbit_pool(t,inc):
 index={b:i for i,b in enumerate(inc.blocks)}; pool=set()
 multipliers=[a for a in range(1,t.v) if __import__('math').gcd(a,t.v)==1]
 for b in inc.blocks[::max(1,len(inc.blocks)//2500)]:
  for a in multipliers[:8]:
   for sh in range(t.v):pool.add(index[tuple(sorted(((a*x+sh)%t.v for x in b)))])
 return pool
def cp(t,inc,pool,seconds,seed):
 ids=sorted(pool); m=cp_model.CpModel(); x={i:m.new_bool_var(f'b{i}') for i in ids}
 for containing in inc.blocks_by_t:
  q=[x[i] for i in containing if i in x]
  if not q:return [],'POOL_MISS'
  m.add(sum(q)>=1)
 m.add(sum(x.values())<=t.upper-1); m.minimize(sum(x.values()))
 s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=seconds; s.parameters.num_search_workers=4; s.parameters.random_seed=seed&0x7fffffff
 st=s.solve(m); name=s.status_name(st)
 return ([i for i in ids if st in (cp_model.FEASIBLE,cp_model.OPTIMAL) and s.value(x[i])],name)
def solve(t,out):
 inc=incidence(t); base=int(hashlib.sha256(f"{SEEDS['v4']}|{t.name}".encode()).hexdigest(),16); best=[]
 for i in range(36):
  g=greedy(inc,base+i*0x9e3779b97f4a7c15)
  if not best or len(g)<len(best):best=g
  if len(best)<t.upper:break
 pool=orbit_pool(t,inc)|set(best)
 for b in best:
  for s in inc.cover_by_block[b]:pool.update(inc.blocks_by_t[s])
 sel,st=cp(t,inc,pool,240,base)
 if not sel:sel,st=cp(t,inc,range(len(inc.blocks)),1500,base)
 blocks=[inc.blocks[i] for i in sel]; ok,msg=verify(t,blocks) if blocks else (False,'none')
 r={'protocol':'LEXIGEN World Covering Record v4','target':asdict(t),'goal_blocks':t.upper-1,'greedy_best':len(best) if best else None,'cp_status':st,'pool_size':len(pool),'result_blocks':len(blocks) if blocks else None,'valid':ok,'verification':msg,'record_candidate':ok,'blocks_zero_based':[list(b) for b in blocks] if ok else []}
 (out/f"{t.name.replace('(','_').replace(')','').replace(',','_')}.json").write_text(json.dumps(r,indent=2)); return r
