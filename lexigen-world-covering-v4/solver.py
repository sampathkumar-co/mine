from __future__ import annotations
import hashlib, random, time
from dataclasses import asdict
from pathlib import Path
from ortools.sat.python import cp_model
from common import *


def greedy(inc: Incidence, seed: int, randomized: bool) -> list[int]:
    rng=random.Random(seed); uncovered=set(range(len(inc.tsets))); chosen=[]; available=set(range(len(inc.blocks)))
    while uncovered:
        best=[]; gain=-1
        sample=list(available)
        if randomized: rng.shuffle(sample)
        for b in sample:
            g=sum(s in uncovered for s in inc.cover_by_block[b])
            if g>gain: gain=g; best=[b]
            elif g==gain: best.append(b)
        if gain<=0:return []
        b=rng.choice(best) if randomized else min(best)
        chosen.append(b); available.remove(b); uncovered.difference_update(inc.cover_by_block[b])
    return prune(inc,chosen,seed)


def counts(inc: Incidence, selected: list[int]) -> list[int]:
    c=[0]*len(inc.tsets)
    for b in selected:
        for s in inc.cover_by_block[b]: c[s]+=1
    return c


def prune(inc: Incidence, selected: list[int], seed: int) -> list[int]:
    rng=random.Random(seed); selected=list(dict.fromkeys(selected)); c=counts(inc,selected)
    order=list(selected); rng.shuffle(order)
    order.sort(key=lambda b:sum(c[s]==1 for s in inc.cover_by_block[b]))
    keep=set(selected)
    for b in order:
        if all(c[s]>=2 for s in inc.cover_by_block[b]):
            keep.remove(b)
            for s in inc.cover_by_block[b]: c[s]-=1
    return [b for b in selected if b in keep]


def repair(inc: Incidence, initial: list[int], goal: int, seed: int, deadline: float) -> tuple[list[int],list[dict]]:
    rng=random.Random(seed); best=list(initial); history=[]; rounds=42
    for r in range(rounds):
        if time.monotonic()>=deadline or len(best)<=goal: break
        c=counts(inc,best); excess=len(best)-goal
        order=sorted(best,key=lambda b:(sum(c[s]==1 for s in inc.cover_by_block[b]),rng.random()))
        remove_count=min(len(best),excess+2+(r%7)); removed=set(order[:remove_count]); survivors=[b for b in best if b not in removed]
        sc=counts(inc,survivors); uncovered=[i for i,x in enumerate(sc) if x==0]; budget=goal-len(survivors)
        pool=set()
        for s in uncovered:
            options=list(inc.blocks_by_t[s]); rng.shuffle(options); pool.update(options[:128])
        pool=[b for b in pool if b not in survivors]
        if budget<0 or not uncovered or not pool:
            history.append({'round':r,'status':'empty','best':len(best)}); continue
        pos={b:i for i,b in enumerate(pool)}; model=cp_model.CpModel(); x=[model.new_bool_var(f'x{i}') for i in range(len(pool))]
        feasible=True
        for s in uncovered:
            row=[x[pos[b]] for b in inc.blocks_by_t[s] if b in pos]
            if not row: feasible=False; break
            model.add(sum(row)>=1)
        if not feasible:
            history.append({'round':r,'status':'pool_miss','best':len(best)}); continue
        model.add(sum(x)<=budget); model.minimize(sum(x)); solver=cp_model.CpSolver()
        solver.parameters.max_time_in_seconds=max(1.0,min(12.0,deadline-time.monotonic())); solver.parameters.num_search_workers=4; solver.parameters.random_seed=(seed+r)&0x7fffffff
        status=solver.solve(model); name=solver.status_name(status)
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            candidate=prune(inc,survivors+[pool[i] for i,v in enumerate(x) if solver.value(v)],seed+r)
            if len(candidate)<len(best): best=candidate
        history.append({'round':r,'status':name,'pool':len(pool),'best':len(best)})
    return best,history


def full_cp(target: Target, inc: Incidence, hint: list[int], seed: int):
    goal=target.upper-1; model=cp_model.CpModel(); x=[model.new_bool_var(f'b{i}') for i in range(len(inc.blocks))]
    for row in inc.blocks_by_t:model.add(sum(x[i] for i in row)>=1)
    model.add(sum(x)<=goal); model.add(x[0]==1); model.minimize(sum(x))
    for i in hint:model.add_hint(x[i],1)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=1050; solver.parameters.num_search_workers=4; solver.parameters.random_seed=seed&0x7fffffff; solver.parameters.cp_model_presolve=True
    status=solver.solve(model); selected=[]
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):selected=[i for i,v in enumerate(x) if solver.value(v)]
    return selected,solver.status_name(status),solver.response_stats()


def solve_target(target: Target, out: Path) -> dict:
    start=time.time(); inc=build_incidence(target); base=int(hashlib.sha256(f'{SEED_MATERIAL}|{target.name}'.encode()).hexdigest(),16); goal=target.upper-1
    best=[]; runs=[]
    for a in range(48):
        cand=greedy(inc,(base+a*0x9E3779B97F4A7C15)&((1<<64)-1),a>=12); runs.append({'attempt':a,'blocks':len(cand) if cand else None})
        if cand and (not best or len(cand)<len(best)):best=cand
        if best and len(best)<=goal:break
    repaired,repair_runs=repair(inc,best,goal,base^0xA5A5A5A5A5A5A5A5,time.monotonic()+360) if best else ([],[])
    selected=[]; method='none'; cp_status='not_run'; cp_stats=None
    for candidate,name in ((best,'greedy'),(repaired,'exact_repair')):
        if candidate and len(candidate)<=goal:selected=candidate;method=name;break
    if not selected:
        selected,cp_status,cp_stats=full_cp(target,inc,repaired or best,base);method='full_cp_sat'
    blocks=[inc.blocks[i] for i in selected] if selected else []; valid,msg=verify_design(target,blocks) if blocks else (False,'no design returned'); record=valid and len(blocks)<=goal
    result={'protocol':'LEXIGEN World Covering Record v4','snapshot_md5':SNAPSHOT_MD5,'target':asdict(target),'goal_blocks':goal,'method':method,'greedy_best_blocks':len(best) if best else None,'greedy_runs':runs,'repair_best_blocks':len(repaired) if repaired else None,'repair_runs':repair_runs,'cp_status':cp_status,'cp_stats':cp_stats,'result_blocks':len(blocks) if blocks else None,'valid':valid,'verification':msg,'record_candidate':record,'blocks_zero_based':[list(b) for b in blocks] if record else [],'elapsed_s':time.time()-start}
    (out/f"{target.name.replace('(','_').replace(')','').replace(',','_')}.json").write_text(__import__('json').dumps(result,indent=2)); return result
