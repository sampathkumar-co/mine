use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::time::Instant;

const BUDGET: u64 = 500_000;

#[derive(Clone)]
struct State {
    digest: String,
    n: usize,
    q: usize,
    labels: Vec<i32>,
    masses: Vec<i64>,
    query_ids: Vec<i32>,
    responses: Vec<Vec<i32>>,
    costs: Vec<Vec<i64>>,
}

#[derive(Clone, Copy, Default)]
struct Plan { mass: i64, expected: i64, worst: i64, query: i32 }

#[derive(Default, Clone, Copy)]
struct Stats { calls: u64, memo_hits: u64, expansions: u64, raw: u64, kept: u64, removed: u64 }

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Key { allowed: u32, remaining: u128 }

fn better(a: Plan, b: Plan) -> bool {
    (a.mass, -a.expected, -a.worst, -a.query as i64) > (b.mass, -b.expected, -b.worst, -b.query as i64)
}

fn partition(s: &State, allowed: u32, query: usize) -> Vec<u32> {
    let mut buckets: BTreeMap<i32,u32> = BTreeMap::new();
    for i in 0..s.n {
        if allowed & (1u32 << i) != 0 {
            *buckets.entry(s.responses[i][query]).or_insert(0) |= 1u32 << i;
        }
    }
    let mut v: Vec<u32> = buckets.values().copied().collect();
    v.sort_unstable();
    v
}

fn mass(s: &State, allowed: u32) -> i64 {
    (0..s.n).filter(|&i| allowed & (1u32<<i) != 0).map(|i| s.masses[i]).sum()
}

fn pure(s: &State, allowed: u32) -> bool {
    let mut first: Option<i32> = None;
    for i in 0..s.n {
        if allowed & (1u32<<i) == 0 { continue; }
        match first { None => first = Some(s.labels[i]), Some(v) if v != s.labels[i] => return false, _ => {} }
    }
    true
}

fn cell_cost(s: &State, child: u32, query: usize) -> i64 {
    let i = child.trailing_zeros() as usize;
    s.costs[i][query]
}

fn dominates(aq: usize, av: &[i64], bq: usize, bv: &[i64]) -> bool {
    if av.len() != bv.len() { return false; }
    let strict = av.iter().zip(bv).all(|(a,b)| a <= b) && av.iter().zip(bv).any(|(a,b)| a < b);
    strict || (av == bv && aq < bq)
}

struct Solver<'a> { s: &'a State, memo: HashMap<Key,Plan>, stats: Stats }
impl<'a> Solver<'a> {
    fn canonical(&mut self, allowed: u32, remaining: u128) -> u128 {
        self.stats.raw += remaining.count_ones() as u64;
        let mut groups: BTreeMap<Vec<u32>,Vec<usize>> = BTreeMap::new();
        for q in 0..self.s.q {
            if remaining & (1u128<<q) == 0 { continue; }
            let sig = partition(self.s, allowed, q);
            if sig.len() > 1 { groups.entry(sig).or_default().push(q); }
        }
        let mut out = 0u128;
        for (sig, qs) in groups {
            let vectors: Vec<Vec<i64>> = qs.iter().map(|&q| sig.iter().map(|&c| cell_cost(self.s,c,q)).collect()).collect();
            for (i,&q) in qs.iter().enumerate() {
                let mut drop = false;
                for (j,&other) in qs.iter().enumerate() {
                    if i != j && dominates(other,&vectors[j],q,&vectors[i]) { drop = true; break; }
                }
                if !drop { out |= 1u128<<q; }
            }
        }
        self.stats.kept += out.count_ones() as u64;
        self.stats.removed += remaining.count_ones() as u64 - out.count_ones() as u64;
        out
    }
    fn solve(&mut self, allowed: u32, remaining: u128) -> Result<Plan,()> {
        self.stats.calls += 1;
        let canon = self.canonical(allowed,remaining);
        let key = Key{allowed,remaining:canon};
        if let Some(p)=self.memo.get(&key) { self.stats.memo_hits += 1; return Ok(*p); }
        if pure(self.s,allowed) {
            let p=Plan{mass:mass(self.s,allowed),expected:0,worst:0,query:i32::MAX};
            self.memo.insert(key,p); return Ok(p);
        }
        let mut found=false; let mut best=Plan::default();
        for q in 0..self.s.q {
            if canon & (1u128<<q)==0 { continue; }
            self.stats.expansions += 1; if self.stats.expansions>BUDGET { return Err(()); }
            let children=partition(self.s,allowed,q);
            let mut cand=Plan{mass:0,expected:0,worst:0,query:self.s.query_ids[q]};
            let mut imm=0i64;
            for i in 0..self.s.n { if allowed&(1u32<<i)!=0 { imm += self.s.masses[i]*self.s.costs[i][q]; } }
            cand.expected=imm;
            for child in children {
                let cp=self.solve(child,canon & !(1u128<<q))?;
                cand.mass += cp.mass; cand.expected += cp.expected;
                cand.worst = cand.worst.max(cell_cost(self.s,child,q)+cp.worst);
            }
            if !found || better(cand,best) { best=cand; found=true; }
        }
        let p=if found {best} else {Plan::default()}; self.memo.insert(key,p); Ok(p)
    }
}

fn parse(path:&str)->Vec<State>{
    let text=fs::read_to_string(path).unwrap(); let mut it=text.split_whitespace();
    assert_eq!(it.next(),Some("COUNT")); let count:usize=it.next().unwrap().parse().unwrap(); let mut out=Vec::new();
    for _ in 0..count {
        assert_eq!(it.next(),Some("STATE")); let digest=it.next().unwrap().to_string(); let _seed=it.next(); let n:usize=it.next().unwrap().parse().unwrap(); let q:usize=it.next().unwrap().parse().unwrap();
        assert_eq!(it.next(),Some("LABELS")); let labels=(0..n).map(|_|it.next().unwrap().parse().unwrap()).collect();
        assert_eq!(it.next(),Some("MASSES")); let masses=(0..n).map(|_|it.next().unwrap().parse().unwrap()).collect();
        assert_eq!(it.next(),Some("QUERY_IDS")); let query_ids=(0..q).map(|_|it.next().unwrap().parse().unwrap()).collect();
        let mut responses=vec![vec![0;q];n]; for r in 0..n { assert_eq!(it.next(),Some("RESPONSES")); for c in 0..q { responses[r][c]=it.next().unwrap().parse().unwrap(); } }
        let mut costs=vec![vec![0;q];n]; for r in 0..n { assert_eq!(it.next(),Some("COSTS")); for c in 0..q { costs[r][c]=it.next().unwrap().parse().unwrap(); } }
        assert_eq!(it.next(),Some("END")); out.push(State{digest,n,q,labels,masses,query_ids,responses,costs});
    } out
}

fn main(){
    let a:Vec<String>=env::args().collect(); if a.len()!=5 || a[1]!="--input" || a[3]!="--output" { eprintln!("usage: solver --input FILE --output FILE"); std::process::exit(2); }
    let states=parse(&a[2]); let start=Instant::now(); let mut rows=Vec::new();
    for s in &states { let mut solver=Solver{s,memo:HashMap::new(),stats:Stats::default()}; let allowed=if s.n==32 {u32::MAX} else {(1u32<<s.n)-1}; let remaining=if s.q==128 {u128::MAX} else {(1u128<<s.q)-1}; let p=solver.solve(allowed,remaining); rows.push((s.digest.clone(),p,solver.stats,solver.memo.len())); }
    let mut out=String::from("{\n  \"rows\": [\n");
    for (i,(d,p,st,memo)) in rows.iter().enumerate(){ if i>0 {out.push_str(",\n");} match p { Ok(x)=>out.push_str(&format!("    {{\"digest\":\"{}\",\"solved\":true,\"plan\":[{},{},{}],\"query_expansions\":{},\"calls\":{},\"memo_entries\":{},\"memo_hits\":{},\"raw_queries_considered\":{},\"representative_queries_considered\":{},\"dominated_queries_removed\":{}}}",d,x.mass,x.expected,x.worst,st.expansions,st.calls,memo,st.memo_hits,st.raw,st.kept,st.removed)), Err(_)=>out.push_str(&format!("    {{\"digest\":\"{}\",\"solved\":false}}",d)) } }
    out.push_str(&format!("\n  ],\n  \"total_milliseconds\": {}\n}}\n",start.elapsed().as_secs_f64()*1000.0)); fs::write(&a[4],out).unwrap();
}
