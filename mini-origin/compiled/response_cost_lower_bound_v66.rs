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

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Plan {
    mass: i64,
    expected: i64,
    worst: i64,
    query: i32,
}

#[derive(Default, Clone, Copy)]
struct Stats {
    calls: u64,
    memo_hits: u64,
    expansions: u64,
    raw: u64,
    kept: u64,
    removed: u64,
    bound_evaluations: u64,
    bound_pruned: u64,
    impossible_pruned: u64,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Key {
    allowed: u32,
    remaining: u128,
}

#[derive(Clone)]
struct CandidateBound {
    local_query: usize,
    original_query: i32,
    children: Vec<u32>,
    full_diagnosis_possible: bool,
    expected_lower_bound: i64,
    worst_lower_bound: i64,
}

fn better(a: Plan, b: Plan) -> bool {
    (a.mass, -a.expected, -a.worst, -(a.query as i64))
        > (b.mass, -b.expected, -b.worst, -(b.query as i64))
}

fn partition(s: &State, allowed: u32, query: usize) -> Vec<u32> {
    let mut buckets: BTreeMap<i32, u32> = BTreeMap::new();
    for i in 0..s.n {
        if allowed & (1u32 << i) != 0 {
            *buckets.entry(s.responses[i][query]).or_insert(0) |= 1u32 << i;
        }
    }
    let mut result: Vec<u32> = buckets.values().copied().collect();
    result.sort_unstable();
    result
}

fn subset_mass(s: &State, allowed: u32) -> i64 {
    (0..s.n)
        .filter(|&i| allowed & (1u32 << i) != 0)
        .map(|i| s.masses[i])
        .sum()
}

fn pure(s: &State, allowed: u32) -> bool {
    let mut first: Option<i32> = None;
    for i in 0..s.n {
        if allowed & (1u32 << i) == 0 {
            continue;
        }
        match first {
            None => first = Some(s.labels[i]),
            Some(value) if value != s.labels[i] => return false,
            _ => {}
        }
    }
    true
}

fn cell_cost(s: &State, child: u32, query: usize) -> i64 {
    let index = child.trailing_zeros() as usize;
    s.costs[index][query]
}

fn immediate_expected_cost(s: &State, allowed: u32, query: usize) -> i64 {
    (0..s.n)
        .filter(|&i| allowed & (1u32 << i) != 0)
        .map(|i| s.masses[i] * s.costs[i][query])
        .sum()
}

fn dominates(a_query: usize, a: &[i64], b_query: usize, b: &[i64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let strict = a.iter().zip(b).all(|(left, right)| left <= right)
        && a.iter().zip(b).any(|(left, right)| left < right);
    strict || (a == b && a_query < b_query)
}

struct Solver<'a> {
    state: &'a State,
    memo: HashMap<Key, Plan>,
    stats: Stats,
}

impl<'a> Solver<'a> {
    fn canonical(&mut self, allowed: u32, remaining: u128) -> u128 {
        self.stats.raw += remaining.count_ones() as u64;
        let mut groups: BTreeMap<Vec<u32>, Vec<usize>> = BTreeMap::new();
        for query in 0..self.state.q {
            if remaining & (1u128 << query) == 0 {
                continue;
            }
            let signature = partition(self.state, allowed, query);
            if signature.len() > 1 {
                groups.entry(signature).or_default().push(query);
            }
        }
        let mut output = 0u128;
        for (signature, queries) in groups {
            let vectors: Vec<Vec<i64>> = queries
                .iter()
                .map(|&query| {
                    signature
                        .iter()
                        .map(|&child| cell_cost(self.state, child, query))
                        .collect()
                })
                .collect();
            for (index, &query) in queries.iter().enumerate() {
                let mut drop_query = false;
                for (other_index, &other) in queries.iter().enumerate() {
                    if index != other_index
                        && dominates(
                            other,
                            &vectors[other_index],
                            query,
                            &vectors[index],
                        )
                    {
                        drop_query = true;
                        break;
                    }
                }
                if !drop_query {
                    output |= 1u128 << query;
                }
            }
        }
        self.stats.kept += output.count_ones() as u64;
        self.stats.removed +=
            remaining.count_ones() as u64 - output.count_ones() as u64;
        output
    }

    fn child_first_step_bound(
        &self,
        allowed: u32,
        remaining: u128,
    ) -> Option<(i64, i64)> {
        if pure(self.state, allowed) {
            return Some((0, 0));
        }
        let mut best: Option<(i64, i64)> = None;
        for query in 0..self.state.q {
            if remaining & (1u128 << query) == 0 {
                continue;
            }
            let children = partition(self.state, allowed, query);
            if children.len() <= 1 {
                continue;
            }
            let expected = immediate_expected_cost(self.state, allowed, query);
            let worst = children
                .iter()
                .map(|&child| cell_cost(self.state, child, query))
                .max()
                .unwrap();
            match best {
                None => best = Some((expected, worst)),
                Some((best_expected, best_worst)) => {
                    if expected < best_expected
                        || (expected == best_expected && worst < best_worst)
                    {
                        best = Some((expected, worst));
                    }
                }
            }
        }
        best
    }

    fn candidate_bound(
        &self,
        allowed: u32,
        canonical: u128,
        query: usize,
    ) -> CandidateBound {
        let children = partition(self.state, allowed, query);
        let next_remaining = canonical & !(1u128 << query);
        let mut expected = immediate_expected_cost(self.state, allowed, query);
        let mut worst = 0i64;
        let mut possible = true;
        for &child in &children {
            match self.child_first_step_bound(child, next_remaining) {
                Some((child_expected, child_worst)) => {
                    expected += child_expected;
                    worst = worst.max(
                        cell_cost(self.state, child, query) + child_worst,
                    );
                }
                None => {
                    possible = false;
                    worst = worst.max(cell_cost(self.state, child, query));
                }
            }
        }
        CandidateBound {
            local_query: query,
            original_query: self.state.query_ids[query],
            children,
            full_diagnosis_possible: possible,
            expected_lower_bound: expected,
            worst_lower_bound: worst,
        }
    }

    fn compare_bounds(left: &CandidateBound, right: &CandidateBound) -> Ordering {
        (
            !left.full_diagnosis_possible,
            left.expected_lower_bound,
            left.worst_lower_bound,
            left.original_query,
        )
            .cmp(&(
                !right.full_diagnosis_possible,
                right.expected_lower_bound,
                right.worst_lower_bound,
                right.original_query,
            ))
    }

    fn incumbent_dominates_bound(
        incumbent: Plan,
        state_mass: i64,
        candidate: &CandidateBound,
    ) -> (bool, bool) {
        if incumbent.mass != state_mass {
            return (false, false);
        }
        if !candidate.full_diagnosis_possible {
            return (true, true);
        }
        if candidate.expected_lower_bound > incumbent.expected {
            return (true, false);
        }
        if candidate.expected_lower_bound < incumbent.expected {
            return (false, false);
        }
        if candidate.worst_lower_bound > incumbent.worst {
            return (true, false);
        }
        if candidate.worst_lower_bound < incumbent.worst {
            return (false, false);
        }
        (candidate.original_query > incumbent.query, false)
    }

    fn solve(&mut self, allowed: u32, remaining: u128) -> Result<Plan, ()> {
        self.stats.calls += 1;
        let canonical = self.canonical(allowed, remaining);
        let key = Key {
            allowed,
            remaining: canonical,
        };
        if let Some(plan) = self.memo.get(&key) {
            self.stats.memo_hits += 1;
            return Ok(*plan);
        }
        let state_mass = subset_mass(self.state, allowed);
        if pure(self.state, allowed) {
            let plan = Plan {
                mass: state_mass,
                expected: 0,
                worst: 0,
                query: i32::MAX,
            };
            self.memo.insert(key, plan);
            return Ok(plan);
        }

        let mut candidates = Vec::new();
        for query in 0..self.state.q {
            if canonical & (1u128 << query) == 0 {
                continue;
            }
            candidates.push(self.candidate_bound(allowed, canonical, query));
            self.stats.bound_evaluations += 1;
        }
        candidates.sort_by(Self::compare_bounds);

        let mut best: Option<Plan> = None;
        for candidate in candidates {
            if let Some(incumbent) = best {
                let (prune, impossible) = Self::incumbent_dominates_bound(
                    incumbent,
                    state_mass,
                    &candidate,
                );
                if prune {
                    self.stats.bound_pruned += 1;
                    self.stats.impossible_pruned += impossible as u64;
                    continue;
                }
            }

            self.stats.expansions += 1;
            if self.stats.expansions > BUDGET {
                return Err(());
            }
            let next_remaining =
                canonical & !(1u128 << candidate.local_query);
            let mut plan = Plan {
                mass: 0,
                expected: immediate_expected_cost(
                    self.state,
                    allowed,
                    candidate.local_query,
                ),
                worst: 0,
                query: candidate.original_query,
            };
            for child in candidate.children {
                let child_plan = self.solve(child, next_remaining)?;
                plan.mass += child_plan.mass;
                plan.expected += child_plan.expected;
                plan.worst = plan.worst.max(
                    cell_cost(self.state, child, candidate.local_query)
                        + child_plan.worst,
                );
            }
            match best {
                None => best = Some(plan),
                Some(incumbent) if better(plan, incumbent) => best = Some(plan),
                _ => {}
            }
        }

        let plan = best.unwrap_or_default();
        self.memo.insert(key, plan);
        Ok(plan)
    }
}

fn parse(path: &str) -> Vec<State> {
    let text = fs::read_to_string(path).unwrap();
    let mut tokens = text.split_whitespace();
    assert_eq!(tokens.next(), Some("COUNT"));
    let count: usize = tokens.next().unwrap().parse().unwrap();
    let mut output = Vec::new();
    for _ in 0..count {
        assert_eq!(tokens.next(), Some("STATE"));
        let digest = tokens.next().unwrap().to_string();
        let _seed = tokens.next();
        let n: usize = tokens.next().unwrap().parse().unwrap();
        let q: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!(tokens.next(), Some("LABELS"));
        let labels = (0..n)
            .map(|_| tokens.next().unwrap().parse().unwrap())
            .collect();
        assert_eq!(tokens.next(), Some("MASSES"));
        let masses = (0..n)
            .map(|_| tokens.next().unwrap().parse().unwrap())
            .collect();
        assert_eq!(tokens.next(), Some("QUERY_IDS"));
        let query_ids = (0..q)
            .map(|_| tokens.next().unwrap().parse().unwrap())
            .collect();
        let mut responses = vec![vec![0; q]; n];
        for row in 0..n {
            assert_eq!(tokens.next(), Some("RESPONSES"));
            for column in 0..q {
                responses[row][column] = tokens.next().unwrap().parse().unwrap();
            }
        }
        let mut costs = vec![vec![0; q]; n];
        for row in 0..n {
            assert_eq!(tokens.next(), Some("COSTS"));
            for column in 0..q {
                costs[row][column] = tokens.next().unwrap().parse().unwrap();
            }
        }
        assert_eq!(tokens.next(), Some("END"));
        output.push(State {
            digest,
            n,
            q,
            labels,
            masses,
            query_ids,
            responses,
            costs,
        });
    }
    output
}

fn plan_json(plan: Plan) -> String {
    let query = if plan.query == i32::MAX {
        "null".to_string()
    } else {
        plan.query.to_string()
    };
    format!(
        "[{},{},{},{}]",
        plan.mass, plan.expected, plan.worst, query
    )
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 5
        || arguments[1] != "--input"
        || arguments[3] != "--output"
    {
        eprintln!("usage: solver --input FILE --output FILE");
        std::process::exit(2);
    }
    let states = parse(&arguments[2]);
    let start = Instant::now();
    let mut rows = Vec::new();
    for state in &states {
        let mut solver = Solver {
            state,
            memo: HashMap::new(),
            stats: Stats::default(),
        };
        let allowed = if state.n == 32 {
            u32::MAX
        } else {
            (1u32 << state.n) - 1
        };
        let remaining = if state.q == 128 {
            u128::MAX
        } else {
            (1u128 << state.q) - 1
        };
        let plan = solver.solve(allowed, remaining);
        rows.push((
            state.digest.clone(),
            plan,
            solver.stats,
            solver.memo.len(),
        ));
    }

    let mut output = String::from("{\n  \"rows\": [\n");
    for (index, (digest, plan, stats, memo_entries)) in rows.iter().enumerate() {
        if index > 0 {
            output.push_str(",\n");
        }
        match plan {
            Ok(value) => output.push_str(&format!(
                "    {{\"digest\":\"{}\",\"solved\":true,\"plan\":{},\"query_expansions\":{},\"calls\":{},\"memo_entries\":{},\"memo_hits\":{},\"raw_queries_considered\":{},\"representative_queries_considered\":{},\"dominated_queries_removed\":{},\"bound_evaluations\":{},\"bound_pruned_queries\":{},\"impossible_full_diagnosis_prunes\":{}}}",
                digest,
                plan_json(*value),
                stats.expansions,
                stats.calls,
                memo_entries,
                stats.memo_hits,
                stats.raw,
                stats.kept,
                stats.removed,
                stats.bound_evaluations,
                stats.bound_pruned,
                stats.impossible_pruned,
            )),
            Err(_) => output.push_str(&format!(
                "    {{\"digest\":\"{}\",\"solved\":false}}",
                digest
            )),
        }
    }
    output.push_str(&format!(
        "\n  ],\n  \"total_milliseconds\": {}\n}}\n",
        start.elapsed().as_secs_f64() * 1000.0
    ));
    fs::write(&arguments[4], output).unwrap();
}
