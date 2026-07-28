from __future__ import annotations

from dataclasses import dataclass
import argparse, functools, hashlib, json, math, random
from pathlib import Path

import numpy as np

from . import tree_compiler_v26 as v26


@dataclass(frozen=True)
class Query:
    name: str
    outcomes: tuple[int, ...]


@dataclass(frozen=True)
class Task:
    domain: str
    candidate_count: int
    queries: tuple[Query, ...]


@dataclass(frozen=True)
class Policy:
    query_rule: str
    decoder_rule: str

    def text(self) -> str:
        return f"query={self.query_rule};decoder={self.decoder_rule}"


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_rate: float
    mean_log2_ratio: float


def interval_task(n: int) -> Task:
    queries = []
    for query in range(n):
        outcomes = tuple(0 if candidate < query else 1 if candidate == query else 2 for candidate in range(n))
        queries.append(Query(f"compare-{query}", outcomes))
    return Task("interval", n, tuple(queries))


def tree_task(tree: list[set[int]]) -> Task:
    n = len(tree)
    queries = []
    allowed = set(range(n))
    for node in range(n):
        neighbors = sorted(tree[node])
        outcome_by_candidate = [0] * n
        for index, neighbor in enumerate(neighbors, start=1):
            nodes = v26.component(tree, allowed, neighbor, node)
            for candidate in nodes:
                outcome_by_candidate[candidate] = index
        queries.append(Query(f"vertex-{node}", tuple(outcome_by_candidate)))
    return Task("tree", n, tuple(queries))


def bitcode_task(bits: int) -> Task:
    n = 1 << bits
    queries = []
    for candidate in range(min(n, 16)):
        queries.append(Query(f"distractor-eq-{candidate}", tuple(int(value == candidate) for value in range(n))))
    for bit in range(bits):
        queries.append(Query(f"bit-{bit}", tuple((value >> bit) & 1 for value in range(n))))
    return Task("bitcode", n, tuple(queries))


def subset_task(n: int, seed: int) -> Task:
    rng = np.random.default_rng(seed)
    queries = []
    for index in range(16):
        pivot = int(rng.integers(0, n))
        queries.append(Query(f"distractor-{index}", tuple(int(value == pivot) for value in range(n))))
    bits = math.ceil(math.log2(n))
    for bit in range(bits):
        queries.append(Query(f"code-{bit}", tuple((value >> bit) & 1 for value in range(n))))
    for index in range(4):
        mask = rng.random(n) < 0.5
        queries.append(Query(f"balanced-{index}", tuple(int(value) for value in mask)))
    return Task("subset", n, tuple(queries))


def partitions(query: Query, allowed: frozenset[int]) -> dict[int, frozenset[int]]:
    rows: dict[int, set[int]] = {}
    for candidate in allowed:
        rows.setdefault(query.outcomes[candidate], set()).add(candidate)
    return {outcome: frozenset(values) for outcome, values in rows.items()}


def query_score(task: Task, allowed: frozenset[int], index: int, rule: str) -> tuple[float, ...]:
    parts = partitions(task.queries[index], allowed)
    sizes = sorted((len(values) for values in parts.values()), reverse=True)
    if len(sizes) <= 1:
        return (float("inf"),)
    if rule == "first":
        return (float(index),)
    if rule == "minimax_bucket":
        return (float(sizes[0]), float(sum(size * size for size in sizes)), float(index))
    if rule == "gini":
        return (float(sum(size * size for size in sizes)), float(sizes[0]), float(index))
    if rule == "max_outcomes":
        return (-float(len(sizes)), float(sizes[0]), float(index))
    raise ValueError(rule)


def select_query(task: Task, allowed: frozenset[int], remaining: frozenset[int], rule: str) -> int:
    candidates = [index for index in remaining if len(partitions(task.queries[index], allowed)) > 1]
    if not candidates:
        raise RuntimeError("no separating experiment remains")
    return min(candidates, key=lambda index: query_score(task, allowed, index, rule))


def observe(seed: int, query: Query, candidate: int, replicates: int = 256) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(seed)
    outcomes = sorted(set(query.outcomes))
    correct = query.outcomes[candidate]
    values = []
    for outcome in outcomes:
        strength = float(rng.uniform(0.30, 0.95))
        active = outcome != correct
        mean = strength if active else 0.0
        variance = 1.0 - strength * strength if active else 1.0
        raw = float(rng.normal(mean, math.sqrt(variance / replicates)))
        values.append(raw / strength)
    return outcomes, values


def decode(rule: str, outcomes: list[int], values: list[float]) -> int:
    if rule == "argmin":
        return outcomes[min(range(len(values)), key=lambda index: (values[index], outcomes[index]))]
    if rule == "argmax":
        return outcomes[max(range(len(values)), key=lambda index: (values[index], -outcomes[index]))]
    if rule == "first_below":
        for outcome, value in zip(outcomes, values):
            if value <= 0.20:
                return outcome
        return outcomes[0]
    raise ValueError(rule)


def run_trial(seed: int, task: Task, candidate: int, policy: Policy) -> tuple[bool, int, bool]:
    allowed = frozenset(range(task.candidate_count))
    remaining = frozenset(range(len(task.queries)))
    budget = math.ceil(math.log2(task.candidate_count)) + 3
    for step in range(budget):
        if len(allowed) <= 1:
            break
        try:
            index = select_query(task, allowed, remaining, policy.query_rule)
        except RuntimeError:
            return False, step, True
        query = task.queries[index]
        outcomes, values = observe(seed + step * 7_919, query, candidate)
        outcome = decode(policy.decoder_rule, outcomes, values)
        parts = partitions(query, allowed)
        if outcome not in parts:
            return False, step + 1, True
        allowed = parts[outcome]
        remaining = frozenset(value for value in remaining if value != index)
        if candidate not in allowed:
            return False, step + 1, True
    return len(allowed) == 1 and next(iter(allowed)) == candidate, min(budget, task.candidate_count), False


def evaluate(seed: int, tasks: list[Task], trials_per_task: int, policy: Policy) -> Evaluation:
    rng = np.random.default_rng(seed)
    rows = []
    sizes = []
    for task_index, task in enumerate(tasks):
        candidates = rng.integers(0, task.candidate_count, size=trials_per_task)
        for trial, candidate in enumerate(candidates):
            row = run_trial(seed + task_index * 1_000_003 + trial * 104_729, task, int(candidate), policy)
            rows.append(row)
            sizes.append(task.candidate_count)
    query_counts = [row[1] for row in rows]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean(query_counts)),
        maximum_queries=max(query_counts),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_log2_ratio=float(np.mean([queries / max(1, math.ceil(math.log2(size))) for queries, size in zip(query_counts, sizes)])),
    )


def noiseless_depth(task: Task, policy: Policy, candidate: int) -> int:
    allowed = frozenset(range(task.candidate_count))
    remaining = frozenset(range(len(task.queries)))
    depth = 0
    while len(allowed) > 1:
        index = select_query(task, allowed, remaining, policy.query_rule)
        query = task.queries[index]
        allowed = partitions(query, allowed)[query.outcomes[candidate]]
        remaining = frozenset(value for value in remaining if value != index)
        depth += 1
    return depth


@functools.lru_cache(maxsize=None)
def optimal_depth_cached(outcome_rows: tuple[tuple[int, ...], ...], allowed: tuple[int, ...], remaining: tuple[int, ...]) -> int:
    if len(allowed) <= 1:
        return 0
    best = math.inf
    allowed_set = frozenset(allowed)
    for index in remaining:
        query = Query(str(index), outcome_rows[index])
        parts = partitions(query, allowed_set)
        if len(parts) <= 1:
            continue
        next_remaining = tuple(value for value in remaining if value != index)
        depth = 1 + max(optimal_depth_cached(outcome_rows, tuple(sorted(part)), next_remaining) for part in parts.values())
        best = min(best, depth)
    return int(best) if best < math.inf else 10**9


def exact_certificate(tasks: list[Task], policy: Policy) -> dict[str, object]:
    rows = []
    for task in tasks:
        outcome_rows = tuple(query.outcomes for query in task.queries)
        optimal = optimal_depth_cached(outcome_rows, tuple(range(task.candidate_count)), tuple(range(len(task.queries))))
        greedy = max(noiseless_depth(task, policy, candidate) for candidate in range(task.candidate_count))
        rows.append({"domain": task.domain, "size": task.candidate_count, "optimal": optimal, "greedy": greedy, "gap": greedy - optimal})
    return {"rows": rows, "maximum_gap": max(row["gap"] for row in rows), "passed": all(row["gap"] <= 1 for row in rows)}


def training_tasks(seed: int) -> list[Task]:
    rng = np.random.default_rng(seed)
    tasks = [interval_task(size) for size in (7, 11, 15)]
    for size in (7, 11, 15):
        tasks.extend(tree_task(tree) for tree in (
            v26.path_tree(size), v26.balanced_tree(size), v26.broom_tree(size), v26.random_tree(size, rng)
        ))
    return tasks


def development_tasks(seed: int) -> list[Task]:
    rng = np.random.default_rng(seed)
    tasks = [interval_task(size) for size in (9, 13, 17)]
    for size in (9, 13, 17):
        tasks.extend(tree_task(tree) for tree in (
            v26.star_tree(size), v26.comet_tree(size), v26.random_tree(size, rng), v26.random_tree(size, rng)
        ))
    return tasks


def hidden_tasks(seed: int) -> list[Task]:
    return [
        bitcode_task(6), bitcode_task(8), bitcode_task(10),
        subset_task(63, seed + 1), subset_task(127, seed + 2), subset_task(255, seed + 3),
    ]


def policies() -> tuple[Policy, ...]:
    return tuple(Policy(query, decoder) for query in ("first", "minimax_bucket", "gini", "max_outcomes") for decoder in ("argmin", "argmax", "first_below"))


def random_query_control(seed: int, tasks: list[Task], trials_per_task: int) -> Evaluation:
    rng = random.Random(seed)
    rows = []
    sizes = []
    for task_index, task in enumerate(tasks):
        for trial in range(trials_per_task):
            candidate = rng.randrange(task.candidate_count)
            allowed = frozenset(range(task.candidate_count))
            remaining = list(range(len(task.queries)))
            budget = math.ceil(math.log2(task.candidate_count)) + 3
            invalid = False
            for step in range(budget):
                if len(allowed) <= 1:
                    break
                separating = [index for index in remaining if len(partitions(task.queries[index], allowed)) > 1]
                if not separating:
                    invalid = True
                    break
                index = rng.choice(separating)
                remaining.remove(index)
                query = task.queries[index]
                outcomes, values = observe(seed + task_index * 1_000_003 + trial * 104_729 + step, query, candidate)
                outcome = decode("argmin", outcomes, values)
                allowed = partitions(query, allowed).get(outcome, frozenset())
                if candidate not in allowed:
                    invalid = True
                    break
            rows.append((len(allowed) == 1 and next(iter(allowed), -1) == candidate, budget, invalid))
            sizes.append(task.candidate_count)
    return Evaluation(float(np.mean([row[0] for row in rows])), float(np.mean([row[1] for row in rows])), max(row[1] for row in rows), float(np.mean([row[2] for row in rows])), float(np.mean([row[1] / math.ceil(math.log2(size)) for row, size in zip(rows, sizes)])))


def digest(policy: Policy) -> str:
    return hashlib.sha256(policy.text().encode()).hexdigest()


def run(seed: int = 1501) -> dict[str, object]:
    train = training_tasks(seed * 10_000 + 107)
    development = development_tasks(seed * 10_000 + 1_000_109)
    rows = []
    for policy in policies():
        train_score = evaluate(seed * 10_000 + 2_000_003, train, 24, policy)
        development_score = evaluate(seed * 10_000 + 3_000_007, development, 32, policy)
        score = (development_score.accuracy, -development_score.invalid_rate, -development_score.mean_queries, -len(policy.text()), policy.text())
        rows.append((score, policy, train_score, development_score))
    _, selected, train_score, development_score = max(rows, key=lambda row: row[0])
    frozen_digest = digest(selected)

    # Entirely new diagnosis domains are created only after the partition rule,
    # decoder and digest are frozen.
    hidden = hidden_tasks(seed * 10_000 + 12_000_001)
    candidate = evaluate(seed * 10_000 + 13_000_003, hidden, 80, selected)
    first_control = evaluate(seed * 10_000 + 13_000_005, hidden, 80, Policy("first", "argmin"))
    random_control = random_query_control(seed * 10_000 + 13_000_007, hidden, 80)
    small_hidden = [bitcode_task(4), subset_task(10, seed * 10_000 + 13_000_009)]
    proof = exact_certificate(small_hidden, selected)
    first_gap = candidate.accuracy - first_control.accuracy
    random_gap = candidate.accuracy - random_control.accuracy
    gate = (
        selected.query_rule == "minimax_bucket"
        and selected.decoder_rule == "argmin"
        and development_score.accuracy >= 0.985
        and candidate.accuracy >= 0.985
        and candidate.invalid_rate <= 0.01
        and candidate.mean_log2_ratio <= 1.20
        and first_gap >= 0.20
        and random_gap >= 0.20
        and proof["passed"]
    )
    return {
        "status": "cross_domain_partition_compiler_candidate" if gate else "not_yet",
        "claim_scope": "a topology-blind policy trained on interval and tree partitions transfers without modification to binary-code and subset-diagnosis holdouts by selecting the experiment with the smallest worst-case outcome bucket and decoding the minimum response channel; this is generalized binary search and active diagnosis, not a world breakthrough",
        "seed": seed,
        "candidate_gate": gate,
        "selected_policy": selected.text(),
        "training_score": train_score.__dict__,
        "development_score": development_score.__dict__,
        "frozen_policy_digest": frozen_digest,
        "hidden_domains": [task.domain for task in hidden],
        "candidate": candidate.__dict__,
        "first_query_control": first_control.__dict__,
        "random_query_control": random_control.__dict__,
        "first_accuracy_gap": first_gap,
        "random_accuracy_gap": random_gap,
        "exact_small_task_certificate": proof,
        "policies_evaluated": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "policy": report["selected_policy"],
        "accuracy": report["candidate"]["accuracy"],
        "log2_ratio": report["candidate"]["mean_log2_ratio"],
        "first_gap": report["first_accuracy_gap"],
        "random_gap": report["random_accuracy_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
