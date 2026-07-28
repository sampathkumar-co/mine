from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, heapq, json, math, random
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueryProgram:
    feature: str
    complexity: int

    def text(self) -> str:
        return f"argmin({self.feature})"


@dataclass(frozen=True)
class DecoderProgram:
    rule: str
    threshold: float
    complexity: int

    def text(self) -> str:
        return f"{self.rule}(threshold={self.threshold:.8f})"


@dataclass(frozen=True)
class Policy:
    query: QueryProgram
    decoder: DecoderProgram

    def text(self) -> str:
        return f"query={self.query.text()};decoder={self.decoder.text()}"


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_rate: float
    mean_log2_ratio: float
    half_shrink_violation_rate: float


def empty_tree(n: int) -> list[set[int]]:
    return [set() for _ in range(n)]


def add_edge(tree: list[set[int]], a: int, b: int) -> None:
    tree[a].add(b)
    tree[b].add(a)


def path_tree(n: int) -> list[set[int]]:
    tree = empty_tree(n)
    for i in range(n - 1):
        add_edge(tree, i, i + 1)
    return tree


def balanced_tree(n: int) -> list[set[int]]:
    tree = empty_tree(n)
    for child in range(1, n):
        add_edge(tree, child, (child - 1) // 2)
    return tree


def star_tree(n: int) -> list[set[int]]:
    tree = empty_tree(n)
    for node in range(1, n):
        add_edge(tree, 0, node)
    return tree


def broom_tree(n: int) -> list[set[int]]:
    tree = empty_tree(n)
    handle = max(2, n // 2)
    for i in range(handle - 1):
        add_edge(tree, i, i + 1)
    for node in range(handle, n):
        add_edge(tree, handle - 1, node)
    return tree


def comet_tree(n: int) -> list[set[int]]:
    tree = empty_tree(n)
    leaves = max(2, n // 4)
    for node in range(1, leaves + 1):
        add_edge(tree, 0, node)
    if leaves + 1 < n:
        add_edge(tree, 0, leaves + 1)
        for node in range(leaves + 1, n - 1):
            add_edge(tree, node, node + 1)
    return tree


def random_tree(n: int, rng: np.random.Generator) -> list[set[int]]:
    if n <= 2:
        return path_tree(n)
    sequence = [int(x) for x in rng.integers(0, n, size=n - 2)]
    degree = [1] * n
    for value in sequence:
        degree[value] += 1
    leaves = [i for i, value in enumerate(degree) if value == 1]
    heapq.heapify(leaves)
    tree = empty_tree(n)
    for value in sequence:
        leaf = heapq.heappop(leaves)
        add_edge(tree, leaf, value)
        degree[leaf] -= 1
        degree[value] -= 1
        if degree[value] == 1:
            heapq.heappush(leaves, value)
    a, b = heapq.heappop(leaves), heapq.heappop(leaves)
    add_edge(tree, a, b)
    return tree


def component(tree: list[set[int]], allowed: set[int], start: int, blocked: int) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for neighbor in tree[node]:
            if neighbor == blocked or neighbor not in allowed or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


def component_sizes(tree: list[set[int]], allowed: set[int], node: int) -> list[int]:
    return [len(component(tree, allowed, neighbor, node))
            for neighbor in tree[node] if neighbor in allowed]


def distances(tree: list[set[int]], allowed: set[int], start: int) -> list[int]:
    distance, queue = {start: 0}, [start]
    for node in queue:
        for neighbor in tree[node]:
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return list(distance.values())


def query_feature(tree: list[set[int]], allowed: set[int], node: int, name: str) -> float:
    sizes = component_sizes(tree, allowed, node)
    if name == "largest_component":
        return float(max(sizes, default=0))
    if name == "eccentricity":
        return float(max(distances(tree, allowed, node), default=0))
    if name == "sum_distance":
        return float(sum(distances(tree, allowed, node)))
    if name == "negative_degree":
        return -float(sum(neighbor in allowed for neighbor in tree[node]))
    if name == "node_id":
        return float(node)
    raise ValueError(name)


def select_query(tree: list[set[int]], allowed: set[int], program: QueryProgram) -> int:
    return min(allowed, key=lambda node: (query_feature(tree, allowed, node, program.feature), node))


def next_on_path(tree: list[set[int]], allowed: set[int], query: int, root: int) -> int | None:
    if query == root:
        return None
    for neighbor in tree[query]:
        if neighbor in allowed and root in component(tree, allowed, neighbor, query):
            return neighbor
    raise RuntimeError("root not reachable inside candidate subtree")


def local_responses(seed: int, tree: list[set[int]], allowed: set[int], query: int,
                    root: int, rho: float, replicates: int) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    parent = next_on_path(tree, allowed, query, root)
    rows = []
    for neighbor in sorted(tree[query]):
        if neighbor not in allowed:
            continue
        active = neighbor != parent
        mean = rho if active else 0.0
        variance = 1.0 - rho * rho if active else 1.0
        rows.append((neighbor, float(rng.normal(mean, math.sqrt(variance / replicates)))))
    return rows


def decode(program: DecoderProgram, responses: list[tuple[int, float]]) -> int | None:
    if not responses:
        return None
    if program.rule == "minimum_below":
        neighbor, value = min(responses, key=lambda row: (row[1], row[0]))
        return neighbor if value <= program.threshold else None
    if program.rule == "first_below":
        for neighbor, value in responses:
            if value <= program.threshold:
                return neighbor
        return None
    if program.rule == "maximum_above":
        neighbor, value = max(responses, key=lambda row: (row[1], -row[0]))
        return neighbor if value >= program.threshold else None
    raise ValueError(program.rule)


def run_trial(seed: int, tree: list[set[int]], root: int, policy: Policy,
              rho: float, replicates: int, budget: int | None = None) -> tuple[bool, int, bool, list[dict[str, int]]]:
    allowed, queries, trace = set(range(len(tree))), 0, []
    budget = budget or max(1, math.ceil(math.log2(len(tree))) + 2)
    while len(allowed) > 1 and queries < budget:
        query = select_query(tree, allowed, policy.query)
        largest = int(max(component_sizes(tree, allowed, query), default=0))
        responses = local_responses(seed + queries * 7919, tree, allowed, query, root, rho, replicates)
        move = decode(policy.decoder, responses)
        queries += 1
        trace.append({"candidate_count": len(allowed), "query": query, "largest_component": largest})
        if move is None:
            return query == root, queries, False, trace
        if move not in tree[query] or move not in allowed:
            return False, queries, True, trace
        allowed = component(tree, allowed, move, query)
        if root not in allowed:
            return False, queries, True, trace
    prediction = next(iter(allowed))
    return prediction == root, queries, False, trace


def evaluate(seed: int, tasks: list[tuple[list[set[int]], int]], policy: Policy,
             replicates: int, noisy: bool = True) -> Evaluation:
    rng, rows = np.random.default_rng(seed), []
    for i, (tree, root) in enumerate(tasks):
        rho = float(rng.uniform(0.30, 0.90)) if noisy else 1.0
        rows.append(run_trial(seed + i * 104729, tree, root, policy, rho, replicates))
    ratios = [row[1] / max(1, math.ceil(math.log2(len(tasks[i][0])))) for i, row in enumerate(rows)]
    trace_rows = [trace for row in rows for trace in row[3]]
    violations = [trace["largest_component"] > trace["candidate_count"] // 2 for trace in trace_rows]
    return Evaluation(float(np.mean([row[0] for row in rows])),
                      float(np.mean([row[1] for row in rows])),
                      max(row[1] for row in rows),
                      float(np.mean([row[2] for row in rows])),
                      float(np.mean(ratios)),
                      float(np.mean(violations)) if violations else 0.0)


def make_tasks(seed: int, sizes: tuple[int, ...], roots_per_tree: int,
               random_count: int, include_structured: bool = True) -> list[tuple[list[set[int]], int]]:
    rng, trees = np.random.default_rng(seed), []
    for size in sizes:
        if include_structured:
            trees.extend((path_tree(size), balanced_tree(size), broom_tree(size),
                          star_tree(size), comet_tree(size)))
        trees.extend(random_tree(size, rng) for _ in range(random_count))
    tasks = []
    for tree in trees:
        roots = list(range(len(tree))) if roots_per_tree >= len(tree) else [int(x) for x in rng.choice(len(tree), roots_per_tree, replace=False)]
        tasks.extend((tree, root) for root in roots)
    return tasks


def query_programs() -> tuple[QueryProgram, ...]:
    return (
        QueryProgram("node_id", 1), QueryProgram("negative_degree", 2),
        QueryProgram("eccentricity", 3), QueryProgram("sum_distance", 3),
        QueryProgram("largest_component", 3),
    )


def decoder_programs() -> list[DecoderProgram]:
    thresholds = (0.10, 0.15, 0.20)
    rows = []
    for threshold in thresholds:
        rows.extend((DecoderProgram("minimum_below", threshold, 2),
                     DecoderProgram("first_below", threshold, 3),
                     DecoderProgram("maximum_above", threshold, 3)))
    return rows


def synthesize(seed: int) -> tuple[Policy, Policy, dict[str, object]]:
    training = make_tasks(seed, (7, 11, 15), 4, 1)
    development = make_tasks(seed + 1000003, (9, 13, 17), 5, 2)
    rows = []
    for query in query_programs():
        for decoder in decoder_programs():
            policy = Policy(query, decoder)
            train = evaluate(seed + 2000003, training, policy, 384)
            dev = evaluate(seed + 3000007, development, policy, 384)
            score = (-dev.half_shrink_violation_rate, dev.accuracy, -dev.invalid_rate,
                     -dev.mean_queries, -dev.maximum_queries,
                     -(query.complexity + decoder.complexity), policy.text())
            rows.append((score, policy, train, dev))
    _, selected, selected_train, selected_dev = max(rows, key=lambda row: row[0])
    separator_features = {"largest_component", "sum_distance"}
    controls = [row for row in rows if row[1].query.feature not in separator_features]
    _, reduced, reduced_train, reduced_dev = max(controls, key=lambda row: row[0])
    return selected, reduced, {
        "training_task_count": len(training), "development_task_count": len(development),
        "programs_evaluated": len(rows), "selected_policy": selected.text(),
        "selected_training": selected_train.__dict__, "selected_development": selected_dev.__dict__,
        "reduced_policy": reduced.text(), "reduced_training": reduced_train.__dict__,
        "reduced_development": reduced_dev.__dict__,
    }


def exact_certificate(tasks: list[tuple[list[set[int]], int]], policy: Policy) -> dict[str, object]:
    failures, shrink_failures, max_queries = 0, 0, 0
    for i, (tree, root) in enumerate(tasks):
        correct, queries, invalid, trace = run_trial(9000001 + i, tree, root, policy, 1.0, 4096)
        failures += int(not correct or invalid)
        max_queries = max(max_queries, queries)
        for row in trace:
            if row["largest_component"] > row["candidate_count"] // 2:
                shrink_failures += 1
    return {"task_count": len(tasks), "failures": failures,
            "half_shrink_failures": shrink_failures, "maximum_queries": max_queries,
            "passed": failures == 0 and shrink_failures == 0}


def passive_control(tasks: list[tuple[list[set[int]], int]]) -> float:
    return float(np.mean([1.0 / len(tree) for tree, _ in tasks]))


def random_decoder_control(seed: int, tasks: list[tuple[list[set[int]], int]],
                           query: QueryProgram, trials: int = 12) -> dict[str, float]:
    rng, scores = random.Random(seed), []
    for _ in range(trials):
        correct = 0
        for tree, root in tasks:
            allowed = set(range(len(tree)))
            for _ in range(math.ceil(math.log2(len(tree))) + 2):
                if len(allowed) <= 1:
                    break
                node = select_query(tree, allowed, query)
                choices = [None] + [n for n in tree[node] if n in allowed]
                move = rng.choice(choices)
                if move is None:
                    correct += int(node == root)
                    allowed = set()
                    break
                allowed = component(tree, allowed, move, node)
            if len(allowed) == 1:
                correct += int(next(iter(allowed)) == root)
        scores.append(correct / len(tasks))
    return {"trials": trials, "median_accuracy": float(np.median(scores)), "maximum_accuracy": max(scores)}


def digest(policy: Policy) -> str:
    return hashlib.sha256(policy.text().encode()).hexdigest()


def run(seed: int = 1101) -> dict[str, object]:
    selected, reduced, synthesis = synthesize(seed * 10000 + 83)
    frozen_digest = digest(selected)

    # Hidden families and sizes are generated only after the query expression,
    # decoder, reduced control and digest are frozen.
    hidden_sizes = (17, 31, 63, 127)
    hidden_tasks = make_tasks(seed * 10000 + 11000001, hidden_sizes, 10, 3)
    exact_tasks = make_tasks(seed * 10000 + 12000001, (9, 13, 17, 25), 1000, 2)
    candidate = evaluate(seed * 10000 + 13000003, hidden_tasks, selected, 512)
    reduced_result = evaluate(seed * 10000 + 13000005, hidden_tasks, reduced, 512)
    human = evaluate(seed * 10000 + 13000007, hidden_tasks,
                     Policy(QueryProgram("largest_component", 3), DecoderProgram("minimum_below", 0.15, 2)), 512)
    proof = exact_certificate(exact_tasks, selected)
    random_decoder = random_decoder_control(seed * 10000 + 13000009, hidden_tasks, selected.query)
    passive = passive_control(hidden_tasks)
    reduced_gap = candidate.accuracy - reduced_result.accuracy
    reduced_query_gain = reduced_result.mean_queries - candidate.mean_queries
    reduced_shrink_gap = (
        reduced_result.half_shrink_violation_rate
        - candidate.half_shrink_violation_rate
    )
    random_gap = candidate.accuracy - random_decoder["median_accuracy"]
    passive_gap = candidate.accuracy - passive
    human_query_gap = candidate.mean_queries - human.mean_queries
    gate = (
        selected.query.feature in {"largest_component", "sum_distance"}
        and selected.decoder.rule == "minimum_below"
        and proof["passed"]
        and candidate.accuracy >= 0.985 and candidate.invalid_rate <= 0.01
        and candidate.mean_log2_ratio <= 1.15
        and candidate.half_shrink_violation_rate == 0.0
        and (reduced_query_gain >= 0.10 or reduced_shrink_gap >= 0.05)
        and random_gap >= 0.50 and passive_gap >= 0.80
        and human_query_gap <= 0.10
    )
    return {
        "status": "tree_separator_compiler_candidate" if gate else "not_yet",
        "claim_scope": "a frozen grammar search jointly selects a graph-separator query expression and a raw local-response decoder, then transfers from small development trees to unseen paths, stars, brooms, balanced trees and random trees up to 127 nodes; the induced policy is centroid search with a thresholded minimum-response direction oracle, an established algorithmic pattern rather than a world breakthrough",
        "seed": seed, "candidate_gate": gate, "synthesis": synthesis,
        "selected_policy": {"query": selected.query.text(), "decoder": selected.decoder.text()},
        "reduced_policy": {"query": reduced.query.text(), "decoder": reduced.decoder.text()},
        "frozen_policy_digest": frozen_digest, "hidden_sizes": list(hidden_sizes),
        "candidate": candidate.__dict__, "reduced_control": reduced_result.__dict__,
        "human_control": human.__dict__, "exact_certificate": proof,
        "random_decoder_control": random_decoder, "passive_control_accuracy": passive,
        "reduced_accuracy_gap": reduced_gap,
        "reduced_mean_query_gain": reduced_query_gain,
        "reduced_half_shrink_gap": reduced_shrink_gap,
        "random_accuracy_gap": random_gap,
        "passive_accuracy_gap": passive_gap, "human_mean_query_gap": human_query_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "policy": report["selected_policy"],
                      "accuracy": report["candidate"]["accuracy"],
                      "mean_queries": report["candidate"]["mean_queries"],
                      "reduced_query_gain": report["reduced_mean_query_gain"],
                      "shrink_gap": report["reduced_half_shrink_gap"]}, indent=2))


if __name__ == "__main__":
    main()
