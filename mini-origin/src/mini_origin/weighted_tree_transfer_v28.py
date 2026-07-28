from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np

from . import calibrated_decoder_v27 as v27
from . import tree_compiler_v26 as v26


@dataclass(frozen=True)
class WeightedTree:
    tree: list[set[int]]
    weights: np.ndarray
    strengths: dict[tuple[int, int], float]


@dataclass(frozen=True)
class QueryProgram:
    feature: str
    complexity: int

    def text(self) -> str:
        return f"argmin({self.feature})"


@dataclass(frozen=True)
class DecoderProgram:
    threshold: float
    normalize: bool

    def text(self) -> str:
        return f"minimum_below(threshold={self.threshold:.8f},normalize={self.normalize})"


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
    invalid_rate: float
    mean_mass_ratio: float
    half_mass_violation_rate: float


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def make_weighted_tree(tree: list[set[int]], rng: np.random.Generator) -> WeightedTree:
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=len(tree)).astype(float)
    strengths = {}
    for a, neighbors in enumerate(tree):
        for b in neighbors:
            if a < b:
                strengths[(a, b)] = float(rng.uniform(0.22, 0.95))
    return WeightedTree(tree, weights, strengths)


def distance_map(tree: list[set[int]], allowed: set[int], start: int) -> dict[int, int]:
    result, queue = {start: 0}, [start]
    for node in queue:
        for neighbor in tree[node]:
            if neighbor in allowed and neighbor not in result:
                result[neighbor] = result[node] + 1
                queue.append(neighbor)
    return result


def component_mass(instance: WeightedTree, allowed: set[int], start: int, blocked: int) -> float:
    nodes = v26.component(instance.tree, allowed, start, blocked)
    return float(np.sum(instance.weights[list(nodes)]))


def query_value(instance: WeightedTree, allowed: set[int], node: int, feature: str) -> float:
    if feature == "unweighted_sum_distance":
        return float(sum(distance_map(instance.tree, allowed, node).values()))
    if feature == "weighted_sum_distance":
        distances = distance_map(instance.tree, allowed, node)
        return float(sum(instance.weights[target] * distance for target, distance in distances.items()))
    if feature == "largest_component_mass":
        return max(
            (
                component_mass(instance, allowed, neighbor, node)
                for neighbor in instance.tree[node]
                if neighbor in allowed
            ),
            default=0.0,
        )
    raise ValueError(feature)


def select_query(instance: WeightedTree, allowed: set[int], program: QueryProgram) -> int:
    return min(allowed, key=lambda node: (query_value(instance, allowed, node, program.feature), node))


def responses(
    seed: int,
    instance: WeightedTree,
    allowed: set[int],
    query: int,
    root: int,
    replicates: int,
    normalize: bool,
) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    parent = v26.next_on_path(instance.tree, allowed, query, root)
    rows = []
    for neighbor in sorted(instance.tree[query]):
        if neighbor not in allowed:
            continue
        strength = instance.strengths[edge_key(query, neighbor)]
        active = neighbor != parent
        mean = strength if active else 0.0
        variance = 1.0 - strength * strength if active else 1.0
        value = float(rng.normal(mean, math.sqrt(variance / replicates)))
        rows.append((neighbor, value / strength if normalize else value))
    return rows


def decode(program: DecoderProgram, rows: list[tuple[int, float]]) -> int | None:
    neighbor, value = min(rows, key=lambda row: (row[1], row[0]))
    return neighbor if value <= program.threshold else None


def run_trial(
    seed: int,
    instance: WeightedTree,
    root: int,
    policy: Policy,
    replicates: int,
) -> tuple[bool, int, bool, list[tuple[float, float]]]:
    allowed = set(range(len(instance.tree)))
    total = float(np.sum(instance.weights))
    budget = max(2, math.ceil(math.log2(total / float(np.min(instance.weights)))) + 2)
    trace = []
    for step in range(budget):
        if len(allowed) <= 1:
            break
        before = float(np.sum(instance.weights[list(allowed)]))
        query = select_query(instance, allowed, policy.query)
        component_masses = [
            component_mass(instance, allowed, neighbor, query)
            for neighbor in instance.tree[query]
            if neighbor in allowed
        ]
        largest = max(component_masses, default=0.0)
        rows = responses(
            seed + step * 7_919,
            instance,
            allowed,
            query,
            root,
            replicates,
            policy.decoder.normalize,
        )
        move = decode(policy.decoder, rows)
        trace.append((before, largest))
        if move is None:
            return query == root, step + 1, False, trace
        if move not in allowed or move not in instance.tree[query]:
            return False, step + 1, True, trace
        allowed = v26.component(instance.tree, allowed, move, query)
        if root not in allowed:
            return False, step + 1, True, trace
    return len(allowed) == 1 and next(iter(allowed)) == root, len(trace), False, trace


def make_instances(seed: int, sizes: tuple[int, ...], random_count: int) -> list[WeightedTree]:
    rng = np.random.default_rng(seed)
    result = []
    for size in sizes:
        trees = [
            v26.path_tree(size),
            v26.balanced_tree(size),
            v26.star_tree(size),
            v26.broom_tree(size),
            v26.comet_tree(size),
        ]
        trees.extend(v26.random_tree(size, rng) for _ in range(random_count))
        result.extend(make_weighted_tree(tree, rng) for tree in trees)
    return result


def sample_tasks(seed: int, instances: list[WeightedTree], roots_per_tree: int) -> list[tuple[WeightedTree, int]]:
    rng = np.random.default_rng(seed)
    tasks = []
    for instance in instances:
        probabilities = instance.weights / np.sum(instance.weights)
        roots = rng.choice(len(instance.tree), size=roots_per_tree, replace=True, p=probabilities)
        tasks.extend((instance, int(root)) for root in roots)
    return tasks


def collect_decisions(
    seed: int,
    tasks: list[tuple[WeightedTree, int]],
    query: QueryProgram,
    replicates: int,
) -> list[v27.DecisionExample]:
    examples = []
    for index, (instance, root) in enumerate(tasks):
        allowed = set(range(len(instance.tree)))
        step = 0
        while len(allowed) > 1:
            node = select_query(instance, allowed, query)
            rows = responses(
                seed + index * 104_729 + step * 7_919,
                instance,
                allowed,
                node,
                root,
                replicates,
                normalize=True,
            )
            examples.append(v27.DecisionExample(min(value for _, value in rows), node != root))
            parent = v26.next_on_path(instance.tree, allowed, node, root)
            if parent is None:
                break
            allowed = v26.component(instance.tree, allowed, parent, node)
            step += 1
    return examples


def evaluate(seed: int, tasks: list[tuple[WeightedTree, int]], policy: Policy, replicates: int) -> Evaluation:
    rows = [
        run_trial(seed + index * 104_729, instance, root, policy, replicates)
        for index, (instance, root) in enumerate(tasks)
    ]
    trace = [entry for row in rows for entry in row[3]]
    ratios = [largest / before for before, largest in trace if before > 0]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean([row[1] for row in rows])),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_mass_ratio=float(np.mean(ratios)),
        half_mass_violation_rate=float(np.mean([ratio > 0.5 + 1e-12 for ratio in ratios])),
    )


def separator_certificate(instances: list[WeightedTree], query: QueryProgram) -> dict[str, object]:
    violations = 0
    checked = 0
    for instance in instances:
        stack = [set(range(len(instance.tree)))]
        while stack:
            allowed = stack.pop()
            if len(allowed) <= 1:
                continue
            node = select_query(instance, allowed, query)
            total = float(np.sum(instance.weights[list(allowed)]))
            for neighbor in instance.tree[node]:
                if neighbor not in allowed:
                    continue
                nodes = v26.component(instance.tree, allowed, neighbor, node)
                mass = float(np.sum(instance.weights[list(nodes)]))
                checked += 1
                violations += int(mass > total / 2.0 + 1e-10)
                stack.append(nodes)
    return {"components_checked": checked, "violations": violations, "passed": violations == 0}


def digest(policy: Policy) -> str:
    return hashlib.sha256(policy.text().encode()).hexdigest()


def run(seed: int = 1301) -> dict[str, object]:
    query_candidates = (
        QueryProgram("unweighted_sum_distance", 1),
        QueryProgram("weighted_sum_distance", 2),
        QueryProgram("largest_component_mass", 3),
    )
    training_instances = make_instances(seed * 10_000 + 97, (7, 11, 15), 2)
    development_instances = make_instances(seed * 10_000 + 1_000_099, (9, 13, 17), 2)
    training_tasks = sample_tasks(seed * 10_000 + 2_000_003, training_instances, 6)
    development_tasks = sample_tasks(seed * 10_000 + 3_000_007, development_instances, 8)
    rows = []
    for query in query_candidates:
        training = collect_decisions(seed * 10_000 + 4_000_009, training_tasks, query, 512)
        development = collect_decisions(seed * 10_000 + 5_000_011, development_tasks, query, 512)
        fit = v27.fit_threshold(training, development)
        policy = Policy(query, DecoderProgram(fit.threshold, True))
        score = evaluate(seed * 10_000 + 6_000_013, development_tasks, policy, 512)
        rows.append((score, fit, policy))
    selected_score, fit, selected = max(
        rows,
        key=lambda row: (
            -row[0].half_mass_violation_rate,
            row[0].accuracy,
            -row[0].mean_mass_ratio,
            -row[0].mean_queries,
            -row[2].query.complexity,
        ),
    )
    unweighted = Policy(
        QueryProgram("unweighted_sum_distance", 1),
        DecoderProgram(fit.threshold, True),
    )
    unnormalized = Policy(selected.query, DecoderProgram(fit.threshold, False))
    frozen_digest = digest(selected)

    hidden_instances = make_instances(seed * 10_000 + 12_000_001, (31, 63, 127, 255), 3)
    hidden_tasks = sample_tasks(seed * 10_000 + 13_000_003, hidden_instances, 16)
    candidate = evaluate(seed * 10_000 + 14_000_005, hidden_tasks, selected, 768)
    unweighted_result = evaluate(seed * 10_000 + 14_000_007, hidden_tasks, unweighted, 768)
    unnormalized_result = evaluate(seed * 10_000 + 14_000_009, hidden_tasks, unnormalized, 768)
    proof = separator_certificate(hidden_instances, selected.query)
    query_gain = unweighted_result.mean_queries - candidate.mean_queries
    mass_gain = unweighted_result.mean_mass_ratio - candidate.mean_mass_ratio
    normalization_gap = candidate.accuracy - unnormalized_result.accuracy
    gate = (
        selected.query.feature in {"weighted_sum_distance", "largest_component_mass"}
        and selected.decoder.normalize
        and proof["passed"]
        and fit.development_accuracy >= 0.985
        and candidate.accuracy >= 0.985
        and candidate.invalid_rate <= 0.01
        and candidate.half_mass_violation_rate == 0.0
        and candidate.mean_mass_ratio <= 0.50
        and (query_gain >= 0.10 or mass_gain >= 0.05)
        and normalization_gap >= 0.05
    )
    return {
        "status": "weighted_tree_transfer_candidate" if gate else "not_yet",
        "claim_scope": "a frozen policy learns to query weighted tree medians and normalize heterogeneous edge responses, then transfers to unseen nonuniform priors and causal strengths up to 255 nodes; this is weighted-median search with calibrated measurements, not a world breakthrough",
        "seed": seed,
        "candidate_gate": gate,
        "selected_policy": selected.text(),
        "threshold_fit": fit.__dict__,
        "development_score": selected_score.__dict__,
        "frozen_policy_digest": frozen_digest,
        "candidate": candidate.__dict__,
        "unweighted_control": unweighted_result.__dict__,
        "unnormalized_control": unnormalized_result.__dict__,
        "separator_certificate": proof,
        "unweighted_query_gain": query_gain,
        "unweighted_mass_ratio_gain": mass_gain,
        "normalization_accuracy_gap": normalization_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "policy": report["selected_policy"],
        "accuracy": report["candidate"]["accuracy"],
        "mass_ratio": report["candidate"]["mean_mass_ratio"],
        "normalization_gap": report["normalization_accuracy_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
