from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np

from . import tree_compiler_v26 as v26
from . import weighted_tree_transfer_v28 as v28


@dataclass(frozen=True)
class BudgetProgram:
    batches: tuple[int, ...]
    z: float

    def text(self) -> str:
        return f"batches={','.join(map(str,self.batches))};z={self.z:.2f}"


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    mean_queries: float
    invalid_rate: float
    mean_observations: float
    p95_observations: float
    half_mass_violation_rate: float


def sequential_decision(
    seed: int,
    instance: v28.WeightedTree,
    allowed: set[int],
    query: int,
    root: int,
    program: BudgetProgram,
) -> tuple[int | None, int]:
    rng = np.random.default_rng(seed)
    parent = v26.next_on_path(instance.tree, allowed, query, root)
    neighbors = [neighbor for neighbor in sorted(instance.tree[query]) if neighbor in allowed]
    sums = {neighbor: 0.0 for neighbor in neighbors}
    previous = 0
    observations = 0
    for cumulative in program.batches:
        batch = cumulative - previous
        previous = cumulative
        for neighbor in neighbors:
            strength = instance.strengths[v28.edge_key(query, neighbor)]
            active = neighbor != parent
            mean = strength if active else 0.0
            variance = 1.0 - strength * strength if active else 1.0
            values = rng.normal(mean, math.sqrt(variance), size=batch) / strength
            sums[neighbor] += float(np.sum(values))
            observations += batch
        means = {neighbor: sums[neighbor] / cumulative for neighbor in neighbors}
        errors = {
            neighbor: program.z
            / (instance.strengths[v28.edge_key(query, neighbor)] * math.sqrt(cumulative))
            for neighbor in neighbors
        }
        lower = {neighbor: means[neighbor] - errors[neighbor] for neighbor in neighbors}
        upper = {neighbor: means[neighbor] + errors[neighbor] for neighbor in neighbors}
        candidate = min(neighbors, key=lambda neighbor: (means[neighbor], neighbor))
        others = [neighbor for neighbor in neighbors if neighbor != candidate]
        if upper[candidate] < 0.5 and all(lower[neighbor] > 0.5 for neighbor in others):
            return candidate, observations
        if all(lower[neighbor] > 0.5 for neighbor in neighbors):
            return None, observations
    means = {neighbor: sums[neighbor] / program.batches[-1] for neighbor in neighbors}
    candidate = min(neighbors, key=lambda neighbor: (means[neighbor], neighbor))
    return (candidate if means[candidate] <= 0.5 else None), observations


def fixed_decision(
    seed: int,
    instance: v28.WeightedTree,
    allowed: set[int],
    query: int,
    root: int,
    replicates: int = 512,
) -> tuple[int | None, int]:
    rows = v28.responses(seed, instance, allowed, query, root, replicates, True)
    neighbor, value = min(rows, key=lambda row: (row[1], row[0]))
    return (neighbor if value <= 0.5 else None), replicates * len(rows)


def run_trial(
    seed: int,
    instance: v28.WeightedTree,
    root: int,
    query_program: v28.QueryProgram,
    budget_program: BudgetProgram | None,
) -> tuple[bool, int, bool, int, list[tuple[float, float]]]:
    allowed = set(range(len(instance.tree)))
    total = float(np.sum(instance.weights))
    budget = max(2, math.ceil(math.log2(total / float(np.min(instance.weights)))) + 2)
    observations = 0
    trace = []
    for step in range(budget):
        if len(allowed) <= 1:
            break
        before = float(np.sum(instance.weights[list(allowed)]))
        query = v28.select_query(instance, allowed, query_program)
        component_masses = [
            v28.component_mass(instance, allowed, neighbor, query)
            for neighbor in instance.tree[query]
            if neighbor in allowed
        ]
        largest = max(component_masses, default=0.0)
        if budget_program is None:
            move, cost = fixed_decision(seed + step * 7_919, instance, allowed, query, root)
        else:
            move, cost = sequential_decision(
                seed + step * 7_919,
                instance,
                allowed,
                query,
                root,
                budget_program,
            )
        observations += cost
        trace.append((before, largest))
        if move is None:
            return query == root, step + 1, False, observations, trace
        if move not in allowed or move not in instance.tree[query]:
            return False, step + 1, True, observations, trace
        allowed = v26.component(instance.tree, allowed, move, query)
        if root not in allowed:
            return False, step + 1, True, observations, trace
    return len(allowed) == 1 and next(iter(allowed)) == root, len(trace), False, observations, trace


def evaluate(
    seed: int,
    tasks: list[tuple[v28.WeightedTree, int]],
    query_program: v28.QueryProgram,
    budget_program: BudgetProgram | None,
) -> Evaluation:
    rows = [
        run_trial(seed + index * 104_729, instance, root, query_program, budget_program)
        for index, (instance, root) in enumerate(tasks)
    ]
    trace = [entry for row in rows for entry in row[4]]
    violations = [largest > before / 2.0 + 1e-10 for before, largest in trace]
    observations = [row[3] for row in rows]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean([row[1] for row in rows])),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_observations=float(np.mean(observations)),
        p95_observations=float(np.quantile(observations, 0.95)),
        half_mass_violation_rate=float(np.mean(violations)) if violations else 0.0,
    )


def programs() -> tuple[BudgetProgram, ...]:
    schedules = (
        (16, 32, 64, 128, 256, 512),
        (32, 64, 128, 256, 512),
        (64, 128, 256, 512),
        (32, 96, 192, 384, 512),
    )
    return tuple(BudgetProgram(schedule, z) for schedule in schedules for z in (1.96, 2.58, 3.29))


def digest(query: v28.QueryProgram, program: BudgetProgram) -> str:
    return hashlib.sha256(f"{query.text()}:{program.text()}".encode()).hexdigest()


def run(seed: int = 1401) -> dict[str, object]:
    query = v28.QueryProgram("weighted_sum_distance", 2)
    training_instances = v28.make_instances(seed * 10_000 + 101, (7, 11, 15), 2)
    development_instances = v28.make_instances(seed * 10_000 + 1_000_103, (9, 13, 17), 2)
    training_tasks = v28.sample_tasks(seed * 10_000 + 2_000_003, training_instances, 8)
    development_tasks = v28.sample_tasks(seed * 10_000 + 3_000_007, development_instances, 10)
    rows = []
    for program in programs():
        train = evaluate(seed * 10_000 + 4_000_009, training_tasks, query, program)
        development = evaluate(seed * 10_000 + 5_000_011, development_tasks, query, program)
        score = (
            development.accuracy,
            -development.invalid_rate,
            -development.mean_observations,
            -development.p95_observations,
            program.text(),
        )
        rows.append((score, program, train, development))
    eligible = [row for row in rows if row[3].accuracy >= 0.99 and row[3].invalid_rate <= 0.005]
    selected_row = max(eligible or rows, key=lambda row: row[0])
    _, selected, train_score, development_score = selected_row
    frozen_digest = digest(query, selected)

    hidden_instances = v28.make_instances(seed * 10_000 + 12_000_001, (31, 63, 127), 3)
    hidden_tasks = v28.sample_tasks(seed * 10_000 + 13_000_003, hidden_instances, 16)
    candidate = evaluate(seed * 10_000 + 14_000_005, hidden_tasks, query, selected)
    fixed = evaluate(seed * 10_000 + 14_000_007, hidden_tasks, query, None)
    observation_ratio = candidate.mean_observations / fixed.mean_observations
    p95_ratio = candidate.p95_observations / fixed.p95_observations
    accuracy_gap = candidate.accuracy - fixed.accuracy
    gate = (
        development_score.accuracy >= 0.99
        and candidate.accuracy >= 0.985
        and candidate.invalid_rate <= 0.01
        and candidate.half_mass_violation_rate == 0.0
        and accuracy_gap >= -0.005
        and observation_ratio <= 0.60
        and p95_ratio <= 0.80
        and selected.batches[0] <= 64
        and selected.batches[-1] == 512
    )
    return {
        "status": "adaptive_experiment_budget_candidate" if gate else "not_yet",
        "claim_scope": "the system selects a sequential confidence schedule that preserves weighted-tree diagnosis while purchasing additional intervention samples only for ambiguous local decisions; this is sequential testing and adaptive sampling, not a world breakthrough",
        "seed": seed,
        "candidate_gate": gate,
        "selected_program": selected.text(),
        "training_score": train_score.__dict__,
        "development_score": development_score.__dict__,
        "frozen_budget_digest": frozen_digest,
        "candidate": candidate.__dict__,
        "fixed_512_control": fixed.__dict__,
        "accuracy_gap": accuracy_gap,
        "mean_observation_ratio": observation_ratio,
        "p95_observation_ratio": p95_ratio,
        "programs_evaluated": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "program": report["selected_program"],
        "accuracy": report["candidate"]["accuracy"],
        "mean_observation_ratio": report["mean_observation_ratio"],
        "p95_observation_ratio": report["p95_observation_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
