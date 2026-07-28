from __future__ import annotations

import argparse, hashlib, json, math, random
from pathlib import Path

import numpy as np

from . import cross_domain_compiler_v30 as base


Query = base.Query
Task = base.Task
Policy = base.Policy
Evaluation = base.Evaluation


def run_trial(seed: int, task: Task, candidate: int, policy: Policy) -> tuple[bool, int, bool]:
    allowed = frozenset(range(task.candidate_count))
    remaining = frozenset(range(len(task.queries)))
    budget = math.ceil(math.log2(task.candidate_count)) + 3
    queries = 0
    while len(allowed) > 1 and queries < budget:
        try:
            index = base.select_query(task, allowed, remaining, policy.query_rule)
        except RuntimeError:
            return False, queries, True
        query = task.queries[index]
        outcomes, values = base.observe(seed + queries * 7_919, query, candidate)
        outcome = base.decode(policy.decoder_rule, outcomes, values)
        parts = base.partitions(query, allowed)
        queries += 1
        if outcome not in parts:
            return False, queries, True
        allowed = parts[outcome]
        remaining = frozenset(value for value in remaining if value != index)
        if candidate not in allowed:
            return False, queries, True
    return len(allowed) == 1 and next(iter(allowed), -1) == candidate, queries, False


def evaluate(seed: int, tasks: list[Task], trials_per_task: int, policy: Policy) -> Evaluation:
    rng = np.random.default_rng(seed)
    rows = []
    sizes = []
    for task_index, task in enumerate(tasks):
        candidates = rng.integers(0, task.candidate_count, size=trials_per_task)
        for trial, candidate in enumerate(candidates):
            rows.append(run_trial(
                seed + task_index * 1_000_003 + trial * 104_729,
                task,
                int(candidate),
                policy,
            ))
            sizes.append(task.candidate_count)
    counts = [row[1] for row in rows]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean(counts)),
        maximum_queries=max(counts),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_log2_ratio=float(np.mean([
            count / max(1, math.ceil(math.log2(size)))
            for count, size in zip(counts, sizes)
        ])),
    )


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
            queries = 0
            invalid = False
            while len(allowed) > 1 and queries < budget:
                separating = [
                    index for index in remaining
                    if len(base.partitions(task.queries[index], allowed)) > 1
                ]
                if not separating:
                    invalid = True
                    break
                index = rng.choice(separating)
                remaining.remove(index)
                query = task.queries[index]
                outcomes, values = base.observe(
                    seed + task_index * 1_000_003 + trial * 104_729 + queries,
                    query,
                    candidate,
                )
                outcome = base.decode("argmin", outcomes, values)
                queries += 1
                allowed = base.partitions(query, allowed).get(outcome, frozenset())
                if candidate not in allowed:
                    invalid = True
                    break
            rows.append((len(allowed) == 1 and next(iter(allowed), -1) == candidate, queries, invalid))
            sizes.append(task.candidate_count)
    counts = [row[1] for row in rows]
    return Evaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        mean_queries=float(np.mean(counts)),
        maximum_queries=max(counts),
        invalid_rate=float(np.mean([row[2] for row in rows])),
        mean_log2_ratio=float(np.mean([
            count / max(1, math.ceil(math.log2(size)))
            for count, size in zip(counts, sizes)
        ])),
    )


def reduced_certificate_tasks(seed: int) -> list[Task]:
    bit = base.bitcode_task(3)
    bit = Task(bit.domain, bit.candidate_count, tuple(q for q in bit.queries if q.name.startswith("bit-")))
    subset = base.subset_task(6, seed)
    subset = Task(subset.domain, subset.candidate_count, tuple(q for q in subset.queries if q.name.startswith("code-")))
    return [bit, subset]


def digest(policy: Policy) -> str:
    return hashlib.sha256(policy.text().encode()).hexdigest()


def run(seed: int = 1501) -> dict[str, object]:
    train = base.training_tasks(seed * 10_000 + 107)
    development = base.development_tasks(seed * 10_000 + 1_000_109)
    complexity = {"minimax_bucket": 1, "gini": 2, "max_outcomes": 3, "first": 4}
    rows = []
    for policy in base.policies():
        train_score = evaluate(seed * 10_000 + 2_000_003, train, 24, policy)
        development_score = evaluate(seed * 10_000 + 3_000_007, development, 32, policy)
        score = (
            development_score.accuracy,
            -development_score.invalid_rate,
            -development_score.mean_queries,
            -complexity[policy.query_rule],
            int(policy.decoder_rule == "argmin"),
        )
        rows.append((score, policy, train_score, development_score))
    _, selected, train_score, development_score = max(rows, key=lambda row: row[0])
    frozen_digest = digest(selected)

    # Entirely new diagnosis domains are generated only after the abstract
    # partition rule, response decoder and digest are frozen.
    hidden = base.hidden_tasks(seed * 10_000 + 12_000_001)
    candidate = evaluate(seed * 10_000 + 13_000_003, hidden, 80, selected)
    first_control = evaluate(seed * 10_000 + 13_000_005, hidden, 80, Policy("first", "argmin"))
    random_control = random_query_control(seed * 10_000 + 13_000_007, hidden, 80)
    proof = base.exact_certificate(
        reduced_certificate_tasks(seed * 10_000 + 13_000_009),
        selected,
    )
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
        "mean_queries": report["candidate"]["mean_queries"],
        "log2_ratio": report["candidate"]["mean_log2_ratio"],
        "first_gap": report["first_accuracy_gap"],
        "random_gap": report["random_accuracy_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
