from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "external-data" / "uci"
WEIGHT_BUDGET = 4
BASIS_NAMES = (
    "max_fraction",
    "collision_mass",
    "cube_mass",
    "inverse_outcomes",
    "entropy_deficit",
)


@dataclass(frozen=True)
class Objective:
    weights: tuple[int, int, int, int, int]

    @property
    def complexity(self) -> int:
        return sum(weight != 0 for weight in self.weights)

    def text(self) -> str:
        return "+".join(
            f"{weight}*{name}"
            for weight, name in zip(self.weights, BASIS_NAMES)
            if weight
        ) or "zero"

    def score(self, sizes: tuple[int, ...]) -> float:
        total = float(sum(sizes))
        probabilities = tuple(size / total for size in sizes)
        maximum = max(probabilities)
        collision = sum(value * value for value in probabilities)
        cube = sum(value * value * value for value in probabilities)
        inverse_outcomes = 1.0 / len(probabilities)
        if len(probabilities) <= 1:
            entropy_deficit = 1.0
        else:
            entropy = -sum(
                value * math.log(value)
                for value in probabilities
                if value > 0.0
            )
            entropy_deficit = 1.0 - entropy / math.log(len(probabilities))
        basis = (
            maximum,
            collision,
            cube,
            inverse_outcomes,
            entropy_deficit,
        )
        return sum(
            weight * value
            for weight, value in zip(self.weights, basis)
        ) / WEIGHT_BUDGET


@dataclass(frozen=True)
class ExternalTask:
    name: str
    feature_names: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    outcome_masks: tuple[tuple[tuple[str, int], ...], ...]

    @property
    def candidate_count(self) -> int:
        return len(self.rows)

    @property
    def query_count(self) -> int:
        return len(self.feature_names)

    @property
    def full_mask(self) -> int:
        return (1 << self.candidate_count) - 1

    def outcome_for(self, query: int, candidate: int) -> str:
        return self.rows[candidate][query]

    def masks_for(self, query: int) -> dict[str, int]:
        return dict(self.outcome_masks[query])


@dataclass(frozen=True)
class Evaluation:
    identified_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int


@dataclass(frozen=True)
class ObjectiveClass:
    members: tuple[tuple[int, ...], ...]
    canonical: Objective
    digest: str
    state_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> dict[str, object]:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["datasets"]:
        archive = DATA_ROOT / entry["archive"]
        actual = file_sha256(archive)
        rows.append(
            {
                "name": entry["name"],
                "archive": entry["archive"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual,
                "matched": actual == entry["sha256"],
                "role": entry["role"],
            }
        )
    return {
        "source": manifest["source"],
        "license": manifest["license"],
        "all_hashes_match": all(row["matched"] for row in rows),
        "rows": rows,
    }


def _dedupe(rows: Iterable[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    seen = set()
    output = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            output.append(row)
    return tuple(output)


def make_task(
    name: str,
    feature_names: Iterable[str],
    rows: Iterable[tuple[str, ...]],
) -> ExternalTask:
    unique = _dedupe(rows)
    names = tuple(feature_names)
    if not unique:
        raise ValueError(f"{name}: no rows")
    if any(len(row) != len(names) for row in unique):
        raise ValueError(f"{name}: inconsistent feature count")
    outcomes = []
    for query in range(len(names)):
        masks: dict[str, int] = {}
        for candidate, row in enumerate(unique):
            value = row[query]
            masks[value] = masks.get(value, 0) | (1 << candidate)
        outcomes.append(tuple(sorted(masks.items())))
    return ExternalTask(name, names, unique, tuple(outcomes))


def _read_zip_lines(archive: str, member: str) -> list[str]:
    with zipfile.ZipFile(DATA_ROOT / archive) as handle:
        text = handle.read(member).decode("utf-8", errors="strict")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_monk(problem: int, split: str) -> ExternalTask:
    member = f"monks-{problem}.{split}"
    rows = []
    for line in _read_zip_lines("monks.zip", member):
        fields = line.split()
        rows.append(tuple(fields[1:7]))
    return make_task(
        f"monk-{problem}-{split}",
        tuple(f"a{index}" for index in range(1, 7)),
        rows,
    )


def load_tic_tac_toe() -> ExternalTask:
    rows = [
        tuple(line.split(",")[:9])
        for line in _read_zip_lines("tic-tac-toe.zip", "tic-tac-toe.data")
    ]
    return make_task(
        "tic-tac-toe",
        tuple(f"square-{index}" for index in range(1, 10)),
        rows,
    )


def load_zoo() -> ExternalTask:
    rows = [
        tuple(line.split(",")[1:17])
        for line in _read_zip_lines("zoo.zip", "zoo.data")
    ]
    names = (
        "hair",
        "feathers",
        "eggs",
        "milk",
        "airborne",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
    )
    return make_task("zoo", names, rows)


def load_mushroom() -> ExternalTask:
    rows = [
        tuple(line.split(",")[1:])
        for line in _read_zip_lines("mushroom.zip", "agaricus-lepiota.data")
    ]
    return make_task(
        "mushroom",
        tuple(f"attribute-{index}" for index in range(1, 23)),
        rows,
    )


def grammar() -> tuple[Objective, ...]:
    objectives = []
    for weights in itertools.product(range(WEIGHT_BUDGET + 1), repeat=5):
        if sum(weights) == WEIGHT_BUDGET:
            objectives.append(Objective(tuple(int(value) for value in weights)))
    return tuple(objectives)


def partition_sizes(
    task: ExternalTask,
    allowed: int,
    query: int,
) -> tuple[int, ...]:
    sizes = tuple(
        sorted(
            (
                (allowed & mask).bit_count()
                for _, mask in task.outcome_masks[query]
                if allowed & mask
            ),
            reverse=True,
        )
    )
    return sizes


def select_query(
    task: ExternalTask,
    allowed: int,
    remaining: int,
    objective: Objective,
) -> int:
    candidates = []
    for query in range(task.query_count):
        if not (remaining >> query) & 1:
            continue
        sizes = partition_sizes(task, allowed, query)
        if len(sizes) <= 1:
            continue
        candidates.append(
            (
                objective.score(sizes),
                sizes,
                task.feature_names[query],
                query,
            )
        )
    if not candidates:
        raise RuntimeError("no separating external feature remains")
    return min(candidates)[-1]


def trace(
    task: ExternalTask,
    objective: Objective,
    candidate: int,
) -> tuple[bool, int, tuple[tuple[int, int], ...]]:
    allowed = task.full_mask
    remaining = (1 << task.query_count) - 1
    queries = 0
    states = []
    while allowed.bit_count() > 1:
        states.append((allowed, remaining))
        try:
            query = select_query(task, allowed, remaining, objective)
        except RuntimeError:
            return False, queries, tuple(states)
        outcome = task.outcome_for(query, candidate)
        allowed &= task.masks_for(query)[outcome]
        remaining &= ~(1 << query)
        queries += 1
        if not (allowed >> candidate) & 1:
            raise AssertionError("truth candidate was removed")
    return allowed == (1 << candidate), queries, tuple(states)


def evaluate(task: ExternalTask, objective: Objective) -> Evaluation:
    counts = []
    identified = 0
    for candidate in range(task.candidate_count):
        correct, queries, _ = trace(task, objective, candidate)
        identified += int(correct)
        counts.append(queries)
    return Evaluation(
        identified_fraction=identified / task.candidate_count,
        mean_queries=float(np.mean(counts)),
        worst_queries=max(counts),
        unresolved=task.candidate_count - identified,
        candidates=task.candidate_count,
    )


def state_union(
    tasks: list[ExternalTask],
    objectives: tuple[Objective, ...],
) -> dict[int, set[tuple[int, int]]]:
    output: dict[int, set[tuple[int, int]]] = {}
    for task_index, task in enumerate(tasks):
        states: set[tuple[int, int]] = set()
        for objective in objectives:
            for candidate in range(task.candidate_count):
                _, _, trajectory = trace(task, objective, candidate)
                states.update(trajectory)
        output[task_index] = states
    return output


def objective_signature(
    tasks: list[ExternalTask],
    states: dict[int, set[tuple[int, int]]],
    objective: Objective,
) -> tuple[object, ...]:
    rows: list[object] = []
    for task_index, task in enumerate(tasks):
        for allowed, remaining in sorted(states[task_index]):
            try:
                query = select_query(task, allowed, remaining, objective)
                partition = partition_sizes(task, allowed, query)
            except RuntimeError:
                partition = ()
            rows.append(
                (
                    task_index,
                    allowed.bit_count(),
                    remaining.bit_count(),
                    partition,
                )
            )
    return tuple(rows)


def quotient_objectives(
    tasks: list[ExternalTask],
    objectives: tuple[Objective, ...],
) -> tuple[ObjectiveClass, ...]:
    states = state_union(tasks, objectives)
    groups: dict[str, list[Objective]] = {}
    counts: dict[str, int] = {}
    for objective in objectives:
        signature = objective_signature(tasks, states, objective)
        digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
        groups.setdefault(digest, []).append(objective)
        counts[digest] = len(signature)
    classes = []
    for digest, members in groups.items():
        ordered = tuple(
            sorted(
                members,
                key=lambda value: (value.complexity, value.weights),
            )
        )
        classes.append(
            ObjectiveClass(
                members=tuple(value.weights for value in ordered),
                canonical=ordered[0],
                digest=digest,
                state_count=counts[digest],
            )
        )
    return tuple(
        sorted(
            classes,
            key=lambda value: (
                value.canonical.complexity,
                value.canonical.weights,
            ),
        )
    )


def aggregate_evaluation(
    tasks: list[ExternalTask],
    objective: Objective,
) -> dict[str, object]:
    rows = {task.name: evaluate(task, objective).__dict__ for task in tasks}
    return {
        "tasks": rows,
        "identified_min": min(row["identified_fraction"] for row in rows.values()),
        "mean_queries": float(np.mean([row["mean_queries"] for row in rows.values()])),
        "mean_worst_queries": float(np.mean([row["worst_queries"] for row in rows.values()])),
        "maximum_worst_queries": max(row["worst_queries"] for row in rows.values()),
    }


def select_objective(
    training: list[ExternalTask],
    development: list[ExternalTask],
) -> tuple[ObjectiveClass, dict[str, object]]:
    objectives = grammar()
    classes = quotient_objectives(training + development, objectives)
    rows = []
    for value in classes:
        train = aggregate_evaluation(training, value.canonical)
        dev = aggregate_evaluation(development, value.canonical)
        score = (
            dev["identified_min"],
            -dev["maximum_worst_queries"],
            -dev["mean_worst_queries"],
            -dev["mean_queries"],
            train["identified_min"],
            -train["maximum_worst_queries"],
            -value.canonical.complexity,
            tuple(-weight for weight in value.canonical.weights),
        )
        rows.append((score, value, train, dev))
    _, selected, train, dev = max(rows, key=lambda row: row[0])
    return selected, {
        "grammar_size": len(objectives),
        "quotient_class_count": len(classes),
        "selected_class_size": len(selected.members),
        "selected_weights": list(selected.canonical.weights),
        "selected_expression": selected.canonical.text(),
        "training": train,
        "development": dev,
        "classes": [
            {
                "canonical_weights": list(value.canonical.weights),
                "canonical_expression": value.canonical.text(),
                "member_count": len(value.members),
                "digest": value.digest,
                "state_count": value.state_count,
            }
            for value in classes
        ],
    }


def baseline_objectives() -> dict[str, Objective]:
    return {
        "minimax": Objective((4, 0, 0, 0, 0)),
        "collision": Objective((0, 4, 0, 0, 0)),
        "cube": Objective((0, 0, 4, 0, 0)),
        "max_outcomes": Objective((0, 0, 0, 4, 0)),
        "entropy": Objective((0, 0, 0, 0, 4)),
    }


def objective_digest(
    selected: ObjectiveClass,
    rotation: int,
    manifest: dict[str, object],
) -> str:
    payload = {
        "weights": selected.canonical.weights,
        "class_digest": selected.digest,
        "class_members": selected.members,
        "rotation": rotation,
        "archives": [
            (row["archive"], row["actual_sha256"])
            for row in manifest["rows"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run(seed: int = 1701) -> dict[str, object]:
    manifest = verify_manifest()
    if not manifest["all_hashes_match"]:
        raise RuntimeError("external UCI archive hash mismatch")

    monk_tasks = [load_monk(index, "train") for index in (1, 2, 3)]
    rotation = (seed - 1701) % 3
    development = [monk_tasks[rotation]]
    training = [
        task
        for index, task in enumerate(monk_tasks)
        if index != rotation
    ]
    selected, synthesis = select_objective(training, development)
    frozen_digest = objective_digest(selected, rotation, manifest)

    # Hidden archives are opened only after objective synthesis, quotienting,
    # canonicalization, manifest verification and digest commitment.
    hidden = [
        load_monk(1, "test"),
        load_tic_tac_toe(),
        load_zoo(),
        load_mushroom(),
    ]
    candidate = aggregate_evaluation(hidden, selected.canonical)
    controls = {
        name: aggregate_evaluation(hidden, objective)
        for name, objective in baseline_objectives().items()
    }

    comparisons = {}
    no_worse_worst = True
    strict_wins = 0
    worst_excess = []
    mean_excess = []
    for task in hidden:
        candidate_row = candidate["tasks"][task.name]
        best_worst = min(
            control["tasks"][task.name]["worst_queries"]
            for control in controls.values()
        )
        best_mean = min(
            control["tasks"][task.name]["mean_queries"]
            for control in controls.values()
        )
        gap_worst = candidate_row["worst_queries"] - best_worst
        gap_mean = candidate_row["mean_queries"] - best_mean
        no_worse_worst &= gap_worst <= 0
        strict_wins += int(gap_worst < 0 or gap_mean < -1e-12)
        worst_excess.append(gap_worst)
        mean_excess.append(gap_mean)
        comparisons[task.name] = {
            "candidate_worst": candidate_row["worst_queries"],
            "best_control_worst": best_worst,
            "worst_gap": gap_worst,
            "candidate_mean": candidate_row["mean_queries"],
            "best_control_mean": best_mean,
            "mean_gap": gap_mean,
        }

    named_weights = {
        objective.weights for objective in baseline_objectives().values()
    }
    genuinely_composite = (
        selected.canonical.weights not in named_weights
        and selected.canonical.complexity >= 2
    )
    gate = (
        manifest["all_hashes_match"]
        and synthesis["training"]["identified_min"] == 1.0
        and synthesis["development"]["identified_min"] == 1.0
        and candidate["identified_min"] == 1.0
        and genuinely_composite
        and no_worse_worst
        and max(worst_excess) <= 0
        and float(np.median(mean_excess)) <= 0.10
        and strict_wins >= 2
    )
    return {
        "status": (
            "external_objective_genesis_candidate"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "a weighted partition objective is synthesized from a fixed "
            "five-statistic grammar, quotiented by behavior on external MONK "
            "tasks, frozen before hidden archives are opened, and evaluated "
            "by exact instance-identification trajectories on four UCI "
            "domains; this is an external-benchmark milestone, not a world "
            "breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "manifest_verification": manifest,
        "rotation": rotation,
        "training_tasks": [task.name for task in training],
        "development_tasks": [task.name for task in development],
        "synthesis": synthesis,
        "selected_class": {
            "weights": list(selected.canonical.weights),
            "expression": selected.canonical.text(),
            "member_count": len(selected.members),
            "class_digest": selected.digest,
        },
        "frozen_objective_digest": frozen_digest,
        "hidden_tasks": [
            {
                "name": task.name,
                "candidates": task.candidate_count,
                "queries": task.query_count,
            }
            for task in hidden
        ],
        "candidate": candidate,
        "controls": controls,
        "comparisons": comparisons,
        "genuinely_composite": genuinely_composite,
        "no_worse_worst_case": no_worse_worst,
        "strict_hidden_wins": strict_wins,
        "median_mean_query_excess": float(np.median(mean_excess)),
        "maximum_worst_query_excess": max(worst_excess),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected_class"],
                "hidden": report["candidate"],
                "strict_hidden_wins": report["strict_hidden_wins"],
                "median_mean_query_excess": report[
                    "median_mean_query_excess"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
