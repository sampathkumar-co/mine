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
DEVELOPMENT_ROOT = ROOT / "external-data" / "uci"
HIDDEN_ROOT = ROOT / "external-data" / "uci-v33"
WEIGHT_BUDGET = 4
BASIS_NAMES = (
    "worst_impure_fraction",
    "expected_gini",
    "expected_entropy",
    "unresolved_fraction",
    "inverse_outcomes",
)


@dataclass(frozen=True)
class DiagnosisObjective:
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

    def score(
        self,
        total: int,
        buckets: tuple[tuple[int, tuple[int, ...]], ...],
        label_count: int,
    ) -> float:
        impure = tuple(
            (size, counts)
            for size, counts in buckets
            if sum(count > 0 for count in counts) > 1
        )
        worst_impure = (
            max(size for size, _ in impure) / total
            if impure
            else 0.0
        )
        expected_gini = 0.0
        expected_entropy = 0.0
        unresolved = 0
        entropy_scale = math.log(label_count) if label_count > 1 else 1.0
        for size, counts in buckets:
            probabilities = tuple(
                count / size for count in counts if count > 0
            )
            if len(probabilities) > 1:
                unresolved += size
            gini = 1.0 - sum(value * value for value in probabilities)
            entropy = -sum(
                value * math.log(value)
                for value in probabilities
                if value > 0.0
            )
            expected_gini += (size / total) * gini
            expected_entropy += (size / total) * entropy / entropy_scale
        basis = (
            worst_impure,
            expected_gini,
            expected_entropy,
            unresolved / total,
            1.0 / len(buckets),
        )
        return sum(
            weight * value
            for weight, value in zip(self.weights, basis)
        ) / WEIGHT_BUDGET


@dataclass(frozen=True)
class ClassTask:
    name: str
    feature_names: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...]
    outcome_masks: tuple[tuple[tuple[str, int], ...], ...]
    label_masks: tuple[tuple[str, int], ...]

    @property
    def candidate_count(self) -> int:
        return len(self.rows)

    @property
    def query_count(self) -> int:
        return len(self.feature_names)

    @property
    def full_mask(self) -> int:
        return (1 << self.candidate_count) - 1

    @property
    def label_count(self) -> int:
        return len(self.label_masks)

    def masks_for(self, query: int) -> dict[str, int]:
        return dict(self.outcome_masks[query])

    def label_mask_dict(self) -> dict[str, int]:
        return dict(self.label_masks)


@dataclass(frozen=True)
class Evaluation:
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int


@dataclass(frozen=True)
class ObjectiveClass:
    members: tuple[tuple[int, ...], ...]
    canonical: DiagnosisObjective
    digest: str
    state_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["datasets"]:
        archive = root / entry["archive"]
        actual = file_sha256(archive)
        rows.append(
            {
                "name": entry["name"],
                "archive": entry["archive"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual,
                "matched": actual == entry["sha256"],
            }
        )
    return {
        "source": manifest["source"],
        "license": manifest["license"],
        "all_hashes_match": all(row["matched"] for row in rows),
        "rows": rows,
    }


def _read_zip_lines(root: Path, archive: str, member: str) -> list[str]:
    with zipfile.ZipFile(root / archive) as handle:
        text = handle.read(member).decode("utf-8", errors="strict")
    return [line.strip() for line in text.splitlines() if line.strip()]


def make_task(
    name: str,
    feature_names: Iterable[str],
    rows: Iterable[tuple[str, ...]],
    labels: Iterable[str],
) -> ClassTask:
    names = tuple(feature_names)
    row_list = tuple(rows)
    label_list = tuple(labels)
    if not row_list or len(row_list) != len(label_list):
        raise ValueError(f"{name}: invalid row/label count")
    if any(len(row) != len(names) for row in row_list):
        raise ValueError(f"{name}: inconsistent feature count")
    outcomes = []
    for query in range(len(names)):
        masks: dict[str, int] = {}
        for candidate, row in enumerate(row_list):
            value = row[query]
            masks[value] = masks.get(value, 0) | (1 << candidate)
        outcomes.append(tuple(sorted(masks.items())))
    label_masks: dict[str, int] = {}
    for candidate, label in enumerate(label_list):
        label_masks[label] = label_masks.get(label, 0) | (1 << candidate)
    return ClassTask(
        name,
        names,
        row_list,
        label_list,
        tuple(outcomes),
        tuple(sorted(label_masks.items())),
    )


def load_monk(problem: int, split: str) -> ClassTask:
    lines = _read_zip_lines(
        DEVELOPMENT_ROOT,
        "monks.zip",
        f"monks-{problem}.{split}",
    )
    rows = []
    labels = []
    for line in lines:
        fields = line.split()
        labels.append(fields[0])
        rows.append(tuple(fields[1:7]))
    return make_task(
        f"monk-{problem}-{split}",
        tuple(f"a{index}" for index in range(1, 7)),
        rows,
        labels,
    )


def load_tic_tac_toe() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(
        DEVELOPMENT_ROOT,
        "tic-tac-toe.zip",
        "tic-tac-toe.data",
    ):
        fields = line.split(",")
        rows.append(tuple(fields[:9]))
        labels.append(fields[9])
    return make_task(
        "tic-tac-toe",
        tuple(f"square-{index}" for index in range(1, 10)),
        rows,
        labels,
    )


def load_zoo() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(DEVELOPMENT_ROOT, "zoo.zip", "zoo.data"):
        fields = line.split(",")
        rows.append(tuple(fields[1:17]))
        labels.append(fields[17])
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
    return make_task("zoo", names, rows, labels)


def load_mushroom_subset(count: int = 512) -> ClassTask:
    candidates = []
    for line in _read_zip_lines(
        DEVELOPMENT_ROOT,
        "mushroom.zip",
        "agaricus-lepiota.data",
    ):
        fields = line.split(",")
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        candidates.append((digest, tuple(fields[1:]), fields[0]))
    selected = sorted(candidates)[:count]
    return make_task(
        f"mushroom-development-{count}",
        tuple(f"attribute-{index}" for index in range(1, 23)),
        [row for _, row, _ in selected],
        [label for _, _, label in selected],
    )


def load_car() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(HIDDEN_ROOT, "car.zip", "car.data"):
        fields = line.split(",")
        rows.append(tuple(fields[:6]))
        labels.append(fields[6])
    return make_task(
        "car-evaluation",
        ("buying", "maint", "doors", "persons", "lug_boot", "safety"),
        rows,
        labels,
    )


def load_nursery() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(
        HIDDEN_ROOT,
        "nursery.zip",
        "nursery.data",
    ):
        fields = line.split(",")
        rows.append(tuple(fields[:8]))
        labels.append(fields[8])
    return make_task(
        "nursery",
        (
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
        ),
        rows,
        labels,
    )


def load_balance() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(
        HIDDEN_ROOT,
        "balance.zip",
        "balance-scale.data",
    ):
        fields = line.split(",")
        labels.append(fields[0])
        rows.append(tuple(fields[1:5]))
    return make_task(
        "balance-scale",
        ("left-weight", "left-distance", "right-weight", "right-distance"),
        rows,
        labels,
    )


def load_votes() -> ClassTask:
    rows = []
    labels = []
    for line in _read_zip_lines(
        HIDDEN_ROOT,
        "votes.zip",
        "house-votes-84.data",
    ):
        fields = line.split(",")
        labels.append(fields[0])
        rows.append(tuple(fields[1:17]))
    return make_task(
        "congressional-votes",
        tuple(f"vote-{index}" for index in range(1, 17)),
        rows,
        labels,
    )


def grammar() -> tuple[DiagnosisObjective, ...]:
    return tuple(
        DiagnosisObjective(tuple(int(value) for value in weights))
        for weights in itertools.product(range(WEIGHT_BUDGET + 1), repeat=5)
        if sum(weights) == WEIGHT_BUDGET
    )


def label_counts(task: ClassTask, allowed: int) -> tuple[int, ...]:
    return tuple(
        (allowed & mask).bit_count()
        for _, mask in task.label_masks
    )


def pure_label(task: ClassTask, allowed: int) -> str | None:
    matches = [
        label
        for label, mask in task.label_masks
        if allowed & mask
    ]
    return matches[0] if len(matches) == 1 else None


def bucket_shape(
    task: ClassTask,
    allowed: int,
    query: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    rows = []
    label_masks = task.label_masks
    for _, mask in task.outcome_masks[query]:
        bucket = allowed & mask
        size = bucket.bit_count()
        if not size:
            continue
        counts = tuple(
            sorted(
                (
                    (bucket & label_mask).bit_count()
                    for _, label_mask in label_masks
                ),
                reverse=True,
            )
        )
        rows.append((size, counts))
    return tuple(sorted(rows, reverse=True))


def select_query(
    task: ClassTask,
    allowed: int,
    remaining: int,
    objective: DiagnosisObjective,
) -> int:
    candidates = []
    total = allowed.bit_count()
    for query in range(task.query_count):
        if not (remaining >> query) & 1:
            continue
        buckets = bucket_shape(task, allowed, query)
        if len(buckets) <= 1:
            continue
        candidates.append(
            (
                objective.score(total, buckets, task.label_count),
                buckets,
                task.feature_names[query],
                query,
            )
        )
    if not candidates:
        raise RuntimeError("no separating diagnostic feature remains")
    return min(candidates)[-1]


def trace(
    task: ClassTask,
    objective: DiagnosisObjective,
    candidate: int,
) -> tuple[bool, int, tuple[tuple[int, int], ...]]:
    allowed = task.full_mask
    remaining = (1 << task.query_count) - 1
    queries = 0
    states = []
    while pure_label(task, allowed) is None:
        states.append((allowed, remaining))
        try:
            query = select_query(task, allowed, remaining, objective)
        except RuntimeError:
            return False, queries, tuple(states)
        outcome = task.rows[candidate][query]
        allowed &= task.masks_for(query)[outcome]
        remaining &= ~(1 << query)
        queries += 1
        if not (allowed >> candidate) & 1:
            raise AssertionError("truth row was removed")
    prediction = pure_label(task, allowed)
    return prediction == task.labels[candidate], queries, tuple(states)


def evaluate(task: ClassTask, objective: DiagnosisObjective) -> Evaluation:
    counts = []
    diagnosed = 0
    for candidate in range(task.candidate_count):
        correct, queries, _ = trace(task, objective, candidate)
        diagnosed += int(correct)
        counts.append(queries)
    return Evaluation(
        diagnosed_fraction=diagnosed / task.candidate_count,
        mean_queries=float(np.mean(counts)),
        worst_queries=max(counts),
        unresolved=task.candidate_count - diagnosed,
        candidates=task.candidate_count,
    )


def state_union(
    tasks: list[ClassTask],
    objectives: tuple[DiagnosisObjective, ...],
) -> dict[int, set[tuple[int, int]]]:
    output = {}
    for task_index, task in enumerate(tasks):
        states: set[tuple[int, int]] = set()
        for objective in objectives:
            for candidate in range(task.candidate_count):
                _, _, trajectory = trace(task, objective, candidate)
                states.update(trajectory)
        output[task_index] = states
    return output


def objective_signature(
    tasks: list[ClassTask],
    states: dict[int, set[tuple[int, int]]],
    objective: DiagnosisObjective,
) -> tuple[object, ...]:
    rows = []
    for task_index, task in enumerate(tasks):
        for allowed, remaining in sorted(states[task_index]):
            try:
                query = select_query(task, allowed, remaining, objective)
                shape = bucket_shape(task, allowed, query)
            except RuntimeError:
                shape = ()
            rows.append(
                (
                    task_index,
                    allowed.bit_count(),
                    remaining.bit_count(),
                    label_counts(task, allowed),
                    shape,
                )
            )
    return tuple(rows)


def quotient_objectives(
    tasks: list[ClassTask],
    objectives: tuple[DiagnosisObjective, ...],
) -> tuple[ObjectiveClass, ...]:
    states = state_union(tasks, objectives)
    groups: dict[str, list[DiagnosisObjective]] = {}
    counts = {}
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


def aggregate(
    tasks: list[ClassTask],
    objective: DiagnosisObjective,
) -> dict[str, object]:
    rows = {task.name: evaluate(task, objective).__dict__ for task in tasks}
    return {
        "tasks": rows,
        "diagnosed_min": min(row["diagnosed_fraction"] for row in rows.values()),
        "diagnosed_mean": float(
            np.mean([row["diagnosed_fraction"] for row in rows.values()])
        ),
        "mean_queries": float(
            np.mean([row["mean_queries"] for row in rows.values()])
        ),
        "mean_worst_queries": float(
            np.mean([row["worst_queries"] for row in rows.values()])
        ),
        "maximum_worst_queries": max(
            row["worst_queries"] for row in rows.values()
        ),
    }


def select_objective(
    training: list[ClassTask],
    development: list[ClassTask],
) -> tuple[ObjectiveClass, dict[str, object]]:
    objectives = grammar()
    classes = quotient_objectives(training + development, objectives)
    rows = []
    for value in classes:
        train = aggregate(training, value.canonical)
        dev = aggregate(development, value.canonical)
        score = (
            dev["diagnosed_min"],
            dev["diagnosed_mean"],
            -dev["maximum_worst_queries"],
            -dev["mean_worst_queries"],
            -dev["mean_queries"],
            train["diagnosed_min"],
            train["diagnosed_mean"],
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


def controls() -> dict[str, DiagnosisObjective]:
    return {
        "worst_impurity": DiagnosisObjective((4, 0, 0, 0, 0)),
        "gini": DiagnosisObjective((0, 4, 0, 0, 0)),
        "entropy": DiagnosisObjective((0, 0, 4, 0, 0)),
        "unresolved_mass": DiagnosisObjective((0, 0, 0, 4, 0)),
        "max_outcomes": DiagnosisObjective((0, 0, 0, 0, 4)),
    }


def frozen_digest(
    selected: ObjectiveClass,
    rotation: int,
    hidden_manifest: dict[str, object],
) -> str:
    payload = {
        "weights": selected.canonical.weights,
        "class_digest": selected.digest,
        "class_members": selected.members,
        "rotation": rotation,
        "hidden_archives": [
            (row["archive"], row["actual_sha256"])
            for row in hidden_manifest["rows"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run(seed: int = 1801) -> dict[str, object]:
    development_manifest = verify_manifest(DEVELOPMENT_ROOT)
    hidden_manifest = verify_manifest(HIDDEN_ROOT)
    if not development_manifest["all_hashes_match"]:
        raise RuntimeError("development archive hash mismatch")
    if not hidden_manifest["all_hashes_match"]:
        raise RuntimeError("hidden archive hash mismatch")

    base_tasks = [
        load_monk(1, "train"),
        load_monk(2, "train"),
        load_monk(3, "train"),
    ]
    rotation = (seed - 1801) % 3
    training = [
        task
        for index, task in enumerate(base_tasks)
        if index != rotation
    ]
    training.append(load_tic_tac_toe())
    development = [
        base_tasks[rotation],
        load_zoo(),
        load_mushroom_subset(512),
    ]
    selected, synthesis = select_objective(training, development)
    digest = frozen_digest(selected, rotation, hidden_manifest)

    # Hidden records are opened only after the objective class, canonical
    # expression, rotation and archive commitments are frozen.
    hidden = [
        load_car(),
        load_nursery(),
        load_balance(),
        load_votes(),
    ]
    candidate = aggregate(hidden, selected.canonical)
    specialist_controls = {
        name: aggregate(hidden, objective)
        for name, objective in controls().items()
    }

    comparisons = {}
    strict_wins = 0
    diagnosed_gaps = []
    worst_gaps = []
    mean_gaps = []
    for task in hidden:
        candidate_row = candidate["tasks"][task.name]
        best_diagnosed = max(
            control["tasks"][task.name]["diagnosed_fraction"]
            for control in specialist_controls.values()
        )
        eligible = [
            control["tasks"][task.name]
            for control in specialist_controls.values()
            if control["tasks"][task.name]["diagnosed_fraction"]
            >= best_diagnosed - 1e-12
        ]
        best_worst = min(row["worst_queries"] for row in eligible)
        best_mean = min(row["mean_queries"] for row in eligible)
        diagnosed_gap = candidate_row["diagnosed_fraction"] - best_diagnosed
        worst_gap = candidate_row["worst_queries"] - best_worst
        mean_gap = candidate_row["mean_queries"] - best_mean
        strict_wins += int(
            diagnosed_gap > 1e-12
            or (
                diagnosed_gap >= -1e-12
                and (worst_gap < 0 or mean_gap < -1e-12)
            )
        )
        diagnosed_gaps.append(diagnosed_gap)
        worst_gaps.append(worst_gap)
        mean_gaps.append(mean_gap)
        comparisons[task.name] = {
            "candidate_diagnosed": candidate_row["diagnosed_fraction"],
            "best_control_diagnosed": best_diagnosed,
            "diagnosed_gap": diagnosed_gap,
            "candidate_worst": candidate_row["worst_queries"],
            "best_control_worst": best_worst,
            "worst_gap": worst_gap,
            "candidate_mean": candidate_row["mean_queries"],
            "best_control_mean": best_mean,
            "mean_gap": mean_gap,
        }

    named = {objective.weights for objective in controls().values()}
    composite = (
        selected.canonical.weights not in named
        and selected.canonical.complexity >= 2
    )
    gate = (
        development_manifest["all_hashes_match"]
        and hidden_manifest["all_hashes_match"]
        and synthesis["training"]["diagnosed_min"] >= 0.99
        and synthesis["development"]["diagnosed_min"] >= 0.99
        and candidate["diagnosed_min"] >= 0.98
        and composite
        and min(diagnosed_gaps) >= -0.005
        and max(worst_gaps) <= 1
        and float(np.median(mean_gaps)) <= 0.0
        and strict_wins >= 2
    )
    return {
        "status": (
            "external_class_diagnosis_objective_candidate"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "a label-aware experiment-selection objective is synthesized from "
            "a five-statistic grammar on previously opened external datasets, "
            "quotiented by exact behavior, frozen before four new UCI archives "
            "are opened, and evaluated on zero-error active class diagnosis; "
            "this is an external research candidate, not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "development_manifest": development_manifest,
        "hidden_manifest": hidden_manifest,
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
        "frozen_objective_digest": digest,
        "hidden_tasks": [
            {
                "name": task.name,
                "rows": task.candidate_count,
                "features": task.query_count,
                "labels": task.label_count,
            }
            for task in hidden
        ],
        "candidate": candidate,
        "controls": specialist_controls,
        "comparisons": comparisons,
        "genuinely_composite": composite,
        "strict_hidden_wins": strict_wins,
        "minimum_diagnosed_gap": min(diagnosed_gaps),
        "maximum_worst_query_gap": max(worst_gaps),
        "median_mean_query_gap": float(np.median(mean_gaps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1801)
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
                "candidate": report["candidate"],
                "strict_hidden_wins": report["strict_hidden_wins"],
                "minimum_diagnosed_gap": report["minimum_diagnosed_gap"],
                "median_mean_query_gap": report["median_mean_query_gap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
