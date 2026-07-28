from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from . import external_class_diagnosis_v33 as base


ROOT = Path(__file__).resolve().parents[2]
HIDDEN_ROOT = ROOT / "external-data" / "uci-v34"
OBJECTIVES = base.controls()
OBJECTIVE_NAMES = tuple(OBJECTIVES)
CONDITION_SPECS = (
    ("candidate_count_le", (8.0, 16.0, 32.0, 64.0, 128.0)),
    ("remaining_features_le", (2.0, 4.0, 8.0, 12.0)),
    ("label_count_le", (2.0, 3.0, 4.0, 8.0)),
    ("majority_fraction_ge", (0.55, 0.70, 0.85)),
    ("normalized_entropy_le", (0.25, 0.50, 0.75)),
)


@dataclass(frozen=True)
class Condition:
    metric: str
    threshold: float

    def text(self) -> str:
        suffix = "le" if self.metric.endswith("_le") else "ge"
        return f"{self.metric}:{self.threshold:g}:{suffix}"


@dataclass(frozen=True)
class StatePolicy:
    condition: Condition | None
    when_true: str
    when_false: str

    @property
    def complexity(self) -> int:
        return 1 if self.condition is None else 3

    def text(self) -> str:
        if self.condition is None:
            return self.when_true
        return (
            f"if({self.condition.text()})"
            f"->{self.when_true}:else->{self.when_false}"
        )


@dataclass(frozen=True)
class ProgramEvaluation:
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int
    true_branch_count: int
    false_branch_count: int

    @property
    def uses_both_branches(self) -> bool:
        return self.true_branch_count > 0 and self.false_branch_count > 0


@dataclass(frozen=True)
class ProgramClass:
    members: tuple[str, ...]
    canonical: StatePolicy
    digest: str
    trajectory_count: int


def verify_hidden_manifest() -> dict[str, object]:
    return base.verify_manifest(HIDDEN_ROOT)


def state_metrics(
    task: base.ClassTask,
    allowed: int,
    remaining: int,
) -> dict[str, float]:
    counts = tuple(
        count
        for count in base.label_counts(task, allowed)
        if count > 0
    )
    total = allowed.bit_count()
    label_count = len(counts)
    majority = max(counts) / total
    if label_count <= 1:
        entropy = 0.0
    else:
        probabilities = tuple(count / total for count in counts)
        entropy = -sum(
            value * math.log(value)
            for value in probabilities
            if value > 0.0
        ) / math.log(label_count)
    return {
        "candidate_count_le": float(total),
        "remaining_features_le": float(remaining.bit_count()),
        "label_count_le": float(label_count),
        "majority_fraction_ge": float(majority),
        "normalized_entropy_le": float(entropy),
    }


def condition_holds(
    condition: Condition,
    task: base.ClassTask,
    allowed: int,
    remaining: int,
) -> bool:
    value = state_metrics(task, allowed, remaining)[condition.metric]
    if condition.metric.endswith("_le"):
        return value <= condition.threshold
    if condition.metric.endswith("_ge"):
        return value >= condition.threshold
    raise ValueError(condition.metric)


def active_objective(
    program: StatePolicy,
    task: base.ClassTask,
    allowed: int,
    remaining: int,
) -> tuple[str, bool]:
    if program.condition is None:
        return program.when_true, True
    branch = condition_holds(
        program.condition,
        task,
        allowed,
        remaining,
    )
    return (
        program.when_true if branch else program.when_false,
        branch,
    )


def grammar() -> tuple[StatePolicy, ...]:
    rows = [
        StatePolicy(None, name, name)
        for name in OBJECTIVE_NAMES
    ]
    for metric, thresholds in CONDITION_SPECS:
        for threshold in thresholds:
            condition = Condition(metric, threshold)
            for when_true, when_false in itertools.permutations(
                OBJECTIVE_NAMES,
                2,
            ):
                rows.append(
                    StatePolicy(
                        condition,
                        when_true,
                        when_false,
                    )
                )
    return tuple(rows)


def select_query(
    task: base.ClassTask,
    allowed: int,
    remaining: int,
    program: StatePolicy,
) -> tuple[int, bool, str]:
    objective_name, branch = active_objective(
        program,
        task,
        allowed,
        remaining,
    )
    query = base.select_query(
        task,
        allowed,
        remaining,
        OBJECTIVES[objective_name],
    )
    return query, branch, objective_name


def trace(
    task: base.ClassTask,
    program: StatePolicy,
    candidate: int,
) -> tuple[
    bool,
    int,
    int,
    int,
    tuple[tuple[object, ...], ...],
]:
    allowed = task.full_mask
    remaining = (1 << task.query_count) - 1
    queries = 0
    true_count = 0
    false_count = 0
    trajectory = []
    while base.pure_label(task, allowed) is None:
        try:
            query, branch, objective_name = select_query(
                task,
                allowed,
                remaining,
                program,
            )
        except RuntimeError:
            return (
                False,
                queries,
                true_count,
                false_count,
                tuple(trajectory),
            )
        shape = base.bucket_shape(task, allowed, query)
        trajectory.append(
            (
                allowed.bit_count(),
                remaining.bit_count(),
                base.label_counts(task, allowed),
                branch,
                objective_name,
                shape,
            )
        )
        true_count += int(branch)
        false_count += int(not branch)
        outcome = task.rows[candidate][query]
        allowed &= task.masks_for(query)[outcome]
        remaining &= ~(1 << query)
        queries += 1
        if not (allowed >> candidate) & 1:
            raise AssertionError("truth row was removed")
    prediction = base.pure_label(task, allowed)
    return (
        prediction == task.labels[candidate],
        queries,
        true_count,
        false_count,
        tuple(trajectory),
    )


def analyse_task(
    task: base.ClassTask,
    program: StatePolicy,
) -> tuple[tuple[object, ...], ...]:
    """Compute each reachable diagnostic state exactly once."""
    results: list[tuple[object, ...] | None] = [None] * task.candidate_count

    def assign(
        allowed: int,
        correct: bool,
        queries: int,
        true_count: int,
        false_count: int,
        trajectory: tuple[tuple[object, ...], ...],
    ) -> None:
        mask = allowed
        while mask:
            bit = mask & -mask
            candidate = bit.bit_length() - 1
            results[candidate] = (
                correct,
                queries,
                true_count,
                false_count,
                trajectory,
            )
            mask ^= bit
    def visit(
        allowed: int,
        remaining: int,
        queries: int,
        true_count: int,
        false_count: int,
        trajectory: tuple[tuple[object, ...], ...],
    ) -> None:
        prediction = base.pure_label(task, allowed)
        if prediction is not None:
            mask = allowed
            while mask:
                bit = mask & -mask
                candidate = bit.bit_length() - 1
                results[candidate] = (
                    prediction == task.labels[candidate],
                    queries,
                    true_count,
                    false_count,
                    trajectory,
                )
                mask ^= bit
            return
        try:
            query, branch, objective_name = select_query(
                task, allowed, remaining, program
            )
        except RuntimeError:
            assign(allowed, False, queries, true_count, false_count, trajectory)
            return
        shape = base.bucket_shape(task, allowed, query)
        step = (
            allowed.bit_count(),
            remaining.bit_count(),
            base.label_counts(task, allowed),
            branch,
            objective_name,
            shape,
        )
        next_remaining = remaining & ~(1 << query)
        next_true = true_count + int(branch)
        next_false = false_count + int(not branch)
        covered = 0
        for mask in task.masks_for(query).values():
            child = allowed & mask
            if not child:
                continue
            covered |= child
            visit(
                child,
                next_remaining,
                queries + 1,
                next_true,
                next_false,
                trajectory + (step,),
            )
        if covered != allowed:
            raise AssertionError("query outcomes did not partition state")

    visit(
        task.full_mask,
        (1 << task.query_count) - 1,
        0,
        0,
        0,
        (),
    )
    if any(row is None for row in results):
        raise AssertionError("not every candidate received a trajectory")
    return tuple(row for row in results if row is not None)


def evaluate(
    task: base.ClassTask,
    program: StatePolicy,
) -> ProgramEvaluation:
    rows = analyse_task(task, program)
    diagnosed = sum(int(bool(row[0])) for row in rows)
    query_counts = [int(row[1]) for row in rows]
    return ProgramEvaluation(
        diagnosed_fraction=diagnosed / task.candidate_count,
        mean_queries=float(np.mean(query_counts)),
        worst_queries=max(query_counts),
        unresolved=task.candidate_count - diagnosed,
        candidates=task.candidate_count,
        true_branch_count=sum(int(row[2]) for row in rows),
        false_branch_count=sum(int(row[3]) for row in rows),
    )


def trajectory_signature(
    tasks: list[base.ClassTask],
    program: StatePolicy,
) -> tuple[object, ...]:
    rows = []
    for task_index, task in enumerate(tasks):
        for candidate, result in enumerate(analyse_task(task, program)):
            correct, queries, _, _, trajectory = result
            rows.append(
                (
                    task_index,
                    task.labels[candidate],
                    correct,
                    queries,
                    trajectory,
                )
            )
    return tuple(rows)

def quotient_programs(
    tasks: list[base.ClassTask],
    programs: tuple[StatePolicy, ...],
) -> tuple[ProgramClass, ...]:
    groups: dict[str, list[StatePolicy]] = {}
    counts = {}
    for program in programs:
        signature = trajectory_signature(tasks, program)
        digest = hashlib.sha256(
            repr(signature).encode("utf-8")
        ).hexdigest()
        groups.setdefault(digest, []).append(program)
        counts[digest] = len(signature)
    classes = []
    for digest, members in groups.items():
        ordered = tuple(
            sorted(
                members,
                key=lambda value: (
                    value.complexity,
                    value.text(),
                ),
            )
        )
        classes.append(
            ProgramClass(
                members=tuple(value.text() for value in ordered),
                canonical=ordered[0],
                digest=digest,
                trajectory_count=counts[digest],
            )
        )
    return tuple(
        sorted(
            classes,
            key=lambda value: (
                value.canonical.complexity,
                value.canonical.text(),
            ),
        )
    )


def aggregate(
    tasks: list[base.ClassTask],
    program: StatePolicy,
) -> dict[str, object]:
    rows = {
        task.name: evaluate(task, program).__dict__
        for task in tasks
    }
    return {
        "tasks": rows,
        "diagnosed_min": min(
            row["diagnosed_fraction"] for row in rows.values()
        ),
        "diagnosed_mean": float(
            np.mean(
                [row["diagnosed_fraction"] for row in rows.values()]
            )
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
        "uses_both_branches": any(
            row["true_branch_count"] > 0
            and row["false_branch_count"] > 0
            for row in rows.values()
        ),
    }


def deterministic_subset(
    task: base.ClassTask,
    count: int,
    salt: str,
) -> base.ClassTask:
    if task.candidate_count <= count:
        return task
    rows = []
    for index, (features, label) in enumerate(
        zip(task.rows, task.labels)
    ):
        digest = hashlib.sha256(
            (
                salt
                + "|"
                + label
                + "|"
                + "|".join(features)
                + f"|{index}"
            ).encode("utf-8")
        ).hexdigest()
        rows.append((digest, features, label))
    selected = sorted(rows)[:count]
    return base.make_task(
        f"{task.name}-subset-{count}",
        task.feature_names,
        [features for _, features, _ in selected],
        [label for _, _, label in selected],
    )


def opened_training_and_development(
    rotation: int,
) -> tuple[list[base.ClassTask], list[base.ClassTask]]:
    monks = [
        base.load_monk(1, "train"),
        base.load_monk(2, "train"),
        base.load_monk(3, "train"),
    ]
    training = [
        task
        for index, task in enumerate(monks)
        if index != rotation
    ]
    training.extend(
        [
            deterministic_subset(
                base.load_tic_tac_toe(),
                512,
                "tic-training",
            ),
            deterministic_subset(
                base.load_car(),
                768,
                "car-training",
            ),
            base.load_balance(),
        ]
    )
    development = [
        monks[rotation],
        base.load_zoo(),
        base.load_mushroom_subset(512),
        deterministic_subset(
            base.load_nursery(),
            1024,
            "nursery-development",
        ),
        base.load_votes(),
    ]
    return training, development


def select_program(
    training: list[base.ClassTask],
    development: list[base.ClassTask],
) -> tuple[ProgramClass, dict[str, object]]:
    programs = grammar()
    classes = quotient_programs(
        training + development,
        programs,
    )
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
            int(dev["uses_both_branches"]),
            -value.canonical.complexity,
            value.canonical.text(),
        )
        rows.append((score, value, train, dev))
    _, selected, train, dev = max(rows, key=lambda row: row[0])
    return selected, {
        "program_count": len(programs),
        "quotient_class_count": len(classes),
        "selected_class_size": len(selected.members),
        "selected_program": selected.canonical.text(),
        "training": train,
        "development": dev,
        "classes": [
            {
                "canonical": value.canonical.text(),
                "member_count": len(value.members),
                "digest": value.digest,
                "trajectory_count": value.trajectory_count,
            }
            for value in classes
        ],
    }


def _read_hidden_lines(archive: str, member: str) -> list[str]:
    with zipfile.ZipFile(HIDDEN_ROOT / archive) as handle:
        text = handle.read(member).decode("utf-8", errors="strict")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_breast_cancer() -> base.ClassTask:
    rows = []
    labels = []
    for line in _read_hidden_lines(
        "breast-cancer.zip",
        "breast-cancer.data",
    ):
        fields = line.split(",")
        labels.append(fields[0])
        rows.append(tuple(fields[1:10]))
    return base.make_task(
        "breast-cancer",
        tuple(f"attribute-{index}" for index in range(1, 10)),
        rows,
        labels,
    )


def load_lymphography() -> base.ClassTask:
    rows = []
    labels = []
    for line in _read_hidden_lines(
        "lymphography.zip",
        "lymphography.data",
    ):
        fields = line.split(",")
        labels.append(fields[0])
        rows.append(tuple(fields[1:]))
    return base.make_task(
        "lymphography",
        tuple(
            f"attribute-{index}"
            for index in range(1, len(rows[0]) + 1)
        ),
        rows,
        labels,
    )


def load_primary_tumor() -> base.ClassTask:
    rows = []
    labels = []
    for line in _read_hidden_lines(
        "primary-tumor.zip",
        "primary-tumor.data",
    ):
        fields = line.split(",")
        labels.append(fields[0])
        rows.append(tuple(fields[1:]))
    return base.make_task(
        "primary-tumor",
        tuple(
            f"attribute-{index}"
            for index in range(1, len(rows[0]) + 1)
        ),
        rows,
        labels,
    )


def load_soybean_large() -> base.ClassTask:
    rows = []
    labels = []
    for member in (
        "soybean-large.data",
        "soybean-large.test",
    ):
        for line in _read_hidden_lines(
            "soybean-large.zip",
            member,
        ):
            fields = line.split(",")
            labels.append(fields[0])
            rows.append(tuple(fields[1:]))
    return base.make_task(
        "soybean-large",
        tuple(
            f"attribute-{index}"
            for index in range(1, len(rows[0]) + 1)
        ),
        rows,
        labels,
    )


def program_digest(
    selected: ProgramClass,
    rotation: int,
    hidden_manifest: dict[str, object],
) -> str:
    payload = {
        "program": selected.canonical.text(),
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


def constant_programs() -> dict[str, StatePolicy]:
    return {
        name: StatePolicy(None, name, name)
        for name in OBJECTIVE_NAMES
    }


def run(seed: int = 1901) -> dict[str, object]:
    development_manifest = base.verify_manifest(
        base.DEVELOPMENT_ROOT
    )
    v33_manifest = base.verify_manifest(base.HIDDEN_ROOT)
    hidden_manifest = verify_hidden_manifest()
    if not development_manifest["all_hashes_match"]:
        raise RuntimeError("v32 archive hash mismatch")
    if not v33_manifest["all_hashes_match"]:
        raise RuntimeError("v33 archive hash mismatch")
    if not hidden_manifest["all_hashes_match"]:
        raise RuntimeError("v34 archive hash mismatch")

    rotation = (seed - 1901) % 3
    training, development = opened_training_and_development(
        rotation
    )
    selected, synthesis = select_program(
        training,
        development,
    )
    digest = program_digest(
        selected,
        rotation,
        hidden_manifest,
    )

    # Hidden medical and biological records are opened only after the
    # state-program class, canonical member and archive commitments freeze.
    hidden = [
        load_breast_cancer(),
        load_lymphography(),
        load_primary_tumor(),
        load_soybean_large(),
    ]
    candidate = aggregate(hidden, selected.canonical)
    controls = {
        name: aggregate(hidden, program)
        for name, program in constant_programs().items()
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
            for control in controls.values()
        )
        eligible = [
            control["tasks"][task.name]
            for control in controls.values()
            if control["tasks"][task.name]["diagnosed_fraction"]
            >= best_diagnosed - 1e-12
        ]
        best_worst = min(
            row["worst_queries"] for row in eligible
        )
        best_mean = min(
            row["mean_queries"] for row in eligible
        )
        diagnosed_gap = (
            candidate_row["diagnosed_fraction"]
            - best_diagnosed
        )
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
            "candidate_diagnosed": candidate_row[
                "diagnosed_fraction"
            ],
            "best_control_diagnosed": best_diagnosed,
            "diagnosed_gap": diagnosed_gap,
            "candidate_worst": candidate_row["worst_queries"],
            "best_control_worst": best_worst,
            "worst_gap": worst_gap,
            "candidate_mean": candidate_row["mean_queries"],
            "best_control_mean": best_mean,
            "mean_gap": mean_gap,
        }

    state_dependent = (
        selected.canonical.condition is not None
        and selected.canonical.when_true
        != selected.canonical.when_false
        and candidate["uses_both_branches"]
    )
    gate = (
        development_manifest["all_hashes_match"]
        and v33_manifest["all_hashes_match"]
        and hidden_manifest["all_hashes_match"]
        and synthesis["training"]["diagnosed_min"] >= 0.95
        and synthesis["development"]["diagnosed_min"] >= 0.95
        and candidate["diagnosed_min"] >= 0.90
        and state_dependent
        and min(diagnosed_gaps) >= -0.005
        and max(worst_gaps) <= 1
        and float(np.median(mean_gaps)) <= -0.05
        and strict_wins >= 2
    )
    return {
        "status": (
            "state_dependent_external_policy_candidate"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "a one-branch state-dependent experiment-selection program is "
            "synthesized and trajectory-quotiented on previously opened UCI "
            "datasets, frozen before four new medical and biological archives "
            "are opened, and compared with all constant specialist policies; "
            "this is an external candidate, not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "development_manifest": development_manifest,
        "v33_manifest": v33_manifest,
        "hidden_manifest": hidden_manifest,
        "rotation": rotation,
        "training_tasks": [task.name for task in training],
        "development_tasks": [task.name for task in development],
        "synthesis": synthesis,
        "selected_class": {
            "program": selected.canonical.text(),
            "condition": (
                selected.canonical.condition.text()
                if selected.canonical.condition
                else None
            ),
            "when_true": selected.canonical.when_true,
            "when_false": selected.canonical.when_false,
            "member_count": len(selected.members),
            "class_digest": selected.digest,
        },
        "frozen_program_digest": digest,
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
        "controls": controls,
        "comparisons": comparisons,
        "state_dependent": state_dependent,
        "strict_hidden_wins": strict_wins,
        "minimum_diagnosed_gap": min(diagnosed_gaps),
        "maximum_worst_query_gap": max(worst_gaps),
        "median_mean_query_gap": float(np.median(mean_gaps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected_class"],
                "candidate": report["candidate"],
                "strict_hidden_wins": report[
                    "strict_hidden_wins"
                ],
                "minimum_diagnosed_gap": report[
                    "minimum_diagnosed_gap"
                ],
                "median_mean_query_gap": report[
                    "median_mean_query_gap"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
