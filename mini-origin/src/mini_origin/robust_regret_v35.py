from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import state_policy_v34 as v34


@dataclass(frozen=True)
class RegretProfile:
    diagnosed_regret_max: float
    worst_query_regret_max: int
    mean_query_regret_median: float
    mean_query_regret_max: float
    strict_wins: int
    no_harm_tasks: int
    uses_both_branches: bool


def control_baselines(
    tasks: list[object],
) -> dict[str, dict[str, float | int]]:
    """Evaluate each constant specialist once per task."""
    controls = v34.constant_programs()
    baselines: dict[str, dict[str, float | int]] = {}
    for task in tasks:
        rows = [
            v34.evaluate(task, program)
            for program in controls.values()
        ]
        best_diagnosed = max(
            row.diagnosed_fraction for row in rows
        )
        eligible = [
            row
            for row in rows
            if row.diagnosed_fraction >= best_diagnosed - 1e-12
        ]
        baselines[task.name] = {
            "best_diagnosed": best_diagnosed,
            "best_worst": min(
                row.worst_queries for row in eligible
            ),
            "best_mean": min(
                row.mean_queries for row in eligible
            ),
        }
    return baselines


def task_regret(
    task: object,
    candidate: v34.StatePolicy,
    baseline: dict[str, float | int],
) -> dict[str, float | int | bool]:
    candidate_row = v34.evaluate(task, candidate)
    best_diagnosed = float(baseline["best_diagnosed"])
    best_worst = int(baseline["best_worst"])
    best_mean = float(baseline["best_mean"])
    diagnosed_regret = (
        best_diagnosed - candidate_row.diagnosed_fraction
    )
    worst_regret = candidate_row.worst_queries - best_worst
    mean_regret = candidate_row.mean_queries - best_mean
    strict_win = int(
        diagnosed_regret < -1e-12
        or (
            diagnosed_regret <= 1e-12
            and (worst_regret < 0 or mean_regret < -1e-12)
        )
    )
    no_harm = int(
        diagnosed_regret <= 0.005
        and worst_regret <= 1
        and mean_regret <= 0.10
    )
    return {
        "diagnosed_regret": float(diagnosed_regret),
        "worst_query_regret": int(worst_regret),
        "mean_query_regret": float(mean_regret),
        "strict_win": strict_win,
        "no_harm": no_harm,
        "uses_both_branches": candidate_row.uses_both_branches,
        "candidate_diagnosed": candidate_row.diagnosed_fraction,
        "candidate_worst": candidate_row.worst_queries,
        "candidate_mean": candidate_row.mean_queries,
        "best_control_diagnosed": best_diagnosed,
        "best_control_worst": best_worst,
        "best_control_mean": best_mean,
    }


def profile_program(
    tasks: list[object],
    program: v34.StatePolicy,
    baselines: dict[str, dict[str, float | int]],
) -> tuple[RegretProfile, dict[str, object]]:
    rows = {
        task.name: task_regret(
            task,
            program,
            baselines[task.name],
        )
        for task in tasks
    }
    diagnosed = [
        float(row["diagnosed_regret"])
        for row in rows.values()
    ]
    worst = [
        int(row["worst_query_regret"])
        for row in rows.values()
    ]
    mean = [
        float(row["mean_query_regret"])
        for row in rows.values()
    ]
    profile = RegretProfile(
        diagnosed_regret_max=float(max(diagnosed)),
        worst_query_regret_max=int(max(worst)),
        mean_query_regret_median=float(np.median(mean)),
        mean_query_regret_max=float(max(mean)),
        strict_wins=sum(
            int(row["strict_win"])
            for row in rows.values()
        ),
        no_harm_tasks=sum(
            int(row["no_harm"])
            for row in rows.values()
        ),
        uses_both_branches=any(
            bool(row["uses_both_branches"])
            for row in rows.values()
        ),
    )
    return profile, rows


def robust_score(
    profile: RegretProfile,
    program: v34.StatePolicy,
) -> tuple[float | int | str, ...]:
    """Lexicographic minimax regret; thresholds are not fitted to hidden data."""
    return (
        -profile.diagnosed_regret_max,
        -profile.worst_query_regret_max,
        -profile.mean_query_regret_max,
        -profile.mean_query_regret_median,
        profile.no_harm_tasks,
        profile.strict_wins,
        int(profile.uses_both_branches),
        -program.complexity,
        program.text(),
    )


def select_robust_program(
    tasks: list[object],
) -> tuple[v34.ProgramClass, RegretProfile, dict[str, object]]:
    programs = v34.grammar()
    classes = v34.quotient_programs(tasks, programs)
    baselines = control_baselines(tasks)
    rows = []
    for value in classes:
        profile, details = profile_program(
            tasks,
            value.canonical,
            baselines,
        )
        rows.append(
            (
                robust_score(profile, value.canonical),
                value,
                profile,
                details,
            )
        )
    _, selected, profile, details = max(
        rows,
        key=lambda row: row[0],
    )
    return selected, profile, {
        "program_count": len(programs),
        "quotient_class_count": len(classes),
        "selected_program": selected.canonical.text(),
        "selected_class_size": len(selected.members),
        "profile": profile.__dict__,
        "task_regrets": details,
    }


def opened_domain_pool() -> list[object]:
    tasks: list[object] = []
    for rotation in range(3):
        training, development = (
            v34.opened_training_and_development(rotation)
        )
        tasks.extend(training)
        tasks.extend(development)
    tasks.extend([
        v34.load_breast_cancer(),
        v34.load_lymphography(),
        v34.load_primary_tumor(),
        v34.load_soybean_large(),
    ])
    unique = {}
    for task in tasks:
        unique[task.name] = task
    return [unique[name] for name in sorted(unique)]


def run() -> dict[str, object]:
    tasks = opened_domain_pool()
    selected, profile, synthesis = select_robust_program(tasks)
    digest = hashlib.sha256(
        json.dumps(
            {
                "program": selected.canonical.text(),
                "class_digest": selected.digest,
                "domains": [task.name for task in tasks],
                "selector": "lexicographic_minimax_regret_v1",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    development_gate = (
        profile.diagnosed_regret_max <= 0.005
        and profile.worst_query_regret_max <= 1
        and profile.mean_query_regret_max <= 0.10
        and profile.no_harm_tasks == len(tasks)
    )
    return {
        "status": (
            "robust_regret_selector_ready"
            if development_gate
            else "not_yet"
        ),
        "claim_scope": (
            "a state-dependent policy is selected by worst held-out-domain regret "
            "against all constant specialists across every previously opened UCI "
            "domain; no new external holdout has been evaluated, so this is a "
            "selection-protocol improvement rather than a breakthrough"
        ),
        "development_gate": development_gate,
        "domain_count": len(tasks),
        "domains": [task.name for task in tasks],
        "selected_class": {
            "program": selected.canonical.text(),
            "member_count": len(selected.members),
            "class_digest": selected.digest,
        },
        "profile": profile.__dict__,
        "synthesis": synthesis,
        "frozen_selector_digest": digest,
        "next_external_gate": {
            "required_fresh_domains": 4,
            "minimum_diagnosed_gap": -0.005,
            "maximum_worst_query_regret": 1,
            "maximum_mean_query_regret": 0.10,
            "strict_wins_required": 2,
            "replication_seeds": 5,
            "passing_seeds_required": 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "selected": report["selected_class"],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
