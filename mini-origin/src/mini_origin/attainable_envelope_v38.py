from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from . import exact_tail_v36 as v36
from . import robust_regret_v35 as v35
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


THRESHOLDS = (4, 6, 8, 10, 12, 16)
FEATURE_LIMITS = (4, 6, 8, 10, 12)
OBJECTIVE_RANK = {
    name: index
    for index, name in enumerate(v34.OBJECTIVE_NAMES)
}


@dataclass(frozen=True)
class AttainableRow:
    objective: str
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    candidates: int
    exact_query_uses: int

    def metric(self) -> tuple[float, int, float]:
        return (
            self.diagnosed_fraction,
            -self.worst_queries,
            -self.mean_queries,
        )

    def selection_score(self) -> tuple[float, int, float, int, int]:
        return (
            *self.metric(),
            -self.exact_query_uses,
            -OBJECTIVE_RANK[self.objective],
        )


def constant_rows(task: object) -> tuple[AttainableRow, ...]:
    rows = []
    for objective in v34.OBJECTIVE_NAMES:
        result = v34.evaluate(
            task,
            v34.StatePolicy(None, objective, objective),
        )
        rows.append(
            AttainableRow(
                objective=objective,
                diagnosed_fraction=result.diagnosed_fraction,
                mean_queries=result.mean_queries,
                worst_queries=result.worst_queries,
                candidates=result.candidates,
                exact_query_uses=0,
            )
        )
    return tuple(rows)


def repaired_rows(
    task: object,
    threshold: int,
    feature_limit: int,
    exact: v36.ExactPlanner,
    fallbacks: dict[str, v37.FallbackPlanner],
) -> tuple[AttainableRow, ...]:
    rows = []
    for objective in v34.OBJECTIVE_NAMES:
        result = v37.evaluate(
            task,
            v37.SafePolicy(
                threshold,
                feature_limit,
                objective,
            ),
            exact,
            fallbacks[objective],
        )
        rows.append(
            AttainableRow(
                objective=objective,
                diagnosed_fraction=result.diagnosed_fraction,
                mean_queries=result.mean_queries,
                worst_queries=result.worst_queries,
                candidates=result.candidates,
                exact_query_uses=result.exact_query_uses,
            )
        )
    return tuple(rows)


def choose_attainable(
    rows: tuple[AttainableRow, ...],
) -> AttainableRow:
    return max(rows, key=lambda row: row.selection_score())


def compare_task(
    task: object,
    threshold: int,
    feature_limit: int,
    exact: v36.ExactPlanner,
    fallbacks: dict[str, v37.FallbackPlanner],
    baseline: AttainableRow,
) -> dict[str, object]:
    candidate = choose_attainable(
        repaired_rows(
            task,
            threshold,
            feature_limit,
            exact,
            fallbacks,
        )
    )
    lex_no_harm = candidate.metric() >= baseline.metric()
    strict_win = candidate.metric() > baseline.metric()
    diagnosed_gap = (
        candidate.diagnosed_fraction
        - baseline.diagnosed_fraction
    )
    worst_gap = candidate.worst_queries - baseline.worst_queries
    mean_gap = candidate.mean_queries - baseline.mean_queries
    total_query_saving = (
        baseline.mean_queries - candidate.mean_queries
    ) * task.candidate_count
    tied_diagnosis = abs(diagnosed_gap) <= 1e-12
    tied_worst = worst_gap == 0
    coordinate_certificate = (
        (not tied_diagnosis or worst_gap <= 0)
        and (
            not (tied_diagnosis and tied_worst)
            or mean_gap <= 1e-12
        )
    )
    return {
        "baseline": baseline.__dict__,
        "candidate": candidate.__dict__,
        "lex_no_harm": lex_no_harm,
        "strict_win": strict_win,
        "diagnosed_gap": diagnosed_gap,
        "worst_query_gap": worst_gap,
        "mean_query_gap": mean_gap,
        "total_query_saving": total_query_saving,
        "coordinate_certificate": coordinate_certificate,
    }


def profile_configuration(
    tasks: list[object],
    threshold: int,
    feature_limit: int,
    exact_planners: dict[str, v36.ExactPlanner],
    fallback_planners: dict[str, dict[str, v37.FallbackPlanner]],
    baselines: dict[str, AttainableRow],
) -> tuple[dict[str, object], dict[str, object]]:
    details = {
        task.name: compare_task(
            task,
            threshold,
            feature_limit,
            exact_planners[task.name],
            fallback_planners[task.name],
            baselines[task.name],
        )
        for task in tasks
    }
    mean_gaps = [
        float(row["mean_query_gap"])
        for row in details.values()
    ]
    profile = {
        "lex_no_harm_tasks": sum(
            int(row["lex_no_harm"])
            for row in details.values()
        ),
        "coordinate_certificate_tasks": sum(
            int(row["coordinate_certificate"])
            for row in details.values()
        ),
        "strict_wins": sum(
            int(row["strict_win"])
            for row in details.values()
        ),
        "minimum_diagnosed_gap": min(
            float(row["diagnosed_gap"])
            for row in details.values()
        ),
        "maximum_tied_diagnosis_worst_gap": max(
            int(row["worst_query_gap"])
            for row in details.values()
            if abs(float(row["diagnosed_gap"])) <= 1e-12
        ),
        "maximum_fully_tied_mean_gap": max(
            float(row["mean_query_gap"])
            for row in details.values()
            if abs(float(row["diagnosed_gap"])) <= 1e-12
            and int(row["worst_query_gap"]) == 0
        ),
        "median_mean_query_gap": float(np.median(mean_gaps)),
        "aggregate_total_query_saving": float(sum(
            float(row["total_query_saving"])
            for row in details.values()
        )),
        "exact_query_uses": sum(
            int(row["candidate"]["exact_query_uses"])
            for row in details.values()
        ),
    }
    return profile, details


def configuration_score(
    profile: dict[str, object],
    threshold: int,
    feature_limit: int,
) -> tuple[float | int, ...]:
    return (
        int(profile["lex_no_harm_tasks"]),
        int(profile["coordinate_certificate_tasks"]),
        int(profile["strict_wins"]),
        float(profile["aggregate_total_query_saving"]),
        -float(profile["median_mean_query_gap"]),
        int(profile["exact_query_uses"] > 0),
        -threshold,
        -feature_limit,
    )


def run() -> dict[str, object]:
    tasks = v35.opened_domain_pool()
    baselines = {
        task.name: choose_attainable(constant_rows(task))
        for task in tasks
    }
    exact_planners = {
        task.name: v36.ExactPlanner(task)
        for task in tasks
    }
    fallback_planners = {
        task.name: {
            objective: v37.FallbackPlanner(task, objective)
            for objective in v34.OBJECTIVE_NAMES
        }
        for task in tasks
    }
    rows = []
    for threshold, feature_limit in itertools.product(
        THRESHOLDS,
        FEATURE_LIMITS,
    ):
        profile, details = profile_configuration(
            tasks,
            threshold,
            feature_limit,
            exact_planners,
            fallback_planners,
            baselines,
        )
        rows.append(
            (
                configuration_score(
                    profile,
                    threshold,
                    feature_limit,
                ),
                threshold,
                feature_limit,
                profile,
                details,
            )
        )
    _, threshold, feature_limit, profile, details = max(
        rows,
        key=lambda row: row[0],
    )
    gate = (
        int(profile["lex_no_harm_tasks"]) == len(tasks)
        and int(profile["coordinate_certificate_tasks"]) == len(tasks)
        and int(profile["strict_wins"]) >= 3
        and float(profile["minimum_diagnosed_gap"]) >= -1e-12
        and int(profile["maximum_tied_diagnosis_worst_gap"]) <= 0
        and float(profile["maximum_fully_tied_mean_gap"]) <= 1e-12
        and float(profile["aggregate_total_query_saving"]) >= 20.0
        and int(profile["exact_query_uses"]) > 0
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "threshold": threshold,
                "feature_limit": feature_limit,
                "baseline": "single_attainable_lexicographic_specialist_v1",
                "portfolio": "safe_exact_envelope_v1",
                "domains": [task.name for task in tasks],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "attainable_envelope_ready"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "a single attainable specialist is selected per opened task by a "
            "fixed lexicographic rule, and the best safely repaired specialist "
            "is compared against that real baseline; no fresh external records "
            "are evaluated in this development study"
        ),
        "development_gate": gate,
        "domain_count": len(tasks),
        "configuration_count": len(THRESHOLDS) * len(FEATURE_LIMITS),
        "selected_configuration": {
            "candidate_threshold": threshold,
            "feature_limit": feature_limit,
        },
        "profile": profile,
        "task_comparisons": details,
        "frozen_envelope_digest": digest,
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
        "selected_configuration": report[
            "selected_configuration"
        ],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
