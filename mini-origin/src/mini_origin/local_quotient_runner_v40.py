from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import attainable_envelope_v38 as v38
from . import local_quotient_v40 as v40
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


def shared_candidate_rows(
    task: object,
    threshold: int,
) -> tuple[v38.AttainableRow, ...]:
    """Compatibility helper used by the proof test."""
    exact = v40.LocalQuotientPlanner(task)
    fallbacks = {
        objective: v37.FallbackPlanner(task, objective)
        for objective in v34.OBJECTIVE_NAMES
    }
    rows, _ = cached_candidate_rows(
        task,
        threshold,
        exact,
        fallbacks,
    )
    return rows


def cached_candidate_rows(
    task: object,
    threshold: int,
    exact: v40.LocalQuotientPlanner,
    fallbacks: dict[str, v37.FallbackPlanner],
) -> tuple[
    tuple[v38.AttainableRow, ...],
    dict[str, v40.QuotientEvaluation],
]:
    rows = []
    evaluations = {}
    for objective in v34.OBJECTIVE_NAMES:
        result = v40.evaluate(
            task,
            v40.QuotientPolicy(threshold, objective),
            exact,
            fallbacks[objective],
        )
        evaluations[objective] = result
        rows.append(v38.AttainableRow(
            objective=objective,
            diagnosed_fraction=result.diagnosed_fraction,
            mean_queries=result.mean_queries,
            worst_queries=result.worst_queries,
            candidates=result.candidates,
            exact_query_uses=result.exact_query_uses,
        ))
    return tuple(rows), evaluations


def cached_compare_task(
    task: object,
    threshold: int,
    exact: v40.LocalQuotientPlanner,
    fallbacks: dict[str, v37.FallbackPlanner],
) -> dict[str, object]:
    baseline = v38.choose_attainable(v38.constant_rows(task))
    candidates, evaluations = cached_candidate_rows(
        task,
        threshold,
        exact,
        fallbacks,
    )
    candidate = v38.choose_attainable(candidates)
    selected = evaluations[candidate.objective]
    diagnosed_gap = (
        candidate.diagnosed_fraction - baseline.diagnosed_fraction
    )
    worst_gap = candidate.worst_queries - baseline.worst_queries
    mean_gap = candidate.mean_queries - baseline.mean_queries
    coordinate_certificate = (
        (abs(diagnosed_gap) > 1e-12 or worst_gap <= 0)
        and (
            not (abs(diagnosed_gap) <= 1e-12 and worst_gap == 0)
            or mean_gap <= 1e-12
        )
    )
    return {
        "baseline": baseline.__dict__,
        "candidate": candidate.__dict__,
        "metric_no_harm": candidate.metric() >= baseline.metric(),
        "coordinate_certificate": coordinate_certificate,
        "strict_win": candidate.metric() > baseline.metric(),
        "diagnosed_gap": diagnosed_gap,
        "worst_query_gap": worst_gap,
        "mean_query_gap": mean_gap,
        "total_query_saving": (
            baseline.mean_queries - candidate.mean_queries
        ) * task.candidate_count,
        "selected_diagnostics": selected.__dict__,
    }


def cached_profile(
    tasks: list[object],
    threshold: int,
    exact_planners: dict[str, v40.LocalQuotientPlanner],
    fallback_planners: dict[str, dict[str, v37.FallbackPlanner]],
) -> tuple[dict[str, object], dict[str, object]]:
    details = {
        task.name: cached_compare_task(
            task,
            threshold,
            exact_planners[task.name],
            fallback_planners[task.name],
        )
        for task in tasks
    }
    profile = {
        "no_harm_tasks": sum(
            int(row["metric_no_harm"])
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
        "new_domain_strict_wins": sum(
            int(row["strict_win"])
            for name, row in details.items()
            if name in v40.NEW_DOMAIN_NAMES
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
        "aggregate_total_query_saving": float(sum(
            float(row["total_query_saving"])
            for row in details.values()
        )),
        "exact_query_uses": sum(
            int(row["candidate"]["exact_query_uses"])
            for row in details.values()
        ),
        "raw_queries_seen": sum(
            int(row["selected_diagnostics"]["raw_queries_seen"])
            for row in details.values()
        ),
        "quotient_queries_seen": sum(
            int(row["selected_diagnostics"]["quotient_queries_seen"])
            for row in details.values()
        ),
    }
    return profile, details


def run() -> dict[str, object]:
    tasks, verification = v40.opened_tasks()
    exact_planners = {
        task.name: v40.LocalQuotientPlanner(task)
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
    for threshold in v40.THRESHOLDS:
        profile, details = cached_profile(
            tasks,
            threshold,
            exact_planners,
            fallback_planners,
        )
        rows.append((
            v40.threshold_score(profile, threshold),
            threshold,
            profile,
            details,
        ))
    _, threshold, profile, details = max(
        rows,
        key=lambda row: row[0],
    )
    gate = (
        verification["all_hashes_match"]
        and int(profile["no_harm_tasks"]) == len(tasks)
        and int(profile["coordinate_certificate_tasks"]) == len(tasks)
        and int(profile["strict_wins"]) >= 5
        and int(profile["new_domain_strict_wins"]) >= 1
        and float(profile["minimum_diagnosed_gap"]) >= -1e-12
        and int(profile["maximum_tied_diagnosis_worst_gap"]) <= 0
        and float(profile["maximum_fully_tied_mean_gap"]) <= 1e-12
        and float(profile["aggregate_total_query_saving"]) >= 100.0
        and int(profile["exact_query_uses"]) > 0
        and int(profile["quotient_queries_seen"]) < int(
            profile["raw_queries_seen"]
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "threshold": threshold,
                "compiler": "local_partition_quotient_exact_subtree_v1",
                "domains": [task.name for task in tasks],
                "v39_manifest": verification["rows"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "local_quotient_compiler_ready"
            if gate
            else "not_yet"
        ),
        "claim_scope": (
            "remaining experiments are quotiented by their state-local candidate "
            "partition before exact dynamic programming, and any accepted exact "
            "subtree is executed completely; all evaluated datasets are now "
            "opened development evidence, not a fresh external breakthrough"
        ),
        "development_gate": gate,
        "domain_count": len(tasks),
        "threshold_count": len(v40.THRESHOLDS),
        "selected_threshold": threshold,
        "profile": profile,
        "task_comparisons": details,
        "archive_verification": verification,
        "frozen_compiler_digest": digest,
        "cache_reuse": "exact_and_fallback_planners_shared_across_thresholds",
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
        "selected_threshold": report["selected_threshold"],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
