from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from . import attainable_envelope_v38 as v38
from . import exact_tail_v36 as v36
from . import external_envelope_v39 as v39
from . import robust_regret_v35 as v35
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


THRESHOLDS = (4, 6, 8, 10, 12, 16, 20, 24)
NEW_DOMAIN_NAMES = {
    "hayes-roth",
    "spect-heart",
    "audiology-standardized",
    "dermatology",
}


@dataclass(frozen=True)
class QuotientPolicy:
    candidate_threshold: int
    fallback: str

    def text(self) -> str:
        return (
            f"local_quotient_exact_if(candidates<="
            f"{self.candidate_threshold}):else->{self.fallback}"
        )


@dataclass(frozen=True)
class QuotientEvaluation:
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int
    exact_query_uses: int
    quotient_states: int
    raw_queries_seen: int
    quotient_queries_seen: int


class LocalQuotientPlanner:
    def __init__(self, task: object) -> None:
        self.task = task
        self.cache: dict[tuple[int, int], v36.Plan] = {}
        self.raw_queries_seen = 0
        self.quotient_queries_seen = 0

    def partition_signature(
        self,
        allowed: int,
        query: int,
    ) -> tuple[int, ...]:
        return tuple(sorted(
            child
            for mask in self.task.masks_for(query).values()
            if (child := allowed & mask)
        ))

    def canonical_remaining(
        self,
        allowed: int,
        remaining: int,
    ) -> int:
        representatives: dict[tuple[int, ...], int] = {}
        query_bits = remaining
        raw = 0
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            raw += 1
            signature = self.partition_signature(allowed, query)
            if len(signature) <= 1:
                continue
            previous = representatives.get(signature)
            if previous is None or query < previous:
                representatives[signature] = query
        canonical = 0
        for query in representatives.values():
            canonical |= 1 << query
        self.raw_queries_seen += raw
        self.quotient_queries_seen += len(representatives)
        return canonical

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        canonical = self.canonical_remaining(allowed, remaining)
        key = (allowed, canonical)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates = []
        query_bits = canonical
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            children = self.partition_signature(allowed, query)
            next_remaining = canonical & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(v36.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                total_queries=size + sum(
                    row.total_queries for row in child_plans
                ),
                query=query,
            ))
        plan = (
            max(candidates, key=lambda row: row.score())
            if candidates
            else v36.Plan(0, 0, 0, None)
        )
        self.cache[key] = plan
        return plan


def analyse_task(
    task: object,
    policy: QuotientPolicy,
    exact: LocalQuotientPlanner,
    fallback: v37.FallbackPlanner,
) -> tuple[tuple[bool, int, int], ...]:
    results: list[tuple[bool, int, int] | None] = [
        None
    ] * task.candidate_count

    def assign(
        allowed: int,
        prediction: str | None,
        queries: int,
        exact_uses: int,
    ) -> None:
        mask = allowed
        while mask:
            bit = mask & -mask
            candidate = bit.bit_length() - 1
            results[candidate] = (
                prediction == task.labels[candidate],
                queries,
                exact_uses,
            )
            mask ^= bit

    def visit(
        allowed: int,
        remaining: int,
        queries: int,
        exact_uses: int,
        exact_mode: bool,
    ) -> None:
        prediction = v34.base.pure_label(task, allowed)
        if prediction is not None:
            assign(allowed, prediction, queries, exact_uses)
            return
        next_exact_mode = exact_mode
        if exact_mode:
            plan = exact.solve(allowed, remaining)
            query = plan.query
            use_exact = True
        else:
            fallback_plan = fallback.solve(allowed, remaining)
            query = fallback_plan.query
            use_exact = False
            if allowed.bit_count() <= policy.candidate_threshold:
                exact_plan = exact.solve(allowed, remaining)
                if (
                    exact_plan.query is not None
                    and exact_plan.score() > fallback_plan.score()
                ):
                    query = exact_plan.query
                    use_exact = True
                    next_exact_mode = True
        if query is None:
            assign(allowed, None, queries, exact_uses)
            return
        next_remaining = remaining & ~(1 << query)
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
                exact_uses + int(use_exact),
                next_exact_mode,
            )
        if covered != allowed:
            raise AssertionError("query outcomes did not partition state")

    visit(
        task.full_mask,
        (1 << task.query_count) - 1,
        0,
        0,
        False,
    )
    if any(row is None for row in results):
        raise AssertionError("missing candidate result")
    return tuple(row for row in results if row is not None)


def evaluate(
    task: object,
    policy: QuotientPolicy,
    exact: LocalQuotientPlanner,
    fallback: v37.FallbackPlanner,
) -> QuotientEvaluation:
    rows = analyse_task(task, policy, exact, fallback)
    diagnosed = sum(int(row[0]) for row in rows)
    queries = [row[1] for row in rows]
    return QuotientEvaluation(
        diagnosed_fraction=diagnosed / task.candidate_count,
        mean_queries=float(np.mean(queries)),
        worst_queries=max(queries),
        unresolved=task.candidate_count - diagnosed,
        candidates=task.candidate_count,
        exact_query_uses=sum(row[2] for row in rows),
        quotient_states=len(exact.cache),
        raw_queries_seen=exact.raw_queries_seen,
        quotient_queries_seen=exact.quotient_queries_seen,
    )


def candidate_rows(
    task: object,
    threshold: int,
) -> tuple[v38.AttainableRow, ...]:
    rows = []
    for objective in v34.OBJECTIVE_NAMES:
        exact = LocalQuotientPlanner(task)
        fallback = v37.FallbackPlanner(task, objective)
        result = evaluate(
            task,
            QuotientPolicy(threshold, objective),
            exact,
            fallback,
        )
        rows.append(v38.AttainableRow(
            objective=objective,
            diagnosed_fraction=result.diagnosed_fraction,
            mean_queries=result.mean_queries,
            worst_queries=result.worst_queries,
            candidates=result.candidates,
            exact_query_uses=result.exact_query_uses,
        ))
    return tuple(rows)


def compare_task(task: object, threshold: int) -> dict[str, object]:
    baseline = v38.choose_attainable(v38.constant_rows(task))
    candidates = candidate_rows(task, threshold)
    candidate = v38.choose_attainable(candidates)
    metric_no_harm = candidate.metric() >= baseline.metric()
    strict_win = candidate.metric() > baseline.metric()
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
    total_saving = (
        baseline.mean_queries - candidate.mean_queries
    ) * task.candidate_count
    selected_result = None
    for objective in v34.OBJECTIVE_NAMES:
        if objective != candidate.objective:
            continue
        exact = LocalQuotientPlanner(task)
        fallback = v37.FallbackPlanner(task, objective)
        selected_result = evaluate(
            task,
            QuotientPolicy(threshold, objective),
            exact,
            fallback,
        )
        break
    assert selected_result is not None
    return {
        "baseline": baseline.__dict__,
        "candidate": candidate.__dict__,
        "metric_no_harm": metric_no_harm,
        "coordinate_certificate": coordinate_certificate,
        "strict_win": strict_win,
        "diagnosed_gap": diagnosed_gap,
        "worst_query_gap": worst_gap,
        "mean_query_gap": mean_gap,
        "total_query_saving": total_saving,
        "selected_diagnostics": selected_result.__dict__,
    }


def opened_tasks() -> tuple[list[object], dict[str, object]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verification = v39.download_and_verify(root)
        external = v39.load_tasks(root)
    return v35.opened_domain_pool() + external, verification


def profile_threshold(
    tasks: list[object],
    threshold: int,
) -> tuple[dict[str, object], dict[str, object]]:
    details = {
        task.name: compare_task(task, threshold)
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
            if name in NEW_DOMAIN_NAMES
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


def threshold_score(
    profile: dict[str, object],
    threshold: int,
) -> tuple[float | int, ...]:
    return (
        int(profile["no_harm_tasks"]),
        int(profile["coordinate_certificate_tasks"]),
        int(profile["new_domain_strict_wins"]),
        int(profile["strict_wins"]),
        float(profile["aggregate_total_query_saving"]),
        int(profile["exact_query_uses"] > 0),
        -threshold,
    )


def run() -> dict[str, object]:
    tasks, verification = opened_tasks()
    rows = []
    for threshold in THRESHOLDS:
        profile, details = profile_threshold(tasks, threshold)
        rows.append((
            threshold_score(profile, threshold),
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
        "threshold_count": len(THRESHOLDS),
        "selected_threshold": threshold,
        "profile": profile,
        "task_comparisons": details,
        "archive_verification": verification,
        "frozen_compiler_digest": digest,
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
