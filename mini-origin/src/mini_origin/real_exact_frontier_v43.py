from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import exact_quotient_certificate_v42 as v42
from . import state_policy_v34 as v34


FRONTIER_BUDGET = 250_000
MAX_STATES_PER_TASK = 10
MIN_CANDIDATES = 10
MAX_CANDIDATES = 24
MIN_RAW_QUERIES = 20
MAX_QUOTIENT_REPRESENTATIVES = 16
MIN_REDUNDANCY = 8
BUDGET_LADDER = (10_000, 50_000, 250_000)


def expected_elimination_numerator(
    task: object,
    allowed: int,
    query: int,
) -> int:
    """Uniform-prior expected eliminations, scaled by |V|."""
    size = allowed.bit_count()
    return sum(
        bucket_size * (size - bucket_size)
        for bucket_size in (
            child.bit_count()
            for mask in task.masks_for(query).values()
            if (child := allowed & mask)
        )
    )


def gini_approximation_certificate(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    separating = []
    query_bits = remaining
    while query_bits:
        bit = query_bits & -query_bits
        query = bit.bit_length() - 1
        query_bits ^= bit
        signature = v42.partition_signature(task, allowed, query)
        if len(signature) <= 1:
            continue
        separating.append(query)
    if not separating:
        return {
            "checked": False,
            "passed": True,
            "reason": "no separating query",
        }
    selected = v34.base.select_query(
        task,
        allowed,
        remaining,
        v34.OBJECTIVES["gini"],
    )
    values = {
        query: expected_elimination_numerator(
            task, allowed, query
        )
        for query in separating
    }
    optimum = max(values.values())
    return {
        "checked": True,
        "passed": values[selected] == optimum,
        "selected_query": selected,
        "selected_value": values[selected],
        "maximum_value": optimum,
        "maximizer_count": sum(
            int(value == optimum) for value in values.values()
        ),
    }


def structural_rank(
    task_name: str,
    allowed: int,
    remaining: int,
    representatives: int,
) -> tuple[int, int, int, str]:
    return (
        -allowed.bit_count(),
        -representatives,
        -remaining.bit_count(),
        hashlib.sha256(
            f"{task_name}:{allowed}:{remaining}".encode("utf-8")
        ).hexdigest(),
    )


def select_frontier_states(task: object) -> tuple[
    list[tuple[int, int, int]],
    dict[str, int],
]:
    states = v42.collect_policy_states(task)
    candidates = []
    for allowed, remaining in states:
        size = allowed.bit_count()
        raw = remaining.bit_count()
        representatives = v42.quotient_representative_count(
            task, allowed, remaining
        )
        redundancy = raw - representatives
        if (
            MIN_CANDIDATES <= size <= MAX_CANDIDATES
            and raw >= MIN_RAW_QUERIES
            and representatives <= MAX_QUOTIENT_REPRESENTATIVES
            and redundancy >= MIN_REDUNDANCY
        ):
            candidates.append((allowed, remaining, representatives))
    candidates.sort(
        key=lambda row: structural_rank(
            task.name,
            row[0],
            row[1],
            row[2],
        )
    )
    return candidates[:MAX_STATES_PER_TASK], {
        "harvested_states": len(states),
        "frontier_candidates": len(candidates),
        "selected_states": min(
            len(candidates), MAX_STATES_PER_TASK
        ),
    }


def solve_frontier_state(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    approximation = gini_approximation_certificate(
        task, allowed, remaining
    )
    quotient_solved = False
    plain_solved = False
    quotient_result = None
    plain_result = None
    try:
        quotient_result = v42.CountingQuotientPlanner(
            task, FRONTIER_BUDGET
        ).result(allowed, remaining)
        quotient_solved = True
    except v42.BudgetExceeded:
        pass
    if quotient_solved:
        try:
            plain_result = v42.CountingPlainPlanner(
                task, FRONTIER_BUDGET
            ).result(allowed, remaining)
            plain_solved = True
        except v42.BudgetExceeded:
            pass
    matched = (
        quotient_solved
        and plain_solved
        and v42.plan_metrics(quotient_result.plan)
        == v42.plan_metrics(plain_result.plan)
    )
    if quotient_solved:
        quotient_expansions = (
            quotient_result.stats.query_expansions
        )
        if plain_solved:
            plain_expansions_lower_bound = (
                plain_result.stats.query_expansions
            )
        else:
            plain_expansions_lower_bound = FRONTIER_BUDGET + 1
        ratio_lower_bound = (
            plain_expansions_lower_bound
            / max(1, quotient_expansions)
        )
    else:
        quotient_expansions = None
        plain_expansions_lower_bound = None
        ratio_lower_bound = None
    ladder = {}
    for budget in BUDGET_LADDER:
        ladder[str(budget)] = {
            "quotient_solved": (
                quotient_solved
                and quotient_result.stats.query_expansions <= budget
            ),
            "plain_solved": (
                plain_solved
                and plain_result.stats.query_expansions <= budget
            ),
        }
    return {
        "candidate_count": allowed.bit_count(),
        "raw_remaining_queries": remaining.bit_count(),
        "quotient_representatives": (
            v42.quotient_representative_count(
                task, allowed, remaining
            )
        ),
        "gini_approximation_certificate": approximation,
        "quotient_solved": quotient_solved,
        "plain_solved": plain_solved,
        "matched_if_both": matched,
        "quotient_plan": (
            v42.plan_metrics(quotient_result.plan)
            if quotient_result is not None else None
        ),
        "plain_plan": (
            v42.plan_metrics(plain_result.plan)
            if plain_result is not None else None
        ),
        "quotient_stats": (
            quotient_result.stats.__dict__
            if quotient_result is not None else None
        ),
        "plain_stats": (
            plain_result.stats.__dict__
            if plain_result is not None else None
        ),
        "expansion_ratio_lower_bound": ratio_lower_bound,
        "budget_ladder": ladder,
    }


def run() -> dict[str, object]:
    tasks, verification = v42.load_all_opened_tasks()
    rows = []
    task_summaries = []
    for task in tasks:
        selected, summary = select_frontier_states(task)
        summary["task"] = task.name
        task_summaries.append(summary)
        for allowed, remaining, representatives in selected:
            row = solve_frontier_state(task, allowed, remaining)
            assert row["quotient_representatives"] == representatives
            row["task"] = task.name
            row["state_digest"] = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            rows.append(row)
    solved = [row for row in rows if row["quotient_solved"]]
    both = [
        row for row in solved if row["plain_solved"]
    ]
    quotient_only = [
        row for row in solved if not row["plain_solved"]
    ]
    ratios = [
        float(row["expansion_ratio_lower_bound"])
        for row in solved
    ]
    approximation_checked = [
        row["gini_approximation_certificate"]
        for row in rows
        if row["gini_approximation_certificate"]["checked"]
    ]
    ladder = {
        str(budget): {
            "quotient_solved": sum(
                int(row["budget_ladder"][str(budget)][
                    "quotient_solved"
                ])
                for row in rows
            ),
            "plain_solved": sum(
                int(row["budget_ladder"][str(budget)][
                    "plain_solved"
                ])
                for row in rows
            ),
        }
        for budget in BUDGET_LADDER
    }
    gate = (
        verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(rows) >= 50
        and len(solved) >= 40
        and len(quotient_only) >= 10
        and len(both) >= 20
        and all(row["matched_if_both"] for row in both)
        and len(approximation_checked) == len(rows)
        and all(row["passed"] for row in approximation_checked)
        and float(np.median(ratios)) >= 3.0
        and float(np.quantile(ratios, 0.9)) >= 10.0
        and ladder["50000"]["quotient_solved"] >= (
            ladder["50000"]["plain_solved"] + 10
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "budget": FRONTIER_BUDGET,
                "budget_ladder": BUDGET_LADDER,
                "selection": {
                    "max_states_per_task": MAX_STATES_PER_TASK,
                    "candidate_range": [
                        MIN_CANDIDATES, MAX_CANDIDATES
                    ],
                    "minimum_raw_queries": MIN_RAW_QUERIES,
                    "maximum_quotient_representatives": (
                        MAX_QUOTIENT_REPRESENTATIVES
                    ),
                    "minimum_redundancy": MIN_REDUNDANCY,
                },
                "task_names": [task.name for task in tasks],
                "verification": verification,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "real_exact_frontier_candidate"
            if gate else "not_yet"
        ),
        "claim_scope": (
            "the state-local quotient is evaluated on structurally selected "
            "hard real states under a matched 250,000-expansion exact-search "
            "budget, and the uniform-prior expected-elimination greedy criterion "
            "is certified to coincide with the Gini control; a pass demonstrates "
            "a real exact-search frontier extension, while novelty still requires "
            "comparison with independent identification-tree implementations"
        ),
        "development_gate": gate,
        "protocol": {
            "frontier_budget": FRONTIER_BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
            "max_states_per_task": MAX_STATES_PER_TASK,
            "candidate_range": [MIN_CANDIDATES, MAX_CANDIDATES],
            "minimum_raw_queries": MIN_RAW_QUERIES,
            "maximum_quotient_representatives": (
                MAX_QUOTIENT_REPRESENTATIVES
            ),
            "minimum_redundancy": MIN_REDUNDANCY,
        },
        "task_count": len(tasks),
        "task_summaries": task_summaries,
        "frontier_state_count": len(rows),
        "quotient_solved_count": len(solved),
        "plain_solved_count": len(both),
        "quotient_only_solved_count": len(quotient_only),
        "both_plan_match_count": sum(
            int(row["matched_if_both"]) for row in both
        ),
        "gini_certificate_count": len(approximation_checked),
        "gini_certificate_pass_count": sum(
            int(row["passed"]) for row in approximation_checked
        ),
        "expansion_ratio_lower_bound_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "expansion_ratio_lower_bound_p90": (
            float(np.quantile(ratios, 0.9)) if ratios else None
        ),
        "budget_ladder_results": ladder,
        "rows": rows,
        "archive_verification": verification,
        "frozen_frontier_digest": digest,
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
        "frontier_states": report["frontier_state_count"],
        "quotient_solved": report["quotient_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "quotient_only": report["quotient_only_solved_count"],
        "median_ratio": report[
            "expansion_ratio_lower_bound_median"
        ],
        "p90_ratio": report[
            "expansion_ratio_lower_bound_p90"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
