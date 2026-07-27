from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import state_invention_v8 as v8
from . import mechanism_quotient_v17 as v17


def run(seed: int = 181) -> v17.QuotientResult:
    rng = np.random.default_rng(seed)
    raw_programs = v8.dynamic_programs(rng, count=900)
    quotient, metadata = v17.quotient_candidates(raw_programs)
    if not quotient:
        raise RuntimeError("all candidate mechanisms collapsed into known classes")

    development = v17._intervention_scenarios(seed * 10_000)
    ranked: list[
        tuple[float, v17.QuotientCandidate, dict[str, float]]
    ] = []
    for candidate in quotient:
        scores = v17._scenario_scores(candidate.program, development)
        score = v17._robust(scores.values()) - 0.0015 * candidate.program.complexity()
        ranked.append((score, candidate, scores))
    ranked.sort(key=lambda value: value[0], reverse=True)

    # Freeze the candidate using development evidence only. Hidden tasks are
    # not evaluated for any alternative candidate.
    development_score, best, development_scores = ranked[0]
    history = [
        {
            "rank": index + 1,
            "program": candidate.program.text(),
            "development_score": score,
            "known_distance": candidate.known_distance,
            "nearest_known": candidate.nearest_known,
        }
        for index, (score, candidate, _) in enumerate(ranked[:10])
    ]
    history.append(
        {
            "stage": "candidate_frozen_before_hidden",
            "program": best.program.text(),
            "development_score": development_score,
            "development_scores": development_scores,
        }
    )

    hidden = v17._hidden_scenarios(seed * 10_000)
    best_hidden = v17._scenario_scores(best.program, hidden)
    baseline_name, baseline_program, baseline_hidden = v17._known_baseline_scores(hidden)
    no_spawn = v8.no_spawn_control(best.program)
    no_spawn_hidden = v17._scenario_scores(no_spawn, hidden)
    metadata = dict(metadata)
    metadata["candidate_selection"] = "development_only"
    metadata["hidden_candidates_evaluated"] = 1

    return v17.QuotientResult(
        seed=seed,
        candidate=best,
        baseline_name=baseline_name,
        baseline_program=baseline_program,
        hidden=best_hidden,
        baseline_hidden=baseline_hidden,
        no_spawn_hidden=no_spawn_hidden,
        quotient_metadata=metadata,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed)
    payload = result.to_dict()
    payload["claim_scope"] = (
        "development-only mechanism-quotiented state search; hidden tasks evaluate exactly "
        "one frozen candidate after behavioural equivalence classes matching named controls "
        "have been excluded; external novelty still requires symbolic equivalence checks, "
        "broader baselines, independent implementation and outside review"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strict_hidden_score": payload["strict_hidden_score"],
                "strict_baseline_score": payload["strict_baseline_score"],
                "strict_no_spawn_score": payload["strict_no_spawn_score"],
                "known_mechanism_distance": payload["known_mechanism_distance"],
                "hidden_candidates_evaluated": payload["quotient"][
                    "hidden_candidates_evaluated"
                ],
                "program": payload["program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
