from __future__ import annotations

import argparse
import json
from pathlib import Path

from .language_v6 import (
    Expr,
    LanguageResult,
    RankedProgram,
    _hand_delta,
    _hand_hebb,
    _select_counterexample,
    _unique,
    base_atoms,
    binary,
    counterexample_pool,
    evaluate,
    hidden_scenarios,
    mine_macros,
    rank_programs,
    terminal,
    training_scenarios,
    unary,
)


def compact_task_specific_programs() -> list[Expr]:
    """Deep enough to expose reusable residuals, small enough for exact search."""
    teacher = terminal("teacher")
    pred = terminal("pred")
    peer = terminal("peer")
    signal_atoms = _unique(
        [
            teacher,
            pred,
            peer,
            terminal("c1"),
            terminal("cm1"),
            terminal("c01"),
            unary("neg", teacher),
            unary("neg", pred),
            unary("neg", peer),
            unary("tanh", teacher),
            unary("tanh", pred),
            unary("tanh", peer),
            unary("clip", teacher),
            unary("clip", pred),
            unary("clip", peer),
        ]
    )
    elig = terminal("elig")
    shrink = binary("mul", terminal("c01"), terminal("weight"))
    programs: list[Expr] = []
    for left in signal_atoms:
        for right in signal_atoms:
            for op in ("add", "mul"):
                feature = binary(op, left, right)
                credit = binary("mul", elig, feature)
                programs.append(credit)
                programs.append(binary("add", credit, unary("neg", shrink)))
    return _unique(programs + signal_atoms)


def compact_shallow_programs(atoms: list[Expr]) -> list[Expr]:
    """All candidates share the same shallow construction budget."""
    elig = terminal("elig")
    weight = terminal("weight")
    shrink = binary("mul", terminal("c01"), weight)
    values: list[Expr] = list(atoms)
    credit_terms: list[Expr] = []
    for feature in atoms:
        credit = binary("mul", elig, feature)
        credit_terms.append(credit)
        values.append(credit)
        values.append(binary("add", credit, unary("neg", shrink)))

    # Permit a second small local correction while keeping learned macros atomic.
    corrections = [
        binary("mul", terminal("c01"), feature)
        for feature in atoms
    ]
    for credit in credit_terms:
        for correction in corrections:
            values.append(binary("add", credit, correction))
    return _unique(values)


def _programs(ranked: list[RankedProgram]) -> list[Expr]:
    return [item.program for item in ranked]


def run_fast_language_search(seed: int = 61) -> LanguageResult:
    family_scenarios = training_scenarios(seed * 10_000)
    deep = compact_task_specific_programs()
    per_family = {
        family: rank_programs(deep, scenarios, limit=16)
        for family, scenarios in family_scenarios.items()
    }
    macros = mine_macros(per_family, limit=8)

    base = base_atoms()
    expanded = compact_shallow_programs(_unique(base + macros))
    no_library = compact_shallow_programs(base)
    curriculum = [scenario for values in family_scenarios.values() for scenario in values]

    # Screen once, then spend the counterexample budget only on plausible programs.
    expanded_shortlist = _programs(rank_programs(expanded, curriculum, limit=72))
    control_shortlist = _programs(rank_programs(no_library, curriculum, limit=72))
    best = rank_programs(expanded_shortlist, curriculum, limit=1)[0]
    control = rank_programs(control_shortlist, curriculum, limit=1)[0]

    pool = counterexample_pool(seed * 10_000)
    used = {scenario.label() for scenario in curriculum}
    history: list[dict[str, object]] = []
    for iteration in range(3):
        counterexample = _select_counterexample(best.program, pool, used)
        history.append(
            {
                "iteration": iteration,
                "curriculum_size": len(curriculum),
                "best_program": best.program.text(),
                "best_score": best.score,
                "no_library_program": control.program.text(),
                "no_library_score": control.score,
                "counterexample": (
                    counterexample.label() if counterexample is not None else None
                ),
            }
        )
        if counterexample is None:
            break
        curriculum.append(counterexample)
        used.add(counterexample.label())
        best = rank_programs(expanded_shortlist, curriculum, limit=1)[0]
        control = rank_programs(control_shortlist, curriculum, limit=1)[0]

    hidden = hidden_scenarios(seed * 10_000)
    return LanguageResult(
        seed=seed,
        macros=macros,
        best_program=best.program,
        no_library_program=control.program,
        hidden_scores=evaluate(best.program, hidden),
        no_library_hidden_scores=evaluate(control.program, hidden),
        delta_hidden_scores=evaluate(_hand_delta(), hidden),
        hebb_hidden_scores=evaluate(_hand_hebb(), hidden),
        search_history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_fast_language_search(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "best_program": payload["best_program"],
                "macros": payload["macros"],
                "strict_hidden_score": payload["strict_hidden_score"],
                "no_library_strict_score": payload["no_library_strict_score"],
                "delta_strict_score": payload["delta_strict_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
