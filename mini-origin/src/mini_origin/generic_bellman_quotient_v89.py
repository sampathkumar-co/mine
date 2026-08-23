from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Hashable, Protocol


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v89-generic-bellman-quotient-preregistration.json"
)
SEED_START = 89_001
SEEDS_PER_FAMILY = 48


@dataclass(frozen=True)
class Outcome:
    successor: Hashable
    weight: int
    cost: int


@dataclass(frozen=True)
class Action:
    action_id: str
    outcomes: tuple[Outcome, ...]
    tie_rank: int = 0


class BellmanProblem(Protocol):
    name: str
    initial_state: Hashable

    def state_key(self, state: Hashable) -> str: ...

    def terminal(self, state: Hashable) -> bool: ...

    def actions(self, state: Hashable) -> tuple[Action, ...]: ...


@dataclass(frozen=True)
class Objective:
    expected_cost: int
    worst_cost: int


@dataclass
class SolverStats:
    states_solved: int = 0
    action_expansions: int = 0
    raw_actions_considered: int = 0
    representative_actions_considered: int = 0
    dominated_actions_removed: int = 0
    equivalence_classes: int = 0


@dataclass(frozen=True)
class SolveResult:
    objective: Objective
    stats: SolverStats


def _ordered_outcomes(
    problem: BellmanProblem,
    action: Action,
) -> tuple[Outcome, ...]:
    if not action.outcomes:
        raise ValueError(f"action {action.action_id} has no outcomes")
    if any(outcome.weight <= 0 for outcome in action.outcomes):
        raise ValueError("outcome weights must be positive")
    if any(outcome.cost < 0 for outcome in action.outcomes):
        raise ValueError("negative immediate costs are outside v0.89")
    ordered = tuple(
        sorted(action.outcomes, key=lambda row: problem.state_key(row.successor))
    )
    keys = [problem.state_key(row.successor) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "v0.89 requires at most one outcome per canonical successor state"
        )
    return ordered


def action_signature(
    problem: BellmanProblem,
    action: Action,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (problem.state_key(outcome.successor), outcome.weight)
        for outcome in _ordered_outcomes(problem, action)
    )


def action_cost_vector(
    problem: BellmanProblem,
    action: Action,
) -> tuple[int, ...]:
    return tuple(
        outcome.cost for outcome in _ordered_outcomes(problem, action)
    )


def vector_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return (
        len(left) == len(right)
        and all(a <= b for a, b in zip(left, right))
        and any(a < b for a, b in zip(left, right))
    )


def quotient_actions(
    problem: BellmanProblem,
    actions: tuple[Action, ...],
) -> tuple[tuple[Action, ...], int, int]:
    """Return exact representatives under frozen v0.89 action dominance.

    Only actions with identical canonical successor/weight signatures are ever
    compared. Within such a class, componentwise immediate-cost dominance is
    safe for any continuation value determined solely by the successor state
    and any monotone nonnegative path-cost aggregation. Equal cost vectors use
    only the preregistered deterministic tie rule.
    """
    groups: dict[tuple[tuple[str, int], ...], list[Action]] = {}
    for action in actions:
        groups.setdefault(action_signature(problem, action), []).append(action)

    kept: list[Action] = []
    removed = 0
    for signature in sorted(groups):
        group = groups[signature]
        vectors = {
            action.action_id: action_cost_vector(problem, action)
            for action in group
        }
        for action in group:
            vector = vectors[action.action_id]
            dominated = False
            for other in group:
                if other.action_id == action.action_id:
                    continue
                other_vector = vectors[other.action_id]
                if vector_dominates(other_vector, vector):
                    dominated = True
                    break
                if other_vector == vector and (
                    other.tie_rank,
                    other.action_id,
                ) < (action.tie_rank, action.action_id):
                    dominated = True
                    break
            if dominated:
                removed += 1
            else:
                kept.append(action)

    kept.sort(key=lambda row: (row.tie_rank, row.action_id))
    return tuple(kept), removed, len(groups)


class ExactBellmanSolver:
    def __init__(self, problem: BellmanProblem, *, use_quotient: bool) -> None:
        self.problem = problem
        self.use_quotient = use_quotient
        self.memo: dict[Hashable, Objective | None] = {}
        self.active: set[Hashable] = set()
        self.stats = SolverStats()

    def solve_state(self, state: Hashable) -> Objective | None:
        if state in self.memo:
            return self.memo[state]
        if state in self.active:
            raise ValueError(
                "v0.89 certificate problems must be acyclic under progress actions"
            )
        if self.problem.terminal(state):
            answer = Objective(0, 0)
            self.memo[state] = answer
            return answer

        self.active.add(state)
        raw_actions = self.problem.actions(state)
        self.stats.raw_actions_considered += len(raw_actions)
        if self.use_quotient:
            actions, removed, classes = quotient_actions(
                self.problem, raw_actions
            )
            self.stats.dominated_actions_removed += removed
            self.stats.equivalence_classes += classes
        else:
            actions = raw_actions
            self.stats.equivalence_classes += len(raw_actions)
        self.stats.representative_actions_considered += len(actions)

        incumbent: tuple[int, int, int, str] | None = None
        incumbent_objective: Objective | None = None
        for action in actions:
            self.stats.action_expansions += 1
            ordered = _ordered_outcomes(self.problem, action)
            child_rows: list[tuple[Outcome, Objective]] = []
            feasible = True
            for outcome in ordered:
                child = self.solve_state(outcome.successor)
                if child is None:
                    feasible = False
                    break
                child_rows.append((outcome, child))
            if not feasible:
                continue
            expected = sum(
                outcome.weight * outcome.cost + child.expected_cost
                for outcome, child in child_rows
            )
            worst = max(
                outcome.cost + child.worst_cost
                for outcome, child in child_rows
            )
            score = (expected, worst, action.tie_rank, action.action_id)
            if incumbent is None or score < incumbent:
                incumbent = score
                incumbent_objective = Objective(expected, worst)

        self.active.remove(state)
        self.stats.states_solved += 1
        self.memo[state] = incumbent_objective
        return incumbent_objective

    def result(self) -> SolveResult:
        objective = self.solve_state(self.problem.initial_state)
        if objective is None:
            raise RuntimeError(f"{self.problem.name} has no feasible exact plan")
        return SolveResult(objective, self.stats)


class DiagnosisProblem:
    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        self.name = f"adaptive_diagnosis:{seed}"
        self.initial_state = (1 << 8) - 1
        self.mass = tuple(rng.randint(1, 5) for _ in range(8))

        bit_patterns = [
            tuple((hypothesis >> bit) & 1 for hypothesis in range(8))
            for bit in range(3)
        ]
        patterns = [
            bit_patterns[0],
            bit_patterns[1],
            bit_patterns[2],
            bit_patterns[0],
            bit_patterns[1],
            tuple(int(hypothesis in {0, 1, 4, 5}) for hypothesis in range(8)),
            tuple(int(hypothesis in {0, 2, 5, 7}) for hypothesis in range(8)),
            tuple(int(hypothesis in {0, 3, 4, 6}) for hypothesis in range(8)),
        ]
        base_costs = [
            (rng.randint(1, 5), rng.randint(1, 5)) for _ in range(3)
        ]
        self.patterns = tuple(patterns)
        self.costs = (
            base_costs[0],
            base_costs[1],
            base_costs[2],
            (base_costs[0][0] + rng.randint(1, 4), base_costs[0][1] + rng.randint(1, 4)),
            (base_costs[1][0] + rng.randint(1, 4), base_costs[1][1] + rng.randint(1, 4)),
            *((rng.randint(1, 8), rng.randint(1, 8)) for _ in range(3)),
        )

    def state_key(self, state: Hashable) -> str:
        return f"H:{int(state):08x}"

    def terminal(self, state: Hashable) -> bool:
        return int(state).bit_count() <= 1

    def actions(self, state: Hashable) -> tuple[Action, ...]:
        allowed = int(state)
        rows: list[Action] = []
        for query, pattern in enumerate(self.patterns):
            cells = []
            for response in (0, 1):
                child = 0
                for hypothesis, value in enumerate(pattern):
                    if value == response and allowed & (1 << hypothesis):
                        child |= 1 << hypothesis
                if child:
                    cells.append((response, child))
            if len(cells) <= 1:
                continue
            outcomes = tuple(
                Outcome(
                    successor=child,
                    weight=sum(
                        self.mass[hypothesis]
                        for hypothesis in range(8)
                        if child & (1 << hypothesis)
                    ),
                    cost=self.costs[query][response],
                )
                for response, child in cells
            )
            rows.append(Action(f"q{query:02d}", outcomes, query))
        return tuple(rows)


class SetCoverProblem:
    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        self.name = f"weighted_set_cover:{seed}"
        self.initial_state = (1 << 9) - 1
        sets: list[tuple[int, int, str]] = []
        for element in range(9):
            mask = 1 << element
            cost = rng.randint(1, 5)
            sets.append((mask, cost, f"s{element:02d}a"))
            sets.append((mask, cost + rng.randint(1, 3), f"s{element:02d}b"))
        for index in range(6):
            a = rng.randrange(9)
            b = (a + rng.randint(1, 8)) % 9
            mask = (1 << a) | (1 << b)
            cost = rng.randint(2, 7)
            sets.append((mask, cost, f"p{index:02d}a"))
            sets.append((mask, cost + rng.randint(1, 3), f"p{index:02d}b"))
        self.sets = tuple(sets)

    def state_key(self, state: Hashable) -> str:
        return f"U:{int(state):09x}"

    def terminal(self, state: Hashable) -> bool:
        return int(state) == 0

    def actions(self, state: Hashable) -> tuple[Action, ...]:
        uncovered = int(state)
        rows = []
        for rank, (mask, cost, action_id) in enumerate(self.sets):
            successor = uncovered & ~mask
            if successor == uncovered:
                continue
            rows.append(
                Action(
                    action_id,
                    (Outcome(successor, 1, cost),),
                    rank,
                )
            )
        return tuple(rows)


class ShortestPathProblem:
    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        self.name = f"acyclic_shortest_path:{seed}"
        self.layers = 5
        self.width = 3
        self.initial_state = (0, 0)
        self.target = (self.layers, 0)
        edges: dict[tuple[int, int], list[tuple[tuple[int, int], int, str]]] = {}
        for layer in range(self.layers):
            for node in range(self.width):
                source = (layer, node)
                targets = (
                    [self.target]
                    if layer == self.layers - 1
                    else [(layer + 1, child) for child in range(self.width)]
                )
                rows = []
                for child_index, target in enumerate(targets):
                    cost = rng.randint(1, 9)
                    rows.append((target, cost, f"e{layer}{node}{child_index}a"))
                    rows.append((target, cost + rng.randint(1, 4), f"e{layer}{node}{child_index}b"))
                edges[source] = rows
        self.edges = edges

    def state_key(self, state: Hashable) -> str:
        layer, node = state  # type: ignore[misc]
        return f"N:{int(layer):02d}:{int(node):02d}"

    def terminal(self, state: Hashable) -> bool:
        return state == self.target

    def actions(self, state: Hashable) -> tuple[Action, ...]:
        rows = []
        for rank, (target, cost, action_id) in enumerate(self.edges.get(state, [])):
            rows.append(
                Action(
                    action_id,
                    (Outcome(target, 1, cost),),
                    rank,
                )
            )
        return tuple(rows)


def _evaluate_problem(problem: BellmanProblem) -> dict[str, object]:
    raw = ExactBellmanSolver(problem, use_quotient=False).result()
    quotient = ExactBellmanSolver(problem, use_quotient=True).result()
    return {
        "problem": problem.name,
        "raw_objective": [
            raw.objective.expected_cost,
            raw.objective.worst_cost,
        ],
        "quotient_objective": [
            quotient.objective.expected_cost,
            quotient.objective.worst_cost,
        ],
        "matched": raw.objective == quotient.objective,
        "raw_action_expansions": raw.stats.action_expansions,
        "quotient_action_expansions": quotient.stats.action_expansions,
        "action_expansion_reduction": (
            raw.stats.action_expansions - quotient.stats.action_expansions
        ),
        "dominated_actions_removed": quotient.stats.dominated_actions_removed,
        "equivalence_classes": quotient.stats.equivalence_classes,
    }


def evaluate(
    *,
    seed_start: int = SEED_START,
    seeds_per_family: int = SEEDS_PER_FAMILY,
) -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_v89_implementation_or_evaluation":
        raise RuntimeError("v0.89 preregistration status changed")

    factories = {
        "adaptive_diagnosis": DiagnosisProblem,
        "weighted_set_cover": SetCoverProblem,
        "acyclic_shortest_path": ShortestPathProblem,
    }
    family_rows: dict[str, list[dict[str, object]]] = {}
    for family, factory in factories.items():
        family_rows[family] = [
            _evaluate_problem(factory(seed))
            for seed in range(seed_start, seed_start + seeds_per_family)
        ]

    summaries = {}
    total_mismatches = 0
    for family, rows in family_rows.items():
        mismatches = sum(int(not bool(row["matched"])) for row in rows)
        total_mismatches += mismatches
        raw_expansions = sum(int(row["raw_action_expansions"]) for row in rows)
        quotient_expansions = sum(
            int(row["quotient_action_expansions"]) for row in rows
        )
        summaries[family] = {
            "instances": len(rows),
            "objective_mismatches": mismatches,
            "instances_with_nonzero_quotient_reduction": sum(
                int(int(row["action_expansion_reduction"]) > 0) for row in rows
            ),
            "raw_action_expansions": raw_expansions,
            "quotient_action_expansions": quotient_expansions,
            "aggregate_action_expansion_reduction": raw_expansions - quotient_expansions,
            "aggregate_action_expansion_reduction_fraction": (
                (raw_expansions - quotient_expansions) / raw_expansions
                if raw_expansions else 0.0
            ),
            "dominated_actions_removed": sum(
                int(row["dominated_actions_removed"]) for row in rows
            ),
        }

    passed = (
        total_mismatches == 0
        and all(
            int(summary["instances"]) == seeds_per_family
            and int(summary["instances_with_nonzero_quotient_reduction"]) >= 1
            and int(summary["aggregate_action_expansion_reduction"]) > 0
            for summary in summaries.values()
        )
    )
    return {
        "status": "generic_bellman_quotient_transfer_pass" if passed else "generic_bellman_quotient_transfer_rejected",
        "version": "v0.89",
        "seed_start": seed_start,
        "seeds_per_family": seeds_per_family,
        "instance_count": seeds_per_family * len(factories),
        "objective_mismatches": total_mismatches,
        "families": summaries,
        "passed": passed,
        "claim_boundary": {
            "synthetic_transfer_certificate_only": True,
            "external_generalization_claim": False,
            "novel_theorem_claim": False,
            "breakthrough_claim": False,
        },
        "rows": family_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": evidence["status"],
        "passed": evidence["passed"],
        "objective_mismatches": evidence["objective_mismatches"],
        "families": evidence["families"],
    }, indent=2, sort_keys=True))
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
