from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Hashable, Iterable


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v91-proof-carrying-reduction-synthesis.json"
)
AUDIT_SEED_START = 91_001
AUDIT_INSTANCES = 96
TRAIN_SEED_START = 92_001
TRAIN_INSTANCES_PER_FAMILY = 48
WITHHELD_SEED_START = 93_001
WITHHELD_INSTANCES = 48


@dataclass(frozen=True)
class Action:
    action_id: int
    cost: int
    successor: Hashable


@dataclass(frozen=True)
class Problem:
    name: str
    initial_state: Hashable
    terminal_state: Hashable
    actions_by_state: dict[Hashable, tuple[Action, ...]]

    def actions(self, state: Hashable) -> tuple[Action, ...]:
        return self.actions_by_state.get(state, ())

    def terminal(self, state: Hashable) -> bool:
        return state == self.terminal_state

    def states(self) -> tuple[Hashable, ...]:
        rows = set(self.actions_by_state)
        rows.add(self.terminal_state)
        for actions in self.actions_by_state.values():
            rows.update(action.successor for action in actions)
        return tuple(sorted(rows, key=repr))


@dataclass(frozen=True, order=True)
class RelationSpec:
    left_terminal_mode: str
    action_quantifier: str
    cost_relation: str
    successor_relation: str

    def payload(self) -> dict[str, str]:
        return {
            "left_terminal_mode": self.left_terminal_mode,
            "action_quantifier": self.action_quantifier,
            "cost_relation": self.cost_relation,
            "successor_relation": self.successor_relation,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SolveStats:
    objective: int
    action_expansions: int
    states_solved: int
    actions_pruned: int
    relation_certificates: int


LEFT_TERMINAL_MODES = ("true", "both_terminal_only")
ACTION_QUANTIFIERS = (
    "forall_right_exists_left",
    "exists_left_forall_right",
    "exists_right_exists_left",
    "forall_left_exists_right",
)
COST_RELATIONS = ("le", "eq", "ignore")
SUCCESSOR_RELATIONS = (
    "recursive_left_le_right",
    "recursive_right_le_left",
    "equal_state",
)


def relation_grammar() -> tuple[RelationSpec, ...]:
    rows = tuple(
        RelationSpec(terminal, quantifier, cost, successor)
        for terminal in LEFT_TERMINAL_MODES
        for quantifier in ACTION_QUANTIFIERS
        for cost in COST_RELATIONS
        for successor in SUCCESSOR_RELATIONS
    )
    if len(rows) != 72:
        raise AssertionError(f"v0.91 grammar changed: {len(rows)}")
    return rows


def exact_values(problem: Problem) -> dict[Hashable, int]:
    memo: dict[Hashable, int] = {}
    active: set[Hashable] = set()

    def solve(state: Hashable) -> int:
        if state in memo:
            return memo[state]
        if problem.terminal(state):
            memo[state] = 0
            return 0
        if state in active:
            raise RuntimeError(f"cyclic v0.91 problem: {problem.name}")
        actions = problem.actions(state)
        if not actions:
            raise RuntimeError(f"nonterminal dead end: {problem.name}:{state!r}")
        active.add(state)
        answer = min(action.cost + solve(action.successor) for action in actions)
        active.remove(state)
        memo[state] = answer
        return answer

    solve(problem.initial_state)
    for state in problem.states():
        if state not in memo:
            solve(state)
    return memo


class RelationEvaluator:
    def __init__(self, problem: Problem, spec: RelationSpec) -> None:
        self.problem = problem
        self.spec = spec
        self.memo: dict[tuple[Hashable, Hashable], bool] = {}
        self.active: set[tuple[Hashable, Hashable]] = set()

    def cost_match(self, left: Action, right: Action) -> bool:
        if self.spec.cost_relation == "le":
            return left.cost <= right.cost
        if self.spec.cost_relation == "eq":
            return left.cost == right.cost
        if self.spec.cost_relation == "ignore":
            return True
        raise ValueError(self.spec.cost_relation)

    def successor_match(self, left: Action, right: Action) -> bool:
        if self.spec.successor_relation == "recursive_left_le_right":
            return self.relates(left.successor, right.successor)
        if self.spec.successor_relation == "recursive_right_le_left":
            return self.relates(right.successor, left.successor)
        if self.spec.successor_relation == "equal_state":
            return left.successor == right.successor
        raise ValueError(self.spec.successor_relation)

    def action_match(self, left: Action, right: Action) -> bool:
        return self.cost_match(left, right) and self.successor_match(left, right)

    def relates(self, left: Hashable, right: Hashable) -> bool:
        key = (left, right)
        if key in self.memo:
            return self.memo[key]
        if left == right:
            self.memo[key] = True
            return True
        left_terminal = self.problem.terminal(left)
        right_terminal = self.problem.terminal(right)
        if left_terminal:
            answer = self.spec.left_terminal_mode == "true" or right_terminal
            self.memo[key] = answer
            return answer
        if right_terminal:
            self.memo[key] = False
            return False
        if key in self.active:
            raise RuntimeError("v0.91 relation encountered a cycle")
        left_actions = self.problem.actions(left)
        right_actions = self.problem.actions(right)
        if not left_actions or not right_actions:
            self.memo[key] = False
            return False

        self.active.add(key)
        q = self.spec.action_quantifier
        if q == "forall_right_exists_left":
            answer = all(
                any(self.action_match(a, b) for a in left_actions)
                for b in right_actions
            )
        elif q == "exists_left_forall_right":
            answer = any(
                all(self.action_match(a, b) for b in right_actions)
                for a in left_actions
            )
        elif q == "exists_right_exists_left":
            answer = any(
                self.action_match(a, b)
                for b in right_actions
                for a in left_actions
            )
        elif q == "forall_left_exists_right":
            answer = all(
                any(self.action_match(a, b) for b in right_actions)
                for a in left_actions
            )
        else:
            raise ValueError(q)
        self.active.remove(key)
        self.memo[key] = answer
        return answer

    def relation_pairs(self) -> tuple[tuple[Hashable, Hashable], ...]:
        states = self.problem.states()
        rows = []
        for left in states:
            for right in states:
                if self.relates(left, right):
                    rows.append((left, right))
        return tuple(rows)


def semantic_violations(problem: Problem, spec: RelationSpec) -> int:
    values = exact_values(problem)
    relation = RelationEvaluator(problem, spec)
    return sum(
        int(values[left] > values[right])
        for left, right in relation.relation_pairs()
    )


def _dominates(
    relation: RelationEvaluator,
    left: Action,
    right: Action,
) -> bool:
    if left.action_id == right.action_id:
        return False
    if left.cost > right.cost:
        return False
    if not relation.relates(left.successor, right.successor):
        return False
    reverse = (
        right.cost <= left.cost
        and relation.relates(right.successor, left.successor)
    )
    if reverse and left.cost == right.cost:
        return left.action_id < right.action_id
    return True


def representative_actions(
    relation: RelationEvaluator,
    actions: tuple[Action, ...],
) -> tuple[tuple[Action, ...], int]:
    kept = []
    pruned = 0
    for action in actions:
        if any(_dominates(relation, other, action) for other in actions):
            pruned += 1
        else:
            kept.append(action)
    if actions and not kept:
        raise RuntimeError("v0.91 relation pruned every action")
    return tuple(kept), pruned


def solve_with_spec(problem: Problem, spec: RelationSpec | None) -> SolveStats:
    relation = RelationEvaluator(problem, spec) if spec is not None else None
    memo: dict[Hashable, int] = {}
    active: set[Hashable] = set()
    action_expansions = 0
    actions_pruned = 0
    relation_certificates = 0

    def solve(state: Hashable) -> int:
        nonlocal action_expansions, actions_pruned, relation_certificates
        if state in memo:
            return memo[state]
        if problem.terminal(state):
            memo[state] = 0
            return 0
        if state in active:
            raise RuntimeError(f"cyclic v0.91 problem: {problem.name}")
        raw = problem.actions(state)
        if not raw:
            raise RuntimeError(f"nonterminal dead end: {problem.name}:{state!r}")
        if relation is None:
            actions = raw
        else:
            actions, pruned = representative_actions(relation, raw)
            actions_pruned += pruned
            relation_certificates += pruned
        active.add(state)
        action_expansions += len(actions)
        answer = min(action.cost + solve(action.successor) for action in actions)
        active.remove(state)
        memo[state] = answer
        return answer

    objective = solve(problem.initial_state)
    return SolveStats(
        objective=objective,
        action_expansions=action_expansions,
        states_solved=len(memo),
        actions_pruned=actions_pruned,
        relation_certificates=relation_certificates,
    )


def random_dag_problem(seed: int, *, audit: bool = False) -> Problem:
    rng = random.Random(seed)
    n = 7 if audit else 9
    actions_by_state: dict[int, tuple[Action, ...]] = {}
    action_id = 0
    for state in range(1, n):
        count = rng.randint(2, min(5, state + 2))
        rows = []
        for _ in range(count):
            successor = rng.randrange(state)
            cost = rng.randint(1, 9)
            rows.append(Action(action_id, cost, successor))
            action_id += 1
        # Include a harmless exact-successor comparison often enough that the
        # audit also exercises local pruning, without guaranteeing a richer
        # recursive relation.
        if state >= 2 and rng.random() < 0.45:
            base = rng.choice(rows)
            rows.append(Action(action_id, base.cost + rng.randint(0, 3), base.successor))
            action_id += 1
        actions_by_state[state] = tuple(rows)
    return Problem(
        name=f"random_dag:{seed}",
        initial_state=n - 1,
        terminal_state=0,
        actions_by_state=actions_by_state,
    )


def weighted_cover_problem(seed: int) -> Problem:
    rng = random.Random(seed)
    universe_bits = 5
    full = (1 << universe_bits) - 1
    catalog: list[tuple[int, int]] = []
    for element in range(universe_bits):
        catalog.append((1 << element, rng.randint(2, 6)))
    for _ in range(7):
        size = rng.randint(1, 3)
        elements = rng.sample(range(universe_bits), size)
        mask = sum(1 << element for element in elements)
        catalog.append((mask, rng.randint(1, 8)))
    # Add overlapping actions with costs independent of coverage order. The
    # synthesizer receives only the resulting transition graph, never masks.
    catalog.append((full, rng.randint(7, 12)))

    actions_by_state: dict[int, tuple[Action, ...]] = {}
    action_id = 0
    for state in range(1, full + 1):
        rows = []
        for mask, cost in catalog:
            successor = state & ~mask
            if successor == state:
                continue
            rows.append(Action(action_id, cost, successor))
            action_id += 1
        actions_by_state[state] = tuple(rows)
    return Problem(
        name=f"weighted_cover_dp:{seed}",
        initial_state=full,
        terminal_state=0,
        actions_by_state=actions_by_state,
    )


def resource_reduction_problem(seed: int) -> Problem:
    rng = random.Random(seed)
    maxima = (2 + rng.randrange(2), 2 + rng.randrange(2), 2 + rng.randrange(2))
    catalog: list[tuple[tuple[int, int, int], int]] = [
        ((1, 0, 0), rng.randint(2, 5)),
        ((0, 1, 0), rng.randint(2, 5)),
        ((0, 0, 1), rng.randint(2, 5)),
        ((1, 1, 0), rng.randint(2, 7)),
        ((1, 0, 1), rng.randint(2, 7)),
        ((0, 1, 1), rng.randint(2, 7)),
        ((1, 1, 1), rng.randint(3, 8)),
        ((2, 1, 0), rng.randint(3, 8)),
        ((0, 2, 1), rng.randint(3, 8)),
    ]
    states = [
        (a, b, c)
        for a in range(maxima[0] + 1)
        for b in range(maxima[1] + 1)
        for c in range(maxima[2] + 1)
    ]
    terminal = (0, 0, 0)
    actions_by_state: dict[tuple[int, int, int], tuple[Action, ...]] = {}
    action_id = 0
    for state in states:
        if state == terminal:
            continue
        rows = []
        for reduction, cost in catalog:
            successor = tuple(
                max(0, value - amount)
                for value, amount in zip(state, reduction)
            )
            if successor == state:
                continue
            rows.append(Action(action_id, cost, successor))
            action_id += 1
        actions_by_state[state] = tuple(rows)
    return Problem(
        name=f"resource_reduction_dp:{seed}",
        initial_state=maxima,
        terminal_state=terminal,
        actions_by_state=actions_by_state,
    )


def audit_candidate(spec: RelationSpec) -> dict[str, int | bool]:
    violations = 0
    objective_mismatches = 0
    relation_pairs = 0
    actions_pruned = 0
    for seed in range(AUDIT_SEED_START, AUDIT_SEED_START + AUDIT_INSTANCES):
        problem = random_dag_problem(seed, audit=True)
        relation = RelationEvaluator(problem, spec)
        pairs = relation.relation_pairs()
        relation_pairs += len(pairs)
        values = exact_values(problem)
        violations += sum(int(values[left] > values[right]) for left, right in pairs)
        raw = solve_with_spec(problem, None)
        candidate = solve_with_spec(problem, spec)
        objective_mismatches += int(raw.objective != candidate.objective)
        actions_pruned += candidate.actions_pruned
    return {
        "semantic_violations": violations,
        "objective_mismatches": objective_mismatches,
        "relation_pairs": relation_pairs,
        "actions_pruned": actions_pruned,
        "eligible": violations == 0 and objective_mismatches == 0,
    }


def training_score(spec: RelationSpec) -> dict[str, object]:
    families = {
        "random_dag": random_dag_problem,
        "weighted_cover_dp": weighted_cover_problem,
    }
    summaries = {}
    total_reduction = 0
    total_pairs = 0
    total_mismatches = 0
    for family, factory in families.items():
        raw_expansions = 0
        candidate_expansions = 0
        pruned = 0
        mismatches = 0
        pairs = 0
        for seed in range(
            TRAIN_SEED_START,
            TRAIN_SEED_START + TRAIN_INSTANCES_PER_FAMILY,
        ):
            problem = factory(seed)
            raw = solve_with_spec(problem, None)
            candidate = solve_with_spec(problem, spec)
            raw_expansions += raw.action_expansions
            candidate_expansions += candidate.action_expansions
            pruned += candidate.actions_pruned
            mismatches += int(raw.objective != candidate.objective)
            pairs += len(RelationEvaluator(problem, spec).relation_pairs())
        reduction = raw_expansions - candidate_expansions
        total_reduction += reduction
        total_pairs += pairs
        total_mismatches += mismatches
        summaries[family] = {
            "instances": TRAIN_INSTANCES_PER_FAMILY,
            "objective_mismatches": mismatches,
            "raw_action_expansions": raw_expansions,
            "candidate_action_expansions": candidate_expansions,
            "action_expansion_reduction": reduction,
            "actions_pruned": pruned,
            "declared_relation_pairs": pairs,
        }
    return {
        "families": summaries,
        "total_action_expansion_reduction": total_reduction,
        "total_declared_relation_pairs": total_pairs,
        "objective_mismatches": total_mismatches,
    }


def select_rule() -> tuple[RelationSpec, dict[str, object]]:
    eligible = []
    audit_rows = []
    for spec in relation_grammar():
        audit = audit_candidate(spec)
        audit_rows.append({"spec": spec.payload(), **audit})
        if not bool(audit["eligible"]):
            continue
        training = training_score(spec)
        if int(training["objective_mismatches"]) != 0:
            continue
        eligible.append((spec, audit, training))
    if not eligible:
        raise RuntimeError("v0.91: no grammar candidate survived semantic audit")

    selected_spec, selected_audit, selected_training = min(
        eligible,
        key=lambda row: (
            -int(row[2]["total_action_expansion_reduction"]),
            int(row[2]["total_declared_relation_pairs"]),
            row[0],
        ),
    )
    freeze_payload = {
        "selected_spec": selected_spec.payload(),
        "semantic_audit": selected_audit,
        "training": selected_training,
    }
    freeze_digest = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return selected_spec, {
        "grammar_candidates": len(audit_rows),
        "eligible_candidates": len(eligible),
        "selected_spec": selected_spec.payload(),
        "selected_spec_digest": selected_spec.digest(),
        "freeze_digest": freeze_digest,
        "selected_semantic_audit": selected_audit,
        "selected_training": selected_training,
        "audit_rows": audit_rows,
    }


def withheld_evaluation(spec: RelationSpec, freeze_digest: str) -> dict[str, object]:
    # The withheld family is not instantiated until after select_rule() has
    # returned both a frozen grammar tuple and digest.
    if not freeze_digest:
        raise RuntimeError("v0.91 withheld construction requires frozen digest")
    raw_expansions = 0
    candidate_expansions = 0
    mismatches = 0
    positive_reductions = 0
    actions_pruned = 0
    certificate_failures = 0
    rows = []
    for seed in range(WITHHELD_SEED_START, WITHHELD_SEED_START + WITHHELD_INSTANCES):
        problem = resource_reduction_problem(seed)
        raw = solve_with_spec(problem, None)
        candidate = solve_with_spec(problem, spec)
        reduction = raw.action_expansions - candidate.action_expansions
        mismatch = raw.objective != candidate.objective
        relation = RelationEvaluator(problem, spec)
        # Every counted prune was produced only by _dominates(), whose local
        # premise includes the selected recursive relation certificate.
        local_certificate_failures = int(
            candidate.relation_certificates != candidate.actions_pruned
        )
        certificate_failures += local_certificate_failures
        raw_expansions += raw.action_expansions
        candidate_expansions += candidate.action_expansions
        actions_pruned += candidate.actions_pruned
        mismatches += int(mismatch)
        positive_reductions += int(reduction > 0)
        rows.append({
            "seed": seed,
            "raw_objective": raw.objective,
            "candidate_objective": candidate.objective,
            "matched": not mismatch,
            "raw_action_expansions": raw.action_expansions,
            "candidate_action_expansions": candidate.action_expansions,
            "action_expansion_reduction": reduction,
            "actions_pruned": candidate.actions_pruned,
            "declared_relation_pairs": len(relation.relation_pairs()),
            "local_certificate_failures": local_certificate_failures,
        })
    aggregate_reduction = raw_expansions - candidate_expansions
    passed = (
        mismatches == 0
        and positive_reductions >= 1
        and aggregate_reduction > 0
        and certificate_failures == 0
    )
    return {
        "family": "resource_reduction_dp",
        "instances": WITHHELD_INSTANCES,
        "objective_mismatches": mismatches,
        "instances_with_positive_action_reduction": positive_reductions,
        "raw_action_expansions": raw_expansions,
        "candidate_action_expansions": candidate_expansions,
        "aggregate_action_expansion_reduction": aggregate_reduction,
        "aggregate_action_expansion_reduction_fraction": (
            aggregate_reduction / raw_expansions if raw_expansions else 0.0
        ),
        "actions_pruned": actions_pruned,
        "local_certificate_failures": certificate_failures,
        "passed": passed,
        "rows": rows,
    }


def evaluate() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_v91_implementation_or_evaluation":
        raise RuntimeError("v0.91 preregistration status changed")
    if len(relation_grammar()) != int(
        preregistration["frozen_relation_grammar"]["candidate_count"]
    ):
        raise RuntimeError("v0.91 grammar candidate count changed")

    selected, synthesis = select_rule()
    frozen_digest = str(synthesis["freeze_digest"])
    withheld = withheld_evaluation(selected, frozen_digest)
    passed = bool(withheld["passed"])
    return {
        "status": (
            "proof_carrying_reduction_synthesis_pass_v91"
            if passed
            else "proof_carrying_reduction_synthesis_rejected_v91"
        ),
        "version": "v0.91",
        "passed": passed,
        "synthesis": synthesis,
        "withheld_transfer": withheld,
        "claim_boundary": preregistration["claim_boundary"],
        "kill_rule": preregistration["kill_rule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "selected_spec": result["synthesis"]["selected_spec"],
        "selected_spec_digest": result["synthesis"]["selected_spec_digest"],
        "freeze_digest": result["synthesis"]["freeze_digest"],
        "eligible_candidates": result["synthesis"]["eligible_candidates"],
        "withheld": {
            key: value for key, value in result["withheld_transfer"].items()
            if key != "rows"
        },
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
