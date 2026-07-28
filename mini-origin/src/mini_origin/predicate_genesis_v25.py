from __future__ import annotations

from dataclasses import dataclass
import argparse, hashlib, itertools, json, math, random
from pathlib import Path

import numpy as np

from .intervention_genesis_v22 import (
    PositionRule, information_lower_bound, intervention_response,
    observational_equivalence_certificate, optimal_worst_case_queries,
    synthesize_position_rule,
)
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .outcome_semantics_v24 import (
    ResponseExample, canonical_semantic_universe, generate_balanced_examples,
)


@dataclass(frozen=True)
class Predicate:
    kind: str
    channel: str
    threshold: float = 0.0
    complexity: int = 1

    def text(self) -> str:
        if self.kind == "present":
            return f"present({self.channel})"
        if self.kind == "greater":
            return f"value({self.channel})>{self.threshold:.8f}"
        a, b = self.channel.split("-")
        return f"value({a})-value({b})>{self.threshold:.8f}"

    def evaluate(self, left: float | None, right: float | None) -> bool:
        if self.kind == "present":
            return (left if self.channel == "left" else right) is not None
        if self.kind == "greater":
            value = left if self.channel == "left" else right
            return value is not None and value > self.threshold
        if left is None or right is None:
            return False
        value = right - left if self.channel == "right-left" else left - right
        return value > self.threshold


@dataclass(frozen=True)
class Program:
    predicates: tuple[Predicate, ...]
    mapping: tuple[tuple[int, int], ...]
    default: int
    train_macro: float
    dev_macro: float
    dev_min: float
    complexity: int

    def text(self) -> str:
        ps = ";".join(p.text() for p in self.predicates)
        table = ",".join(f"{c}:{ACTION_NAMES[a]}" for c, a in self.mapping)
        return f"{ps};default={ACTION_NAMES[self.default]};map={table}"


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_accuracy: float
    minimum_state_accuracy: float
    unseen_code_rate: float
    per_state_accuracy: dict[str, float]


@dataclass(frozen=True)
class LoopEvaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_transition_rate: float
    mean_remaining_candidates: float


def arrays(examples: list[ResponseExample]) -> dict[str, object]:
    names = tuple(sorted({e.state for e in examples}))
    ids = {name: i for i, name in enumerate(names)}
    return {
        "left": np.array([0.0 if e.left is None else e.left for e in examples]),
        "right": np.array([0.0 if e.right is None else e.right for e in examples]),
        "lp": np.array([e.left is not None for e in examples]),
        "rp": np.array([e.right is not None for e in examples]),
        "action": np.array([e.action for e in examples], dtype=np.int8),
        "state": np.array([ids[e.state] for e in examples], dtype=np.int8),
        "names": names,
    }


def predicate_bits(predicate: Predicate, data: dict[str, object]) -> np.ndarray:
    left, right = data["left"], data["right"]
    lp, rp = data["lp"], data["rp"]
    if predicate.kind == "present":
        return lp if predicate.channel == "left" else rp
    if predicate.kind == "greater":
        values, present = (left, lp) if predicate.channel == "left" else (right, rp)
        return present & (values > predicate.threshold)
    difference = right - left if predicate.channel == "right-left" else left - right
    return lp & rp & (difference > predicate.threshold)


def threshold_candidates(examples: list[ResponseExample]) -> list[float]:
    values = [v for e in examples for v in (e.left, e.right) if v is not None]
    result = {float(v) for v in np.quantile(values, np.linspace(0.04, 0.96, 17))}
    result.update((0.0, 0.10, 0.15, 0.20, 0.25, 0.30))
    return sorted(result)


def predicate_grammar(
    training: list[ResponseExample], development: list[ResponseExample]
) -> list[Predicate]:
    raw = [Predicate("present", "left"), Predicate("present", "right")]
    for t in threshold_candidates(training):
        raw.extend((
            Predicate("greater", "left", t, 2),
            Predicate("greater", "right", t, 2),
            Predicate("difference", "right-left", t, 3),
            Predicate("difference", "left-right", t, 3),
        ))
    train, dev = arrays(training), arrays(development)
    unique: dict[bytes, Predicate] = {}
    for p in raw:
        signature = np.concatenate((predicate_bits(p, train), predicate_bits(p, dev)))
        key = np.packbits(signature).tobytes()
        old = unique.get(key)
        if old is None or (p.complexity, p.text()) < (old.complexity, old.text()):
            unique[key] = p
    return sorted(unique.values(), key=lambda p: (p.complexity, p.text()))


def encode(predicates: tuple[Predicate, ...], left: float | None, right: float | None) -> int:
    return sum(int(p.evaluate(left, right)) << i for i, p in enumerate(predicates))


def codes(bits: list[np.ndarray], indices: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(len(bits[0]), dtype=np.int16)
    for i, index in enumerate(indices):
        out |= bits[index].astype(np.int16) << i
    return out


def fit_table(code: np.ndarray, action: np.ndarray, count: int) -> tuple[tuple[tuple[int, int], ...], int]:
    table = []
    for value in range(1 << count):
        mask = code == value
        if np.any(mask):
            table.append((value, int(np.argmax(np.bincount(action[mask], minlength=3)))))
    return tuple(table), int(np.argmax(np.bincount(action, minlength=3)))


def metrics(
    code: np.ndarray, data: dict[str, object], table: tuple[tuple[int, int], ...], default: int
) -> Evaluation:
    maximum = max(int(np.max(code)), max((c for c, _ in table), default=0))
    lookup = np.full(maximum + 1, default, dtype=np.int8)
    observed = np.zeros(maximum + 1, dtype=np.bool_)
    for value, action in table:
        lookup[value], observed[value] = action, True
    correct = lookup[code] == data["action"]
    per_state = {}
    for i, name in enumerate(data["names"]):
        per_state[name] = float(np.mean(correct[data["state"] == i]))
    return Evaluation(
        float(np.mean(correct)), float(np.mean(list(per_state.values()))),
        min(per_state.values()), float(np.mean(~observed[code])), per_state,
    )


def synthesize(
    training: list[ResponseExample], development: list[ResponseExample]
) -> tuple[Program, Program, dict[str, object], list[Predicate]]:
    grammar = predicate_grammar(training, development)
    train, dev = arrays(training), arrays(development)
    train_bits = [predicate_bits(p, train) for p in grammar]
    dev_bits = [predicate_bits(p, dev) for p in grammar]
    rows: list[tuple[Program, Evaluation]] = []
    best_by_count: dict[int, tuple[Program, Evaluation]] = {}
    for count in (1, 2):
        for indices in itertools.combinations(range(len(grammar)), count):
            tc, dc = codes(train_bits, indices), codes(dev_bits, indices)
            table, default = fit_table(tc, train["action"], count)
            tm, dm = metrics(tc, train, table, default), metrics(dc, dev, table, default)
            predicates = tuple(grammar[i] for i in indices)
            program = Program(predicates, table, default, tm.macro_accuracy,
                              dm.macro_accuracy, dm.minimum_state_accuracy,
                              sum(p.complexity for p in predicates))
            rows.append((program, dm))
            score = (dm.minimum_state_accuracy, dm.macro_accuracy, tm.macro_accuracy,
                     -program.complexity, tuple(p.text() for p in predicates))
            old = best_by_count.get(count)
            if old is None:
                best_by_count[count] = (program, dm)
            else:
                op, om = old
                old_score = (om.minimum_state_accuracy, om.macro_accuracy, op.train_macro,
                             -op.complexity, tuple(p.text() for p in op.predicates))
                if score > old_score:
                    best_by_count[count] = (program, dm)
    selected, selected_metrics = max(
        rows, key=lambda item: (item[1].minimum_state_accuracy, item[1].macro_accuracy,
                                item[0].train_macro, -len(item[0].predicates),
                                -item[0].complexity,
                                tuple(p.text() for p in item[0].predicates)))
    reduced, reduced_metrics = best_by_count[1]
    evidence = {
        "grammar_size": len(grammar),
        "subsets_evaluated": len(grammar) + math.comb(len(grammar), 2),
        "selected_predicates": [p.text() for p in selected.predicates],
        "selected_development": selected_metrics.__dict__,
        "reduced_predicates": [p.text() for p in reduced.predicates],
        "reduced_development": reduced_metrics.__dict__,
    }
    return selected, reduced, evidence, grammar


def evaluate_examples(program: Program, examples: list[ResponseExample]) -> Evaluation:
    data = arrays(examples)
    code = np.array([encode(program.predicates, e.left, e.right) for e in examples])
    return metrics(code, data, program.mapping, program.default)


def prototypes() -> dict[str, tuple[float | None, float | None, int]]:
    result = {
        "left_boundary_equal": (None, 1.0, ACCEPT),
        "left_boundary_right": (None, 0.0, RIGHT),
        "interior_left": (0.0, 1.0, LEFT),
        "interior_equal": (1.0, 1.0, ACCEPT),
        "interior_right": (1.0, 0.0, RIGHT),
    }
    assert {s.name: s.action for s in canonical_semantic_universe()} == {
        name: row[2] for name, row in result.items()
    }
    return result


def conflict(predicates: tuple[Predicate, ...]) -> dict[str, object] | None:
    seen: dict[int, tuple[str, int]] = {}
    for name, (left, right, action) in prototypes().items():
        code = encode(predicates, left, right)
        if code in seen and seen[code][1] != action:
            previous = seen[code]
            return {"code": code, "first_state": previous[0],
                    "first_action": ACTION_NAMES[previous[1]], "second_state": name,
                    "second_action": ACTION_NAMES[action]}
        seen[code] = (name, action)
    return None


def certificate(grammar: list[Predicate], selected: Program) -> dict[str, object]:
    single_witnesses = []
    consistent_pairs = []
    for p in grammar:
        single_witnesses.append({"predicate": p.text(), "conflict": conflict((p,))})
    for pair in itertools.combinations(grammar, 2):
        if conflict(pair) is None:
            consistent_pairs.append([p.text() for p in pair])
    return {
        "minimum_predicate_count": 2 if consistent_pairs else None,
        "selected_is_consistent": conflict(selected.predicates) is None,
        "all_single_predicates_refuted": all(row["conflict"] is not None for row in single_witnesses),
        "consistent_pair_count": len(consistent_pairs),
        "selected_predicates": [p.text() for p in selected.predicates],
        "single_collision_witnesses": single_witnesses,
    }


def decode(program: Program, left: float | None, right: float | None) -> int:
    return dict(program.mapping).get(encode(program.predicates, left, right), program.default)


def run_trial(seed: int, dimension: int, root: int, rho: float, rule: PositionRule,
              program: Program, replicates: int) -> tuple[bool, int, bool, int]:
    low, high, queries = 0, dimension - 1, 0
    budget = information_lower_bound(dimension)
    while low < high and queries < budget:
        query = low + rule.offset(high - low + 1)
        left, right = intervention_response(seed + queries * 7919, dimension, root,
                                            rho, query, replicates)
        action, queries = decode(program, left, right), queries + 1
        if action == ACCEPT:
            return query == root, queries, False, 1
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        if low > high:
            return False, queries, True, 0
    remaining = max(1, high - low + 1)
    return low + (remaining - 1) // 2 == root, queries, False, remaining


def evaluate_loop(seed: int, rule: PositionRule, program: Program,
                  dimensions: tuple[int, ...], trials: int, replicates: int) -> LoopEvaluation:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(trials):
        dimension = int(rng.choice(dimensions))
        rows.append(run_trial(seed + i * 104729, dimension,
                              int(rng.integers(0, dimension)),
                              float(rng.uniform(0.28, 0.94)), rule, program, replicates))
    return LoopEvaluation(float(np.mean([r[0] for r in rows])),
                          float(np.mean([r[1] for r in rows])), max(r[1] for r in rows),
                          float(np.mean([r[2] for r in rows])),
                          float(np.mean([r[3] for r in rows])))


def hand_program(threshold: float = 0.15) -> Program:
    ps = (Predicate("present", "left"), Predicate("greater", "left", threshold, 2),
          Predicate("greater", "right", threshold, 2))
    table = {encode(ps, left, right): action for left, right, action in prototypes().values()}
    return Program(ps, tuple(sorted(table.items())), ACCEPT, 1.0, 1.0, 1.0, 5)


def random_control(seed: int, rule: PositionRule, template: Program,
                   dimensions: tuple[int, ...]) -> dict[str, float]:
    rng, scores = random.Random(seed), []
    keys = [code for code, _ in template.mapping]
    for i in range(48):
        program = Program(template.predicates,
                          tuple(sorted((code, rng.choice((LEFT, ACCEPT, RIGHT))) for code in keys)),
                          rng.choice((LEFT, ACCEPT, RIGHT)), 0.0, 0.0, 0.0, template.complexity)
        scores.append(evaluate_loop(seed + i * 1000003, rule, program,
                                    dimensions, 420, 512).accuracy)
    return {"trials": 48, "median_accuracy": float(np.median(scores)), "maximum_accuracy": max(scores)}


def digest(rule: PositionRule, program: Program) -> str:
    return hashlib.sha256(f"{rule.name}:{program.text()}".encode()).hexdigest()


def run(seed: int = 1001) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    rule, query_evidence = synthesize_position_rule()
    training = generate_balanced_examples(seed * 10000 + 79, 700, (3,4,5,6,7,8), 384, 0.30, 0.90)
    development = generate_balanced_examples(seed * 10000 + 1000081, 350, (4,6,9,11), 384, 0.28, 0.92)
    selected, reduced, synthesis, grammar = synthesize(training, development)
    proof = certificate(grammar, selected)
    frozen_digest = digest(rule, selected)

    # Hidden evidence appears only after predicates, table, reduced control,
    # grammar-relative proof and digest are frozen.
    hidden_dimensions = (9, 13, 21, 37, 63)
    hidden = generate_balanced_examples(seed * 10000 + 10000001, 1000,
                                        hidden_dimensions, 512, 0.26, 0.95)
    candidate_semantic, reduced_semantic = evaluate_examples(selected, hidden), evaluate_examples(reduced, hidden)
    candidate_loop = evaluate_loop(seed * 10000 + 10000003, rule, selected, hidden_dimensions, 3000, 512)
    reduced_loop = evaluate_loop(seed * 10000 + 10000005, rule, reduced, hidden_dimensions, 3000, 512)
    human_loop = evaluate_loop(seed * 10000 + 10000007, rule, hand_program(), hidden_dimensions, 3000, 512)
    random_book = random_control(seed * 10000 + 10000009, rule, selected, hidden_dimensions)
    semantic_gap = candidate_semantic.macro_accuracy - reduced_semantic.macro_accuracy
    loop_gap = candidate_loop.accuracy - reduced_loop.accuracy
    human_gap = candidate_loop.accuracy - human_loop.accuracy
    random_gap = candidate_loop.accuracy - random_book["median_accuracy"]
    predicate_texts = [p.text() for p in selected.predicates]
    gate = (
        equivalence["exact_within_tolerance"] and rule.name == "lower_midpoint"
        and query_evidence["all_training_depths_meet_lower_bound"]
        and len(selected.predicates) == 2
        and any(p.kind == "greater" and p.channel == "right" for p in selected.predicates)
        and any(p.kind == "difference" and p.channel == "right-left" for p in selected.predicates)
        and proof["minimum_predicate_count"] == 2 and proof["selected_is_consistent"]
        and proof["all_single_predicates_refuted"]
        and selected.dev_macro >= 0.985 and selected.dev_min >= 0.97
        and candidate_semantic.macro_accuracy >= 0.985
        and candidate_semantic.minimum_state_accuracy >= 0.97
        and candidate_semantic.unseen_code_rate <= 0.01 and semantic_gap >= 0.15
        and candidate_loop.accuracy >= 0.985 and candidate_loop.invalid_transition_rate <= 0.01
        and loop_gap >= 0.02 and human_gap >= -0.01 and random_gap >= 0.45
        and all(optimal_worst_case_queries(d) == information_lower_bound(d) for d in hidden_dimensions)
    )
    return {
        "status": "raw_predicate_genesis_candidate" if gate else "not_yet",
        "claim_scope": "an enumerative MDL search receives raw optional response values and action labels, synthesizes a two-predicate relational alphabet and lookup controller, freezes them, and transfers to unseen causal-chain sizes; the discovered right-activity and right-minus-left predicates compress the three named v0.24 bits, but remain grammar-bounded classical controller synthesis rather than a world breakthrough",
        "seed": seed, "candidate_gate": gate, "observational_equivalence": equivalence,
        "query_synthesis": query_evidence, "predicate_synthesis": synthesis,
        "grammar_relative_certificate": proof,
        "selected_program": {"predicates": predicate_texts,
                             "mapping": [{"code": c, "action": ACTION_NAMES[a]} for c,a in selected.mapping],
                             "default_action": ACTION_NAMES[selected.default], "complexity": selected.complexity,
                             "training_macro_accuracy": selected.train_macro,
                             "development_macro_accuracy": selected.dev_macro,
                             "development_minimum_state_accuracy": selected.dev_min},
        "reduced_program": {"predicates": [p.text() for p in reduced.predicates],
                            "mapping": [{"code": c, "action": ACTION_NAMES[a]} for c,a in reduced.mapping],
                            "default_action": ACTION_NAMES[reduced.default], "complexity": reduced.complexity,
                            "development_macro_accuracy": reduced.dev_macro,
                            "development_minimum_state_accuracy": reduced.dev_min},
        "frozen_program_digest": frozen_digest, "hidden_dimensions": list(hidden_dimensions),
        "candidate_semantic": candidate_semantic.__dict__, "best_reduced_semantic": reduced_semantic.__dict__,
        "candidate_closed_loop": candidate_loop.__dict__, "best_reduced_closed_loop": reduced_loop.__dict__,
        "human_closed_loop": human_loop.__dict__, "random_codebook_control": random_book,
        "semantic_gap": semantic_gap, "closed_loop_gap": loop_gap,
        "human_gap": human_gap, "random_gap": random_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "predicates": report["selected_program"]["predicates"],
                      "semantic_accuracy": report["candidate_semantic"]["macro_accuracy"],
                      "minimum_state_accuracy": report["candidate_semantic"]["minimum_state_accuracy"],
                      "closed_loop_accuracy": report["candidate_closed_loop"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
