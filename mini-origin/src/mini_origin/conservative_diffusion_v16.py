from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import rotating_sketch_v12 as v12
from . import adversarial_flat_v15 as v15


WRITE_BUDGET = 0.20


@dataclass(frozen=True)
class DiffusionProgram:
    injection_count: int
    injection_mode: str
    mixing_mode: str
    mixing_alpha: float
    load_weight: float
    leverage_weight: float
    random_weight: float
    decode_ridge: float
    seed_salt: int

    def text(self) -> str:
        return (
            f"inject={self.injection_count}:{self.injection_mode};"
            f"mix={self.mixing_mode}:{self.mixing_alpha:.3f};"
            f"load={self.load_weight:.3f};leverage={self.leverage_weight:.3f};"
            f"random={self.random_weight:.3f};ridge={self.decode_ridge:.1e};"
            f"salt={self.seed_salt}"
        )


@dataclass
class DiffusedStatistics:
    gram: np.ndarray
    response: np.ndarray
    writes: np.ndarray


def operation_budget(cells: int) -> int:
    return max(3, int(math.floor(WRITE_BUDGET * cells)))


def _rng(
    program: DiffusionProgram,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
    stream: int,
) -> np.random.Generator:
    return np.random.default_rng(
        scenario.seed
        + program.seed_salt * 1_000_003
        + context * 10_007
        + occurrence * 101
        + stream * 1_009
    )


def _leverage(statistics: DiffusedStatistics, context: int) -> np.ndarray:
    return np.trace(statistics.gram[:, context], axis1=1, axis2=2)


def select_injection_cells(
    program: DiffusionProgram,
    statistics: DiffusedStatistics,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
) -> np.ndarray:
    count = min(program.injection_count, operation_budget(scenario.cells))
    rng = _rng(program, scenario, context, occurrence, 1)
    if program.injection_mode == "random":
        return np.sort(rng.choice(scenario.cells, size=count, replace=False))

    loads = np.sum(statistics.writes, axis=1).astype(np.float64)
    leverage = _leverage(statistics, context)
    random_signal = rng.normal(size=scenario.cells)
    score = (
        -program.load_weight * loads
        -program.leverage_weight * leverage
        + program.random_weight * random_signal
    )
    if program.injection_mode == "underloaded":
        score -= 4.0 * (loads - np.min(loads))
    elif program.injection_mode != "flat":
        raise ValueError(program.injection_mode)
    return np.sort(np.argsort(score)[::-1][:count])


def _disjoint_pairs(order: np.ndarray, pair_count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for index in range(0, len(order) - 1, 2):
        left = int(order[index])
        right = int(order[index + 1])
        if left == right or left in used or right in used:
            continue
        pairs.append((left, right))
        used.add(left)
        used.add(right)
        if len(pairs) >= pair_count:
            break
    return pairs


def select_mixing_pairs(
    program: DiffusionProgram,
    statistics: DiffusedStatistics,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
) -> list[tuple[int, int]]:
    budget = operation_budget(scenario.cells)
    pair_count = max(0, (budget - program.injection_count) // 2)
    if pair_count == 0:
        return []
    rng = _rng(program, scenario, context, occurrence, 2)

    if program.mixing_mode == "random":
        return _disjoint_pairs(rng.permutation(scenario.cells), pair_count)

    if program.mixing_mode == "ring":
        start = int((context * 7 + occurrence * 3 + program.seed_salt) % scenario.cells)
        order = np.asarray(
            [(start + index) % scenario.cells for index in range(scenario.cells)],
            dtype=int,
        )
        return _disjoint_pairs(order, pair_count)

    leverage = _leverage(statistics, context)
    loads = np.sum(statistics.writes, axis=1).astype(np.float64)
    jitter = program.random_weight * rng.normal(size=scenario.cells)
    importance = (
        program.leverage_weight * leverage
        + program.load_weight * loads
        + jitter
    )

    if program.mixing_mode == "high_low":
        high = list(np.argsort(importance)[::-1])
        low = list(np.argsort(importance))
        pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        for left in high:
            if int(left) in used:
                continue
            right = next(
                (
                    int(value)
                    for value in low
                    if int(value) not in used and int(value) != int(left)
                ),
                None,
            )
            if right is None:
                break
            pairs.append((int(left), right))
            used.add(int(left))
            used.add(right)
            if len(pairs) >= pair_count:
                break
        return pairs

    if program.mixing_mode == "underloaded":
        order = np.argsort(loads + 0.05 * leverage + jitter)
        interleaved: list[int] = []
        left = 0
        right = len(order) - 1
        while left <= right:
            interleaved.append(int(order[right]))
            right -= 1
            if left <= right:
                interleaved.append(int(order[left]))
                left += 1
        return _disjoint_pairs(np.asarray(interleaved, dtype=int), pair_count)

    raise ValueError(program.mixing_mode)


def conservative_mix(
    statistics: DiffusedStatistics,
    context: int,
    pairs: list[tuple[int, int]],
    alpha: float,
) -> None:
    for left, right in pairs:
        left_gram = statistics.gram[left, context].copy()
        right_gram = statistics.gram[right, context].copy()
        left_response = statistics.response[left, context].copy()
        right_response = statistics.response[right, context].copy()
        statistics.gram[left, context] = (
            (1.0 - alpha) * left_gram + alpha * right_gram
        )
        statistics.gram[right, context] = (
            alpha * left_gram + (1.0 - alpha) * right_gram
        )
        statistics.response[left, context] = (
            (1.0 - alpha) * left_response + alpha * right_response
        )
        statistics.response[right, context] = (
            alpha * left_response + (1.0 - alpha) * right_response
        )
        statistics.writes[left, context] += 1
        statistics.writes[right, context] += 1


def collect_statistics(
    program: DiffusionProgram,
    scenario: v12.SketchScenario,
) -> tuple[v12.CellStatistics, np.ndarray, float]:
    rng = np.random.default_rng(scenario.seed)
    target = v12._normalised_rows(rng, scenario.contexts, scenario.dimension)
    bases = np.stack(
        [v12._feature_basis(rng, scenario.dimension) for _ in range(scenario.contexts)]
    )
    statistics = DiffusedStatistics(
        gram=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension),
            dtype=np.float64,
        ),
        response=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension),
            dtype=np.float64,
        ),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
    )
    total_operations = 0

    for occurrence in range(scenario.examples_per_context):
        for context_value in rng.permutation(scenario.contexts):
            context = int(context_value)
            feature = v12._feature(
                rng,
                scenario.dimension,
                scenario.condition,
                bases[context],
            )
            teacher = float(
                target[context] @ feature + rng.normal(0.0, scenario.noise)
            )
            selected = select_injection_cells(
                program, statistics, scenario, context, occurrence
            )
            outer = np.outer(feature, feature) / max(1, len(selected))
            response = feature * teacher / max(1, len(selected))
            statistics.gram[selected, context] += outer
            statistics.response[selected, context] += response
            statistics.writes[selected, context] += 1
            total_operations += len(selected)

            pairs = select_mixing_pairs(
                program, statistics, scenario, context, occurrence
            )
            conservative_mix(
                statistics,
                context,
                pairs,
                program.mixing_alpha,
            )
            total_operations += 2 * len(pairs)

    examples = scenario.examples_per_context * scenario.contexts
    average_operation_fraction = total_operations / max(1, examples * scenario.cells)
    return (
        v12.CellStatistics(
            gram=statistics.gram,
            response=statistics.response,
            writes=statistics.writes,
        ),
        target,
        float(average_operation_fraction),
    )


@dataclass(frozen=True)
class DiffusionEvaluation:
    score: float
    pre_damage: float
    post_damage: float
    retention: float
    operation_fraction: float
    attack_surface: float


def evaluate_program(
    program: DiffusionProgram,
    scenario: v12.SketchScenario,
) -> DiffusionEvaluation:
    statistics, target, operations = collect_statistics(program, scenario)
    alive_all = np.ones(scenario.cells, dtype=bool)
    pre_estimate = v12.decode(statistics, alive_all, program.decode_ridge)
    pre = float(np.min(v12.mapping_scores(pre_estimate, target)))
    alive = v12.targeted_delete(
        statistics, target, scenario, program.decode_ridge
    )
    post_estimate = v12.decode(statistics, alive, program.decode_ridge)
    post = float(np.min(v12.mapping_scores(post_estimate, target)))
    retention = post / max(pre, 1e-12)
    leverage = np.trace(statistics.gram, axis1=2, axis2=3)
    delete_count = min(
        scenario.cells - 1,
        max(1, int(round(scenario.cells * scenario.damage_fraction))),
    )
    surfaces: list[float] = []
    for context in range(scenario.contexts):
        values = np.maximum(leverage[:, context], 0.0)
        total = float(np.sum(values))
        top = np.sort(values)[-delete_count:]
        surfaces.append(float(np.sum(top) / max(total, 1e-12)))
    surface = max(surfaces)
    score = float(
        np.clip(
            0.66 * post
            + 0.16 * min(retention, 1.0)
            + 0.10 * (1.0 - operations)
            + 0.05 * pre
            + 0.03 * (1.0 - surface),
            0.0,
            1.0,
        )
    )
    return DiffusionEvaluation(
        score=score,
        pre_damage=pre,
        post_damage=post,
        retention=float(retention),
        operation_fraction=operations,
        attack_surface=surface,
    )


def random_gossip_control() -> DiffusionProgram:
    return DiffusionProgram(1, "random", "random", 0.50, 0.0, 0.0, 0.0, 2.5e-5, 16_001)


def ring_gossip_control() -> DiffusionProgram:
    return DiffusionProgram(1, "random", "ring", 0.50, 0.0, 0.0, 0.0, 2.5e-5, 16_002)


def hand_flat_diffusion() -> DiffusionProgram:
    return DiffusionProgram(1, "flat", "high_low", 0.50, 0.15, 1.0, 0.02, 2.5e-5, 16_003)


def random_programs(
    rng: np.random.Generator,
    count: int = 120,
) -> list[DiffusionProgram]:
    values = [
        random_gossip_control(),
        ring_gossip_control(),
        hand_flat_diffusion(),
    ]
    for _ in range(count):
        values.append(
            DiffusionProgram(
                injection_count=int(rng.choice((1, 2))),
                injection_mode=str(rng.choice(("random", "underloaded", "flat"))),
                mixing_mode=str(rng.choice(("random", "ring", "high_low", "underloaded"))),
                mixing_alpha=float(rng.uniform(0.20, 0.50)),
                load_weight=float(10.0 ** rng.uniform(-2.0, 0.3)),
                leverage_weight=float(10.0 ** rng.uniform(-2.0, 0.5)),
                random_weight=float(rng.uniform(0.0, 0.15)),
                decode_ridge=float(10.0 ** rng.uniform(-6.0, -2.5)),
                seed_salt=int(rng.integers(1, 2_000_000)),
            )
        )
    unique: dict[str, DiffusionProgram] = {}
    for value in values:
        unique.setdefault(value.text(), value)
    return list(unique.values())


def robust_score(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(ordered.size * 0.4)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


def training_scenarios(seed: int) -> list[v12.SketchScenario]:
    return [
        v12.SketchScenario(seed + 1, 4, 6, 3.0, 34, 16.0, 0.025, 0.54),
        v12.SketchScenario(seed + 2, 5, 7, 3.1, 38, 24.0, 0.030, 0.56),
        v12.SketchScenario(seed + 3, 6, 8, 3.2, 42, 32.0, 0.035, 0.58),
    ]


def counterexamples(seed: int) -> list[v12.SketchScenario]:
    return [
        v12.SketchScenario(seed + 101, 7, 9, 3.2, 46, 40.0, 0.045, 0.59),
        v12.SketchScenario(seed + 102, 8, 10, 3.3, 50, 50.0, 0.050, 0.61),
        v12.SketchScenario(seed + 103, 9, 11, 3.4, 54, 60.0, 0.055, 0.62),
    ]


def hidden_scenarios(seed: int) -> list[v12.SketchScenario]:
    return [
        v12.SketchScenario(seed + 501, 9, 12, 3.3, 68, 55.0, 0.055, 0.60),
        v12.SketchScenario(seed + 502, 11, 14, 3.4, 76, 70.0, 0.065, 0.62),
        v12.SketchScenario(seed + 503, 14, 17, 3.5, 84, 90.0, 0.075, 0.64),
    ]


@dataclass(frozen=True)
class RankedProgram:
    program: DiffusionProgram
    score: float
    evaluations: dict[str, DiffusionEvaluation]


def rank_programs(
    programs: Iterable[DiffusionProgram],
    scenarios: list[v12.SketchScenario],
    limit: int,
) -> list[RankedProgram]:
    rows: list[RankedProgram] = []
    for program in programs:
        evaluations = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        score = robust_score(value.score for value in evaluations.values())
        rows.append(RankedProgram(program, score, evaluations))
    rows.sort(key=lambda value: value.score, reverse=True)
    return rows[:limit]


def iid_program() -> v12.SketchProgram:
    return v12.SketchProgram(WRITE_BUDGET, "iid", 1, 1, 1, 2.5e-5, 710_020)


def dense_program() -> v12.SketchProgram:
    return v12.SketchProgram(1.0, "balanced", 1, 1, 1, 2.5e-5, 715_100)


def v15_candidate() -> v15.FlatProgram:
    return v15.FlatProgram(
        density=WRITE_BUDGET,
        candidate_sets=6,
        leverage_weight=1.514,
        directional_weight=0.894,
        load_weight=0.170,
        overlap_weight=0.309,
        random_weight=0.051,
        decode_ridge=1.6e-5,
        seed_salt=468_979,
    )


@dataclass
class DiffusionResult:
    seed: int
    best: DiffusionProgram
    hidden: dict[str, DiffusionEvaluation]
    dense: dict[str, v12.SketchEvaluation]
    iid: dict[str, v12.SketchEvaluation]
    random_gossip: dict[str, DiffusionEvaluation]
    ring_gossip: dict[str, DiffusionEvaluation]
    flat_v15: dict[str, v15.FlatEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def dense_fraction(self) -> float:
        return self.strict_post / max(self.strict_dense, 1e-12)

    @property
    def strict_specialist(self) -> float:
        return float(
            max(
                min(value.post_damage for value in self.iid.values()),
                min(value.post_damage for value in self.random_gossip.values()),
                min(value.post_damage for value in self.ring_gossip.values()),
                min(value.post_damage for value in self.flat_v15.values()),
            )
        )

    @property
    def strict_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def max_operations(self) -> float:
        return float(max(value.operation_fraction for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.dense_fraction >= 0.985
            and self.strict_post >= self.strict_specialist + 0.025
            and self.strict_retention >= 0.95
            and self.max_operations <= WRITE_BUDGET + 1e-12
        )

    def to_dict(self) -> dict[str, object]:
        def values(items):
            return {label: value.__dict__ for label, value in items.items()}

        return {
            "status": (
                "conservative_diffusion_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "fixed-operation conservative diffusion spreads online sufficient statistics "
                "before targeted post-training erasure; the mechanism must beat iid, random/ring "
                "gossip and the strongest fixed-budget v0.15 router, and still requires a separate "
                "greedy verifier, coding-theory baselines and external reproduction"
            ),
            "seed": self.seed,
            "best_program": self.best.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "fraction_of_dense": self.dense_fraction,
            "strict_specialist_post_damage": self.strict_specialist,
            "strict_retention": self.strict_retention,
            "max_operation_fraction": self.max_operations,
            "hidden": values(self.hidden),
            "dense": values(self.dense),
            "iid": values(self.iid),
            "random_gossip": values(self.random_gossip),
            "ring_gossip": values(self.ring_gossip),
            "flat_v15": values(self.flat_v15),
            "history": self.history,
        }


def run(seed: int = 171) -> DiffusionResult:
    rng = np.random.default_rng(seed)
    candidates = random_programs(rng)
    curriculum = training_scenarios(seed * 10_000)
    ranked = rank_programs(candidates, curriculum, limit=32)
    shortlist = [value.program for value in ranked]
    best = ranked[0]
    history: list[dict[str, object]] = []

    for iteration, scenario in enumerate(counterexamples(seed * 10_000)):
        pre = evaluate_program(best.program, scenario)
        history.append(
            {
                "iteration": iteration,
                "counterexample": scenario.label(),
                "pre_score": pre.score,
                "pre_post_damage": pre.post_damage,
                "program": best.program.text(),
            }
        )
        curriculum.append(scenario)
        best = rank_programs(shortlist, curriculum, limit=1)[0]

    hidden = hidden_scenarios(seed * 10_000)
    return DiffusionResult(
        seed=seed,
        best=best.program,
        hidden={
            scenario.label(): evaluate_program(best.program, scenario)
            for scenario in hidden
        },
        dense={
            scenario.label(): v12.evaluate_program(dense_program(), scenario)
            for scenario in hidden
        },
        iid={
            scenario.label(): v12.evaluate_program(iid_program(), scenario)
            for scenario in hidden
        },
        random_gossip={
            scenario.label(): evaluate_program(random_gossip_control(), scenario)
            for scenario in hidden
        },
        ring_gossip={
            scenario.label(): evaluate_program(ring_gossip_control(), scenario)
            for scenario in hidden
        },
        flat_v15={
            scenario.label(): v15.evaluate_program(v15_candidate(), scenario)
            for scenario in hidden
        },
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strict_post_damage": payload["strict_post_damage"],
                "fraction_of_dense": payload["fraction_of_dense"],
                "strict_specialist_post_damage": payload[
                    "strict_specialist_post_damage"
                ],
                "strict_retention": payload["strict_retention"],
                "max_operation_fraction": payload["max_operation_fraction"],
                "best_program": payload["best_program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
