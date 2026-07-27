from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


WRITE_LIMIT = 0.40


@dataclass(frozen=True)
class SketchProgram:
    density: float
    schedule: str
    stride: int
    context_stride: int
    phase_stride: int
    ridge: float
    seed_salt: int

    def text(self) -> str:
        return (
            f"density={self.density:.3f};schedule={self.schedule};stride={self.stride};"
            f"context_stride={self.context_stride};phase_stride={self.phase_stride};"
            f"ridge={self.ridge:.1e};salt={self.seed_salt}"
        )


@dataclass(frozen=True)
class SketchScenario:
    seed: int
    contexts: int
    dimension: int
    redundancy: float
    examples_per_context: int
    condition: float
    noise: float
    damage_fraction: float

    @property
    def cells(self) -> int:
        return max(4, int(math.ceil(self.contexts * self.redundancy)))

    def label(self) -> str:
        return (
            f"ctx{self.contexts}:d{self.dimension}:r{self.redundancy:.2f}:"
            f"cond{self.condition:.1f}:damage{self.damage_fraction:.2f}:s{self.seed}"
        )


def _normalised_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _feature_basis(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q


def _feature(
    rng: np.random.Generator,
    dimension: int,
    condition: float,
    basis: np.ndarray,
) -> np.ndarray:
    scales = np.geomspace(1.0, 1.0 / max(condition, 1.0), dimension)
    value = (rng.normal(size=dimension) * np.sqrt(scales)) @ basis.T
    return value / max(np.linalg.norm(value), 1e-12)


def _coprime_step(value: int, modulus: int) -> int:
    value = max(1, value % modulus)
    while math.gcd(value, modulus) != 1:
        value += 1
        if value >= modulus:
            value = 1
    return value


def selected_cells(
    program: SketchProgram,
    scenario: SketchScenario,
    context: int,
    occurrence: int,
) -> np.ndarray:
    cells = scenario.cells
    count = max(1, min(cells, int(math.floor(program.density * cells))))
    rng = np.random.default_rng(
        scenario.seed
        + program.seed_salt * 1_000_003
        + context * 10_007
        + occurrence * 101
    )

    if program.schedule == "iid":
        return np.sort(rng.choice(cells, size=count, replace=False))

    stride = _coprime_step(program.stride, cells)
    context_offset = (context * program.context_stride) % cells
    phase_offset = (occurrence * program.phase_stride) % cells

    if program.schedule == "cyclic":
        start = (context_offset + phase_offset) % cells
        return np.asarray(
            sorted({(start + index * stride) % cells for index in range(count)}),
            dtype=int,
        )

    if program.schedule == "balanced":
        # A deterministic permutation changes slowly with occurrence. Every cell
        # receives nearly the same number of writes for every context.
        epoch = occurrence // max(1, cells)
        position = occurrence % cells
        start = (
            context_offset
            + position * stride
            + epoch * program.phase_stride
        ) % cells
        return np.asarray(
            sorted({(start + index) % cells for index in range(count)}),
            dtype=int,
        )

    if program.schedule == "affine":
        a = _coprime_step(program.stride + occurrence * program.phase_stride, cells)
        b = (context_offset + occurrence) % cells
        order = np.asarray([(a * index + b) % cells for index in range(cells)])
        return np.sort(order[:count])

    if program.schedule == "antithetic":
        half = (occurrence // 2) * 2
        base_rng = np.random.default_rng(
            scenario.seed
            + program.seed_salt * 1_000_003
            + context * 10_007
            + half * 101
        )
        order = base_rng.permutation(cells)
        if occurrence % 2 == 0:
            return np.sort(order[:count])
        start = count
        if start + count <= cells:
            return np.sort(order[start : start + count])
        return np.sort(np.concatenate([order[start:], order[: count - (cells - start)]]))

    raise ValueError(program.schedule)


@dataclass
class CellStatistics:
    gram: np.ndarray
    response: np.ndarray
    writes: np.ndarray


def collect_statistics(
    program: SketchProgram,
    scenario: SketchScenario,
) -> tuple[CellStatistics, np.ndarray]:
    rng = np.random.default_rng(scenario.seed)
    target = _normalised_rows(rng, scenario.contexts, scenario.dimension)
    bases = np.stack(
        [_feature_basis(rng, scenario.dimension) for _ in range(scenario.contexts)]
    )
    gram = np.zeros(
        (scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension),
        dtype=np.float64,
    )
    response = np.zeros(
        (scenario.cells, scenario.contexts, scenario.dimension),
        dtype=np.float64,
    )
    writes = np.zeros((scenario.cells, scenario.contexts), dtype=np.int32)

    for occurrence in range(scenario.examples_per_context):
        context_order = rng.permutation(scenario.contexts)
        for context in context_order:
            feature = _feature(
                rng,
                scenario.dimension,
                scenario.condition,
                bases[context],
            )
            teacher = float(
                target[context] @ feature + rng.normal(0.0, scenario.noise)
            )
            selected = selected_cells(
                program,
                scenario,
                int(context),
                occurrence,
            )
            outer = np.outer(feature, feature)
            gram[selected, context] += outer
            response[selected, context] += feature * teacher
            writes[selected, context] += 1
    return CellStatistics(gram=gram, response=response, writes=writes), target


def decode(
    statistics: CellStatistics,
    alive: np.ndarray,
    ridge: float,
) -> np.ndarray:
    aggregate_gram = np.sum(statistics.gram[alive], axis=0)
    aggregate_response = np.sum(statistics.response[alive], axis=0)
    contexts, dimension = aggregate_response.shape
    estimate = np.zeros((contexts, dimension), dtype=np.float64)
    identity = np.eye(dimension)
    for context in range(contexts):
        estimate[context] = np.linalg.solve(
            aggregate_gram[context] + ridge * identity,
            aggregate_response[context],
        )
    return estimate


def mapping_scores(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.mean((estimate - target) ** 2, axis=1)
    return np.exp(-10.0 * error)


def targeted_delete(
    statistics: CellStatistics,
    target: np.ndarray,
    scenario: SketchScenario,
    ridge: float,
) -> np.ndarray:
    """Delete cells most informative for one context; choose the worst attack."""
    delete_count = min(
        scenario.cells - 1,
        int(round(scenario.cells * scenario.damage_fraction)),
    )
    best_alive = np.ones(scenario.cells, dtype=bool)
    best_score = float("inf")
    leverage = np.trace(statistics.gram, axis1=2, axis2=3)

    for attacked_context in range(scenario.contexts):
        order = np.argsort(leverage[:, attacked_context])[::-1]
        alive = np.ones(scenario.cells, dtype=bool)
        alive[order[:delete_count]] = False
        estimate = decode(statistics, alive, ridge)
        score = float(np.min(mapping_scores(estimate, target)))
        if score < best_score:
            best_score = score
            best_alive = alive
    return best_alive


@dataclass(frozen=True)
class SketchEvaluation:
    score: float
    pre_damage: float
    post_damage: float
    retention: float
    write_fraction: float
    coverage_min: int
    coverage_spread: int


def evaluate_program(
    program: SketchProgram,
    scenario: SketchScenario,
) -> SketchEvaluation:
    statistics, target = collect_statistics(program, scenario)
    all_alive = np.ones(scenario.cells, dtype=bool)
    pre_estimate = decode(statistics, all_alive, program.ridge)
    pre_scores = mapping_scores(pre_estimate, target)
    pre = float(np.min(pre_scores))

    alive = targeted_delete(statistics, target, scenario, program.ridge)
    post_estimate = decode(statistics, alive, program.ridge)
    post_scores = mapping_scores(post_estimate, target)
    post = float(np.min(post_scores))
    retention = post / max(pre, 1e-12)
    write_fraction = max(1, int(math.floor(program.density * scenario.cells))) / scenario.cells
    per_context_coverage = np.count_nonzero(statistics.writes, axis=0)
    coverage_min = int(np.min(per_context_coverage))
    cell_load = np.sum(statistics.writes, axis=1)
    coverage_spread = int(np.max(cell_load) - np.min(cell_load))
    score = float(
        np.clip(
            0.64 * post
            + 0.16 * min(retention, 1.0)
            + 0.12 * (1.0 - write_fraction)
            + 0.08 * pre,
            0.0,
            1.0,
        )
    )
    return SketchEvaluation(
        score=score,
        pre_damage=pre,
        post_damage=post,
        retention=float(retention),
        write_fraction=float(write_fraction),
        coverage_min=coverage_min,
        coverage_spread=coverage_spread,
    )


def dense_program() -> SketchProgram:
    return SketchProgram(1.0, "balanced", 1, 1, 1, 1e-5, 1)


def static_program() -> SketchProgram:
    return SketchProgram(WRITE_LIMIT, "cyclic", 1, 0, 0, 1e-5, 1)


def hand_rotating_program() -> SketchProgram:
    return SketchProgram(WRITE_LIMIT, "balanced", 1, 3, 5, 1e-5, 7_777)


def random_programs(rng: np.random.Generator, count: int = 280) -> list[SketchProgram]:
    values: list[SketchProgram] = []
    for _ in range(count):
        values.append(
            SketchProgram(
                density=float(rng.uniform(0.18, WRITE_LIMIT)),
                schedule=str(rng.choice(("iid", "cyclic", "balanced", "affine", "antithetic"))),
                stride=int(rng.integers(1, 97)),
                context_stride=int(rng.integers(0, 97)),
                phase_stride=int(rng.integers(0, 97)),
                ridge=float(10.0 ** rng.uniform(-6.5, -2.5)),
                seed_salt=int(rng.integers(1, 2_000_000)),
            )
        )
    return values


def robust_score(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(ordered.size * 0.4)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


def training_scenarios(seed: int) -> list[SketchScenario]:
    return [
        SketchScenario(seed + 1, 4, 6, 2.8, 38, 10.0, 0.025, 0.52),
        SketchScenario(seed + 2, 5, 7, 2.9, 42, 16.0, 0.030, 0.55),
        SketchScenario(seed + 3, 6, 8, 3.0, 46, 22.0, 0.035, 0.57),
    ]


def counterexamples(seed: int) -> list[SketchScenario]:
    return [
        SketchScenario(seed + 101, 7, 9, 3.0, 50, 28.0, 0.040, 0.58),
        SketchScenario(seed + 102, 8, 10, 3.1, 54, 34.0, 0.045, 0.60),
        SketchScenario(seed + 103, 9, 11, 3.2, 58, 40.0, 0.050, 0.61),
    ]


def hidden_scenarios(seed: int) -> list[SketchScenario]:
    return [
        SketchScenario(seed + 501, 8, 11, 3.1, 62, 36.0, 0.050, 0.58),
        SketchScenario(seed + 502, 10, 13, 3.2, 68, 45.0, 0.060, 0.60),
        SketchScenario(seed + 503, 12, 15, 3.3, 74, 55.0, 0.070, 0.62),
    ]


@dataclass(frozen=True)
class RankedSketch:
    program: SketchProgram
    score: float
    evaluations: dict[str, SketchEvaluation]


def rank_programs(
    programs: Iterable[SketchProgram],
    scenarios: list[SketchScenario],
    limit: int,
) -> list[RankedSketch]:
    ranked: list[RankedSketch] = []
    for program in programs:
        evaluations = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        if any(value.write_fraction > WRITE_LIMIT + 1e-12 for value in evaluations.values()):
            continue
        score = robust_score(value.score for value in evaluations.values())
        ranked.append(RankedSketch(program, score, evaluations))
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


@dataclass
class RotatingSketchResult:
    seed: int
    program: SketchProgram
    hidden: dict[str, SketchEvaluation]
    dense: dict[str, SketchEvaluation]
    static: dict[str, SketchEvaluation]
    hand: dict[str, SketchEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def strict_static(self) -> float:
        return float(min(value.post_damage for value in self.static.values()))

    @property
    def strict_hand(self) -> float:
        return float(min(value.post_damage for value in self.hand.values()))

    @property
    def max_write(self) -> float:
        return float(max(value.write_fraction for value in self.hidden.values()))

    @property
    def min_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.strict_post >= 0.90
            and self.min_retention >= 0.95
            and self.max_write <= WRITE_LIMIT
            and self.strict_post >= self.strict_dense - 0.04
            and self.strict_post >= self.strict_static + 0.30
            and self.strict_post >= self.strict_hand + 0.025
        )

    def to_dict(self) -> dict[str, object]:
        def serialize(values: dict[str, SketchEvaluation]) -> dict[str, dict[str, float | int]]:
            return {
                key: {
                    "score": value.score,
                    "pre_damage": value.pre_damage,
                    "post_damage": value.post_damage,
                    "retention": value.retention,
                    "write_fraction": value.write_fraction,
                    "coverage_min": value.coverage_min,
                    "coverage_spread": value.coverage_spread,
                }
                for key, value in values.items()
            }

        return {
            "status": (
                "rotating_sparse_sketch_external_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided synthesis of temporally rotating sparse writes into distributed "
                "online sufficient statistics, enabling immediate frozen-learning recovery after targeted "
                "cell deletion under a hard per-example write budget; external significance requires "
                "specialist baselines, independent reproduction and peer review"
            ),
            "seed": self.seed,
            "program": self.program.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "strict_static_post_damage": self.strict_static,
            "strict_hand_post_damage": self.strict_hand,
            "max_write_fraction": self.max_write,
            "min_retention": self.min_retention,
            "hidden": serialize(self.hidden),
            "dense": serialize(self.dense),
            "static": serialize(self.static),
            "hand": serialize(self.hand),
            "history": self.history,
        }


def run_search(seed: int = 121) -> RotatingSketchResult:
    rng = np.random.default_rng(seed)
    curriculum = training_scenarios(seed * 10_000)
    ranked = rank_programs(random_programs(rng, 280), curriculum, 64)
    if not ranked:
        raise RuntimeError("no write-budget-feasible sketch programs")
    shortlist = [value.program for value in ranked]
    history: list[dict[str, object]] = []
    for iteration, scenario in enumerate(counterexamples(seed * 10_000)):
        champion = rank_programs(shortlist, curriculum, 1)[0]
        outcome = evaluate_program(champion.program, scenario)
        hand = evaluate_program(hand_rotating_program(), scenario)
        history.append(
            {
                "iteration": iteration,
                "program": champion.program.text(),
                "curriculum_score": champion.score,
                "counterexample": scenario.label(),
                "counterexample_post_damage": outcome.post_damage,
                "counterexample_hand_gap": outcome.post_damage - hand.post_damage,
            }
        )
        curriculum.append(scenario)
    program = rank_programs(shortlist, curriculum, 1)[0].program
    hidden = hidden_scenarios(seed * 10_000)
    return RotatingSketchResult(
        seed=seed,
        program=program,
        hidden={scenario.label(): evaluate_program(program, scenario) for scenario in hidden},
        dense={scenario.label(): evaluate_program(dense_program(), scenario) for scenario in hidden},
        static={scenario.label(): evaluate_program(static_program(), scenario) for scenario in hidden},
        hand={scenario.label(): evaluate_program(hand_rotating_program(), scenario) for scenario in hidden},
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_search(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "program": payload["program"],
        "strict_post_damage": payload["strict_post_damage"],
        "strict_dense_post_damage": payload["strict_dense_post_damage"],
        "strict_static_post_damage": payload["strict_static_post_damage"],
        "strict_hand_post_damage": payload["strict_hand_post_damage"],
        "max_write_fraction": payload["max_write_fraction"],
        "min_retention": payload["min_retention"],
    }, indent=2))


if __name__ == "__main__":
    main()
