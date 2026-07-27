from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import rotating_sketch_v12 as v12
from . import information_routing_v14 as v14


WRITE_LIMIT = 0.20


@dataclass(frozen=True)
class FlatProgram:
    density: float
    candidate_sets: int
    leverage_weight: float
    directional_weight: float
    load_weight: float
    overlap_weight: float
    random_weight: float
    decode_ridge: float
    seed_salt: int

    def text(self) -> str:
        return (
            f"density={self.density:.3f};candidates={self.candidate_sets};"
            f"leverage={self.leverage_weight:.3f};directional={self.directional_weight:.3f};"
            f"load={self.load_weight:.3f};overlap={self.overlap_weight:.3f};"
            f"random={self.random_weight:.3f};ridge={self.decode_ridge:.1e};"
            f"salt={self.seed_salt}"
        )


@dataclass
class FlatStatistics:
    gram: np.ndarray
    response: np.ndarray
    writes: np.ndarray
    previous: list[np.ndarray]


def _top_fraction(values: np.ndarray, delete_count: int) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    if total <= 1e-12:
        return 1.0
    count = min(len(values), max(1, delete_count))
    top = np.partition(values, len(values) - count)[-count:]
    return float(np.sum(top) / total)


def _candidate_subsets(
    program: FlatProgram,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
) -> list[np.ndarray]:
    cells = scenario.cells
    count = max(1, min(cells, int(math.floor(program.density * cells))))
    rng = np.random.default_rng(
        scenario.seed
        + program.seed_salt * 1_000_003
        + context * 10_007
        + occurrence * 101
    )
    return [
        np.sort(rng.choice(cells, size=count, replace=False))
        for _ in range(program.candidate_sets)
    ]


def _selection_score(
    program: FlatProgram,
    statistics: FlatStatistics,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
    feature: np.ndarray,
    selected: np.ndarray,
) -> float:
    cells = scenario.cells
    delete_count = min(
        cells - 1,
        max(1, int(round(cells * scenario.damage_fraction))),
    )
    feature_norm_four = float(np.dot(feature, feature) ** 2)

    leverage = np.trace(
        statistics.gram[:, context], axis1=1, axis2=2
    ).astype(np.float64)
    leverage[selected] += float(np.dot(feature, feature))
    leverage_risk = _top_fraction(leverage, delete_count)

    directional = np.einsum(
        "cij,i,j->c",
        statistics.gram[:, context],
        feature,
        feature,
    )
    directional[selected] += feature_norm_four
    directional_risk = _top_fraction(directional, delete_count)

    load = np.sum(statistics.writes, axis=1).astype(np.float64)
    load[selected] += 1.0
    load_spread = float(
        (np.max(load) - np.min(load)) / max(1.0, occurrence + 1.0)
    )

    previous = statistics.previous[context]
    overlap = (
        len(np.intersect1d(previous, selected)) / max(1, len(selected))
        if previous.size
        else 0.0
    )
    rng = np.random.default_rng(
        scenario.seed
        + program.seed_salt * 2_000_003
        + context * 20_011
        + occurrence * 211
        + int(np.sum(selected)) * 17
    )
    random_signal = float(rng.normal())
    return float(
        -program.leverage_weight * leverage_risk
        -program.directional_weight * directional_risk
        -program.load_weight * load_spread
        -program.overlap_weight * overlap
        +program.random_weight * random_signal
    )


def select_cells(
    program: FlatProgram,
    statistics: FlatStatistics,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
    feature: np.ndarray,
) -> np.ndarray:
    candidates = _candidate_subsets(program, scenario, context, occurrence)
    scored = [
        (
            _selection_score(
                program,
                statistics,
                scenario,
                context,
                occurrence,
                feature,
                selected,
            ),
            selected,
        )
        for selected in candidates
    ]
    scored.sort(key=lambda value: value[0], reverse=True)
    return scored[0][1]


def collect_statistics(
    program: FlatProgram,
    scenario: v12.SketchScenario,
) -> tuple[v12.CellStatistics, np.ndarray]:
    rng = np.random.default_rng(scenario.seed)
    target = v12._normalised_rows(rng, scenario.contexts, scenario.dimension)
    bases = np.stack(
        [v12._feature_basis(rng, scenario.dimension) for _ in range(scenario.contexts)]
    )
    statistics = FlatStatistics(
        gram=np.zeros(
            (
                scenario.cells,
                scenario.contexts,
                scenario.dimension,
                scenario.dimension,
            ),
            dtype=np.float64,
        ),
        response=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension),
            dtype=np.float64,
        ),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
        previous=[np.asarray([], dtype=int) for _ in range(scenario.contexts)],
    )

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
            selected = select_cells(
                program,
                statistics,
                scenario,
                context,
                occurrence,
                feature,
            )
            outer = np.outer(feature, feature)
            statistics.gram[selected, context] += outer
            statistics.response[selected, context] += feature * teacher
            statistics.writes[selected, context] += 1
            statistics.previous[context] = selected

    return (
        v12.CellStatistics(
            gram=statistics.gram,
            response=statistics.response,
            writes=statistics.writes,
        ),
        target,
    )


@dataclass(frozen=True)
class FlatEvaluation:
    score: float
    pre_damage: float
    post_damage: float
    retention: float
    write_fraction: float
    attack_surface: float


def attack_surface(
    statistics: v12.CellStatistics,
    scenario: v12.SketchScenario,
) -> float:
    delete_count = min(
        scenario.cells - 1,
        max(1, int(round(scenario.cells * scenario.damage_fraction))),
    )
    leverage = np.trace(statistics.gram, axis1=2, axis2=3)
    return float(
        max(
            _top_fraction(leverage[:, context], delete_count)
            for context in range(scenario.contexts)
        )
    )


def evaluate_program(
    program: FlatProgram,
    scenario: v12.SketchScenario,
) -> FlatEvaluation:
    statistics, target = collect_statistics(program, scenario)
    all_alive = np.ones(scenario.cells, dtype=bool)
    pre_estimate = v12.decode(statistics, all_alive, program.decode_ridge)
    pre = float(np.min(v12.mapping_scores(pre_estimate, target)))
    alive = v12.targeted_delete(
        statistics,
        target,
        scenario,
        program.decode_ridge,
    )
    post_estimate = v12.decode(statistics, alive, program.decode_ridge)
    post = float(np.min(v12.mapping_scores(post_estimate, target)))
    retention = post / max(pre, 1e-12)
    write_fraction = max(
        1, int(math.floor(program.density * scenario.cells))
    ) / scenario.cells
    surface = attack_surface(statistics, scenario)
    score = float(
        np.clip(
            0.62 * post
            + 0.16 * min(retention, 1.0)
            + 0.10 * (1.0 - write_fraction)
            + 0.07 * pre
            + 0.05 * (1.0 - surface),
            0.0,
            1.0,
        )
    )
    return FlatEvaluation(
        score=score,
        pre_damage=pre,
        post_damage=post,
        retention=float(retention),
        write_fraction=float(write_fraction),
        attack_surface=surface,
    )


def hand_flat_program() -> FlatProgram:
    return FlatProgram(
        density=WRITE_LIMIT,
        candidate_sets=12,
        leverage_weight=1.0,
        directional_weight=0.8,
        load_weight=0.20,
        overlap_weight=0.15,
        random_weight=0.015,
        decode_ridge=2.5e-5,
        seed_salt=15_015,
    )


def random_programs(
    rng: np.random.Generator,
    count: int = 140,
) -> list[FlatProgram]:
    values = [hand_flat_program()]
    for _ in range(count):
        values.append(
            FlatProgram(
                density=float(rng.uniform(0.12, WRITE_LIMIT)),
                candidate_sets=int(rng.choice((4, 6, 8, 12, 16))),
                leverage_weight=float(10.0 ** rng.uniform(-1.0, 0.5)),
                directional_weight=float(10.0 ** rng.uniform(-1.0, 0.5)),
                load_weight=float(10.0 ** rng.uniform(-2.0, 0.2)),
                overlap_weight=float(rng.uniform(0.0, 0.6)),
                random_weight=float(rng.uniform(0.0, 0.12)),
                decode_ridge=float(10.0 ** rng.uniform(-6.0, -2.5)),
                seed_salt=int(rng.integers(1, 2_000_000)),
            )
        )
    return values


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
class RankedFlat:
    program: FlatProgram
    score: float
    evaluations: dict[str, FlatEvaluation]


def rank_programs(
    programs: Iterable[FlatProgram],
    scenarios: list[v12.SketchScenario],
    limit: int,
) -> list[RankedFlat]:
    ranked: list[RankedFlat] = []
    for program in programs:
        evaluations = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        score = robust_score(value.score for value in evaluations.values())
        ranked.append(RankedFlat(program, score, evaluations))
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


def _iid(density: float = WRITE_LIMIT) -> v12.SketchProgram:
    return v12.SketchProgram(density, "iid", 1, 1, 1, 2.5e-5, 510_020)


def _balanced(density: float = WRITE_LIMIT) -> v12.SketchProgram:
    return v12.SketchProgram(density, "balanced", 1, 3, 5, 2.5e-5, 515_020)


def _antithetic(density: float = WRITE_LIMIT) -> v12.SketchProgram:
    return v12.SketchProgram(density, "antithetic", 1, 1, 1, 2.5e-5, 520_020)


@dataclass
class FlatResult:
    seed: int
    best: FlatProgram
    hidden: dict[str, FlatEvaluation]
    iid: dict[str, v12.SketchEvaluation]
    balanced: dict[str, v12.SketchEvaluation]
    antithetic: dict[str, v12.SketchEvaluation]
    hand_information: dict[str, v14.RoutingEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_iid(self) -> float:
        return float(min(value.post_damage for value in self.iid.values()))

    @property
    def strict_specialist(self) -> float:
        controls = [
            min(value.post_damage for value in self.balanced.values()),
            min(value.post_damage for value in self.antithetic.values()),
            min(value.post_damage for value in self.hand_information.values()),
        ]
        return float(max(controls))

    @property
    def strict_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.strict_post >= 0.86
            and self.strict_post >= self.strict_iid + 0.04
            and self.strict_post >= self.strict_specialist + 0.03
            and self.strict_retention >= 0.90
            and max(value.write_fraction for value in self.hidden.values()) <= WRITE_LIMIT + 1e-12
        )

    def to_dict(self) -> dict[str, object]:
        def flat_values(values):
            return {
                label: value.__dict__
                for label, value in values.items()
            }

        return {
            "status": (
                "counterfactual_flat_memory_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "online sparse routing chooses among random placements by minimising the "
                "counterfactual information fraction removable by a targeted deleter; "
                "outside novelty still requires coding-theory baselines, independent "
                "implementation, external reproduction and peer review"
            ),
            "seed": self.seed,
            "best_program": self.best.text(),
            "strict_post_damage": self.strict_post,
            "strict_iid_post_damage": self.strict_iid,
            "strict_specialist_post_damage": self.strict_specialist,
            "strict_retention": self.strict_retention,
            "hidden": flat_values(self.hidden),
            "iid": flat_values(self.iid),
            "balanced": flat_values(self.balanced),
            "antithetic": flat_values(self.antithetic),
            "hand_information": flat_values(self.hand_information),
            "history": self.history,
        }


def run(seed: int = 151) -> FlatResult:
    rng = np.random.default_rng(seed)
    candidates = random_programs(rng)
    curriculum = training_scenarios(seed * 10_000)
    ranked = rank_programs(candidates, curriculum, limit=36)
    shortlist = [value.program for value in ranked]
    best = ranked[0]
    history: list[dict[str, object]] = []

    for iteration, counterexample in enumerate(counterexamples(seed * 10_000)):
        pre = evaluate_program(best.program, counterexample)
        history.append(
            {
                "iteration": iteration,
                "counterexample": counterexample.label(),
                "pre_score": pre.score,
                "pre_post_damage": pre.post_damage,
                "program": best.program.text(),
            }
        )
        curriculum.append(counterexample)
        best = rank_programs(shortlist, curriculum, limit=1)[0]

    hidden = hidden_scenarios(seed * 10_000)
    return FlatResult(
        seed=seed,
        best=best.program,
        hidden={
            scenario.label(): evaluate_program(best.program, scenario)
            for scenario in hidden
        },
        iid={
            scenario.label(): v12.evaluate_program(_iid(), scenario)
            for scenario in hidden
        },
        balanced={
            scenario.label(): v12.evaluate_program(_balanced(), scenario)
            for scenario in hidden
        },
        antithetic={
            scenario.label(): v12.evaluate_program(_antithetic(), scenario)
            for scenario in hidden
        },
        hand_information={
            scenario.label(): v14.evaluate_program(v14.hand_information_program(), scenario)
            for scenario in hidden
        },
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=151)
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
                "strict_iid_post_damage": payload["strict_iid_post_damage"],
                "strict_specialist_post_damage": payload["strict_specialist_post_damage"],
                "strict_retention": payload["strict_retention"],
                "best_program": payload["best_program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
