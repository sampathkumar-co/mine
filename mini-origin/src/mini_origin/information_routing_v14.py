from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import rotating_sketch_v12 as v12


WRITE_LIMIT = 0.20


@dataclass(frozen=True)
class RoutingProgram:
    density: float
    mode: str
    uncertainty_weight: float
    load_weight: float
    random_weight: float
    selection_ridge: float
    decode_ridge: float
    seed_salt: int

    def text(self) -> str:
        return (
            f"density={self.density:.3f};mode={self.mode};"
            f"uncertainty={self.uncertainty_weight:.3f};load={self.load_weight:.3f};"
            f"random={self.random_weight:.3f};select_ridge={self.selection_ridge:.1e};"
            f"decode_ridge={self.decode_ridge:.1e};salt={self.seed_salt}"
        )


@dataclass
class AdaptiveStatistics:
    gram: np.ndarray
    response: np.ndarray
    writes: np.ndarray
    diagonal: np.ndarray


def _select(
    program: RoutingProgram,
    statistics: AdaptiveStatistics,
    scenario: v12.SketchScenario,
    context: int,
    occurrence: int,
    feature: np.ndarray,
) -> np.ndarray:
    cells = scenario.cells
    count = max(1, min(cells, int(math.floor(program.density * cells))))
    squared = feature * feature
    uncertainty = np.sum(
        squared[None, :]
        / (statistics.diagonal[:, context] + program.selection_ridge),
        axis=1,
    )
    load = statistics.writes[:, context].astype(np.float64)
    load /= max(1.0, float(occurrence + 1))
    rng = np.random.default_rng(
        scenario.seed
        + program.seed_salt * 1_000_003
        + context * 10_007
        + occurrence * 101
    )
    random_signal = rng.normal(size=cells)
    score = (
        program.uncertainty_weight * uncertainty
        - program.load_weight * load
        + program.random_weight * random_signal
    )

    if program.mode == "topk":
        order = np.argsort(score)[::-1]
        return np.sort(order[:count])

    if program.mode == "hybrid":
        deterministic_count = max(1, count // 2)
        order = np.argsort(score)[::-1]
        selected = list(int(value) for value in order[:deterministic_count])
        remaining = np.asarray(
            [index for index in range(cells) if index not in selected],
            dtype=int,
        )
        random_count = count - deterministic_count
        if random_count > 0:
            selected.extend(
                int(value)
                for value in rng.choice(
                    remaining,
                    size=random_count,
                    replace=False,
                )
            )
        return np.sort(np.asarray(selected, dtype=int))

    if program.mode == "softmax":
        centered = score - np.max(score)
        temperature = max(0.05, program.random_weight + 0.10)
        probabilities = np.exp(np.clip(centered / temperature, -40.0, 40.0))
        probabilities /= np.sum(probabilities)
        return np.sort(
            rng.choice(cells, size=count, replace=False, p=probabilities)
        )

    if program.mode == "underloaded":
        # First protect load balance, then information geometry.
        minimum_load = np.min(statistics.writes[:, context])
        underloaded = np.flatnonzero(
            statistics.writes[:, context] <= minimum_load + 1
        )
        if len(underloaded) >= count:
            local_order = underloaded[np.argsort(score[underloaded])[::-1]]
            return np.sort(local_order[:count])
        order = np.argsort(score)[::-1]
        selected = list(int(value) for value in underloaded)
        selected.extend(
            int(value)
            for value in order
            if int(value) not in selected
        )
        return np.sort(np.asarray(selected[:count], dtype=int))

    raise ValueError(program.mode)


def collect_statistics(
    program: RoutingProgram,
    scenario: v12.SketchScenario,
) -> tuple[v12.CellStatistics, np.ndarray]:
    rng = np.random.default_rng(scenario.seed)
    target = v12._normalised_rows(rng, scenario.contexts, scenario.dimension)
    bases = np.stack(
        [v12._feature_basis(rng, scenario.dimension) for _ in range(scenario.contexts)]
    )
    statistics = AdaptiveStatistics(
        gram=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension),
            dtype=np.float64,
        ),
        response=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension),
            dtype=np.float64,
        ),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
        diagonal=np.zeros(
            (scenario.cells, scenario.contexts, scenario.dimension),
            dtype=np.float64,
        ),
    )

    for occurrence in range(scenario.examples_per_context):
        for context in rng.permutation(scenario.contexts):
            context = int(context)
            feature = v12._feature(
                rng,
                scenario.dimension,
                scenario.condition,
                bases[context],
            )
            teacher = float(
                target[context] @ feature + rng.normal(0.0, scenario.noise)
            )
            selected = _select(
                program,
                statistics,
                scenario,
                context,
                occurrence,
                feature,
            )
            squared = feature * feature
            outer = np.outer(feature, feature)
            statistics.gram[selected, context] += outer
            statistics.response[selected, context] += feature * teacher
            statistics.writes[selected, context] += 1
            statistics.diagonal[selected, context] += squared

    return (
        v12.CellStatistics(
            gram=statistics.gram,
            response=statistics.response,
            writes=statistics.writes,
        ),
        target,
    )


@dataclass(frozen=True)
class RoutingEvaluation:
    score: float
    pre_damage: float
    post_damage: float
    retention: float
    write_fraction: float
    coverage_min: int
    coverage_spread: int


def evaluate_program(
    program: RoutingProgram,
    scenario: v12.SketchScenario,
) -> RoutingEvaluation:
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
    write_fraction = max(1, int(math.floor(program.density * scenario.cells))) / scenario.cells
    coverage = np.count_nonzero(statistics.writes, axis=0)
    load = np.sum(statistics.writes, axis=1)
    score = float(
        np.clip(
            0.66 * post
            + 0.16 * min(retention, 1.0)
            + 0.10 * (1.0 - write_fraction)
            + 0.08 * pre,
            0.0,
            1.0,
        )
    )
    return RoutingEvaluation(
        score=score,
        pre_damage=pre,
        post_damage=post,
        retention=float(retention),
        write_fraction=float(write_fraction),
        coverage_min=int(np.min(coverage)),
        coverage_spread=int(np.max(load) - np.min(load)),
    )


def iid_program(density: float = WRITE_LIMIT) -> v12.SketchProgram:
    return v12.SketchProgram(density, "iid", 1, 1, 1, 2.5e-5, 410_020)


def hand_information_program() -> RoutingProgram:
    return RoutingProgram(
        density=WRITE_LIMIT,
        mode="topk",
        uncertainty_weight=1.0,
        load_weight=0.15,
        random_weight=0.0,
        selection_ridge=0.025,
        decode_ridge=2.5e-5,
        seed_salt=8_888,
    )


def random_programs(rng: np.random.Generator, count: int = 150) -> list[RoutingProgram]:
    values: list[RoutingProgram] = []
    for _ in range(count):
        values.append(
            RoutingProgram(
                density=float(rng.uniform(0.10, WRITE_LIMIT)),
                mode=str(rng.choice(("topk", "hybrid", "softmax", "underloaded"))),
                uncertainty_weight=float(10.0 ** rng.uniform(-1.0, 0.6)),
                load_weight=float(10.0 ** rng.uniform(-2.0, 0.5)),
                random_weight=float(rng.uniform(0.0, 1.2)),
                selection_ridge=float(10.0 ** rng.uniform(-3.0, -0.5)),
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
        v12.SketchScenario(seed + 1, 4, 6, 3.0, 38, 16.0, 0.025, 0.54),
        v12.SketchScenario(seed + 2, 5, 7, 3.1, 44, 24.0, 0.030, 0.56),
        v12.SketchScenario(seed + 3, 6, 8, 3.2, 50, 32.0, 0.035, 0.58),
    ]


def counterexamples(seed: int) -> list[v12.SketchScenario]:
    return [
        v12.SketchScenario(seed + 101, 7, 9, 3.2, 56, 40.0, 0.045, 0.59),
        v12.SketchScenario(seed + 102, 8, 10, 3.3, 62, 50.0, 0.050, 0.61),
        v12.SketchScenario(seed + 103, 9, 11, 3.4, 68, 60.0, 0.055, 0.62),
    ]


def hidden_scenarios(seed: int) -> list[v12.SketchScenario]:
    return [
        v12.SketchScenario(seed + 501, 9, 12, 3.3, 74, 55.0, 0.055, 0.60),
        v12.SketchScenario(seed + 502, 11, 14, 3.4, 82, 70.0, 0.065, 0.62),
        v12.SketchScenario(seed + 503, 14, 17, 3.5, 92, 90.0, 0.075, 0.64),
    ]


@dataclass(frozen=True)
class RankedRouting:
    program: RoutingProgram
    score: float
    evaluations: dict[str, RoutingEvaluation]


def rank_programs(
    programs: Iterable[RoutingProgram],
    scenarios: list[v12.SketchScenario],
    limit: int,
) -> list[RankedRouting]:
    ranked: list[RankedRouting] = []
    for program in programs:
        evaluations = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        if any(value.write_fraction > WRITE_LIMIT + 1e-12 for value in evaluations.values()):
            continue
        ranked.append(
            RankedRouting(
                program,
                robust_score(value.score for value in evaluations.values()),
                evaluations,
            )
        )
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


@dataclass
class InformationRoutingResult:
    seed: int
    program: RoutingProgram
    hidden: dict[str, RoutingEvaluation]
    dense: dict[str, v12.SketchEvaluation]
    iid: dict[str, v12.SketchEvaluation]
    hand: dict[str, RoutingEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def strict_iid(self) -> float:
        return float(min(value.post_damage for value in self.iid.values()))

    @property
    def strict_hand(self) -> float:
        return float(min(value.post_damage for value in self.hand.values()))

    @property
    def min_relative_dense(self) -> float:
        return float(
            min(
                self.hidden[key].post_damage / max(self.dense[key].post_damage, 1e-12)
                for key in self.hidden
            )
        )

    @property
    def min_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def max_write(self) -> float:
        return float(max(value.write_fraction for value in self.hidden.values()))

    @property
    def candidate(self) -> bool:
        return (
            self.min_relative_dense >= 0.98
            and self.min_retention >= 0.95
            and self.max_write <= WRITE_LIMIT
            and self.strict_post >= self.strict_iid + 0.04
            and self.strict_post >= self.strict_hand + 0.02
        )

    def to_dict(self) -> dict[str, object]:
        def serialize(values):
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
                "information_balanced_routing_external_candidate"
                if self.candidate
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided routing of online examples to physical memory cells using local "
                "information geometry under a hard write budget and targeted deletion; leverage-based "
                "sampling and experimental design are established, so external significance requires "
                "specialist comparisons, independent reproduction and peer review"
            ),
            "seed": self.seed,
            "program": self.program.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "strict_iid_post_damage": self.strict_iid,
            "strict_hand_post_damage": self.strict_hand,
            "min_relative_dense": self.min_relative_dense,
            "min_retention": self.min_retention,
            "max_write_fraction": self.max_write,
            "hidden": serialize(self.hidden),
            "dense": serialize(self.dense),
            "iid": serialize(self.iid),
            "hand": serialize(self.hand),
            "history": self.history,
        }


def run_search(seed: int = 141) -> InformationRoutingResult:
    rng = np.random.default_rng(seed)
    curriculum = training_scenarios(seed * 10_000)
    ranked = rank_programs(random_programs(rng, 150), curriculum, 40)
    if not ranked:
        raise RuntimeError("no information-routing candidates")
    shortlist = [value.program for value in ranked]
    history: list[dict[str, object]] = []
    for iteration, scenario in enumerate(counterexamples(seed * 10_000)):
        champion = rank_programs(shortlist, curriculum, 1)[0]
        outcome = evaluate_program(champion.program, scenario)
        iid = v12.evaluate_program(iid_program(), scenario)
        hand = evaluate_program(hand_information_program(), scenario)
        history.append({
            "iteration": iteration,
            "program": champion.program.text(),
            "curriculum_score": champion.score,
            "counterexample": scenario.label(),
            "post_damage": outcome.post_damage,
            "iid_gap": outcome.post_damage - iid.post_damage,
            "hand_gap": outcome.post_damage - hand.post_damage,
        })
        curriculum.append(scenario)
    program = rank_programs(shortlist, curriculum, 1)[0].program
    hidden = hidden_scenarios(seed * 10_000)
    return InformationRoutingResult(
        seed=seed,
        program=program,
        hidden={scenario.label(): evaluate_program(program, scenario) for scenario in hidden},
        dense={scenario.label(): v12.evaluate_program(v12.dense_program(), scenario) for scenario in hidden},
        iid={scenario.label(): v12.evaluate_program(iid_program(), scenario) for scenario in hidden},
        hand={scenario.label(): evaluate_program(hand_information_program(), scenario) for scenario in hidden},
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=141)
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
        "strict_iid_post_damage": payload["strict_iid_post_damage"],
        "strict_hand_post_damage": payload["strict_hand_post_damage"],
        "min_relative_dense": payload["min_relative_dense"],
        "min_retention": payload["min_retention"],
        "max_write_fraction": payload["max_write_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
