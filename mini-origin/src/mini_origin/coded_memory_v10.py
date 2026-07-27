from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CodeProgram:
    density: float
    systematic: bool
    coefficient_mode: str
    balanced: bool
    ridge: float
    learning_rate: float
    seed_salt: int

    def text(self) -> str:
        return (
            f"density={self.density:.3f};systematic={int(self.systematic)};"
            f"coeff={self.coefficient_mode};balanced={int(self.balanced)};"
            f"ridge={self.ridge:.1e};lr={self.learning_rate:.3f};salt={self.seed_salt}"
        )


@dataclass(frozen=True)
class CodeScenario:
    seed: int
    contexts: int
    dimension: int
    redundancy: float
    examples_per_context: int
    noise: float
    damage_fraction: float

    def label(self) -> str:
        return (
            f"ctx{self.contexts}:d{self.dimension}:r{self.redundancy:.2f}:"
            f"damage{self.damage_fraction:.2f}:s{self.seed}"
        )


def _normalised_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _parity_value(rng: np.random.Generator, mode: str, degree: int) -> np.ndarray:
    if mode == "rademacher":
        return rng.choice((-1.0, 1.0), size=degree) / math.sqrt(degree)
    if mode == "gaussian":
        value = rng.normal(size=degree)
        return value / max(np.linalg.norm(value), 1e-12)
    if mode == "positive":
        return np.ones(degree, dtype=np.float64) / math.sqrt(degree)
    raise ValueError(mode)


def make_code_matrix(
    program: CodeProgram,
    scenario: CodeScenario,
) -> np.ndarray:
    contexts = scenario.contexts
    cells = max(contexts + 1, int(math.ceil(contexts * scenario.redundancy)))
    rng = np.random.default_rng(scenario.seed + 1_000_003 * program.seed_salt)
    matrix = np.zeros((cells, contexts), dtype=np.float64)
    start = 0
    if program.systematic:
        systematic_rows = min(contexts, cells)
        matrix[:systematic_rows, :systematic_rows] = np.eye(systematic_rows)
        start = systematic_rows

    degree = int(np.clip(round(program.density * contexts), 2, contexts))
    for row in range(start, cells):
        chosen = rng.choice(contexts, size=degree, replace=False)
        matrix[row, chosen] = _parity_value(rng, program.coefficient_mode, degree)

    if program.balanced:
        target_degree = max(2, int(math.ceil(program.density * cells)))
        for context in range(contexts):
            current = int(np.count_nonzero(matrix[:, context]))
            missing = max(0, target_degree - current)
            if missing == 0:
                continue
            candidate_rows = np.flatnonzero(matrix[:, context] == 0.0)
            if candidate_rows.size == 0:
                continue
            chosen_rows = rng.choice(
                candidate_rows,
                size=min(missing, candidate_rows.size),
                replace=False,
            )
            signs = rng.choice((-1.0, 1.0), size=len(chosen_rows))
            matrix[chosen_rows, context] = signs / math.sqrt(max(2, degree))

    # Normalize each encoded cell to prevent coefficient scale from changing
    # the learning rate or the attack purely through row magnitude.
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    active = row_norms[:, 0] > 1e-12
    matrix[active] /= row_norms[active]
    return matrix


def replication_matrix(scenario: CodeScenario) -> np.ndarray:
    contexts = scenario.contexts
    cells = max(contexts + 1, int(math.ceil(contexts * scenario.redundancy)))
    matrix = np.zeros((cells, contexts), dtype=np.float64)
    for row in range(cells):
        matrix[row, row % contexts] = 1.0
    return matrix


def dense_matrix(scenario: CodeScenario) -> np.ndarray:
    return make_code_matrix(
        CodeProgram(
            density=1.0,
            systematic=False,
            coefficient_mode="gaussian",
            balanced=True,
            ridge=1e-6,
            learning_rate=0.22,
            seed_salt=991,
        ),
        scenario,
    )


def uncoded_matrix(scenario: CodeScenario) -> np.ndarray:
    contexts = scenario.contexts
    cells = max(contexts + 1, int(math.ceil(contexts * scenario.redundancy)))
    matrix = np.zeros((cells, contexts), dtype=np.float64)
    matrix[:contexts] = np.eye(contexts)
    return matrix


def _decode(
    code: np.ndarray,
    memory: np.ndarray,
    alive: np.ndarray,
    ridge: float,
) -> np.ndarray:
    active_code = code[alive]
    active_memory = memory[alive]
    gram = active_code.T @ active_code
    return np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        active_code.T @ active_memory,
    )


def _mapping_score(estimate: np.ndarray, target: np.ndarray) -> float:
    error = np.mean((estimate - target) ** 2, axis=1)
    return float(np.mean(np.exp(-10.0 * error)))


def _adversarial_delete(code: np.ndarray, fraction: float) -> np.ndarray:
    """Greedily remove rows that most reduce the remaining minimum singular value."""
    cells, contexts = code.shape
    alive = np.ones(cells, dtype=bool)
    max_delete = max(0, cells - contexts)
    delete_count = min(max_delete, int(round(cells * fraction)))
    for _ in range(delete_count):
        candidates = np.flatnonzero(alive)
        best_row = None
        best_value = float("inf")
        best_rank = contexts
        for row in candidates:
            trial = alive.copy()
            trial[row] = False
            active = code[trial]
            rank = int(np.linalg.matrix_rank(active, tol=1e-8))
            singular = np.linalg.svd(active, compute_uv=False)
            minimum = float(singular[-1]) if singular.size else 0.0
            # Rank loss dominates; otherwise minimize conditioning.
            value = minimum + 100.0 * rank
            if rank < best_rank or (rank == best_rank and value < best_value):
                best_rank = rank
                best_value = value
                best_row = int(row)
        if best_row is None:
            break
        alive[best_row] = False
    return alive


@dataclass(frozen=True)
class CodeEvaluation:
    score: float
    pre_damage: float
    post_damage: float
    retention: float
    write_fraction: float
    surviving_rank: int


def evaluate_matrix(
    code: np.ndarray,
    ridge: float,
    learning_rate: float,
    scenario: CodeScenario,
) -> CodeEvaluation:
    rng = np.random.default_rng(scenario.seed)
    target = _normalised_rows(rng, scenario.contexts, scenario.dimension)
    memory = np.zeros((code.shape[0], scenario.dimension), dtype=np.float64)
    alive = np.ones(code.shape[0], dtype=bool)

    total_steps = scenario.contexts * scenario.examples_per_context
    order = np.tile(np.arange(scenario.contexts), scenario.examples_per_context)
    rng.shuffle(order)
    for context in order:
        feature = _normalised_rows(rng, 1, scenario.dimension)[0]
        teacher = float(target[context] @ feature + rng.normal(0.0, scenario.noise))
        estimate = _decode(code, memory, alive, ridge)
        prediction = float(estimate[context] @ feature)
        delta = learning_rate * (teacher - prediction) * feature
        # Exact encoded write for a change to one logical context mapping.
        memory += code[:, context, None] * delta[None, :]

    pre_estimate = _decode(code, memory, alive, ridge)
    pre = _mapping_score(pre_estimate, target)
    alive = _adversarial_delete(code, scenario.damage_fraction)
    post_estimate = _decode(code, memory, alive, ridge)
    post = _mapping_score(post_estimate, target)
    retention = post / max(pre, 1e-12)
    write_fraction = float(np.count_nonzero(code) / code.size)
    rank = int(np.linalg.matrix_rank(code[alive], tol=1e-8))
    score = float(
        np.clip(
            0.62 * post
            + 0.16 * min(retention, 1.0)
            + 0.14 * (1.0 - write_fraction)
            + 0.08 * pre,
            0.0,
            1.0,
        )
    )
    return CodeEvaluation(
        score=score,
        pre_damage=pre,
        post_damage=post,
        retention=float(retention),
        write_fraction=write_fraction,
        surviving_rank=rank,
    )


def evaluate_program(program: CodeProgram, scenario: CodeScenario) -> CodeEvaluation:
    return evaluate_matrix(
        make_code_matrix(program, scenario),
        program.ridge,
        program.learning_rate,
        scenario,
    )


def evaluate_replication(scenario: CodeScenario) -> CodeEvaluation:
    return evaluate_matrix(replication_matrix(scenario), 1e-6, 0.22, scenario)


def evaluate_dense(scenario: CodeScenario) -> CodeEvaluation:
    return evaluate_matrix(dense_matrix(scenario), 1e-6, 0.22, scenario)


def evaluate_uncoded(scenario: CodeScenario) -> CodeEvaluation:
    return evaluate_matrix(uncoded_matrix(scenario), 1e-6, 0.22, scenario)


def robust_score(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    tail = ordered[: max(1, int(math.ceil(len(ordered) * 0.4)))]
    return float(0.60 * ordered[0] + 0.25 * np.mean(tail) + 0.15 * np.mean(ordered))


def random_programs(rng: np.random.Generator, count: int = 260) -> list[CodeProgram]:
    values = []
    for _ in range(count):
        values.append(
            CodeProgram(
                density=float(rng.uniform(0.12, 0.58)),
                systematic=bool(rng.integers(0, 2)),
                coefficient_mode=str(rng.choice(("rademacher", "gaussian", "positive"))),
                balanced=bool(rng.integers(0, 2)),
                ridge=float(10.0 ** rng.uniform(-7.0, -2.5)),
                learning_rate=float(rng.uniform(0.10, 0.38)),
                seed_salt=int(rng.integers(1, 50_000)),
            )
        )
    return values


def training_scenarios(seed: int) -> list[CodeScenario]:
    return [
        CodeScenario(seed + 1, 4, 6, 2.50, 45, 0.025, 0.45),
        CodeScenario(seed + 2, 5, 8, 2.50, 50, 0.030, 0.50),
        CodeScenario(seed + 3, 6, 9, 2.60, 55, 0.035, 0.52),
    ]


def counterexamples(seed: int) -> list[CodeScenario]:
    return [
        CodeScenario(seed + 101, 7, 10, 2.55, 60, 0.040, 0.54),
        CodeScenario(seed + 102, 8, 11, 2.65, 65, 0.045, 0.56),
        CodeScenario(seed + 103, 9, 12, 2.70, 70, 0.050, 0.58),
    ]


def hidden_scenarios(seed: int) -> list[CodeScenario]:
    return [
        CodeScenario(seed + 501, 8, 12, 2.60, 75, 0.050, 0.55),
        CodeScenario(seed + 502, 10, 14, 2.70, 80, 0.060, 0.58),
        CodeScenario(seed + 503, 12, 16, 2.80, 85, 0.070, 0.60),
    ]


@dataclass(frozen=True)
class RankedCode:
    program: CodeProgram
    score: float
    evaluations: dict[str, CodeEvaluation]


def rank_programs(
    programs: Iterable[CodeProgram],
    scenarios: list[CodeScenario],
    limit: int,
) -> list[RankedCode]:
    ranked = []
    for program in programs:
        evaluations = {
            scenario.label(): evaluate_program(program, scenario)
            for scenario in scenarios
        }
        quality = robust_score(value.score for value in evaluations.values())
        # Directly reward write sparsity only after preserving full logical rank.
        rank_fraction = min(
            value.surviving_rank / scenario.contexts
            for value, scenario in zip(evaluations.values(), scenarios)
        )
        score = quality if rank_fraction >= 1.0 else quality - 0.30 * (1.0 - rank_fraction)
        ranked.append(RankedCode(program, float(score), evaluations))
    ranked.sort(key=lambda value: value.score, reverse=True)
    return ranked[:limit]


@dataclass
class CodedMemoryResult:
    seed: int
    program: CodeProgram
    hidden: dict[str, CodeEvaluation]
    dense: dict[str, CodeEvaluation]
    replication: dict[str, CodeEvaluation]
    uncoded: dict[str, CodeEvaluation]
    history: list[dict[str, object]]

    @property
    def strict_post(self) -> float:
        return float(min(value.post_damage for value in self.hidden.values()))

    @property
    def strict_dense(self) -> float:
        return float(min(value.post_damage for value in self.dense.values()))

    @property
    def strict_replication(self) -> float:
        return float(min(value.post_damage for value in self.replication.values()))

    @property
    def strict_uncoded(self) -> float:
        return float(min(value.post_damage for value in self.uncoded.values()))

    @property
    def max_write_fraction(self) -> float:
        return float(max(value.write_fraction for value in self.hidden.values()))

    @property
    def min_retention(self) -> float:
        return float(min(value.retention for value in self.hidden.values()))

    @property
    def external_candidate(self) -> bool:
        return (
            self.strict_post >= 0.90
            and self.min_retention >= 0.95
            and self.max_write_fraction <= 0.40
            and self.strict_post >= self.strict_dense - 0.04
            and self.strict_post >= self.strict_replication + 0.15
            and self.strict_post >= self.strict_uncoded + 0.35
        )

    def to_dict(self) -> dict[str, object]:
        def serialize(values: dict[str, CodeEvaluation]) -> dict[str, dict[str, float | int]]:
            return {
                key: {
                    "score": value.score,
                    "pre_damage": value.pre_damage,
                    "post_damage": value.post_damage,
                    "retention": value.retention,
                    "write_fraction": value.write_fraction,
                    "surviving_rank": value.surviving_rank,
                }
                for key, value in values.items()
            }

        return {
            "status": (
                "sparse_online_code_external_candidate"
                if self.external_candidate
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided synthesis of a sparse online memory code that learns arbitrary "
                "context mappings, immediately decodes them after targeted cell deletion with learning "
                "frozen, and approaches dense-code recovery at lower write cost; external acceptance "
                "requires independent reproduction and peer review"
            ),
            "seed": self.seed,
            "program": self.program.text(),
            "strict_post_damage": self.strict_post,
            "strict_dense_post_damage": self.strict_dense,
            "strict_replication_post_damage": self.strict_replication,
            "strict_uncoded_post_damage": self.strict_uncoded,
            "max_write_fraction": self.max_write_fraction,
            "min_retention": self.min_retention,
            "hidden": serialize(self.hidden),
            "dense": serialize(self.dense),
            "replication": serialize(self.replication),
            "uncoded": serialize(self.uncoded),
            "history": self.history,
        }


def run_coded_memory_search(seed: int = 101) -> CodedMemoryResult:
    rng = np.random.default_rng(seed)
    curriculum = training_scenarios(seed * 10_000)
    candidates = random_programs(rng, count=260)
    ranked = rank_programs(candidates, curriculum, limit=72)
    shortlist = [value.program for value in ranked]
    history: list[dict[str, object]] = []

    for iteration, scenario in enumerate(counterexamples(seed * 10_000)):
        champion = rank_programs(shortlist, curriculum, limit=1)[0]
        current = evaluate_program(champion.program, scenario)
        history.append(
            {
                "iteration": iteration,
                "program": champion.program.text(),
                "curriculum_score": champion.score,
                "counterexample": scenario.label(),
                "counterexample_post_damage": current.post_damage,
                "counterexample_write_fraction": current.write_fraction,
            }
        )
        curriculum.append(scenario)

    champion = rank_programs(shortlist, curriculum, limit=1)[0].program
    hidden = hidden_scenarios(seed * 10_000)
    return CodedMemoryResult(
        seed=seed,
        program=champion,
        hidden={scenario.label(): evaluate_program(champion, scenario) for scenario in hidden},
        dense={scenario.label(): evaluate_dense(scenario) for scenario in hidden},
        replication={scenario.label(): evaluate_replication(scenario) for scenario in hidden},
        uncoded={scenario.label(): evaluate_uncoded(scenario) for scenario in hidden},
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_coded_memory_search(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "program": payload["program"],
                "strict_post_damage": payload["strict_post_damage"],
                "strict_dense_post_damage": payload["strict_dense_post_damage"],
                "strict_replication_post_damage": payload["strict_replication_post_damage"],
                "strict_uncoded_post_damage": payload["strict_uncoded_post_damage"],
                "max_write_fraction": payload["max_write_fraction"],
                "min_retention": payload["min_retention"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
