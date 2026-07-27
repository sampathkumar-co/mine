from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Callable

import numpy as np

from .observable_genesis_v20 import Expr, Separator, fit_perfect_separator, predict


@dataclass(frozen=True)
class RawWorld:
    label: int
    x: np.ndarray
    y: np.ndarray
    pair_id: int


@dataclass(frozen=True)
class MatrixDiscovery:
    dimension: int
    matrix: np.ndarray
    invariance_error: float
    nonzero_count: int


@dataclass(frozen=True)
class IndexRule:
    name: str
    complexity: int
    predicate: Callable[[int, int, int], int]

    def matrix(self, dimension: int) -> np.ndarray:
        return np.asarray(
            [
                [self.predicate(i, j, dimension) for j in range(dimension)]
                for i in range(dimension)
            ],
            dtype=np.float64,
        )


def _orthonormal_plane(
    rng: np.random.Generator,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, 2)))
    return basis[:, 0], basis[:, 1]


def make_raw_pair(
    seed: int,
    pair_id: int,
    dimension: int,
    rho: float,
    theta: float,
    replicates: int,
    noise: float,
) -> tuple[RawWorld, RawWorld, float]:
    rng = np.random.default_rng(seed)
    u, v = _orthonormal_plane(rng, dimension)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    x0 = math.cos(phase) * u + math.sin(phase) * v
    tangent = -math.sin(phase) * u + math.cos(phase) * v
    radial_sign = -1.0 if rng.random() < 0.5 else 1.0
    radial_y = radial_sign * rho * x0
    rotational_y = rho * (
        math.cos(theta) * x0 + math.sin(theta) * tangent
    )

    radial_x = np.repeat(x0[None, :], replicates, axis=0)
    rotational_x = np.repeat(x0[None, :], replicates, axis=0)
    radial_target = np.repeat(radial_y[None, :], replicates, axis=0)
    rotational_target = np.repeat(rotational_y[None, :], replicates, axis=0)

    if noise > 0.0:
        shared_x_noise = rng.normal(0.0, noise, size=radial_x.shape)
        radial_x = radial_x + shared_x_noise
        rotational_x = rotational_x + shared_x_noise
        radial_target = radial_target + rng.normal(
            0.0, noise, size=radial_target.shape
        )
        rotational_target = rotational_target + rng.normal(
            0.0, noise, size=rotational_target.shape
        )

    default_difference = max(
        abs(float(np.dot(x0, x0)) - float(np.dot(x0, x0))),
        abs(float(np.dot(radial_y, radial_y)) - float(np.dot(rotational_y, rotational_y))),
    )
    return (
        RawWorld(0, radial_x, radial_target, pair_id),
        RawWorld(1, rotational_x, rotational_target, pair_id),
        default_difference,
    )


def make_raw_dataset(
    seed: int,
    pairs: int,
    dimensions: tuple[int, ...],
    rho_range: tuple[float, float],
    theta_range: tuple[float, float],
    replicates: int,
    noise: float,
) -> tuple[list[RawWorld], float]:
    worlds: list[RawWorld] = []
    maximum_default_difference = 0.0
    for pair_id in range(pairs):
        rng = np.random.default_rng(seed + pair_id * 3_571)
        dimension = int(rng.choice(dimensions))
        rho = float(rng.uniform(*rho_range))
        base_theta = float(rng.uniform(*theta_range))
        theta = base_theta if rng.random() < 0.5 else math.pi - base_theta
        radial, rotational, difference = make_raw_pair(
            seed + pair_id * 65_537,
            pair_id,
            dimension,
            rho,
            theta,
            replicates,
            noise,
        )
        worlds.extend((radial, rotational))
        maximum_default_difference = max(maximum_default_difference, difference)
    return worlds, maximum_default_difference


def symmetric_binary_matrices(dimension: int) -> list[np.ndarray]:
    entries = [(i, j) for i in range(dimension) for j in range(i, dimension)]
    matrices: list[np.ndarray] = []
    for bit_pattern in range(1, 1 << len(entries)):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for bit, (i, j) in enumerate(entries):
            if bit_pattern & (1 << bit):
                matrix[i, j] = 1.0
                matrix[j, i] = 1.0
        matrices.append(matrix)
    return matrices


def random_orthogonal(
    rng: np.random.Generator,
    dimension: int,
) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def invariance_error(
    matrix: np.ndarray,
    seed: int,
    trials: int = 12,
) -> float:
    rng = np.random.default_rng(seed)
    denominator = max(float(np.linalg.norm(matrix)), 1e-12)
    errors = []
    for _ in range(trials):
        rotation = random_orthogonal(rng, matrix.shape[0])
        transformed = rotation.T @ matrix @ rotation
        errors.append(float(np.linalg.norm(transformed - matrix)) / denominator)
    return max(errors)


def discover_invariant_matrix(
    dimension: int,
    seed: int,
) -> MatrixDiscovery:
    candidates = []
    for matrix in symmetric_binary_matrices(dimension):
        error = invariance_error(matrix, seed + int(np.sum(matrix)) * 101)
        if error > 1e-10:
            continue
        nonzero = int(np.count_nonzero(matrix))
        candidates.append((nonzero, error, matrix))
    if not candidates:
        raise RuntimeError(f"no invariant matrix found for dimension {dimension}")
    candidates.sort(key=lambda value: (value[0], value[1]))
    nonzero, error, matrix = candidates[0]
    return MatrixDiscovery(dimension, matrix, error, nonzero)


def index_rules() -> list[IndexRule]:
    return [
        IndexRule("always_zero", 1, lambda i, j, d: 0),
        IndexRule("always_one", 1, lambda i, j, d: 1),
        IndexRule("same_index", 2, lambda i, j, d: int(i == j)),
        IndexRule("different_index", 2, lambda i, j, d: int(i != j)),
        IndexRule("adjacent", 3, lambda i, j, d: int(abs(i - j) == 1)),
        IndexRule("cyclic_adjacent", 4, lambda i, j, d: int((i - j) % d in (1, d - 1))),
        IndexRule("same_parity", 3, lambda i, j, d: int((i - j) % 2 == 0)),
        IndexRule("upper_triangle", 2, lambda i, j, d: int(i <= j)),
        IndexRule("lower_triangle", 2, lambda i, j, d: int(i >= j)),
        IndexRule("anti_diagonal", 3, lambda i, j, d: int(i + j == d - 1)),
    ]


def induce_size_polymorphic_rule(
    discoveries: list[MatrixDiscovery],
) -> IndexRule:
    candidates = []
    for rule in index_rules():
        if all(
            np.array_equal(rule.matrix(item.dimension), item.matrix)
            for item in discoveries
        ):
            candidates.append(rule)
    if not candidates:
        raise RuntimeError("no dimension-polymorphic rule explains the discovered matrices")
    candidates.sort(key=lambda rule: (rule.complexity, rule.name))
    return candidates[0]


def measurement_values(
    world: RawWorld,
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "e0": np.einsum("bi,bi->b", world.x, world.x),
        "e1": np.einsum("bi,bi->b", world.y, world.y),
        "invented": np.einsum("bi,ij,bj->b", world.x, matrix, world.y),
    }


def evaluate_expr(expr: Expr, atoms: dict[str, np.ndarray]) -> np.ndarray:
    if expr.op == "atom":
        assert expr.name is not None
        return atoms[expr.name]
    assert expr.left is not None
    left = evaluate_expr(expr.left, atoms)
    if expr.op == "square":
        return left * left
    if expr.op == "abs":
        return np.abs(left)
    assert expr.right is not None
    right = evaluate_expr(expr.right, atoms)
    if expr.op == "add":
        return left + right
    if expr.op == "sub":
        return left - right
    if expr.op == "mul":
        return left * right
    raise ValueError(expr.op)


def expression_outputs(
    expr: Expr,
    worlds: list[RawWorld],
    rule: IndexRule,
) -> np.ndarray:
    values = []
    matrices: dict[int, np.ndarray] = {}
    for world in worlds:
        dimension = world.x.shape[1]
        matrix = matrices.setdefault(dimension, rule.matrix(dimension))
        raw = evaluate_expr(expr, measurement_values(world, matrix))
        values.append(float(np.mean(raw)))
    return np.asarray(values, dtype=np.float64)


def expression_levels(max_complexity: int = 4) -> dict[int, list[Expr]]:
    atoms = [Expr.atom(name) for name in ("e0", "e1", "invented")]
    levels: dict[int, list[Expr]] = {1: atoms}
    seen = {expr.text() for expr in atoms}
    for complexity in range(2, max_complexity + 1):
        candidates: list[Expr] = []
        for child in levels.get(complexity - 1, []):
            candidates.extend((Expr("square", child), Expr("abs", child)))
        for left_complexity in range(1, complexity - 1):
            right_complexity = complexity - 1 - left_complexity
            for left in levels.get(left_complexity, []):
                for right in levels.get(right_complexity, []):
                    if left.text() <= right.text():
                        candidates.extend((Expr("add", left, right), Expr("mul", left, right)))
                    candidates.append(Expr("sub", left, right))
        unique: list[Expr] = []
        for candidate in candidates:
            if candidate.text() in seen:
                continue
            seen.add(candidate.text())
            unique.append(candidate)
        levels[complexity] = unique
    return levels


@dataclass(frozen=True)
class ScalarDiscovery:
    expression: Expr
    separator: Separator
    complexity: int
    lower_complexity_solutions: int


def discover_scalar_observable(
    worlds: list[RawWorld],
    rule: IndexRule,
    max_complexity: int = 4,
) -> ScalarDiscovery:
    labels = np.asarray([world.label for world in worlds], dtype=np.int64)
    levels = expression_levels(max_complexity)
    solution_counts: dict[int, int] = {}
    for complexity in range(1, max_complexity + 1):
        solutions = []
        for expression in levels[complexity]:
            values = expression_outputs(expression, worlds, rule)
            separator = fit_perfect_separator(values, labels)
            if separator is None:
                continue
            solutions.append((separator.training_margin, expression.text(), expression, separator))
        solution_counts[complexity] = len(solutions)
        if solutions:
            solutions.sort(key=lambda value: (value[0], value[1]), reverse=True)
            _, _, expression, separator = solutions[0]
            lower = sum(solution_counts.get(value, 0) for value in range(1, complexity))
            return ScalarDiscovery(expression, separator, complexity, lower)
    raise RuntimeError("no scalar observable found")


def accuracy(
    expression: Expr,
    separator: Separator,
    worlds: list[RawWorld],
    rule: IndexRule,
) -> float:
    labels = np.asarray([world.label for world in worlds], dtype=np.int64)
    outputs = expression_outputs(expression, worlds, rule)
    return float(np.mean(predict(outputs, separator) == labels))


def fit_control(
    training: list[RawWorld],
    hidden: list[RawWorld],
    expression: Expr,
    matrices: dict[int, np.ndarray],
) -> float:
    def outputs(worlds: list[RawWorld]) -> np.ndarray:
        values = []
        for world in worlds:
            matrix = matrices[world.x.shape[1]]
            raw = evaluate_expr(expression, measurement_values(world, matrix))
            values.append(float(np.mean(raw)))
        return np.asarray(values, dtype=np.float64)

    labels = np.asarray([world.label for world in training], dtype=np.int64)
    train_values = outputs(training)
    separator = fit_perfect_separator(train_values, labels)
    if separator is None:
        candidates = sorted(set(float(value) for value in train_values))
        best = (0.0, Separator(float(np.median(train_values)), 1, 0.0))
        for direction in (-1, 1):
            thresholds = [candidates[0] - 1e-9]
            thresholds.extend(
                0.5 * (left + right)
                for left, right in zip(candidates, candidates[1:])
            )
            thresholds.append(candidates[-1] + 1e-9)
            for threshold in thresholds:
                candidate = Separator(threshold, direction, 0.0)
                score = float(np.mean(predict(train_values, candidate) == labels))
                if score > best[0]:
                    best = (score, candidate)
        separator = best[1]
    hidden_labels = np.asarray([world.label for world in hidden], dtype=np.int64)
    return float(np.mean(predict(outputs(hidden), separator) == hidden_labels))


def random_matrix_baseline(
    seed: int,
    training: list[RawWorld],
    hidden: list[RawWorld],
    expression: Expr,
    trials: int = 96,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dimensions = sorted({world.x.shape[1] for world in training + hidden})
    scores = []
    for _ in range(trials):
        matrices = {}
        for dimension in dimensions:
            raw = rng.normal(size=(dimension, dimension))
            matrices[dimension] = 0.5 * (raw + raw.T)
            norm = float(np.linalg.norm(matrices[dimension]))
            matrices[dimension] /= max(norm, 1e-12)
        scores.append(fit_control(training, hidden, expression, matrices))
    return {
        "trials": trials,
        "median": float(np.median(scores)),
        "maximum": max(scores),
    }


def fixed_operator_controls(
    training: list[RawWorld],
    hidden: list[RawWorld],
    expression: Expr,
) -> dict[str, float]:
    dimensions = sorted({world.x.shape[1] for world in training + hidden})
    controls: dict[str, dict[int, np.ndarray]] = {
        "all_ones": {
            dimension: np.ones((dimension, dimension), dtype=np.float64) / dimension
            for dimension in dimensions
        },
        "cyclic_shift": {
            dimension: np.roll(np.eye(dimension), 1, axis=1)
            for dimension in dimensions
        },
        "alternating_diagonal": {
            dimension: np.diag(
                [1.0 if index % 2 == 0 else -1.0 for index in range(dimension)]
            )
            for dimension in dimensions
        },
    }
    return {
        name: fit_control(training, hidden, expression, matrices)
        for name, matrices in controls.items()
    }


def digest(rule: IndexRule, expression: Expr) -> str:
    return hashlib.sha256(
        f"{rule.name}:{expression.text()}".encode("utf-8")
    ).hexdigest()


def run(seed: int = 601) -> dict[str, object]:
    discoveries = [
        discover_invariant_matrix(dimension, seed * 1_000 + dimension)
        for dimension in (2, 3, 4)
    ]
    rule = induce_size_polymorphic_rule(discoveries)

    training, training_default_difference = make_raw_dataset(
        seed * 10_000,
        pairs=84,
        dimensions=(2, 3, 4),
        rho_range=(0.48, 0.98),
        theta_range=(0.12, 1.45),
        replicates=6,
        noise=0.0005,
    )
    scalar = discover_scalar_observable(training, rule, max_complexity=4)
    frozen_digest = digest(rule, scalar.expression)

    hidden, hidden_default_difference = make_raw_dataset(
        seed * 10_000 + 7_000_000,
        pairs=180,
        dimensions=(5, 8, 12, 16),
        rho_range=(0.42, 0.995),
        theta_range=(0.08, 1.52),
        replicates=12,
        noise=0.002,
    )
    hidden_accuracy = accuracy(
        scalar.expression,
        scalar.separator,
        hidden,
        rule,
    )
    controls = fixed_operator_controls(training, hidden, scalar.expression)
    random_control = random_matrix_baseline(
        seed + 77,
        training,
        hidden,
        scalar.expression,
    )

    hidden_invariance = max(
        invariance_error(rule.matrix(dimension), seed + dimension * 19)
        for dimension in (5, 8, 12, 16)
    )
    maximum_control = max(controls.values())
    default_difference = max(training_default_difference, hidden_default_difference)
    candidate_gate = (
        rule.name == "same_index"
        and max(item.invariance_error for item in discoveries) <= 1e-10
        and hidden_invariance <= 1e-10
        and scalar.lower_complexity_solutions == 0
        and hidden_accuracy >= 0.97
        and hidden_accuracy >= maximum_control + 0.15
        and hidden_accuracy >= random_control["median"] + 0.15
        and default_difference <= 1e-10
    )
    return {
        "status": (
            "size_polymorphic_operator_invention_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "the system exhaustively discovers coordinate matrices on small dimensions, induces a "
            "size-polymorphic index rule, verifies rotation invariance, freezes the new cross-time "
            "operator, and only then synthesizes and tests a scalar observable on unseen dimensions; "
            "the discovered same-index contraction is the classical inner product, so this is not a "
            "world breakthrough without genuinely new operators, broader systems and external review"
        ),
        "seed": seed,
        "operator_rule": rule.name,
        "operator_rule_complexity": rule.complexity,
        "training_matrix_discoveries": [
            {
                "dimension": item.dimension,
                "matrix": item.matrix.tolist(),
                "invariance_error": item.invariance_error,
                "nonzero_count": item.nonzero_count,
            }
            for item in discoveries
        ],
        "hidden_invariance_error": hidden_invariance,
        "expression": scalar.expression.text(),
        "expression_complexity": scalar.complexity,
        "lower_complexity_solution_count": scalar.lower_complexity_solutions,
        "frozen_operator_observable_digest": frozen_digest,
        "default_indistinguishability_max_difference": default_difference,
        "hidden_accuracy": hidden_accuracy,
        "fixed_operator_controls": controls,
        "random_matrix_control": random_control,
        "candidate_gate": candidate_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "operator_rule": report["operator_rule"],
                "expression": report["expression"],
                "hidden_accuracy": report["hidden_accuracy"],
                "hidden_invariance_error": report["hidden_invariance_error"],
                "fixed_controls": report["fixed_operator_controls"],
                "random_median": report["random_matrix_control"]["median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
