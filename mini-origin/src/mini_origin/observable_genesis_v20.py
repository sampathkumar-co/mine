from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np


DEFAULT_PRIMITIVES = ("e0", "e1")
ALL_PRIMITIVES = (
    "e0",
    "e1",
    "cross",
    "sum_product",
    "l1_delta",
    "max_product",
)


@dataclass(frozen=True)
class ObservableWorld:
    label: int
    primitives: dict[str, np.ndarray]
    pair_id: int


@dataclass(frozen=True)
class Expr:
    op: str
    left: "Expr | None" = None
    right: "Expr | None" = None
    name: str | None = None

    @staticmethod
    def atom(name: str) -> "Expr":
        return Expr("atom", name=name)

    def complexity(self) -> int:
        if self.op == "atom":
            return 1
        if self.right is None:
            assert self.left is not None
            return 1 + self.left.complexity()
        assert self.left is not None
        return 1 + self.left.complexity() + self.right.complexity()

    def primitives(self) -> frozenset[str]:
        if self.op == "atom":
            assert self.name is not None
            return frozenset((self.name,))
        assert self.left is not None
        result = set(self.left.primitives())
        if self.right is not None:
            result.update(self.right.primitives())
        return frozenset(result)

    def text(self) -> str:
        if self.op == "atom":
            assert self.name is not None
            return self.name
        assert self.left is not None
        if self.op == "square":
            return f"square({self.left.text()})"
        if self.op == "abs":
            return f"abs({self.left.text()})"
        assert self.right is not None
        symbol = {"add": "+", "sub": "-", "mul": "*"}[self.op]
        return f"({self.left.text()}{symbol}{self.right.text()})"


def _normalised_plane(rng: np.random.Generator, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, 2)))
    return basis[:, 0], basis[:, 1]


def _primitive_arrays(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    dimension = x.shape[1]
    return {
        "e0": np.sum(x * x, axis=1),
        "e1": np.sum(y * y, axis=1),
        "cross": np.sum(x * y, axis=1),
        "sum_product": np.sum(x, axis=1) * np.sum(y, axis=1) / max(1, dimension),
        "l1_delta": np.sum(np.abs(x - y), axis=1) / math.sqrt(max(1, dimension)),
        "max_product": np.max(np.abs(x), axis=1) * np.max(np.abs(y), axis=1),
    }


def make_world_pair(
    seed: int,
    pair_id: int,
    dimension: int,
    rho: float,
    theta: float,
    replicates: int,
    noise: float,
) -> tuple[ObservableWorld, ObservableWorld, float]:
    rng = np.random.default_rng(seed)
    u, v = _normalised_plane(rng, dimension)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    x0 = math.cos(phase) * u + math.sin(phase) * v
    tangent = -math.sin(phase) * u + math.cos(phase) * v
    radial_sign = -1.0 if rng.random() < 0.5 else 1.0

    radial_y = radial_sign * rho * x0
    rotational_y = rho * (math.cos(theta) * x0 + math.sin(theta) * tangent)

    x_radial = np.repeat(x0[None, :], replicates, axis=0)
    x_rotational = np.repeat(x0[None, :], replicates, axis=0)
    y_radial = np.repeat(radial_y[None, :], replicates, axis=0)
    y_rotational = np.repeat(rotational_y[None, :], replicates, axis=0)

    if noise > 0.0:
        x_noise = rng.normal(0.0, noise, size=x_radial.shape)
        radial_noise = rng.normal(0.0, noise, size=y_radial.shape)
        rotational_noise = rng.normal(0.0, noise, size=y_rotational.shape)
        x_radial = x_radial + x_noise
        x_rotational = x_rotational + x_noise
        y_radial = y_radial + radial_noise
        y_rotational = y_rotational + rotational_noise

    radial = ObservableWorld(0, _primitive_arrays(x_radial, y_radial), pair_id)
    rotational = ObservableWorld(1, _primitive_arrays(x_rotational, y_rotational), pair_id)

    exact_radial = _primitive_arrays(x0[None, :], radial_y[None, :])
    exact_rotational = _primitive_arrays(x0[None, :], rotational_y[None, :])
    default_difference = max(
        float(np.max(np.abs(exact_radial[name] - exact_rotational[name])))
        for name in DEFAULT_PRIMITIVES
    )
    return radial, rotational, default_difference


def make_dataset(
    seed: int,
    pairs: int,
    dimensions: tuple[int, ...],
    rho_range: tuple[float, float],
    theta_range: tuple[float, float],
    replicates: int,
    noise: float,
) -> tuple[list[ObservableWorld], float]:
    worlds: list[ObservableWorld] = []
    maximum_default_difference = 0.0
    for pair_id in range(pairs):
        rng = np.random.default_rng(seed + pair_id * 7_919)
        dimension = int(rng.choice(dimensions))
        rho = float(rng.uniform(*rho_range))
        base_theta = float(rng.uniform(*theta_range))
        theta = base_theta if rng.random() < 0.5 else math.pi - base_theta
        radial, rotational, default_difference = make_world_pair(
            seed + pair_id * 104_729,
            pair_id,
            dimension,
            rho,
            theta,
            replicates,
            noise,
        )
        worlds.extend((radial, rotational))
        maximum_default_difference = max(maximum_default_difference, default_difference)
    return worlds, maximum_default_difference


def evaluate_expression_array(expr: Expr, atoms: dict[str, np.ndarray]) -> np.ndarray:
    if expr.op == "atom":
        assert expr.name is not None
        return atoms[expr.name]
    assert expr.left is not None
    left = evaluate_expression_array(expr.left, atoms)
    if expr.op == "square":
        return left * left
    if expr.op == "abs":
        return np.abs(left)
    assert expr.right is not None
    right = evaluate_expression_array(expr.right, atoms)
    if expr.op == "add":
        return left + right
    if expr.op == "sub":
        return left - right
    if expr.op == "mul":
        return left * right
    raise ValueError(expr.op)


def expression_outputs(expr: Expr, worlds: list[ObservableWorld]) -> np.ndarray:
    values = []
    for world in worlds:
        raw = evaluate_expression_array(expr, world.primitives)
        value = float(np.mean(raw))
        if not math.isfinite(value):
            value = math.nan
        values.append(value)
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class Separator:
    threshold: float
    direction: int
    training_margin: float


def fit_perfect_separator(values: np.ndarray, labels: np.ndarray) -> Separator | None:
    if not np.all(np.isfinite(values)):
        return None
    zero = values[labels == 0]
    one = values[labels == 1]
    positive_gap = float(np.min(one) - np.max(zero))
    negative_gap = float(np.min(zero) - np.max(one))
    if positive_gap <= 0.0 and negative_gap <= 0.0:
        return None
    if positive_gap >= negative_gap:
        threshold = 0.5 * (float(np.min(one)) + float(np.max(zero)))
        direction = 1
        gap = positive_gap
    else:
        threshold = 0.5 * (float(np.min(zero)) + float(np.max(one)))
        direction = -1
        gap = negative_gap
    scale = float(np.std(values)) + 1e-12
    return Separator(threshold, direction, gap / scale)


def predict(values: np.ndarray, separator: Separator) -> np.ndarray:
    return (
        separator.direction * values
        > separator.direction * separator.threshold
    ).astype(np.int64)


def semantic_key(values: np.ndarray) -> bytes:
    if not np.all(np.isfinite(values)):
        return b"invalid"
    scale = float(np.max(np.abs(values)))
    normalised = values if scale <= 1e-12 else values / scale
    return np.round(normalised, 8).tobytes()


def enumerate_expressions(
    worlds: list[ObservableWorld],
    max_complexity: int = 6,
    per_level_cap: int = 12_000,
) -> tuple[dict[int, list[Expr]], dict[str, np.ndarray]]:
    outputs: dict[str, np.ndarray] = {}
    by_complexity: dict[int, list[Expr]] = {1: []}
    seen: set[bytes] = set()

    for name in ALL_PRIMITIVES:
        expr = Expr.atom(name)
        values = expression_outputs(expr, worlds)
        key = semantic_key(values)
        if key not in seen:
            seen.add(key)
            outputs[expr.text()] = values
            by_complexity[1].append(expr)

    for complexity in range(2, max_complexity + 1):
        candidates: list[Expr] = []
        for child in by_complexity.get(complexity - 1, []):
            candidates.append(Expr("square", left=child))
            candidates.append(Expr("abs", left=child))

        for left_complexity in range(1, complexity - 1):
            right_complexity = complexity - 1 - left_complexity
            if right_complexity < 1:
                continue
            left_values = by_complexity.get(left_complexity, [])
            right_values = by_complexity.get(right_complexity, [])
            for left in left_values:
                for right in right_values:
                    if left.text() <= right.text():
                        candidates.append(Expr("add", left, right))
                        candidates.append(Expr("mul", left, right))
                    candidates.append(Expr("sub", left, right))

        accepted: list[Expr] = []
        for expr in candidates:
            try:
                values = expression_outputs(expr, worlds)
            except (FloatingPointError, OverflowError, ValueError):
                continue
            if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e14:
                continue
            key = semantic_key(values)
            if key in seen:
                continue
            seen.add(key)
            outputs[expr.text()] = values
            accepted.append(expr)
            if len(accepted) >= per_level_cap:
                break
        by_complexity[complexity] = accepted
    return by_complexity, outputs


@dataclass(frozen=True)
class SynthesisedObservable:
    expression: Expr
    separator: Separator
    first_solution_complexity: int
    lower_complexity_solutions: int


def synthesise_observable(
    worlds: list[ObservableWorld],
    max_complexity: int = 6,
) -> tuple[SynthesisedObservable | None, dict[int, int], dict[str, np.ndarray]]:
    labels = np.asarray([world.label for world in worlds], dtype=np.int64)
    by_complexity, outputs = enumerate_expressions(worlds, max_complexity=max_complexity)
    solutions_by_complexity: dict[int, int] = {}

    for complexity in range(1, max_complexity + 1):
        solutions: list[tuple[float, int, str, Expr, Separator]] = []
        for expr in by_complexity.get(complexity, []):
            separator = fit_perfect_separator(outputs[expr.text()], labels)
            if separator is None:
                continue
            new_primitive_count = len(expr.primitives() - set(DEFAULT_PRIMITIVES))
            solutions.append(
                (
                    separator.training_margin,
                    -new_primitive_count,
                    expr.text(),
                    expr,
                    separator,
                )
            )
        solutions_by_complexity[complexity] = len(solutions)
        if solutions:
            solutions.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            _, _, _, expression, separator = solutions[0]
            lower = sum(
                solutions_by_complexity.get(value, 0)
                for value in range(1, complexity)
            )
            return (
                SynthesisedObservable(expression, separator, complexity, lower),
                solutions_by_complexity,
                outputs,
            )
    return None, solutions_by_complexity, outputs


def accuracy(expr: Expr, separator: Separator, worlds: list[ObservableWorld]) -> float:
    labels = np.asarray([world.label for world in worlds], dtype=np.int64)
    values = expression_outputs(expr, worlds)
    return float(np.mean(predict(values, separator) == labels))


def best_atomic_accuracy(
    training: list[ObservableWorld],
    hidden: list[ObservableWorld],
) -> tuple[str, float]:
    train_labels = np.asarray([world.label for world in training], dtype=np.int64)
    hidden_labels = np.asarray([world.label for world in hidden], dtype=np.int64)
    best_name = ""
    best_accuracy = 0.0
    for name in ALL_PRIMITIVES:
        expr = Expr.atom(name)
        train_values = expression_outputs(expr, training)
        candidates = sorted(set(float(value) for value in train_values))
        thresholds = [candidates[0] - 1e-9]
        thresholds.extend(
            0.5 * (left + right)
            for left, right in zip(candidates, candidates[1:])
        )
        thresholds.append(candidates[-1] + 1e-9)
        for direction in (-1, 1):
            for threshold in thresholds:
                separator = Separator(threshold, direction, 0.0)
                predictions = predict(train_values, separator)
                train_accuracy = float(np.mean(predictions == train_labels))
                if train_accuracy < 0.80:
                    continue
                hidden_accuracy = float(
                    np.mean(
                        predict(expression_outputs(expr, hidden), separator)
                        == hidden_labels
                    )
                )
                if hidden_accuracy > best_accuracy:
                    best_accuracy = hidden_accuracy
                    best_name = name
    return best_name, best_accuracy


def polynomial_control(
    training: list[ObservableWorld],
    hidden: list[ObservableWorld],
) -> float:
    def design(worlds: list[ObservableWorld]) -> np.ndarray:
        base = np.stack(
            [
                np.asarray(
                    [float(np.mean(world.primitives[name])) for name in ALL_PRIMITIVES]
                )
                for world in worlds
            ]
        )
        columns = [np.ones(len(worlds))]
        columns.extend(base[:, index] for index in range(base.shape[1]))
        for left in range(base.shape[1]):
            for right in range(left, base.shape[1]):
                columns.append(base[:, left] * base[:, right])
        return np.stack(columns, axis=1)

    train_x = design(training)
    hidden_x = design(hidden)
    train_y = np.asarray([world.label for world in training], dtype=np.float64) * 2.0 - 1.0
    ridge = 1e-4 * np.eye(train_x.shape[1])
    weights = np.linalg.solve(train_x.T @ train_x + ridge, train_x.T @ train_y)
    predictions = (hidden_x @ weights > 0.0).astype(np.int64)
    hidden_labels = np.asarray([world.label for world in hidden], dtype=np.int64)
    return float(np.mean(predictions == hidden_labels))


def random_formula_baseline(
    seed: int,
    training: list[ObservableWorld],
    hidden: list[ObservableWorld],
    complexity: int,
    trials: int = 256,
) -> dict[str, float]:
    labels = np.asarray([world.label for world in training], dtype=np.int64)
    by_complexity, outputs = enumerate_expressions(training, max_complexity=complexity)
    expressions = list(by_complexity.get(complexity, []))
    rng = random.Random(seed)
    if len(expressions) > trials:
        expressions = rng.sample(expressions, trials)
    accuracies: list[float] = []
    for expr in expressions:
        separator = fit_perfect_separator(outputs[expr.text()], labels)
        if separator is None:
            # Fit a median threshold without using hidden labels.
            values = outputs[expr.text()]
            separator = Separator(float(np.median(values)), 1, 0.0)
        accuracies.append(accuracy(expr, separator, hidden))
    return {
        "trials": len(accuracies),
        "median": float(np.median(accuracies)) if accuracies else 0.0,
        "maximum": max(accuracies, default=0.0),
    }


def expression_digest(expr: Expr) -> str:
    return hashlib.sha256(expr.text().encode("utf-8")).hexdigest()


def run(seed: int = 501) -> dict[str, object]:
    training, training_default_difference = make_dataset(
        seed * 10_000,
        pairs=72,
        dimensions=(2, 3, 4),
        rho_range=(0.48, 0.98),
        theta_range=(0.12, 1.45),
        replicates=6,
        noise=0.0005,
    )
    synthesised, solution_counts, _ = synthesise_observable(training, max_complexity=6)
    if synthesised is None:
        return {
            "status": "not_yet",
            "seed": seed,
            "reason": "no separating observable found",
            "solution_counts": solution_counts,
        }

    frozen_expression = synthesised.expression
    frozen_separator = synthesised.separator
    frozen_digest = expression_digest(frozen_expression)

    hidden, hidden_default_difference = make_dataset(
        seed * 10_000 + 5_000_000,
        pairs=160,
        dimensions=(5, 8, 12, 16),
        rho_range=(0.42, 0.995),
        theta_range=(0.08, 1.52),
        replicates=12,
        noise=0.002,
    )
    hidden_accuracy = accuracy(frozen_expression, frozen_separator, hidden)
    atomic_name, atomic_accuracy = best_atomic_accuracy(training, hidden)
    polynomial_accuracy = polynomial_control(training, hidden)
    random_baseline = random_formula_baseline(
        seed + 91,
        training,
        hidden,
        synthesised.first_solution_complexity,
    )

    default_certificate = max(
        training_default_difference,
        hidden_default_difference,
    )
    new_primitives = sorted(
        frozen_expression.primitives() - set(DEFAULT_PRIMITIVES)
    )
    candidate_gate = (
        default_certificate <= 1e-10
        and synthesised.lower_complexity_solutions == 0
        and bool(new_primitives)
        and hidden_accuracy >= 0.97
        and hidden_accuracy >= atomic_accuracy + 0.10
        and hidden_accuracy >= random_baseline["median"] + 0.15
    )
    return {
        "status": (
            "proof_carrying_observable_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "counterexample-certified synthesis of the smallest executable observable that "
            "separates mechanisms indistinguishable under the original measurement interface; "
            "the prototype uses linear dynamical worlds and a fixed arithmetic grammar, so an "
            "external breakthrough claim additionally requires broader physical systems, stronger "
            "measurement-design controls, independent implementation and peer review"
        ),
        "seed": seed,
        "expression": frozen_expression.text(),
        "expression_digest": frozen_digest,
        "complexity": synthesised.first_solution_complexity,
        "new_primitives": new_primitives,
        "training_margin": frozen_separator.training_margin,
        "separator": {
            "threshold": frozen_separator.threshold,
            "direction": frozen_separator.direction,
        },
        "default_indistinguishability_max_difference": default_certificate,
        "certificate_pair_count": 72 + 160,
        "lower_complexity_solution_count": synthesised.lower_complexity_solutions,
        "solutions_by_complexity": solution_counts,
        "hidden_accuracy": hidden_accuracy,
        "best_atomic_primitive": atomic_name,
        "best_atomic_accuracy": atomic_accuracy,
        "degree_two_polynomial_accuracy": polynomial_accuracy,
        "random_equal_complexity": random_baseline,
        "candidate_gate": candidate_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "expression": report.get("expression"),
                "complexity": report.get("complexity"),
                "hidden_accuracy": report.get("hidden_accuracy"),
                "best_atomic_accuracy": report.get("best_atomic_accuracy"),
                "degree_two_polynomial_accuracy": report.get("degree_two_polynomial_accuracy"),
                "random_median": report.get("random_equal_complexity", {}).get("median"),
                "default_difference": report.get("default_indistinguishability_max_difference"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
