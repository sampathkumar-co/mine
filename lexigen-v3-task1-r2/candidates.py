from __future__ import annotations

from typing import Callable

import numpy as np

Problem = dict[str, object]
Solution = dict[str, list[float]]


def _data(problem: Problem) -> tuple[np.ndarray, np.ndarray, int, float]:
    x0 = np.asarray(problem["x0"], dtype=np.float64)
    scenarios = np.asarray(problem["loss_scenarios"], dtype=np.float64)
    beta = float(problem.get("beta", 0.95))
    kappa = float(problem.get("kappa", 0.1))
    k = int((1.0 - beta) * scenarios.shape[0])
    if x0.ndim != 1 or scenarios.ndim != 2 or scenarios.shape[1] != x0.size:
        raise ValueError("invalid CVaR projection dimensions")
    if k <= 0:
        raise ValueError("top-k count must be positive")
    return x0, scenarios, k, kappa * k


def _oracle(
    scenarios: np.ndarray,
    x: np.ndarray,
    k: int,
) -> tuple[float, np.ndarray, tuple[int, ...]]:
    losses = scenarios @ x
    indices = np.argpartition(losses, losses.size - k)[-k:]
    support = np.sum(scenarios[indices], axis=0, dtype=np.float64)
    signature = tuple(sorted(int(index) for index in indices))
    return float(np.sum(losses[indices], dtype=np.float64)), support, signature


def _solve_nonnegative_dual(
    normals: np.ndarray,
    x0: np.ndarray,
    bound: float,
    dual_tolerance: float,
    zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = normals.shape[0]
    gram = normals @ normals.T
    linear = normals @ x0 - bound
    weights = np.zeros(count, dtype=np.float64)
    passive = np.zeros(count, dtype=bool)
    maximum_iterations = max(100, 12 * count * count)

    for _ in range(maximum_iterations):
        reduced_gradient = linear - gram @ weights
        inactive = ~passive
        if not np.any(inactive):
            return x0 - normals.T @ weights, weights
        inactive_scores = np.where(inactive, reduced_gradient, -np.inf)
        entering = int(np.argmax(inactive_scores))
        if inactive_scores[entering] <= dual_tolerance:
            return x0 - normals.T @ weights, weights
        passive[entering] = True

        while True:
            active_indices = np.flatnonzero(passive)
            trial = np.zeros(count, dtype=np.float64)
            active_gram = gram[np.ix_(active_indices, active_indices)]
            active_linear = linear[active_indices]
            try:
                trial_active = np.linalg.solve(active_gram, active_linear)
            except np.linalg.LinAlgError:
                trial_active = np.linalg.lstsq(active_gram, active_linear, rcond=1e-12)[0]
            trial[active_indices] = trial_active
            if np.all(trial_active > zero_tolerance):
                weights = trial
                break

            nonpositive = trial_active <= zero_tolerance
            current_active = weights[active_indices]
            denominators = current_active[nonpositive] - trial_active[nonpositive]
            valid = denominators > 0.0
            if not np.any(valid):
                passive[active_indices[nonpositive]] = False
                weights[~passive] = 0.0
                continue
            ratios = current_active[nonpositive][valid] / denominators[valid]
            step = float(np.min(ratios))
            weights += step * (trial - weights)
            dropping = passive & (weights <= zero_tolerance)
            passive[dropping] = False
            weights[dropping] = 0.0

    raise RuntimeError("nonnegative active-set dual did not converge")


def _feasibility_polish(
    scenarios: np.ndarray,
    x: np.ndarray,
    k: int,
    target: float,
    maximum_steps: int = 24,
) -> np.ndarray:
    result = np.asarray(x, dtype=np.float64).copy()
    for _ in range(maximum_steps):
        value, support, _ = _oracle(scenarios, result, k)
        if value <= target:
            return result
        norm_squared = float(support @ support)
        if norm_squared <= 0.0:
            return result
        result -= ((value - target) / norm_squared) * support
    return result


def _active_bundle(
    problem: Problem,
    maximum_cuts: int,
    feasibility_margin: float,
    dual_tolerance: float,
    zero_tolerance: float,
    initial_signatures: list[tuple[int, ...]] | None = None,
    initial_normals: list[np.ndarray] | None = None,
) -> Solution:
    x0, scenarios, k, alpha = _data(problem)
    target = alpha - k * feasibility_margin
    x = x0.copy()
    normals = list(initial_normals or [])
    signatures = set(initial_signatures or [])
    if normals:
        x, _ = _solve_nonnegative_dual(
            np.vstack(normals), x0, target, dual_tolerance, zero_tolerance
        )

    for _ in range(maximum_cuts):
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target:
            return {"x_proj": x.tolist()}
        if signature in signatures:
            x = _feasibility_polish(scenarios, x, k, target)
            return {"x_proj": x.tolist()}
        signatures.add(signature)
        normals.append(support)
        x, _ = _solve_nonnegative_dual(
            np.vstack(normals), x0, target, dual_tolerance, zero_tolerance
        )

    x = _feasibility_polish(scenarios, x, k, target)
    return {"x_proj": x.tolist()}


def active_bundle(problem: Problem) -> Solution:
    return _active_bundle(
        problem,
        maximum_cuts=64,
        feasibility_margin=5e-5,
        dual_tolerance=1e-11,
        zero_tolerance=1e-12,
    )


def support_fixed_point(problem: Problem) -> Solution:
    x0, scenarios, k, alpha = _data(problem)
    target = alpha - k * 5e-5
    x = x0.copy()
    previous_signature: tuple[int, ...] | None = None
    for _ in range(32):
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target:
            return {"x_proj": x.tolist()}
        norm_squared = float(support @ support)
        if norm_squared <= 0.0:
            break
        multiplier = max(float((support @ x0 - target) / norm_squared), 0.0)
        candidate = x0 - multiplier * support
        if signature == previous_signature:
            x = candidate
            break
        previous_signature = signature
        x = candidate
    x = _feasibility_polish(scenarios, x, k, target)
    return {"x_proj": x.tolist()}


def hybrid_bundle(problem: Problem) -> Solution:
    x0, scenarios, k, alpha = _data(problem)
    target = alpha - k * 5e-5
    x = x0.copy()
    signatures: list[tuple[int, ...]] = []
    normals: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(12):
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target:
            return {"x_proj": x.tolist()}
        if signature in seen:
            break
        seen.add(signature)
        signatures.append(signature)
        normals.append(support)
        norm_squared = float(support @ support)
        if norm_squared <= 0.0:
            break
        multiplier = max(float((support @ x0 - target) / norm_squared), 0.0)
        x = x0 - multiplier * support

    return _active_bundle(
        problem,
        maximum_cuts=48,
        feasibility_margin=5e-5,
        dual_tolerance=1e-11,
        zero_tolerance=1e-12,
        initial_signatures=signatures,
        initial_normals=normals,
    )


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "active_bundle": active_bundle,
    "support_fixed_point": support_fixed_point,
    "hybrid_bundle": hybrid_bundle,
}
