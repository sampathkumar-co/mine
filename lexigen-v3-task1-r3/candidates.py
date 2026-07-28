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
    dual_tolerance: float = 1e-11,
    zero_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    count = normals.shape[0]
    gram = normals @ normals.T
    linear = normals @ x0 - bound
    weights = np.zeros(count, dtype=np.float64)
    passive = np.zeros(count, dtype=bool)

    for _ in range(max(100, 12 * count * count)):
        reduced_gradient = linear - gram @ weights
        scores = np.where(~passive, reduced_gradient, -np.inf)
        entering = int(np.argmax(scores))
        if scores[entering] <= dual_tolerance:
            return x0 - normals.T @ weights, weights
        passive[entering] = True

        while True:
            active = np.flatnonzero(passive)
            trial = np.zeros(count, dtype=np.float64)
            active_gram = gram[np.ix_(active, active)]
            active_linear = linear[active]
            try:
                trial_active = np.linalg.solve(active_gram, active_linear)
            except np.linalg.LinAlgError:
                trial_active = np.linalg.lstsq(active_gram, active_linear, rcond=1e-12)[0]
            trial[active] = trial_active
            if np.all(trial_active > zero_tolerance):
                weights = trial
                break

            nonpositive = trial_active <= zero_tolerance
            current = weights[active]
            denominators = current[nonpositive] - trial_active[nonpositive]
            valid = denominators > 0.0
            if not np.any(valid):
                passive[active[nonpositive]] = False
                weights[~passive] = 0.0
                continue
            step = float(np.min(current[nonpositive][valid] / denominators[valid]))
            weights += step * (trial - weights)
            dropping = passive & (weights <= zero_tolerance)
            passive[dropping] = False
            weights[dropping] = 0.0

    raise RuntimeError("nonnegative active-set dual did not converge")


def _polish(
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


def _collect_supports(
    scenarios: np.ndarray,
    x0: np.ndarray,
    k: int,
    target: float,
    maximum_supports: int,
) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
    x = x0.copy()
    normals: list[np.ndarray] = []
    signatures: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(maximum_supports):
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target or signature in seen:
            break
        seen.add(signature)
        normals.append(support)
        signatures.append(signature)
        norm_squared = float(support @ support)
        if norm_squared <= 0.0:
            break
        multiplier = max(float((support @ x0 - target) / norm_squared), 0.0)
        x = x0 - multiplier * support
    return normals, signatures


def _bounded_bundle(
    problem: Problem,
    initial_supports: int,
    corrective_cuts: int,
) -> Solution:
    x0, scenarios, k, alpha = _data(problem)
    target = alpha - k * 5e-5
    normals, signatures = _collect_supports(
        scenarios, x0, k, target, initial_supports
    )
    if not normals:
        return {"x_proj": x0.tolist()}
    x, _ = _solve_nonnegative_dual(np.vstack(normals), x0, target)
    current = set(signatures)

    for _ in range(corrective_cuts):
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target:
            return {"x_proj": x.tolist()}
        if signature in current:
            break
        normals.append(support)
        signatures.append(signature)
        current.add(signature)
        x, _ = _solve_nonnegative_dual(np.vstack(normals), x0, target)

    x = _polish(scenarios, x, k, target)
    return {"x_proj": x.tolist()}


def bundle8_plus2(problem: Problem) -> Solution:
    return _bounded_bundle(problem, initial_supports=8, corrective_cuts=2)


def bundle16_plus2(problem: Problem) -> Solution:
    return _bounded_bundle(problem, initial_supports=16, corrective_cuts=2)


def pruned_hybrid6(problem: Problem) -> Solution:
    x0, scenarios, k, alpha = _data(problem)
    target = alpha - k * 5e-5
    normals, signatures = _collect_supports(scenarios, x0, k, target, 6)
    if not normals:
        return {"x_proj": x0.tolist()}
    x, weights = _solve_nonnegative_dual(np.vstack(normals), x0, target)

    for _ in range(32):
        keep = weights > 1e-10
        if np.any(keep):
            normals = [normal for normal, retained in zip(normals, keep) if retained]
            signatures = [signature for signature, retained in zip(signatures, keep) if retained]
        value, support, signature = _oracle(scenarios, x, k)
        if value <= target:
            return {"x_proj": x.tolist()}
        if signature in set(signatures):
            break
        normals.append(support)
        signatures.append(signature)
        x, weights = _solve_nonnegative_dual(np.vstack(normals), x0, target)

    x = _polish(scenarios, x, k, target)
    return {"x_proj": x.tolist()}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "bundle8_plus2": bundle8_plus2,
    "bundle16_plus2": bundle16_plus2,
    "pruned_hybrid6": pruned_hybrid6,
}
