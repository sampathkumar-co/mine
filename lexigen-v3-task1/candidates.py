from __future__ import annotations

from typing import Callable

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import minimize

Problem = dict[str, object]
Solution = dict[str, list[float]]


def _data(problem: Problem) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    x0 = np.asarray(problem["x0"], dtype=np.float64)
    scenarios = np.asarray(problem["loss_scenarios"], dtype=np.float64)
    beta = float(problem.get("beta", 0.95))
    kappa = float(problem.get("kappa", 0.1))
    k = int((1.0 - beta) * scenarios.shape[0])
    if x0.ndim != 1 or scenarios.ndim != 2 or scenarios.shape[1] != x0.size:
        raise ValueError("invalid CVaR projection dimensions")
    if k <= 0:
        raise ValueError("top-k count must be positive")
    return x0, scenarios, k, kappa, kappa * k


def _topk_support(scenarios: np.ndarray, x: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    losses = scenarios @ x
    indices = np.argpartition(losses, losses.size - k)[-k:]
    support = np.sum(scenarios[indices], axis=0, dtype=np.float64)
    return float(np.sum(losses[indices], dtype=np.float64)), support


def _polish_feasibility(
    scenarios: np.ndarray,
    x: np.ndarray,
    k: int,
    alpha: float,
    target_average_margin: float = 5e-5,
    maximum_steps: int = 20,
) -> np.ndarray:
    target = alpha - k * target_average_margin
    result = np.asarray(x, dtype=np.float64).copy()
    for _ in range(maximum_steps):
        value, support = _topk_support(scenarios, result, k)
        if value <= target:
            break
        norm_squared = float(support @ support)
        if norm_squared <= 0.0:
            break
        result -= ((value - target) / norm_squared) * support
    return result


def _osqp(problem: Problem, eps_abs: float, eps_rel: float, max_iter: int) -> Solution:
    x0, scenarios, k, kappa, alpha = _data(problem)
    initial_value, _ = _topk_support(scenarios, x0, k)
    if initial_value <= alpha:
        return {"x_proj": x0.tolist()}

    scenario_count, dimension = scenarios.shape
    variable_count = dimension + 1 + scenario_count
    quadratic = sparse.diags(
        np.concatenate((np.full(dimension, 2.0), np.zeros(1 + scenario_count))),
        format="csc",
    )
    linear = np.zeros(variable_count, dtype=np.float64)
    linear[:dimension] = -2.0 * x0
    scenario_constraints = sparse.hstack(
        (
            sparse.csc_matrix(scenarios),
            sparse.csc_matrix(-np.ones((scenario_count, 1), dtype=np.float64)),
            -sparse.eye(scenario_count, format="csc"),
        ),
        format="csc",
    )
    nonnegative_excess = sparse.hstack(
        (
            sparse.csc_matrix((scenario_count, dimension + 1)),
            sparse.eye(scenario_count, format="csc"),
        ),
        format="csc",
    )
    cvar_row = sparse.csc_matrix(
        np.concatenate((
            np.zeros(dimension, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.full(scenario_count, 1.0 / k, dtype=np.float64),
        ))[None, :]
    )
    constraint_matrix = sparse.vstack(
        (scenario_constraints, nonnegative_excess, cvar_row),
        format="csc",
    )
    lower = np.concatenate((
        np.full(scenario_count, -np.inf),
        np.zeros(scenario_count),
        np.full(1, -np.inf),
    ))
    upper = np.concatenate((
        np.zeros(scenario_count),
        np.full(scenario_count, np.inf),
        np.asarray([kappa], dtype=np.float64),
    ))

    solver = osqp.OSQP()
    solver.setup(
        P=quadratic,
        q=linear,
        A=constraint_matrix,
        l=lower,
        u=upper,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polishing=True,
        verbose=False,
        adaptive_rho=True,
        check_termination=25,
        scaled_termination=True,
    )
    result = solver.solve(raise_error=False)
    if result.x is None or result.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    projected = _polish_feasibility(scenarios, result.x[:dimension], k, alpha)
    return {"x_proj": projected.tolist()}


def osqp_strict(problem: Problem) -> Solution:
    return _osqp(problem, eps_abs=2e-6, eps_rel=2e-6, max_iter=30000)


def osqp_balanced(problem: Problem) -> Solution:
    return _osqp(problem, eps_abs=1e-5, eps_rel=1e-5, max_iter=15000)


def cutting_plane(problem: Problem) -> Solution:
    x0, scenarios, k, _kappa, alpha = _data(problem)
    value, support = _topk_support(scenarios, x0, k)
    if value <= alpha:
        return {"x_proj": x0.tolist()}

    cuts: list[np.ndarray] = [support]
    dual = np.zeros(1, dtype=np.float64)
    projected = x0.copy()
    for _ in range(64):
        matrix = np.vstack(cuts)
        gram = matrix @ matrix.T
        violation_at_x0 = matrix @ x0 - alpha
        if dual.size != matrix.shape[0]:
            dual = np.pad(dual, (0, matrix.shape[0] - dual.size))
        result = minimize(
            fun=lambda weights: 0.5 * float(weights @ gram @ weights)
            - float(violation_at_x0 @ weights),
            x0=dual,
            jac=lambda weights: gram @ weights - violation_at_x0,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * matrix.shape[0],
            options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 1000},
        )
        dual = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        projected = x0 - matrix.T @ dual
        value, support = _topk_support(scenarios, projected, k)
        if value <= alpha - k * 5e-5:
            break
        normalised = support / max(float(np.linalg.norm(support)), 1e-300)
        duplicate = any(
            float(np.linalg.norm(normalised - cut / max(float(np.linalg.norm(cut)), 1e-300)))
            < 1e-10
            for cut in cuts
        )
        if duplicate:
            break
        cuts.append(support)

    projected = _polish_feasibility(scenarios, projected, k, alpha)
    return {"x_proj": projected.tolist()}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "osqp_strict": osqp_strict,
    "osqp_balanced": osqp_balanced,
    "cutting_plane": cutting_plane,
}
