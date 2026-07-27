from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Tiny executable language
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()

    def text(self) -> str:
        if not self.args:
            return self.op
        if len(self.args) == 1:
            return f"{self.op}({self.args[0].text()})"
        return f"{self.op}({self.args[0].text()},{self.args[1].text()})"

    def size(self) -> int:
        return 1 + sum(arg.size() for arg in self.args)

    def subtrees(self) -> tuple["Expr", ...]:
        values = [self]
        for arg in self.args:
            values.extend(arg.subtrees())
        return tuple(values)


TERMINALS = ("teacher", "pred", "peer", "elig", "weight", "c1", "cm1", "c01")
UNARY = ("neg", "tanh", "clip")
BINARY = ("add", "mul")


def terminal(name: str) -> Expr:
    return Expr(name)


def unary(op: str, value: Expr) -> Expr:
    return Expr(op, (value,))


def binary(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(op, (left, right))


def _as_array(value: np.ndarray | float, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == shape:
        return array
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape == (shape[0], 1):
        return np.broadcast_to(array, shape)
    raise ValueError(f"cannot broadcast {array.shape} to {shape}")


def execute(expr: Expr, context: dict[str, np.ndarray | float]) -> np.ndarray:
    weight = np.asarray(context["weight"], dtype=np.float64)
    shape = weight.shape
    if not expr.args:
        if expr.op == "c1":
            return np.ones(shape, dtype=np.float64)
        if expr.op == "cm1":
            return -np.ones(shape, dtype=np.float64)
        if expr.op == "c01":
            return np.full(shape, 0.1, dtype=np.float64)
        return _as_array(context[expr.op], shape)

    if len(expr.args) == 1:
        value = execute(expr.args[0], context)
        if expr.op == "neg":
            return -value
        if expr.op == "tanh":
            return np.tanh(value)
        if expr.op == "clip":
            return np.clip(value, -1.0, 1.0)
        raise ValueError(expr.op)

    left = execute(expr.args[0], context)
    right = execute(expr.args[1], context)
    if expr.op == "add":
        return np.clip(left + right, -6.0, 6.0)
    if expr.op == "mul":
        return np.clip(left * right, -6.0, 6.0)
    raise ValueError(expr.op)


def _unique(expressions: Iterable[Expr]) -> list[Expr]:
    by_text: dict[str, Expr] = {}
    for expression in expressions:
        by_text.setdefault(expression.text(), expression)
    return list(by_text.values())


def base_atoms() -> list[Expr]:
    raw = [terminal(name) for name in TERMINALS]
    transformed: list[Expr] = []
    for name in ("teacher", "pred", "peer", "weight"):
        value = terminal(name)
        transformed.extend(unary(op, value) for op in UNARY)
    return _unique(raw + transformed)


def task_specific_programs() -> list[Expr]:
    """Depth-three programs used only to discover reusable subexpressions."""
    atoms = base_atoms()
    inner = [
        binary(op, left, right)
        for op in BINARY
        for left in atoms
        for right in atoms
    ]
    elig = terminal("elig")
    weight = terminal("weight")
    shrink = binary("mul", terminal("c01"), weight)
    programs: list[Expr] = []
    for value in inner:
        credit = binary("mul", elig, value)
        programs.append(credit)
        programs.append(binary("add", credit, unary("neg", shrink)))
    programs.extend(atoms)
    return _unique(programs)


def shallow_programs(atoms: list[Expr]) -> list[Expr]:
    """Equal-depth search space; learned macros count as one atom."""
    values: list[Expr] = list(atoms)
    for op in BINARY:
        for left in atoms:
            for right in atoms:
                values.append(binary(op, left, right))
    shrink = binary("mul", terminal("c01"), terminal("weight"))
    for value in list(values):
        values.append(binary("add", value, unary("neg", shrink)))
    return _unique(values)


# ---------------------------------------------------------------------------
# Distributed online-learning substrates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    family: str
    seed: int
    dimension: int
    condition: float
    cells: int
    train_steps: int
    noise: float
    dropout: float
    damage: float

    def label(self) -> str:
        return (
            f"{self.family}:d{self.dimension}:c{self.condition:g}:"
            f"drop{self.dropout:.2f}:damage{self.damage:.2f}:s{self.seed}"
        )


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q


def _correlated_vector(
    rng: np.random.Generator,
    dimension: int,
    condition: float,
    basis: np.ndarray,
) -> np.ndarray:
    scales = np.geomspace(1.0, 1.0 / max(condition, 1.0), dimension)
    value = rng.normal(size=dimension) * np.sqrt(scales)
    value = value @ basis.T
    norm = np.linalg.norm(value)
    return value / max(norm, 1e-12)


def _isotropic_vector(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=dimension)
    return value / max(np.linalg.norm(value), 1e-12)


def _damage(alive: np.ndarray, fraction: float, rng: np.random.Generator) -> None:
    count = min(len(alive) - 1, int(round(len(alive) * fraction)))
    if count > 0:
        alive[rng.choice(len(alive), size=count, replace=False)] = False


def _update_cells(
    program: Expr,
    weights: np.ndarray,
    alive: np.ndarray,
    feature: np.ndarray,
    teacher_value: float,
    rng: np.random.Generator,
    noise: float,
    dropout: float,
    step_size: float,
) -> None:
    cells, dimension = weights.shape
    visible = alive & (rng.random(cells) >= dropout)
    if not np.any(visible):
        return
    local_feature = feature + rng.normal(0.0, noise, (cells, dimension))
    local_teacher = teacher_value + rng.normal(0.0, noise, cells)
    prediction = np.einsum("cd,cd->c", weights, local_feature)
    peer = float(np.median(prediction[alive]))
    context: dict[str, np.ndarray | float] = {
        "teacher": local_teacher[:, None],
        "pred": prediction[:, None],
        "peer": peer,
        "elig": local_feature,
        "weight": weights,
    }
    delta = execute(program, context)
    weights[visible] = np.clip(
        weights[visible] + step_size * delta[visible],
        -4.0,
        4.0,
    )


def _similarity(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return float(math.exp(-3.0 * mse))


def regression_score(program: Expr, scenario: Scenario) -> float:
    rng = np.random.default_rng(scenario.seed)
    basis = _orthogonal(rng, scenario.dimension)
    target_weight = _isotropic_vector(rng, scenario.dimension)
    weights = rng.normal(0.0, 0.01, (scenario.cells, scenario.dimension))
    alive = np.ones(scenario.cells, dtype=bool)

    for _ in range(scenario.train_steps):
        feature = _correlated_vector(
            rng,
            scenario.dimension,
            scenario.condition,
            basis,
        )
        teacher_value = float(target_weight @ feature)
        _update_cells(
            program,
            weights,
            alive,
            feature,
            teacher_value,
            rng,
            scenario.noise,
            scenario.dropout,
            step_size=0.075,
        )

    _damage(alive, scenario.damage, rng)
    test = np.stack(
        [_isotropic_vector(rng, scenario.dimension) for _ in range(64)]
    )
    target = test @ target_weight
    cell_predictions = np.einsum("cd,nd->cn", weights[alive], test)
    prediction = np.median(cell_predictions, axis=0)
    return _similarity(prediction, target)


def _nonlinear_features(raw: np.ndarray, dimension: int) -> np.ndarray:
    x, y = float(raw[0]), float(raw[1])
    values = [
        x,
        y,
        x * y,
        math.sin(x),
        math.sin(y),
        math.cos(x - y),
        x * x - y * y,
        1.0,
    ]
    while len(values) < dimension:
        index = len(values)
        values.append(math.sin((index + 1) * x + (index + 2) * y))
    feature = np.asarray(values[:dimension], dtype=np.float64)
    return feature / max(np.linalg.norm(feature), 1e-12)


def nonlinear_score(program: Expr, scenario: Scenario) -> float:
    rng = np.random.default_rng(scenario.seed)
    weights = rng.normal(0.0, 0.01, (scenario.cells, scenario.dimension))
    alive = np.ones(scenario.cells, dtype=bool)

    for _ in range(scenario.train_steps):
        raw = rng.normal(size=2)
        feature = _nonlinear_features(raw, scenario.dimension)
        teacher_value = math.tanh(1.8 * raw[0] * raw[1] + 0.4 * math.sin(raw[0]))
        _update_cells(
            program,
            weights,
            alive,
            feature,
            teacher_value,
            rng,
            scenario.noise,
            scenario.dropout,
            step_size=0.065,
        )

    _damage(alive, scenario.damage, rng)
    raw_test = rng.normal(size=(72, 2))
    test = np.stack(
        [_nonlinear_features(value, scenario.dimension) for value in raw_test]
    )
    target = np.tanh(
        1.8 * raw_test[:, 0] * raw_test[:, 1]
        + 0.4 * np.sin(raw_test[:, 0])
    )
    cell_predictions = np.einsum("cd,nd->cn", weights[alive], test)
    prediction = np.median(cell_predictions, axis=0)
    return _similarity(prediction, target)


def bandit_score(program: Expr, scenario: Scenario) -> float:
    rng = np.random.default_rng(scenario.seed)
    actions = 2
    target = rng.normal(size=(actions, scenario.dimension))
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    basis = _orthogonal(rng, scenario.dimension)
    weights = rng.normal(
        0.0,
        0.01,
        (scenario.cells, actions, scenario.dimension),
    )
    alive = np.ones(scenario.cells, dtype=bool)
    traces = np.zeros((actions, scenario.dimension), dtype=np.float64)

    for step in range(scenario.train_steps):
        feature = _correlated_vector(
            rng,
            scenario.dimension,
            scenario.condition,
            basis,
        )
        estimates = np.mean(
            np.einsum("cad,d->ca", weights[alive], feature),
            axis=0,
        )
        epsilon = max(0.05, 0.25 * (1.0 - step / scenario.train_steps))
        action = (
            int(rng.integers(actions))
            if rng.random() < epsilon
            else int(np.argmax(estimates))
        )
        rewards = target @ feature
        reward = float(rewards[action] + rng.normal(0.0, scenario.noise))
        traces *= 0.65
        traces[action] += feature

        prediction = np.einsum("cd,d->c", weights[:, action], feature)
        peer = float(np.median(prediction[alive]))
        visible = alive & (rng.random(scenario.cells) >= scenario.dropout)
        context: dict[str, np.ndarray | float] = {
            "teacher": (reward + rng.normal(0.0, scenario.noise, scenario.cells))[:, None],
            "pred": prediction[:, None],
            "peer": peer,
            "elig": np.broadcast_to(
                traces[action],
                (scenario.cells, scenario.dimension),
            ),
            "weight": weights[:, action],
        }
        delta = execute(program, context)
        weights[visible, action] = np.clip(
            weights[visible, action] + 0.035 * delta[visible],
            -4.0,
            4.0,
        )

    _damage(alive, scenario.damage, rng)
    regrets: list[float] = []
    for _ in range(160):
        feature = _isotropic_vector(rng, scenario.dimension)
        estimates = np.mean(
            np.einsum("cad,d->ca", weights[alive], feature),
            axis=0,
        )
        action = int(np.argmax(estimates))
        rewards = target @ feature
        regrets.append(float(np.max(rewards) - rewards[action]))
    return float(math.exp(-3.5 * float(np.mean(regrets))))


TASKS: dict[str, Callable[[Expr, Scenario], float]] = {
    "regression": regression_score,
    "nonlinear": nonlinear_score,
    "bandit": bandit_score,
}


def evaluate(program: Expr, scenarios: Iterable[Scenario]) -> dict[str, float]:
    return {
        scenario.label(): TASKS[scenario.family](program, scenario)
        for scenario in scenarios
    }


def robust_score(scores: Iterable[float]) -> float:
    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    if values.size == 0:
        return 0.0
    tail = values[: max(1, int(math.ceil(values.size * 0.4)))]
    return float(0.55 * values[0] + 0.25 * np.mean(tail) + 0.20 * np.mean(values))


# ---------------------------------------------------------------------------
# Search, library induction and counterexample loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedProgram:
    program: Expr
    score: float
    details: dict[str, float]


def rank_programs(
    programs: Iterable[Expr],
    scenarios: list[Scenario],
    limit: int,
) -> list[RankedProgram]:
    ranked: list[RankedProgram] = []
    for program in programs:
        details = evaluate(program, scenarios)
        score = robust_score(details.values()) - 0.0015 * program.size()
        ranked.append(RankedProgram(program, score, details))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def mine_macros(
    per_family: dict[str, list[RankedProgram]],
    limit: int = 8,
) -> list[Expr]:
    usage: dict[str, set[str]] = {}
    quality: dict[str, list[float]] = {}
    expressions: dict[str, Expr] = {}
    for family, ranked in per_family.items():
        for item in ranked:
            for subtree in item.program.subtrees()[1:]:
                if not subtree.args or subtree.size() > 7:
                    continue
                text = subtree.text()
                usage.setdefault(text, set()).add(family)
                quality.setdefault(text, []).append(item.score)
                expressions[text] = subtree

    candidates: list[tuple[float, Expr]] = []
    for text, families in usage.items():
        expression = expressions[text]
        transfer = len(families)
        if transfer < 2:
            continue
        score = (
            2.5 * transfer
            + float(np.mean(quality[text]))
            - 0.04 * expression.size()
        )
        candidates.append((score, expression))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return _unique(expression for _, expression in candidates[:limit])


def training_scenarios(seed: int) -> dict[str, list[Scenario]]:
    return {
        "regression": [Scenario("regression", seed + 1, 4, 14.0, 24, 90, 0.025, 0.22, 0.35)],
        "nonlinear": [Scenario("nonlinear", seed + 2, 6, 1.0, 24, 110, 0.025, 0.24, 0.35)],
        "bandit": [Scenario("bandit", seed + 3, 5, 16.0, 24, 260, 0.035, 0.20, 0.35)],
    }


def counterexample_pool(seed: int) -> list[Scenario]:
    return [
        Scenario("regression", seed + 101, 6, 28.0, 32, 130, 0.045, 0.30, 0.50),
        Scenario("regression", seed + 102, 8, 42.0, 40, 160, 0.060, 0.36, 0.60),
        Scenario("nonlinear", seed + 103, 8, 1.0, 32, 150, 0.045, 0.30, 0.50),
        Scenario("nonlinear", seed + 104, 10, 1.0, 40, 180, 0.060, 0.36, 0.60),
        Scenario("bandit", seed + 105, 7, 28.0, 32, 380, 0.050, 0.28, 0.50),
        Scenario("bandit", seed + 106, 9, 40.0, 40, 460, 0.065, 0.34, 0.60),
    ]


def hidden_scenarios(seed: int) -> list[Scenario]:
    return [
        Scenario("regression", seed + 501, 10, 55.0, 48, 210, 0.075, 0.40, 0.65),
        Scenario("nonlinear", seed + 502, 12, 1.0, 48, 220, 0.075, 0.40, 0.65),
        Scenario("bandit", seed + 503, 11, 55.0, 48, 560, 0.080, 0.38, 0.65),
    ]


def _hand_delta() -> Expr:
    residual = binary("add", terminal("teacher"), unary("neg", terminal("pred")))
    return binary("mul", terminal("elig"), residual)


def _hand_hebb() -> Expr:
    return binary("mul", terminal("elig"), terminal("teacher"))


def _family_best(
    programs: list[Expr],
    family_scenarios: dict[str, list[Scenario]],
) -> dict[str, list[RankedProgram]]:
    return {
        family: rank_programs(programs, scenarios, limit=14)
        for family, scenarios in family_scenarios.items()
    }


def _select_counterexample(
    program: Expr,
    pool: list[Scenario],
    used: set[str],
) -> Scenario | None:
    candidates = [scenario for scenario in pool if scenario.label() not in used]
    if not candidates:
        return None
    scored = [
        (TASKS[scenario.family](program, scenario), scenario)
        for scenario in candidates
    ]
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


@dataclass
class LanguageResult:
    seed: int
    macros: list[Expr]
    best_program: Expr
    no_library_program: Expr
    hidden_scores: dict[str, float]
    no_library_hidden_scores: dict[str, float]
    delta_hidden_scores: dict[str, float]
    hebb_hidden_scores: dict[str, float]
    search_history: list[dict[str, object]]

    @property
    def strict_hidden(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def no_library_strict(self) -> float:
        return float(min(self.no_library_hidden_scores.values()))

    @property
    def delta_strict(self) -> float:
        return float(min(self.delta_hidden_scores.values()))

    @property
    def candidate_external_result(self) -> bool:
        return (
            self.strict_hidden >= 0.72
            and self.strict_hidden >= self.no_library_strict + 0.16
            and self.strict_hidden >= 0.92 * self.delta_strict
            and len(self.macros) >= 1
            and any(
                macro.text() in self.best_program.text()
                for macro in self.macros
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "external_breakthrough_candidate"
                if self.candidate_external_result
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided library induction expands a local-plasticity DSL and "
                "synthesizes one shallow program that transfers across online regression, "
                "nonlinear random-feature learning and delayed contextual reward under cell loss; "
                "external novelty still requires independent reproduction and peer review"
            ),
            "seed": self.seed,
            "macros": [macro.text() for macro in self.macros],
            "best_program": self.best_program.text(),
            "no_library_program": self.no_library_program.text(),
            "strict_hidden_score": self.strict_hidden,
            "no_library_strict_score": self.no_library_strict,
            "delta_strict_score": self.delta_strict,
            "hidden_scores": self.hidden_scores,
            "no_library_hidden_scores": self.no_library_hidden_scores,
            "delta_hidden_scores": self.delta_hidden_scores,
            "hebb_hidden_scores": self.hebb_hidden_scores,
            "search_history": self.search_history,
        }


def run_language_search(seed: int = 61) -> LanguageResult:
    family_scenarios = training_scenarios(seed * 10_000)
    deep = task_specific_programs()
    per_family = _family_best(deep, family_scenarios)
    macros = mine_macros(per_family, limit=8)

    base = base_atoms()
    expanded_atoms = _unique(base + macros)
    expanded_programs = shallow_programs(expanded_atoms)
    no_library_programs = shallow_programs(base)

    curriculum = [scenario for values in family_scenarios.values() for scenario in values]
    pool = counterexample_pool(seed * 10_000)
    used = {scenario.label() for scenario in curriculum}
    history: list[dict[str, object]] = []

    best = rank_programs(expanded_programs, curriculum, limit=1)[0]
    control = rank_programs(no_library_programs, curriculum, limit=1)[0]
    for iteration in range(3):
        counterexample = _select_counterexample(best.program, pool, used)
        history.append(
            {
                "iteration": iteration,
                "curriculum_size": len(curriculum),
                "best_program": best.program.text(),
                "best_score": best.score,
                "no_library_program": control.program.text(),
                "no_library_score": control.score,
                "counterexample": (
                    counterexample.label() if counterexample is not None else None
                ),
            }
        )
        if counterexample is None:
            break
        curriculum.append(counterexample)
        used.add(counterexample.label())
        best = rank_programs(expanded_programs, curriculum, limit=1)[0]
        control = rank_programs(no_library_programs, curriculum, limit=1)[0]

    hidden = hidden_scenarios(seed * 10_000)
    hidden_scores = evaluate(best.program, hidden)
    no_library_scores = evaluate(control.program, hidden)
    delta_scores = evaluate(_hand_delta(), hidden)
    hebb_scores = evaluate(_hand_hebb(), hidden)
    return LanguageResult(
        seed=seed,
        macros=macros,
        best_program=best.program,
        no_library_program=control.program,
        hidden_scores=hidden_scores,
        no_library_hidden_scores=no_library_scores,
        delta_hidden_scores=delta_scores,
        hebb_hidden_scores=hebb_scores,
        search_history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_language_search(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "best_program": payload["best_program"],
                "macros": payload["macros"],
                "strict_hidden_score": payload["strict_hidden_score"],
                "no_library_strict_score": payload["no_library_strict_score"],
                "delta_strict_score": payload["delta_strict_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
