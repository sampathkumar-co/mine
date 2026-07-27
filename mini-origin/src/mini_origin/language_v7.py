from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Executable local-plasticity language with recurrent state terminals
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

    def terminals(self) -> set[str]:
        if not self.args:
            return {self.op}
        values: set[str] = set()
        for arg in self.args:
            values.update(arg.terminals())
        return values


TERMINALS = (
    "teacher",
    "pred",
    "peer",
    "elig",
    "weight",
    "err_ema",
    "abs_ema",
    "momentum",
    "c0",
    "c1",
    "cm1",
    "c01",
    "c001",
)
UNARY = ("neg", "tanh", "clip", "abs", "softsign")
BINARY = ("add", "mul", "sub", "div")


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
        constants = {
            "c0": 0.0,
            "c1": 1.0,
            "cm1": -1.0,
            "c01": 0.1,
            "c001": 0.01,
        }
        if expr.op in constants:
            return np.full(shape, constants[expr.op], dtype=np.float64)
        return _as_array(context[expr.op], shape)

    if len(expr.args) == 1:
        value = execute(expr.args[0], context)
        if expr.op == "neg":
            return -value
        if expr.op == "tanh":
            return np.tanh(value)
        if expr.op == "clip":
            return np.clip(value, -1.0, 1.0)
        if expr.op == "abs":
            return np.abs(value)
        if expr.op == "softsign":
            return value / (1.0 + np.abs(value))
        raise ValueError(expr.op)

    left = execute(expr.args[0], context)
    right = execute(expr.args[1], context)
    if expr.op == "add":
        return np.clip(left + right, -8.0, 8.0)
    if expr.op == "sub":
        return np.clip(left - right, -8.0, 8.0)
    if expr.op == "mul":
        return np.clip(left * right, -8.0, 8.0)
    if expr.op == "div":
        denominator = np.where(
            np.abs(right) < 0.05,
            np.where(right < 0.0, -0.05, 0.05),
            right,
        )
        return np.clip(left / denominator, -8.0, 8.0)
    raise ValueError(expr.op)


def _unique(expressions: Iterable[Expr]) -> list[Expr]:
    by_text: dict[str, Expr] = {}
    for expression in expressions:
        by_text.setdefault(expression.text(), expression)
    return list(by_text.values())


def primitive_signal_atoms() -> list[Expr]:
    names = (
        "teacher",
        "pred",
        "peer",
        "err_ema",
        "abs_ema",
        "c0",
        "c1",
        "cm1",
        "c01",
    )
    atoms = [terminal(name) for name in names]
    for name in ("teacher", "pred", "peer", "err_ema", "abs_ema"):
        value = terminal(name)
        atoms.extend(unary(op, value) for op in UNARY)
    return _unique(atoms)


def task_specific_programs() -> list[Expr]:
    """Deep templates used only for reusable signal discovery."""
    teacher = terminal("teacher")
    pred = terminal("pred")
    peer = terminal("peer")
    err_ema = terminal("err_ema")
    abs_ema = terminal("abs_ema")
    elig = terminal("elig")
    weight = terminal("weight")
    momentum = terminal("momentum")
    c1 = terminal("c1")
    c01 = terminal("c01")
    c001 = terminal("c001")

    residuals = _unique(
        [
            teacher,
            binary("sub", teacher, pred),
            binary("sub", teacher, peer),
            binary("sub", teacher, err_ema),
            binary("sub", binary("sub", teacher, pred), err_ema),
            binary("sub", binary("sub", teacher, peer), err_ema),
        ]
    )
    scales = [
        binary("add", c01, abs_ema),
        binary("add", c1, abs_ema),
        binary("add", c01, unary("abs", pred)),
        binary("add", c1, unary("abs", weight)),
    ]
    signals: list[Expr] = []
    for residual in residuals:
        signals.extend(
            [
                residual,
                unary("tanh", residual),
                unary("softsign", residual),
            ]
        )
        for scale in scales[:3]:
            signals.append(binary("div", residual, scale))
            signals.append(unary("tanh", binary("div", residual, scale)))
    signals = _unique(signals)

    programs: list[Expr] = []
    decay = binary("mul", c001, weight)
    for signal in signals:
        credit = binary("mul", elig, signal)
        programs.extend(
            [
                credit,
                binary("add", credit, binary("mul", c01, momentum)),
                binary("sub", credit, decay),
                binary(
                    "add",
                    binary("sub", credit, decay),
                    binary("mul", c01, momentum),
                ),
            ]
        )
    return _unique(programs)


def mine_macros(
    per_family: dict[str, list["RankedProgram"]],
    limit: int = 10,
) -> list[Expr]:
    usage: dict[str, set[str]] = {}
    quality: dict[str, list[float]] = {}
    expressions: dict[str, Expr] = {}
    for family, ranked in per_family.items():
        for item in ranked:
            for subtree in item.program.subtrees()[1:]:
                terminals = subtree.terminals()
                if (
                    not subtree.args
                    or subtree.size() < 3
                    or subtree.size() > 13
                    or "elig" in terminals
                    or "weight" in terminals
                    or "momentum" in terminals
                ):
                    continue
                text = subtree.text()
                usage.setdefault(text, set()).add(family)
                quality.setdefault(text, []).append(item.score)
                expressions[text] = subtree

    candidates: list[tuple[float, Expr]] = []
    for text, families in usage.items():
        if len(families) < 2:
            continue
        expression = expressions[text]
        compression_gain = max(0, expression.size() - 1)
        score = (
            3.0 * len(families)
            + float(np.mean(quality[text]))
            + 0.10 * compression_gain
            - 0.025 * expression.size()
        )
        candidates.append((score, expression))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return _unique(expression for _, expression in candidates[:limit])


def deployment_programs(signal_atoms: list[Expr]) -> list[Expr]:
    """All candidates have the same shallow deployment grammar."""
    elig = terminal("elig")
    weight = terminal("weight")
    momentum = terminal("momentum")
    c01 = terminal("c01")
    c001 = terminal("c001")
    decay = binary("mul", c001, weight)
    programs: list[Expr] = []
    for signal in signal_atoms:
        credit = binary("mul", elig, signal)
        programs.extend(
            [
                credit,
                binary("add", credit, binary("mul", c01, momentum)),
                binary("sub", credit, decay),
                binary(
                    "add",
                    binary("sub", credit, decay),
                    binary("mul", c01, momentum),
                ),
            ]
        )
    return _unique(programs)


# ---------------------------------------------------------------------------
# Stateful distributed substrate
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
    outlier_rate: float
    delay: int

    def label(self) -> str:
        return (
            f"{self.family}:d{self.dimension}:c{self.condition:g}:"
            f"drop{self.dropout:.2f}:damage{self.damage:.2f}:"
            f"out{self.outlier_rate:.2f}:delay{self.delay}:s{self.seed}"
        )


@dataclass
class CellState:
    err_ema: np.ndarray
    abs_ema: np.ndarray
    momentum: np.ndarray


def _new_state(cells: int, dimension: int) -> CellState:
    return CellState(
        err_ema=np.zeros(cells, dtype=np.float64),
        abs_ema=np.full(cells, 0.2, dtype=np.float64),
        momentum=np.zeros((cells, dimension), dtype=np.float64),
    )


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q


def _correlated_vector(
    rng: np.random.Generator,
    dimension: int,
    condition: float,
    basis: np.ndarray,
    outlier_rate: float,
) -> np.ndarray:
    scales = np.geomspace(1.0, 1.0 / max(condition, 1.0), dimension)
    value = rng.normal(size=dimension) * np.sqrt(scales)
    value = value @ basis.T
    if rng.random() < outlier_rate:
        value *= float(rng.uniform(3.0, 8.0))
    norm = np.linalg.norm(value)
    return value / max(norm, 1e-12)


def _isotropic_vector(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=dimension)
    return value / max(np.linalg.norm(value), 1e-12)


def _damage(alive: np.ndarray, fraction: float, rng: np.random.Generator) -> None:
    count = min(len(alive) - 1, int(round(len(alive) * fraction)))
    if count > 0:
        alive[rng.choice(len(alive), size=count, replace=False)] = False


def _stateful_update(
    program: Expr,
    weights: np.ndarray,
    state: CellState,
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
    local_feature = feature + rng.normal(0.0, noise, (cells, dimension))
    local_teacher = teacher_value + rng.normal(0.0, noise, cells)
    prediction = np.einsum("cd,cd->c", weights, local_feature)
    peer = float(np.median(prediction[alive]))
    residual = local_teacher - prediction

    context: dict[str, np.ndarray | float] = {
        "teacher": local_teacher[:, None],
        "pred": prediction[:, None],
        "peer": peer,
        "elig": local_feature,
        "weight": weights,
        "err_ema": state.err_ema[:, None],
        "abs_ema": state.abs_ema[:, None],
        "momentum": state.momentum,
    }
    delta = execute(program, context)
    if np.any(visible):
        applied = np.clip(step_size * delta[visible], -0.5, 0.5)
        weights[visible] = np.clip(weights[visible] + applied, -5.0, 5.0)
        state.momentum[visible] = (
            0.78 * state.momentum[visible] + 0.22 * delta[visible]
        )

    state.err_ema[alive] = 0.90 * state.err_ema[alive] + 0.10 * residual[alive]
    state.abs_ema[alive] = (
        0.94 * state.abs_ema[alive] + 0.06 * np.abs(residual[alive])
    )


def _similarity(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    return float(math.exp(-3.0 * mse))


def regression_score(program: Expr, scenario: Scenario) -> float:
    rng = np.random.default_rng(scenario.seed)
    basis = _orthogonal(rng, scenario.dimension)
    target_weight = _isotropic_vector(rng, scenario.dimension)
    weights = rng.normal(0.0, 0.01, (scenario.cells, scenario.dimension))
    state = _new_state(scenario.cells, scenario.dimension)
    alive = np.ones(scenario.cells, dtype=bool)

    for _ in range(scenario.train_steps):
        feature = _correlated_vector(
            rng,
            scenario.dimension,
            scenario.condition,
            basis,
            scenario.outlier_rate,
        )
        teacher = float(target_weight @ feature)
        _stateful_update(
            program,
            weights,
            state,
            alive,
            feature,
            teacher,
            rng,
            scenario.noise,
            scenario.dropout,
            0.060,
        )

    _damage(alive, scenario.damage, rng)
    test = np.stack([_isotropic_vector(rng, scenario.dimension) for _ in range(72)])
    target = test @ target_weight
    prediction = np.median(np.einsum("cd,nd->cn", weights[alive], test), axis=0)
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
        math.sin(2.0 * x + y),
        math.cos(x + 2.0 * y),
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
    state = _new_state(scenario.cells, scenario.dimension)
    alive = np.ones(scenario.cells, dtype=bool)

    for _ in range(scenario.train_steps):
        raw = rng.normal(size=2)
        if rng.random() < scenario.outlier_rate:
            raw *= float(rng.uniform(2.0, 4.0))
        feature = _nonlinear_features(raw, scenario.dimension)
        teacher = math.tanh(
            1.5 * raw[0] * raw[1]
            + 0.45 * math.sin(raw[0])
            - 0.25 * math.cos(2.0 * raw[1])
        )
        _stateful_update(
            program,
            weights,
            state,
            alive,
            feature,
            teacher,
            rng,
            scenario.noise,
            scenario.dropout,
            0.052,
        )

    _damage(alive, scenario.damage, rng)
    raw_test = rng.normal(size=(84, 2))
    test = np.stack([_nonlinear_features(value, scenario.dimension) for value in raw_test])
    target = np.tanh(
        1.5 * raw_test[:, 0] * raw_test[:, 1]
        + 0.45 * np.sin(raw_test[:, 0])
        - 0.25 * np.cos(2.0 * raw_test[:, 1])
    )
    prediction = np.median(np.einsum("cd,nd->cn", weights[alive], test), axis=0)
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
    states = [_new_state(scenario.cells, scenario.dimension) for _ in range(actions)]
    alive = np.ones(scenario.cells, dtype=bool)
    pending: list[tuple[int, np.ndarray, float]] = []

    for step in range(scenario.train_steps):
        feature = _correlated_vector(
            rng,
            scenario.dimension,
            scenario.condition,
            basis,
            scenario.outlier_rate,
        )
        estimates = np.mean(
            np.einsum("cad,d->ca", weights[alive], feature),
            axis=0,
        )
        epsilon = max(0.04, 0.28 * (1.0 - step / scenario.train_steps))
        action = (
            int(rng.integers(actions))
            if rng.random() < epsilon
            else int(np.argmax(estimates))
        )
        rewards = target @ feature
        pending.append(
            (
                action,
                feature.copy(),
                float(rewards[action] + rng.normal(0.0, scenario.noise)),
            )
        )
        if len(pending) > max(0, scenario.delay):
            old_action, old_feature, reward = pending.pop(0)
            _stateful_update(
                program,
                weights[:, old_action],
                states[old_action],
                alive,
                old_feature,
                reward,
                rng,
                scenario.noise,
                scenario.dropout,
                0.032,
            )

    while pending:
        old_action, old_feature, reward = pending.pop(0)
        _stateful_update(
            program,
            weights[:, old_action],
            states[old_action],
            alive,
            old_feature,
            reward,
            rng,
            scenario.noise,
            scenario.dropout,
            0.032,
        )

    _damage(alive, scenario.damage, rng)
    regrets: list[float] = []
    for _ in range(180):
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
    tail = values[: max(1, int(math.ceil(values.size * 0.45)))]
    return float(0.58 * values[0] + 0.27 * np.mean(tail) + 0.15 * np.mean(values))


# ---------------------------------------------------------------------------
# Program search and counterexample-guided language expansion
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
    cache: dict[tuple[str, str], float] | None = None,
) -> list[RankedProgram]:
    cache = cache if cache is not None else {}
    ranked: list[RankedProgram] = []
    for program in programs:
        details: dict[str, float] = {}
        program_text = program.text()
        for scenario in scenarios:
            label = scenario.label()
            key = (program_text, label)
            if key not in cache:
                cache[key] = TASKS[scenario.family](program, scenario)
            details[label] = cache[key]
        score = robust_score(details.values()) - 0.0012 * program.size()
        ranked.append(RankedProgram(program, score, details))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def screening_scenarios(seed: int) -> list[Scenario]:
    """Cheap deterministic semantic screen shared by both search spaces."""
    return [
        Scenario("regression", seed + 901, 4, 12.0, 10, 28, 0.02, 0.15, 0.15, 0.02, 0),
        Scenario("nonlinear", seed + 902, 7, 1.0, 10, 34, 0.02, 0.15, 0.15, 0.02, 0),
        Scenario("bandit", seed + 903, 5, 14.0, 10, 70, 0.025, 0.14, 0.15, 0.02, 1),
    ]


def training_scenarios(seed: int) -> dict[str, list[Scenario]]:
    return {
        "regression": [
            Scenario("regression", seed + 1, 4, 18.0, 20, 85, 0.025, 0.20, 0.30, 0.04, 0)
        ],
        "nonlinear": [
            Scenario("nonlinear", seed + 2, 7, 1.0, 20, 110, 0.025, 0.22, 0.30, 0.05, 0)
        ],
        "bandit": [
            Scenario("bandit", seed + 3, 5, 20.0, 20, 240, 0.035, 0.18, 0.30, 0.04, 2)
        ],
    }


def counterexample_pool(seed: int) -> list[Scenario]:
    return [
        Scenario("regression", seed + 101, 6, 35.0, 28, 120, 0.045, 0.28, 0.45, 0.08, 0),
        Scenario("regression", seed + 102, 8, 55.0, 36, 150, 0.060, 0.34, 0.55, 0.12, 0),
        Scenario("nonlinear", seed + 103, 9, 1.0, 28, 145, 0.045, 0.28, 0.45, 0.08, 0),
        Scenario("nonlinear", seed + 104, 11, 1.0, 36, 180, 0.060, 0.34, 0.55, 0.12, 0),
        Scenario("bandit", seed + 105, 7, 35.0, 28, 340, 0.050, 0.26, 0.45, 0.08, 4),
        Scenario("bandit", seed + 106, 9, 50.0, 36, 430, 0.065, 0.32, 0.55, 0.12, 6),
    ]


def hidden_scenarios(seed: int) -> list[Scenario]:
    return [
        Scenario("regression", seed + 501, 11, 75.0, 48, 200, 0.075, 0.40, 0.65, 0.16, 0),
        Scenario("nonlinear", seed + 502, 14, 1.0, 48, 230, 0.075, 0.40, 0.65, 0.16, 0),
        Scenario("bandit", seed + 503, 12, 70.0, 48, 560, 0.080, 0.38, 0.65, 0.16, 8),
    ]


def _family_best(
    programs: list[Expr],
    scenarios: dict[str, list[Scenario]],
) -> dict[str, list[RankedProgram]]:
    return {
        family: rank_programs(programs, family_scenarios, limit=10)
        for family, family_scenarios in scenarios.items()
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


def _hand_delta() -> Expr:
    return binary(
        "mul",
        terminal("elig"),
        binary("sub", terminal("teacher"), terminal("pred")),
    )


def _hand_normalized_delta() -> Expr:
    residual = binary("sub", terminal("teacher"), terminal("pred"))
    scale = binary("add", terminal("c01"), terminal("abs_ema"))
    return binary("mul", terminal("elig"), binary("div", residual, scale))


def _hand_hebb() -> Expr:
    return binary("mul", terminal("elig"), terminal("teacher"))


def replace_subtree(expr: Expr, target: Expr, replacement: Expr) -> Expr:
    if expr == target:
        return replacement
    if not expr.args:
        return expr
    return Expr(
        expr.op,
        tuple(replace_subtree(arg, target, replacement) for arg in expr.args),
    )


@dataclass
class LanguageResult:
    seed: int
    macros: list[Expr]
    best_program: Expr
    no_library_program: Expr
    hidden_scores: dict[str, float]
    no_library_scores: dict[str, float]
    delta_scores: dict[str, float]
    normalized_delta_scores: dict[str, float]
    hebb_scores: dict[str, float]
    macro_ablation_scores: dict[str, float]
    history: list[dict[str, object]]

    @property
    def strict_hidden(self) -> float:
        return float(min(self.hidden_scores.values()))

    @property
    def no_library_strict(self) -> float:
        return float(min(self.no_library_scores.values()))

    @property
    def normalized_delta_strict(self) -> float:
        return float(min(self.normalized_delta_scores.values()))

    @property
    def macro_ablation_strict(self) -> float:
        return float(min(self.macro_ablation_scores.values()))

    @property
    def candidate_external_result(self) -> bool:
        used = [macro for macro in self.macros if macro.text() in self.best_program.text()]
        return (
            self.strict_hidden >= 0.76
            and self.strict_hidden >= self.no_library_strict + 0.12
            and self.strict_hidden >= 0.95 * self.normalized_delta_strict
            and self.strict_hidden >= self.macro_ablation_strict + 0.08
            and len(used) >= 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "external_breakthrough_candidate"
                if self.candidate_external_result
                else "not_yet"
            ),
            "claim_scope": (
                "counterexample-guided induction of recurrent error-normalisation macros "
                "for one shallow local rule spanning regression, nonlinear learning and "
                "delayed contextual reward under severe cell loss; outside novelty still "
                "requires independent reproduction and peer review"
            ),
            "seed": self.seed,
            "macros": [macro.text() for macro in self.macros],
            "best_program": self.best_program.text(),
            "no_library_program": self.no_library_program.text(),
            "strict_hidden_score": self.strict_hidden,
            "no_library_strict_score": self.no_library_strict,
            "normalized_delta_strict_score": self.normalized_delta_strict,
            "macro_ablation_strict_score": self.macro_ablation_strict,
            "hidden_scores": self.hidden_scores,
            "no_library_hidden_scores": self.no_library_scores,
            "delta_hidden_scores": self.delta_scores,
            "normalized_delta_hidden_scores": self.normalized_delta_scores,
            "hebb_hidden_scores": self.hebb_scores,
            "macro_ablation_hidden_scores": self.macro_ablation_scores,
            "history": self.history,
        }


def run_language_search(seed: int = 71) -> LanguageResult:
    family_scenarios = training_scenarios(seed * 10_000)
    deep_programs = task_specific_programs()
    per_family = _family_best(deep_programs, family_scenarios)
    macros = mine_macros(per_family, limit=10)

    base_signals = primitive_signal_atoms()
    expanded_signals = _unique(base_signals + macros)
    expanded_programs = deployment_programs(expanded_signals)
    no_library_programs = deployment_programs(base_signals)

    curriculum = [
        scenario
        for family_values in family_scenarios.values()
        for scenario in family_values
    ]
    pool = counterexample_pool(seed * 10_000)
    used = {scenario.label() for scenario in curriculum}
    history: list[dict[str, object]] = []
    cache: dict[tuple[str, str], float] = {}

    screen = screening_scenarios(seed * 10_000)
    expanded_shortlist = [
        item.program
        for item in rank_programs(expanded_programs, screen, limit=40, cache=cache)
    ]
    no_library_shortlist = [
        item.program
        for item in rank_programs(no_library_programs, screen, limit=40, cache=cache)
    ]

    best = rank_programs(expanded_shortlist, curriculum, limit=1, cache=cache)[0]
    control = rank_programs(no_library_shortlist, curriculum, limit=1, cache=cache)[0]
    for iteration in range(4):
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
        best = rank_programs(expanded_shortlist, curriculum, limit=1, cache=cache)[0]
        control = rank_programs(no_library_shortlist, curriculum, limit=1, cache=cache)[0]

    hidden = hidden_scenarios(seed * 10_000)
    hidden_scores = evaluate(best.program, hidden)
    no_library_scores = evaluate(control.program, hidden)
    delta_scores = evaluate(_hand_delta(), hidden)
    normalized_scores = evaluate(_hand_normalized_delta(), hidden)
    hebb_scores = evaluate(_hand_hebb(), hidden)

    used_macros = [
        macro for macro in macros if macro.text() in best.program.text()
    ]
    ablated = best.program
    for macro in used_macros:
        ablated = replace_subtree(ablated, macro, terminal("c0"))
    ablation_scores = evaluate(ablated, hidden)

    return LanguageResult(
        seed=seed,
        macros=macros,
        best_program=best.program,
        no_library_program=control.program,
        hidden_scores=hidden_scores,
        no_library_scores=no_library_scores,
        delta_scores=delta_scores,
        normalized_delta_scores=normalized_scores,
        hebb_scores=hebb_scores,
        macro_ablation_scores=ablation_scores,
        history=history,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
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
                "strict_hidden_score": payload["strict_hidden_score"],
                "no_library_strict_score": payload["no_library_strict_score"],
                "normalized_delta_strict_score": payload[
                    "normalized_delta_strict_score"
                ],
                "macro_ablation_strict_score": payload[
                    "macro_ablation_strict_score"
                ],
                "best_program": payload["best_program"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
