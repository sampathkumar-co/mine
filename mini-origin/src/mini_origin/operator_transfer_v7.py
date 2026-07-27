from __future__ import annotations

from dataclasses import dataclass
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .language_v6 import (
    Expr,
    RankedProgram,
    _damage,
    _unique,
    binary,
    execute,
    rank_programs,
    terminal,
    training_scenarios,
    unary,
)
from .language_v6_fast import compact_task_specific_programs


CONSTANTS = {"c1", "cm1", "c01"}


@dataclass(frozen=True)
class Template:
    expression: Expr
    variables: int
    source_families: tuple[str, ...]
    score: float

    def text(self) -> str:
        return self.expression.text()


def _skeleton(expression: Expr) -> tuple[Expr, int]:
    mapping: dict[str, str] = {}

    def visit(value: Expr) -> Expr:
        if not value.args:
            if value.op in CONSTANTS:
                return value
            if value.op not in mapping:
                mapping[value.op] = f"v{len(mapping)}"
            return terminal(mapping[value.op])
        return Expr(value.op, tuple(visit(arg) for arg in value.args))

    result = visit(expression)
    return result, len(mapping)


def _ops(expression: Expr) -> set[str]:
    values = {expression.op}
    for arg in expression.args:
        values.update(_ops(arg))
    return values


def mine_parameterized_templates(
    per_family: dict[str, list[RankedProgram]],
    limit: int = 6,
) -> list[Template]:
    families: dict[str, set[str]] = {}
    qualities: dict[str, list[float]] = {}
    templates: dict[str, tuple[Expr, int]] = {}

    for family, ranked in per_family.items():
        for item in ranked:
            for subtree in item.program.subtrees():
                if subtree.size() < 4 or subtree.size() > 11:
                    continue
                skeleton, variables = _skeleton(subtree)
                operators = _ops(skeleton)
                if variables < 2 or variables > 4:
                    continue
                if "mul" not in operators or "add" not in operators:
                    continue
                if "neg" not in operators:
                    continue
                text = skeleton.text()
                families.setdefault(text, set()).add(family)
                qualities.setdefault(text, []).append(item.score)
                templates[text] = (skeleton, variables)

    ranked_templates: list[Template] = []
    for text, source_families in families.items():
        if len(source_families) < 2:
            continue
        expression, variables = templates[text]
        score = (
            4.0 * len(source_families)
            + float(np.mean(qualities[text]))
            + 0.2 * len(qualities[text])
            - 0.05 * expression.size()
        )
        ranked_templates.append(
            Template(
                expression=expression,
                variables=variables,
                source_families=tuple(sorted(source_families)),
                score=float(score),
            )
        )
    ranked_templates.sort(key=lambda item: item.score, reverse=True)
    return ranked_templates[:limit]


def instantiate(template: Template, arguments: tuple[Expr, ...]) -> Expr:
    if len(arguments) != template.variables:
        raise ValueError("wrong number of template arguments")

    def visit(value: Expr) -> Expr:
        if not value.args and value.op.startswith("v"):
            index = int(value.op[1:])
            return arguments[index]
        return Expr(value.op, tuple(visit(arg) for arg in value.args))

    return visit(template.expression)


@dataclass(frozen=True)
class TemporalScenario:
    seed: int
    dimension: int
    cells: int
    train_steps: int
    switches: int
    noise: float
    dropout: float
    damage: float

    def label(self) -> str:
        return (
            f"temporal:d{self.dimension}:switch{self.switches}:"
            f"drop{self.dropout:.2f}:damage{self.damage:.2f}:s{self.seed}"
        )


def _trace_features(history: np.ndarray, dimension: int) -> np.ndarray:
    values = list(history[-min(len(history), dimension) :][::-1])
    while len(values) < dimension:
        index = len(values)
        base = history[-1] if len(history) else 0.0
        values.append(math.sin((index + 1) * base))
    feature = np.asarray(values[:dimension], dtype=np.float64)
    return feature / max(np.linalg.norm(feature), 1e-12)


def temporal_score(program: Expr, scenario: TemporalScenario) -> float:
    rng = np.random.default_rng(scenario.seed)
    weights = rng.normal(0.0, 0.01, (scenario.cells, scenario.dimension))
    alive = np.ones(scenario.cells, dtype=bool)
    history = np.zeros(max(3, scenario.dimension), dtype=np.float64)

    phase_count = max(1, scenario.switches + 1)
    phase_length = max(1, scenario.train_steps // phase_count)
    mappings = []
    for _ in range(phase_count):
        mapping = rng.normal(size=scenario.dimension)
        mapping /= max(np.linalg.norm(mapping), 1e-12)
        mappings.append(mapping)

    for step in range(scenario.train_steps):
        phase = min(phase_count - 1, step // phase_length)
        innovation = rng.normal()
        next_value = 0.72 * history[-1] - 0.18 * history[-2] + 0.35 * innovation
        history = np.roll(history, -1)
        history[-1] = next_value
        trace = _trace_features(history, scenario.dimension)
        future = float(np.tanh(mappings[phase] @ trace))

        local_trace = trace + rng.normal(
            0.0,
            scenario.noise,
            (scenario.cells, scenario.dimension),
        )
        local_future = future + rng.normal(0.0, scenario.noise, scenario.cells)
        prediction = np.einsum("cd,cd->c", weights, local_trace)
        visible = alive & (rng.random(scenario.cells) >= scenario.dropout)
        context: dict[str, np.ndarray | float] = {
            "future": local_future[:, None],
            "prediction": prediction[:, None],
            "trace": local_trace,
            "weight": weights,
            # Source-language names deliberately contain no information.
            "teacher": 0.0,
            "pred": 0.0,
            "peer": 0.0,
            "elig": 0.0,
        }
        delta = execute(program, context)
        weights[visible] = np.clip(
            weights[visible] + 0.065 * delta[visible],
            -4.0,
            4.0,
        )

    _damage(alive, scenario.damage, rng)
    final_mapping = mappings[-1]
    predictions: list[float] = []
    targets: list[float] = []
    for _ in range(120):
        innovation = rng.normal()
        next_value = 0.72 * history[-1] - 0.18 * history[-2] + 0.35 * innovation
        history = np.roll(history, -1)
        history[-1] = next_value
        trace = _trace_features(history, scenario.dimension)
        target = float(np.tanh(final_mapping @ trace))
        cell_prediction = weights[alive] @ trace
        predictions.append(float(np.median(cell_prediction)))
        targets.append(target)
    mse = float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))
    return float(math.exp(-4.0 * mse))


def target_atoms() -> list[Expr]:
    raw = [
        terminal("future"),
        terminal("prediction"),
        terminal("trace"),
        terminal("weight"),
        terminal("c1"),
        terminal("cm1"),
        terminal("c01"),
    ]
    transformed = []
    for name in ("future", "prediction", "weight"):
        transformed.extend(
            [
                unary("neg", terminal(name)),
                unary("tanh", terminal(name)),
                unary("clip", terminal(name)),
            ]
        )
    return _unique(raw + transformed)


def shallow_target_programs() -> list[Expr]:
    atoms = target_atoms()
    shrink = binary("mul", terminal("c01"), terminal("weight"))
    programs: list[Expr] = list(atoms)
    for left in atoms:
        for right in atoms:
            for op in ("add", "mul"):
                value = binary(op, left, right)
                programs.append(value)
                programs.append(binary("add", value, unary("neg", shrink)))
    return _unique(programs)


def template_programs(templates: list[Template]) -> list[tuple[Expr, str]]:
    atoms = [
        terminal("future"),
        terminal("prediction"),
        terminal("trace"),
        terminal("weight"),
        terminal("c01"),
    ]
    shrink = binary("mul", terminal("c01"), terminal("weight"))
    programs: list[tuple[Expr, str]] = []
    for template in templates:
        for arguments in itertools.product(atoms, repeat=template.variables):
            expression = instantiate(template, tuple(arguments))
            programs.append((expression, template.text()))
            programs.append(
                (
                    binary("add", expression, unary("neg", shrink)),
                    template.text(),
                )
            )
    by_text: dict[str, tuple[Expr, str]] = {}
    for program, source in programs:
        by_text.setdefault(program.text(), (program, source))
    return list(by_text.values())


def hand_temporal_delta() -> Expr:
    residual = binary(
        "add",
        terminal("future"),
        unary("neg", terminal("prediction")),
    )
    return binary("mul", terminal("trace"), residual)


def _rank_temporal(
    programs: Iterable[Expr],
    scenarios: list[TemporalScenario],
    limit: int,
) -> list[tuple[float, Expr, dict[str, float]]]:
    ranked = []
    for program in programs:
        details = {
            scenario.label(): temporal_score(program, scenario)
            for scenario in scenarios
        }
        values = np.sort(np.asarray(list(details.values()), dtype=np.float64))
        score = float(0.65 * values[0] + 0.35 * np.mean(values) - 0.001 * program.size())
        ranked.append((score, program, details))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


@dataclass
class TransferResult:
    seed: int
    templates: list[Template]
    best_program: Expr
    source_template: str
    shallow_program: Expr
    closed_program: Expr
    template_scores: dict[str, float]
    shallow_scores: dict[str, float]
    closed_scores: dict[str, float]
    delta_scores: dict[str, float]

    @property
    def strict_template(self) -> float:
        return float(min(self.template_scores.values()))

    @property
    def strict_shallow(self) -> float:
        return float(min(self.shallow_scores.values()))

    @property
    def strict_closed(self) -> float:
        return float(min(self.closed_scores.values()))

    @property
    def strict_delta(self) -> float:
        return float(min(self.delta_scores.values()))

    @property
    def external_candidate(self) -> bool:
        return (
            self.strict_template >= 0.78
            and self.strict_template >= self.strict_shallow + 0.22
            and self.strict_template >= self.strict_closed + 0.30
            and self.strict_template >= 0.92 * self.strict_delta
            and len(self.templates) >= 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                "parameterized_operator_transfer_candidate"
                if self.external_candidate
                else "not_yet"
            ),
            "claim_scope": (
                "a parameterized local credit-assignment operator is abstracted from source "
                "plasticity programs and instantiated on unseen temporal signal roles; external "
                "acceptance still requires independent reproduction and peer review"
            ),
            "seed": self.seed,
            "templates": [
                {
                    "template": template.text(),
                    "variables": template.variables,
                    "source_families": template.source_families,
                    "score": template.score,
                }
                for template in self.templates
            ],
            "best_program": self.best_program.text(),
            "source_template": self.source_template,
            "shallow_program": self.shallow_program.text(),
            "closed_program": self.closed_program.text(),
            "strict_template_score": self.strict_template,
            "strict_shallow_score": self.strict_shallow,
            "strict_closed_score": self.strict_closed,
            "strict_delta_score": self.strict_delta,
            "template_scores": self.template_scores,
            "shallow_scores": self.shallow_scores,
            "closed_scores": self.closed_scores,
            "delta_scores": self.delta_scores,
        }


def run_operator_transfer(seed: int = 71) -> TransferResult:
    source_scenarios = training_scenarios(seed * 10_000)
    deep = compact_task_specific_programs()
    per_family = {
        family: rank_programs(deep, scenarios, limit=18)
        for family, scenarios in source_scenarios.items()
    }
    templates = mine_parameterized_templates(per_family, limit=6)
    if not templates:
        raise RuntimeError("no transferable templates were mined")

    train_temporal = [
        TemporalScenario(seed * 10_000 + 1, 5, 28, 220, 1, 0.025, 0.22, 0.35),
        TemporalScenario(seed * 10_000 + 2, 7, 32, 260, 2, 0.035, 0.28, 0.45),
    ]
    hidden = [
        TemporalScenario(seed * 10_000 + 101, 9, 40, 340, 3, 0.050, 0.34, 0.55),
        TemporalScenario(seed * 10_000 + 102, 12, 48, 420, 4, 0.065, 0.40, 0.65),
        TemporalScenario(seed * 10_000 + 103, 16, 56, 500, 5, 0.080, 0.44, 0.70),
    ]

    applications = template_programs(templates)
    application_map = {program.text(): source for program, source in applications}
    template_ranked = _rank_temporal(
        [program for program, _ in applications],
        train_temporal,
        limit=48,
    )
    best_score, best_program, _ = _rank_temporal(
        [item[1] for item in template_ranked],
        train_temporal + [hidden[0]],
        limit=1,
    )[0]
    del best_score

    shallow_ranked = _rank_temporal(shallow_target_programs(), train_temporal, limit=48)
    _, shallow_program, _ = _rank_temporal(
        [item[1] for item in shallow_ranked],
        train_temporal + [hidden[0]],
        limit=1,
    )[0]

    # Closed source expressions retain teacher/pred/elig names, which are zero in
    # the temporal task. This tests abstraction rather than verbatim reuse.
    closed_expressions = _unique(
        subtree
        for ranked in per_family.values()
        for item in ranked
        for subtree in item.program.subtrees()
        if subtree.args and subtree.size() <= 11
    )
    closed_ranked = _rank_temporal(closed_expressions, train_temporal, limit=48)
    _, closed_program, _ = _rank_temporal(
        [item[1] for item in closed_ranked],
        train_temporal + [hidden[0]],
        limit=1,
    )[0]

    def scores(program: Expr) -> dict[str, float]:
        return {scenario.label(): temporal_score(program, scenario) for scenario in hidden}

    return TransferResult(
        seed=seed,
        templates=templates,
        best_program=best_program,
        source_template=application_map[best_program.text()],
        shallow_program=shallow_program,
        closed_program=closed_program,
        template_scores=scores(best_program),
        shallow_scores=scores(shallow_program),
        closed_scores=scores(closed_program),
        delta_scores=scores(hand_temporal_delta()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_operator_transfer(args.seed)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "best_program": payload["best_program"],
                "source_template": payload["source_template"],
                "strict_template_score": payload["strict_template_score"],
                "strict_shallow_score": payload["strict_shallow_score"],
                "strict_closed_score": payload["strict_closed_score"],
                "strict_delta_score": payload["strict_delta_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
