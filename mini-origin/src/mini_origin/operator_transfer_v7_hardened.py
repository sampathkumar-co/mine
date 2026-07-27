from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import operator_transfer_v7 as base
from .language_v6 import _damage, execute


# Re-exported for tests and research tooling.
Template = base.Template
TemporalScenario = base.TemporalScenario
_skeleton = base._skeleton
instantiate = base.instantiate
hand_temporal_delta = base.hand_temporal_delta


def temporal_score(program, scenario: TemporalScenario) -> float:
    """Baseline-normalized temporal skill; a mean predictor scores zero."""
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
        trace = base._trace_features(history, scenario.dimension)
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
            # Closed source-language programs cannot read target-role signals.
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
        trace = base._trace_features(history, scenario.dimension)
        target = float(np.tanh(final_mapping @ trace))
        cell_prediction = weights[alive] @ trace
        predictions.append(float(np.median(cell_prediction)))
        targets.append(target)

    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    mse = float(np.mean((prediction_array - target_array) ** 2))
    mean_baseline_mse = float(
        np.mean((target_array - float(np.mean(target_array))) ** 2)
    )
    skill = 1.0 - mse / max(mean_baseline_mse, 1e-12)
    return float(np.clip(skill, 0.0, 1.0))


def run_operator_transfer(seed: int = 71):
    # All functions inside the base module resolve this global at call time.
    base.temporal_score = temporal_score
    return base.run_operator_transfer(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_operator_transfer(args.seed)
    payload = result.to_dict()
    payload["metric"] = (
        "1 - model_mse / mean_predictor_mse, clipped to [0,1]"
    )
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
                "metric": payload["metric"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
