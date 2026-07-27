import numpy as np

from mini_origin.resilience_v3 import (
    _robust_score,
    _sample_training_scenarios,
    _validation_scenarios,
)


def test_robust_score_penalizes_single_failure() -> None:
    broad = _robust_score([0.70, 0.72, 0.75, 0.78, 0.80])
    brittle = _robust_score([0.02, 0.92, 0.94, 0.96, 0.98])
    assert broad > brittle


def test_training_scenarios_change_with_generation() -> None:
    first = _sample_training_scenarios(
        np.random.default_rng(3),
        stage_width=20,
        stage_damage=0.20,
        generation=1,
        stage_index=2,
    )
    second = _sample_training_scenarios(
        np.random.default_rng(4),
        stage_width=20,
        stage_damage=0.20,
        generation=2,
        stage_index=2,
    )
    assert first != second
    assert len(first) == 6
    assert len(second) == 6


def test_validation_suite_covers_sizes_and_damage() -> None:
    scenarios = _validation_scenarios(stage_width=36, stage_damage=0.31)
    widths = {scenario.width for scenario in scenarios}
    damage = {scenario.damage for scenario in scenarios}
    assert len(widths) == 3
    assert len(damage) == 2
    assert len(scenarios) == 6
