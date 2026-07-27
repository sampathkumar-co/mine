import numpy as np

from mini_origin import rotating_sketch_v12 as v12
from mini_origin.information_routing_v14 import (
    WRITE_LIMIT,
    AdaptiveStatistics,
    RoutingProgram,
    _select,
    evaluate_program,
    hand_information_program,
)


def _scenario() -> v12.SketchScenario:
    return v12.SketchScenario(
        seed=501,
        contexts=3,
        dimension=5,
        redundancy=3.0,
        examples_per_context=20,
        condition=16.0,
        noise=0.02,
        damage_fraction=0.50,
    )


def _statistics(scenario: v12.SketchScenario) -> AdaptiveStatistics:
    return AdaptiveStatistics(
        gram=np.zeros((scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension)),
        response=np.zeros((scenario.cells, scenario.contexts, scenario.dimension)),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
        diagonal=np.zeros((scenario.cells, scenario.contexts, scenario.dimension)),
    )


def test_selector_respects_hard_write_budget() -> None:
    scenario = _scenario()
    statistics = _statistics(scenario)
    feature = np.ones(scenario.dimension) / np.sqrt(scenario.dimension)
    for mode in ("topk", "hybrid", "softmax", "underloaded"):
        program = RoutingProgram(
            density=WRITE_LIMIT,
            mode=mode,
            uncertainty_weight=1.0,
            load_weight=0.2,
            random_weight=0.3,
            selection_ridge=0.02,
            decode_ridge=1e-5,
            seed_salt=17,
        )
        selected = _select(program, statistics, scenario, 0, 0, feature)
        assert 1 <= len(selected) <= int(np.floor(WRITE_LIMIT * scenario.cells))
        assert len(np.unique(selected)) == len(selected)


def test_information_selector_prefers_missing_direction() -> None:
    scenario = _scenario()
    statistics = _statistics(scenario)
    feature = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    statistics.diagonal[:3, 0, 0] = 100.0
    program = RoutingProgram(
        density=WRITE_LIMIT,
        mode="topk",
        uncertainty_weight=1.0,
        load_weight=0.0,
        random_weight=0.0,
        selection_ridge=0.02,
        decode_ridge=1e-5,
        seed_salt=19,
    )
    selected = _select(program, statistics, scenario, 0, 1, feature)
    assert not any(index < 3 for index in selected)


def test_evaluation_is_deterministic_and_bounded() -> None:
    scenario = _scenario()
    program = hand_information_program()
    first = evaluate_program(program, scenario)
    second = evaluate_program(program, scenario)
    assert first == second
    assert 0.0 <= first.post_damage <= 1.0
    assert first.write_fraction <= WRITE_LIMIT
