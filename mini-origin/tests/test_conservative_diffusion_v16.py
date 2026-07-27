import numpy as np

from mini_origin import rotating_sketch_v12 as v12
from mini_origin.conservative_diffusion_v16 import (
    DiffusedStatistics,
    WRITE_BUDGET,
    conservative_mix,
    operation_budget,
    random_gossip_control,
    select_mixing_pairs,
)


def test_pairwise_mixing_conserves_statistics() -> None:
    statistics = DiffusedStatistics(
        gram=np.zeros((4, 1, 2, 2)),
        response=np.zeros((4, 1, 2)),
        writes=np.zeros((4, 1), dtype=np.int32),
    )
    statistics.gram[0, 0] = np.array([[2.0, 0.5], [0.5, 1.0]])
    statistics.gram[1, 0] = np.array([[1.0, 0.2], [0.2, 3.0]])
    statistics.response[0, 0] = np.array([2.0, -1.0])
    statistics.response[1, 0] = np.array([-0.5, 3.0])
    gram_before = np.sum(statistics.gram, axis=0)
    response_before = np.sum(statistics.response, axis=0)
    conservative_mix(statistics, 0, [(0, 1)], 0.37)
    assert np.allclose(np.sum(statistics.gram, axis=0), gram_before)
    assert np.allclose(np.sum(statistics.response, axis=0), response_before)


def test_mixing_pairs_respect_physical_write_budget() -> None:
    scenario = v12.SketchScenario(11, 3, 4, 4.0, 5, 8.0, 0.01, 0.50)
    program = random_gossip_control()
    statistics = DiffusedStatistics(
        gram=np.zeros((scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension)),
        response=np.zeros((scenario.cells, scenario.contexts, scenario.dimension)),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
    )
    pairs = select_mixing_pairs(program, statistics, scenario, 0, 0)
    physical_writes = program.injection_count + 2 * len(pairs)
    assert physical_writes <= operation_budget(scenario.cells)
    assert physical_writes / scenario.cells <= WRITE_BUDGET + 1.0 / scenario.cells


def test_random_gossip_is_deterministic() -> None:
    scenario = v12.SketchScenario(13, 3, 4, 4.0, 5, 8.0, 0.01, 0.50)
    program = random_gossip_control()
    statistics = DiffusedStatistics(
        gram=np.zeros((scenario.cells, scenario.contexts, scenario.dimension, scenario.dimension)),
        response=np.zeros((scenario.cells, scenario.contexts, scenario.dimension)),
        writes=np.zeros((scenario.cells, scenario.contexts), dtype=np.int32),
    )
    first = select_mixing_pairs(program, statistics, scenario, 1, 2)
    second = select_mixing_pairs(program, statistics, scenario, 1, 2)
    assert first == second


def test_gate_is_operation_and_dense_relative() -> None:
    source = open(
        "src/mini_origin/conservative_diffusion_v16.py",
        encoding="utf-8",
    ).read()
    assert "self.dense_fraction >= 0.985" in source
    assert "self.strict_post >= self.strict_specialist + 0.025" in source
    assert "self.max_operations <= WRITE_BUDGET" in source
