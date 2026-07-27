import numpy as np

from mini_origin.research_v2 import DirectionalGenome
from mini_origin.resilience_v2 import (
    evaluate_resilient_relay,
    genome_from_dict,
    make_obstacle_mask,
)


def test_obstacles_preserve_source_and_destination_columns() -> None:
    mask = make_obstacle_mask(17, height=9, width=16, damage_fraction=0.25)
    assert not np.any(mask[:, 0])
    assert not np.any(mask[:, -1])
    assert np.any(mask[:, 1:-1])


def test_obstacle_generation_is_deterministic() -> None:
    first = make_obstacle_mask(23, 11, 20, 0.20)
    second = make_obstacle_mask(23, 11, 20, 0.20)
    assert np.array_equal(first, second)


def test_genome_round_trip_from_dict() -> None:
    rng = np.random.default_rng(5)
    genome = DirectionalGenome.random(rng)
    restored = genome_from_dict(genome.to_dict())
    assert np.allclose(restored.proposal_weights, genome.proposal_weights)
    assert np.allclose(restored.gate_weights, genome.gate_weights)


def test_resilient_evaluation_is_bounded() -> None:
    rng = np.random.default_rng(9)
    genome = DirectionalGenome.random(rng)
    result = evaluate_resilient_relay(
        genome,
        width=8,
        height=7,
        damage_fraction=0.10,
        seeds=(1,),
    )
    assert 0.0 <= result.score <= 1.0
    assert len(result.case_scores) == 2
