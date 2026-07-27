import numpy as np

from mini_origin.competitive_v4 import (
    CompetitiveGenome,
    CompetitiveSubstrate,
    evaluate_competitive_relay,
    hand_flood_baseline,
)


def test_hand_flood_routes_around_damage() -> None:
    result = evaluate_competitive_relay(
        hand_flood_baseline(),
        width=24,
        height=13,
        damage_fraction=0.25,
        seeds=(101, 103),
    )
    assert result.score > 0.90


def test_zero_signal_rule_does_not_fake_bipolar_relay() -> None:
    zero = CompetitiveGenome(
        magnitude_gains=np.zeros(5),
        direction_bias=np.zeros(5),
        temperature=1.0,
        signal_scale=0.25,
        inertia=0.0,
    )
    result = evaluate_competitive_relay(
        zero,
        width=12,
        height=8,
        damage_fraction=0.15,
        seeds=(7,),
    )
    assert result.score < 0.20


def test_competitive_substrate_keeps_dead_cells_zero() -> None:
    genome = hand_flood_baseline()
    world = CompetitiveSubstrate(genome, 7, 10)
    initial = np.zeros((7, 10))
    initial[:, 0] = 0.9
    dead = np.zeros_like(initial, dtype=bool)
    dead[2:5, 4] = True
    world.reset(initial)
    for _ in range(8):
        world.step(dead)
    assert np.allclose(world.state[dead], 0.0)
