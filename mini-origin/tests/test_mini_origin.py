import numpy as np

from mini_origin.genome import Genome
from mini_origin.search import EvolutionConfig, evolve
from mini_origin.substrate import CellularSubstrate
from mini_origin.tasks import MemoryTask, RelayTask, RepairTask


def zero_genome(channels: int = 2) -> Genome:
    z = np.zeros((channels, channels))
    return Genome(z, z, z, z, np.zeros(channels), leak=1.0)


def persistent_genome(channels: int = 3) -> Genome:
    eye = np.eye(channels) * 3.0
    zero = np.zeros((channels, channels))
    return Genome(eye, zero, zero, zero, np.zeros(channels), leak=0.7)


def test_zero_rule_erases_state() -> None:
    world = CellularSubstrate(zero_genome(), 5, 5)
    world.state.fill(0.8)
    world.step()
    assert np.allclose(world.state, 0.0)


def test_damage_is_reproducible() -> None:
    world_a = CellularSubstrate(zero_genome(), 8, 8)
    world_b = CellularSubstrate(zero_genome(), 8, 8)
    world_a.state.fill(1.0)
    world_b.state.fill(1.0)
    mask_a = world_a.damage(0.25, np.random.default_rng(42))
    mask_b = world_b.damage(0.25, np.random.default_rng(42))
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(world_a.state, world_b.state)


def test_task_scores_are_bounded() -> None:
    genome = persistent_genome()
    tasks = (
        MemoryTask(size=8, steps=3),
        RelayTask(height=6, width=8, steps=4),
        RepairTask(size=8, settle_steps=2, recovery_steps=2),
    )
    for task in tasks:
        score = task.evaluate(genome, seed=5)
        assert 0.0 <= score <= 1.0


def test_persistent_rule_retains_memory() -> None:
    score = MemoryTask(size=8, steps=5).evaluate(persistent_genome(), seed=9)
    assert score > 0.75


def test_short_evolution_is_deterministic() -> None:
    config = EvolutionConfig(
        population_size=8,
        generations=2,
        elite_count=2,
        channels=2,
        evaluation_seeds=(3,),
        seed=17,
    )
    first = evolve(config, tasks=[MemoryTask(size=6, steps=3)])
    second = evolve(config, tasks=[MemoryTask(size=6, steps=3)])
    assert first.best_fitness == second.best_fitness
    assert first.task_scores == second.task_scores
