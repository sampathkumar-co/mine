import numpy as np

from mini_origin.coded_memory_v10 import (
    CodeProgram,
    CodeScenario,
    _adversarial_delete,
    evaluate_dense,
    evaluate_matrix,
    evaluate_replication,
    make_code_matrix,
    replication_matrix,
)


def _program() -> CodeProgram:
    return CodeProgram(
        density=0.34,
        systematic=False,
        coefficient_mode="rademacher",
        balanced=True,
        ridge=1e-6,
        learning_rate=0.22,
        seed_salt=17,
    )


def _scenario() -> CodeScenario:
    return CodeScenario(
        seed=101,
        contexts=6,
        dimension=8,
        redundancy=2.7,
        examples_per_context=55,
        noise=0.02,
        damage_fraction=0.50,
    )


def test_sparse_code_is_full_rank_before_damage() -> None:
    matrix = make_code_matrix(_program(), _scenario())
    assert np.linalg.matrix_rank(matrix) == _scenario().contexts
    assert 0.0 < np.count_nonzero(matrix) / matrix.size < 0.60


def test_adversary_preserves_minimum_decodable_row_count() -> None:
    matrix = make_code_matrix(_program(), _scenario())
    alive = _adversarial_delete(matrix, 0.75)
    assert int(np.sum(alive)) >= _scenario().contexts


def test_dense_code_recovers_after_targeted_deletion() -> None:
    result = evaluate_dense(_scenario())
    assert result.post_damage > 0.85
    assert result.retention > 0.90
    assert result.surviving_rank == _scenario().contexts


def test_replication_is_vulnerable_to_adversarial_row_loss() -> None:
    scenario = _scenario()
    replication = evaluate_replication(scenario)
    dense = evaluate_dense(scenario)
    assert dense.post_damage > replication.post_damage + 0.05


def test_learning_is_frozen_after_damage() -> None:
    scenario = _scenario()
    matrix = make_code_matrix(_program(), scenario)
    first = evaluate_matrix(matrix, 1e-6, 0.22, scenario)
    second = evaluate_matrix(matrix, 1e-6, 0.22, scenario)
    assert first == second
