import numpy as np

from mini_origin.operator_invention_v21 import (
    accuracy,
    discover_invariant_matrix,
    discover_scalar_observable,
    induce_size_polymorphic_rule,
    invariance_error,
    make_raw_dataset,
)


def test_invariant_matrix_search_rediscovers_identity() -> None:
    discoveries = [
        discover_invariant_matrix(dimension, seed=100 + dimension)
        for dimension in (2, 3, 4)
    ]
    for item in discoveries:
        assert np.array_equal(item.matrix, np.eye(item.dimension))
        assert item.invariance_error <= 1e-10


def test_matrix_family_induces_same_index_rule() -> None:
    discoveries = [
        discover_invariant_matrix(dimension, seed=200 + dimension)
        for dimension in (2, 3, 4)
    ]
    rule = induce_size_polymorphic_rule(discoveries)
    assert rule.name == "same_index"
    for dimension in (5, 8, 12):
        assert np.array_equal(rule.matrix(dimension), np.eye(dimension))
        assert invariance_error(rule.matrix(dimension), seed=300 + dimension) <= 1e-10


def test_invented_operator_supports_minimal_scalar_observable() -> None:
    discoveries = [
        discover_invariant_matrix(dimension, seed=400 + dimension)
        for dimension in (2, 3, 4)
    ]
    rule = induce_size_polymorphic_rule(discoveries)
    training, difference = make_raw_dataset(
        seed=500,
        pairs=40,
        dimensions=(2, 3, 4),
        rho_range=(0.48, 0.98),
        theta_range=(0.12, 1.45),
        replicates=6,
        noise=0.0005,
    )
    scalar = discover_scalar_observable(training, rule, max_complexity=4)
    assert difference <= 1e-12
    assert scalar.complexity == 4
    assert scalar.lower_complexity_solutions == 0
    assert "invented" in scalar.expression.primitives()


def test_invented_operator_transfers_to_unseen_dimensions() -> None:
    discoveries = [
        discover_invariant_matrix(dimension, seed=600 + dimension)
        for dimension in (2, 3, 4)
    ]
    rule = induce_size_polymorphic_rule(discoveries)
    training, _ = make_raw_dataset(
        seed=700,
        pairs=48,
        dimensions=(2, 3, 4),
        rho_range=(0.48, 0.98),
        theta_range=(0.12, 1.45),
        replicates=6,
        noise=0.0005,
    )
    scalar = discover_scalar_observable(training, rule, max_complexity=4)
    hidden, _ = make_raw_dataset(
        seed=800,
        pairs=90,
        dimensions=(5, 8, 12, 16),
        rho_range=(0.44, 0.99),
        theta_range=(0.10, 1.50),
        replicates=12,
        noise=0.0015,
    )
    assert accuracy(scalar.expression, scalar.separator, hidden, rule) >= 0.97
