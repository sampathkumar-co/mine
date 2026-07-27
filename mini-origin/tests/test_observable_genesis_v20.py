import numpy as np

from mini_origin.observable_genesis_v20 import (
    DEFAULT_PRIMITIVES,
    Expr,
    accuracy,
    expression_outputs,
    fit_perfect_separator,
    make_dataset,
    make_world_pair,
)


def gram_observable() -> Expr:
    return Expr(
        "sub",
        Expr("mul", Expr.atom("e0"), Expr.atom("e1")),
        Expr("square", Expr.atom("cross")),
    )


def test_default_interface_is_exactly_indistinguishable() -> None:
    radial, rotational, difference = make_world_pair(
        seed=7,
        pair_id=0,
        dimension=5,
        rho=0.73,
        theta=0.81,
        replicates=3,
        noise=0.0,
    )
    assert radial.label != rotational.label
    assert difference <= 1e-12
    for name in DEFAULT_PRIMITIVES:
        assert np.allclose(radial.primitives[name], rotational.primitives[name])


def test_derived_gram_observable_separates_noiseless_pairs() -> None:
    worlds, difference = make_dataset(
        seed=101,
        pairs=24,
        dimensions=(2, 3, 4, 7),
        rho_range=(0.45, 0.98),
        theta_range=(0.15, 1.40),
        replicates=3,
        noise=0.0,
    )
    labels = np.asarray([world.label for world in worlds])
    values = expression_outputs(gram_observable(), worlds)
    separator = fit_perfect_separator(values, labels)
    assert difference <= 1e-12
    assert separator is not None
    assert accuracy(gram_observable(), separator, worlds) == 1.0


def test_observable_transfers_to_unseen_dimensions_with_noise() -> None:
    training, _ = make_dataset(
        seed=202,
        pairs=40,
        dimensions=(2, 3, 4),
        rho_range=(0.48, 0.98),
        theta_range=(0.12, 1.45),
        replicates=6,
        noise=0.0005,
    )
    labels = np.asarray([world.label for world in training])
    expression = gram_observable()
    separator = fit_perfect_separator(expression_outputs(expression, training), labels)
    assert separator is not None

    hidden, _ = make_dataset(
        seed=303,
        pairs=80,
        dimensions=(5, 8, 12),
        rho_range=(0.45, 0.99),
        theta_range=(0.10, 1.50),
        replicates=12,
        noise=0.0015,
    )
    assert accuracy(expression, separator, hidden) >= 0.97


def test_observable_is_not_default_only() -> None:
    expression = gram_observable()
    assert expression.complexity() == 6
    assert expression.primitives() - set(DEFAULT_PRIMITIVES) == {"cross"}
