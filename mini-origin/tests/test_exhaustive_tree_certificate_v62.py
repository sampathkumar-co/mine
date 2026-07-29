from mini_origin.exhaustive_tree_certificate_v62 import (
    Metric,
    all_partitions,
    enumerate_frontier,
    retained_queries,
)
from mini_origin.exhaustive_tree_certificate_gate_v62 import (
    scalar_collapse_counterexample,
)


def test_partition_counts() -> None:
    assert len(all_partitions(3)) == 4
    assert len(all_partitions(4)) == 14


def test_dominated_equivalent_query_preserves_frontier() -> None:
    partitions = ((0, 0, 1), (1, 1, 0), (0, 1, 1))
    labels = (0, 1, 1)
    masses = (1, 2, 3)
    costs = ((1, 1, 3), (2, 2, 3), (2, 1, 1))
    full = (1 << 3) - 1
    assert retained_queries(partitions, costs, full, (0, 1, 2)) == (0, 2)
    plain = enumerate_frontier(partitions, labels, masses, costs, False)
    quotient = enumerate_frontier(partitions, labels, masses, costs, True)
    assert plain == quotient
    assert plain and isinstance(plain[0], Metric)


def test_incomparable_scalar_counterexample() -> None:
    assert scalar_collapse_counterexample()["passed"]
