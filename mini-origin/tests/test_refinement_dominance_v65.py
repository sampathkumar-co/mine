from mini_origin import refinement_dominance_certificate_v65 as certificate
from mini_origin import refinement_dominance_gate_v65 as gate


def test_strict_refinement_relation() -> None:
    fine = certificate.children((0, 1, 2), 0b111)
    coarse = certificate.children((0, 0, 1), 0b111)
    assert certificate.partition_refines(fine, coarse)
    assert not certificate.partition_refines(coarse, fine)


def test_refinement_requires_strict_cost_gain() -> None:
    coarse = (0, 0, 1)
    fine = (0, 1, 2)
    allowed = 0b111
    assert not certificate.query_dominates(
        (coarse, fine), ((1, 1, 1), (1, 1, 1)), allowed, 1, 0
    )[0]
    assert not certificate.query_dominates(
        (coarse, fine), ((1, 1, 1), (2, 2, 2)), allowed, 1, 0
    )[0]
    assert certificate.query_dominates(
        (coarse, fine), ((2, 2, 2), (1, 1, 1)), allowed, 1, 0
    ) == (True, "strict-refinement")


def test_counterexamples_are_locked() -> None:
    assert gate.corrected_counterexamples()["passed"]
