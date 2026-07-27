import numpy as np

from mini_origin.adversarial_flat_v15_1 import (
    WRITE_LIMIT,
    dense_program,
    fixed_budget_programs,
    iid_program,
)


def test_all_search_candidates_have_equal_write_budget() -> None:
    programs = fixed_budget_programs(np.random.default_rng(5), count=30)
    assert programs
    assert all(program.density == WRITE_LIMIT for program in programs)


def test_dense_and_sparse_controls_are_separate() -> None:
    assert dense_program().density == 1.0
    assert iid_program().density == WRITE_LIMIT


def test_corrected_gate_uses_dense_relative_recovery() -> None:
    source = open(
        "src/mini_origin/adversarial_flat_v15_1.py",
        encoding="utf-8",
    ).read()
    assert "dense_fraction >= 0.97" in source
    assert "strict_post >= self.strict_iid + 0.025" in source
    assert "strict_post >= self.strict_specialist + 0.020" in source
    assert "strict_post >= 0.86" not in source
