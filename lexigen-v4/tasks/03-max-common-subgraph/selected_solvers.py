from __future__ import annotations

from collections.abc import Callable

from candidates import CANDIDATES_BY_ARM, Problem, Solution

SELECTED_NAMES = {
    "v4_full": "v4_bit_frontier_closed",
    "v4_no_transfer": "no_transfer_bit_frontier_risk",
    "random_search": "random_sparse_early",
    "template_synthesis": "template_bit_parallel",
    "v3_compatible": "v3_dtype_specialization",
}

SELECTED_SOLVERS: dict[str, tuple[str, Callable[[Problem], Solution]]] = {
    arm: (name, CANDIDATES_BY_ARM[arm][name])
    for arm, name in SELECTED_NAMES.items()
}

if set(SELECTED_SOLVERS) != set(SELECTED_NAMES):
    raise RuntimeError("selected solver arm mismatch")

__all__ = ["SELECTED_NAMES", "SELECTED_SOLVERS"]
