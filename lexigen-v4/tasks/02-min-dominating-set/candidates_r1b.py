from __future__ import annotations

from collections.abc import Sequence

import candidates as _base

Problem = _base.Problem
Solution = _base.Solution
_ORIGINAL_DOMINANCE_REDUCED = _base._dominance_reduced


def _dominance_reduced_r1b(vertices: Sequence[int], masks: Sequence[int]) -> list[int]:
    """Preserve revision-1 reduction semantics, adding only the missing empty-input identity."""
    if not vertices:
        return []
    return _ORIGINAL_DOMINANCE_REDUCED(vertices, masks)


# Candidate functions in the frozen module resolve this global at execution time.
# Patch only the generic helper; candidate compositions, ordering and mechanisms remain unchanged.
_base._dominance_reduced = _dominance_reduced_r1b

CANDIDATES_BY_ARM = _base.CANDIDATES_BY_ARM
