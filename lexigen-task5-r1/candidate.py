from __future__ import annotations

import numpy as np

import lexigen_outer_native

Problem = tuple[np.ndarray, np.ndarray]
Solution = np.ndarray


def solve(problem: Problem) -> Solution:
    left, right = problem
    return lexigen_outer_native.outer(left, right, 8)
