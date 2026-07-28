from __future__ import annotations

from typing import Callable

import numpy as np

import lexigen_outer_native

Problem = tuple[np.ndarray, np.ndarray]
Solution = np.ndarray


def numpy_broadcast(problem: Problem) -> Solution:
    left, right = problem
    return np.multiply(left[:, None], right[None, :])


def native_parallel2(problem: Problem) -> Solution:
    left, right = problem
    return lexigen_outer_native.outer(left, right, 2)


def native_parallel4(problem: Problem) -> Solution:
    left, right = problem
    return lexigen_outer_native.outer(left, right, 4)


def native_parallel8(problem: Problem) -> Solution:
    left, right = problem
    return lexigen_outer_native.outer(left, right, 8)


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "numpy_broadcast": numpy_broadcast,
    "native_parallel2": native_parallel2,
    "native_parallel4": native_parallel4,
    "native_parallel8": native_parallel8,
}
