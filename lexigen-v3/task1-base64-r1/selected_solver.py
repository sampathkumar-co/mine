from __future__ import annotations

from candidates import Problem, Solution, pybase64_simd


def solve(problem: Problem) -> Solution:
    return pybase64_simd(problem)
