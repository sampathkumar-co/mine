from __future__ import annotations

from candidates import Problem, Solution, block_lloyd

CANDIDATE_NAME = "block_lloyd"


def solve(problem: Problem) -> Solution:
    return block_lloyd(problem)
