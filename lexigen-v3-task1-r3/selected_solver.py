from __future__ import annotations

from candidates import Problem, Solution, pruned_hybrid6

CANDIDATE_NAME = "pruned_hybrid6"


def solve(problem: Problem) -> Solution:
    return pruned_hybrid6(problem)
