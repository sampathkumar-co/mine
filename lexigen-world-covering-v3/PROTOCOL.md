# LEXIGEN World Covering Record v3

## Objective

Run one clean, frozen, blind search for a covering design strictly smaller than the April 24, 2026 repository upper bound.

## Target lineage

Before revealing any identity, the selector reproduces the exact frozen v1 target set and the corrected v2 target set. It excludes all six names, then selects three v3 targets using a separately frozen v3 eligibility and score.

v3 limits are `v <= 22`, upper bound `<= 120`, at most 75,000 candidate blocks, at most 6,500 required subsets, and at most 4,500,000 incidence edges. A `(k,t)` pair may occur at most twice.

## Generic engine

The same engine and budget apply to every target:

1. deterministic and seeded randomized greedy construction;
2. redundancy pruning;
3. eight guided fixed-budget repair restarts;
4. a generic restricted CP-SAT model built from incidence-only candidate pooling;
5. a complete CP-SAT feasibility model;
6. universal first-block symmetry breaking only;
7. independent exhaustive verification of serialized blocks.

No selected-target formulas, manually supplied blocks, repository constructions, live lookups, or post-selection changes are permitted.

## Budget

- 180 seconds guided repair per target;
- 180 seconds restricted CP-SAT per target;
- 1,250 seconds full CP-SAT per target;
- four CP-SAT workers;
- exactly one research workflow execution;
- 90-minute hard workflow timeout.

## Acceptance

A candidate must use distinct valid blocks, contain strictly fewer blocks than the frozen upper bound, cover every required subset, and pass the independent verifier.

It is then only a **verified world-record candidate** until compared with current post-snapshot records and independently reviewed. Failure remains failure.
