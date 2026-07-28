# LEXIGEN World Covering Record v2

## Objective

Test whether a frozen generic discovery engine can construct a valid covering design with strictly fewer blocks than the upper bound recorded in the April 24, 2026 La Jolla Coverings Repository snapshot.

This is an isolated successor to v1. It does not rerun or reinterpret v1. It deterministically skips the three v1-ranked targets and selects the next three eligible targets before revealing their identities.

## Frozen external snapshot

- Zenodo record: `10.5281/zenodo.19735294`
- File: `coverdata.json`
- Expected MD5: `b2c626b07f216aac830d344eff5ad523`

The workflow aborts if the downloaded bytes do not match this commitment.

## Blind target selection

The solver, verifier, workflow, dependency lock, and protocol are committed before snapshot access. Eligibility is fixed:

- `10 <= v <= 22`
- `4 <= k <= min(10, v-2)`
- `3 <= t <= min(5, k-1)`
- upper minus lower bound is at least 2
- upper bound is at most 100
- `C(v,k) <= 60,000`
- `C(v,t) <= 5,000`
- `C(v,k) * C(k,t) <= 3,500,000`

Eligible instances are ranked by a frozen opportunity score using gap, age, upper bound, and incidence complexity. A deterministic pair-diversity cap allows at most two selected instances with the same `(k,t)`. The first three are reserved as the v1 slice; v2 uses positions four through six. No replacement is permitted after selection.

## Generic engine

The identical algorithm and limits apply to all three targets:

1. Enumerate every candidate `k`-block and required `t`-subset.
2. Build exact incidence lists.
3. Generate full greedy coverings under deterministic and seeded randomized orderings.
4. Remove redundant blocks.
5. Search directly at `upper_bound - 1` blocks with a generic stochastic replacement engine driven only by uncovered subsets and exact coverage counts.
6. Use the strongest generic construction as a CP-SAT hint.
7. Solve a full set-cover model with a fixed upper limit and minimization objective.
8. Apply only the universally valid symmetry break requiring the lexicographically first block.
9. Independently recompute every required subset from serialized blocks.

No target-specific formulas, manually supplied blocks, live-record lookup for selected parameters, repository constructions, or post-selection changes are allowed.

## Compute budget

- Three targets.
- Six local-search restarts and at most 120 local-search seconds per target.
- At most 1,400 CP-SAT seconds per target.
- Four CP-SAT workers.
- One workflow execution.
- Hard workflow timeout: 90 minutes.

## Acceptance and claim boundary

A result is a verified world-record candidate only if it contains distinct valid blocks, uses strictly fewer blocks than the frozen upper bound, covers every required `t`-subset, and passes the separate verifier.

Any candidate must still be checked against post-April-24-2026 updates and independently reviewed. Failure is preserved. Success is not evidence of AGI, autonomous self-improvement, optimality, or universal mathematical discovery.
