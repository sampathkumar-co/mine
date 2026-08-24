# LEXIGEN V7 Gate -1 R2 — Prospective Abstraction Feasibility Preregistration

## Purpose

R2 is the clean rerun of the V7 Mechanism Genesis kill-test after R1 was invalidated by a deterministic fixture-binding error. R1 workflow run `32699545049` is preserved and must not be interpreted as evidence against the architecture: the locked holdout input vectors and locked behavior hashes came from different fixture instances, so no candidate program could match them.

R2 changes only the oracle packaging needed to make the committed holdout behavior internally self-consistent. The DSL, apprenticeship programs, search budget, random-library control, causal removal replay, pass thresholds, and claim boundary remain the same.

## Isolation

- Parent scientific checkpoint: LEXIGEN V5 final audit `6267ce2acb7ee22306c61188c956f6c69e996af2`.
- R2 starts from the R1 pre-trigger checkpoint `dd32d7f7d3ae2d927480626fd0d2f2172847505f`.
- Work is isolated to `lexigen/v7-gate-minus1-preregister-r2`.
- Do not modify or merge into any `lexigen/world-covering-record-*`, Mini-ORIGIN, Language Genesis, V4, or V5 evidence branch.
- GitHub Actions only; no laptop compute required.

## Scientific boundary

The experiment is synthetic and intentionally small. It is an architecture-feasibility gate, not a benchmark, novelty, or breakthrough claim.

The learner receives only eight low-level vector primitives:
`ABS`, `CLIP_POS`, `CUMSUM`, `DIFF`, `NEG`, `REVERSE`, `SORT`, `UNIQUE`.

No named algorithm such as dynamic programming, branch-and-bound, frontier reduction, FFT, active-set, graph contraction, or task-specific solver is supplied to the learner.

## R2 oracle separation

`oracle_r2.py` is frozen before execution. It deterministically constructs search inputs, validation inputs, and three held-out target behaviors. It writes only input vectors plus SHA-256 behavior signatures to a temporary oracle JSON artifact.

The search engine `gate_minus1.py` never imports the oracle generator and receives only that temporary JSON. The workflow verifies the exact oracle JSON SHA-256 before search. This prevents the R1 mismatch while keeping the target-generating program out of the learner's input path.

## Automatic abstraction induction

The learner mines repeated contiguous primitive subsequences of length 2-3 from apprenticeship programs. A macro is eligible only if it appears in at least two distinct apprenticeship families. Candidates are ranked deterministically by MDL-style savings, source-family support, occurrence count, length, then lexical order. Exactly three learned macros are selected.

## Held-out controls

Each held-out family receives exactly 500 candidate evaluator calls and maximum expanded primitive length 4 for three arms:

1. `learned_library`: atoms plus the three induced macros.
2. `no_library`: atoms only.
3. `random_library`: atoms plus three deterministic random macros matched in count and macro lengths and forbidden from being identical to learned macros.

A candidate must match the search behavior signature and a separately generated validation behavior signature.

## Causal replay

For every successful cross-family learned macro used by the winning program, the same search is rerun with only that macro removed while all low-level atoms and all other learned macros remain available.

This measures whether learned symbolic compression is causally necessary to find the solution within the frozen search budget. It does not claim the underlying semantic operation is impossible to reconstruct with unlimited search.

## Pass / kill gate

All conditions are mandatory:

- at least 3 macros induced automatically;
- at least 2 prospective cross-family held-out transfers;
- learned-library succeeds where no-library fails on at least 2 holdouts;
- learned-library succeeds where equal-size random-library fails on at least 2 holdouts;
- removal of a specifically used learned macro causes budgeted search failure on at least 2 holdouts;
- zero task-specific human solver hints during execution.

Failure of any condition kills this V7 design at Gate -1.

## Claim boundary

Passing R2 would establish only that automatic symbolic abstraction can prospectively change equal-budget search efficiency in this frozen synthetic DSL. It does not establish scientific novelty, real-world generalization, an external algorithmic result, or an AI breakthrough. A full V7 must face substantially harder external and contamination-resistant gates.
