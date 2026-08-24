# LEXIGEN V7 Gate -1 — Prospective Abstraction Feasibility Preregistration

## Purpose

This is a kill-test for the proposed V7 Mechanism Genesis architecture. It is not a benchmark campaign and cannot establish novelty or a breakthrough.

The experiment asks whether a low-level program language can support automatic abstraction induction that prospectively improves equal-budget search on held-out families.

## Isolation

- Parent checkpoint: LEXIGEN V5 final audit `6267ce2acb7ee22306c61188c956f6c69e996af2`.
- Work is isolated to `lexigen/v7-gate-minus1-preregister-r1`.
- Do not modify or merge into any `lexigen/world-covering-record-*`, Mini-ORIGIN, Language Genesis, or prior V4/V5 evidence branch.
- No laptop compute is required. Execution is GitHub Actions only.

## Scientific boundary

The committed experiment is synthetic and intentionally small. Apprenticeship programs are visible to the abstraction learner. Held-out target programs are *not* stored in the experiment specification; only input/output behavior is committed for the holdouts.

The learner receives only eight low-level vector primitives:
`ABS`, `CLIP_POS`, `CUMSUM`, `DIFF`, `NEG`, `REVERSE`, `SORT`, `UNIQUE`.

No named algorithm such as dynamic programming, branch-and-bound, frontier reduction, FFT, active-set, graph contraction, or a task-specific solver is supplied.

## Automatic abstraction induction

The learner mines repeated contiguous primitive subsequences of length 2-3 from apprenticeship programs.

A macro is eligible only if it occurs in at least two distinct apprenticeship families. Candidates are ranked deterministically by MDL-style savings, source-family support, occurrence count, length, then lexical order. Exactly three macros are frozen by this rule.

This stage is deliberately mechanical: if useful abstractions do not emerge automatically, V7 is rejected before a larger implementation.

## Held-out evaluation

Three held-out behavioral tasks are committed under families not used as the source family for the successful transferred macro.

For every holdout, the following arms receive exactly 500 candidate evaluator calls and maximum expanded primitive length 4:

1. `learned_library` — atoms plus the automatically induced three macros.
2. `no_library` — atoms only.
3. `random_library` — atoms plus three deterministic random macros matched in count and macro length and forbidden from being identical to learned macros.

Candidate programs must match all search cases and then all separately committed validation cases.

## Causal replay

For every successful cross-family learned macro used by the winning program, rerun the same search with that specific macro removed while retaining all low-level atoms and all other learned macros.

This is a search-representation ablation, not a claim that the semantic operation is impossible to reconstruct. A causal success means the learned compressed abstraction is necessary to find the solution within the frozen equal evaluator budget.

## Pass / kill gate

All conditions are mandatory:

- at least 3 macros induced automatically;
- at least 2 prospective cross-family held-out transfers;
- learned-library succeeds where no-library fails on at least 2 holdouts;
- learned-library succeeds where equal-size random-library fails on at least 2 holdouts;
- removal of a specifically used learned macro causes budgeted search failure on at least 2 holdouts;
- zero task-specific human solver hints during execution.

If any condition fails, Gate -1 is killed and the full V7 architecture must not be built from this design without a new preregistration.

## Claim boundary

Passing Gate -1 only demonstrates that learned symbolic compression can prospectively change search efficiency in this committed synthetic DSL. It does not establish a new algorithm, real-world generalization, scientific novelty, or a world-level AI breakthrough.

A full V7 would require substantially harder external, cross-domain, causal, contamination-resistant and frontier-result gates.
