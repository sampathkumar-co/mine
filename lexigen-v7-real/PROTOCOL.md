# LEXIGEN V7 — Real Mechanism-Genesis Pilot R1

## Mission
Test whether algorithmic abstractions induced from previously sealed real Lexigen apprenticeship evidence can causally improve autonomous solver discovery on fresh real AlgoTune tasks from semantically distant families.

This is the first real-task successor to the synthetic Gate -1 feasibility pass. It is not a continuation of V5 candidate code and does not alter V5, covering-design, Language Genesis, Mini-ORIGIN, or any unrelated branch.

## Frozen external snapshots
- AlgoTune source commit: `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`
- AlgoTune dataset revision: `bb02811fa47ca1c833baaa344949bcd8fb307ac8`
- Hosted runner: Ubuntu 24.04 / Python 3.12

## Apprenticeship boundary
V7 may learn only from sealed abstract evidence produced before this pilot. No future holdout identity, source, payload, public solver, report, leaderboard data, or task-specific human mechanism idea may enter apprenticeship.

The initial real apprenticeship families are restricted to the four pre-V5 mechanism-source families:
- `graph_discrete`
- `numerical_optimization`
- `linear_algebra`
- `signal_processing`

The learner receives low-level typed mechanism traces, not named V5 recipes or task payloads. Higher-level macros must be induced automatically from repeated typed structure.

## Strong semantic-distance rule
Fresh pilot holdouts may not come from any apprenticeship-source family. `graph_discrete`, `numerical_optimization`, `linear_algebra`, and `signal_processing` are therefore ineligible holdout families for R1.

This is intentionally stricter than V5: a graph-derived abstraction does not receive easy credit on a neighboring graph/discrete task.

## Contamination boundary
Every task previously selected, inspected, benchmarked, or explicitly exposed by the Lexigen benchmark track through V5 is excluded before selection. Selected pilot tasks can never be swapped because they are difficult or fail.

## Holdout selection
After protocol, exclusions, family classifier, selector, apprenticeship traces, abstraction inducer, comparison arms, budgets, and success criteria are frozen:
- deterministically select 6 tasks,
- require 6 distinct eligible families,
- maximum 1 task per family,
- selection seed `LEXIGEN-V7-REAL-MECHANISM-GENESIS-2026-08-24-A`,
- selection may use only task names, repository/dataset path metadata, and committed family rules,
- selected task source/description/manifests/payloads remain closed until the selection transcript is sealed.

## V7 mechanism language
The primitive representation is typed and intentionally lower-level than V5 recipe names. Operator roles include:
- `REPRESENT`
- `RESTRICT`
- `REDUCE`
- `EXECUTE`
- `LIFT`
- `REFINE`
- `CERTIFY`
- `RECOVER`
- `SPECIALIZE`

The abstraction inducer may create a macro only from recurring typed subsequences observed in at least two apprenticeship families. A macro stores its typed structure, source families, support count, and MDL-style compression gain. It may not store task constants or candidate source.

## Comparison arms
Every real holdout receives equal candidate/evaluator/training budgets:
1. `v7_full` — low-level primitives plus automatically induced learned macros.
2. `v7_no_library` — identical primitives/search, learned macros unavailable.
3. `v7_random_library` — same number and arities of macros as `v7_full`, but generated from deterministic random primitive compositions.
4. `v5_compatible` — frozen predecessor behavior where scientifically executable.

## Per-task causal credit
A task earns `causal_transfer_win=true` only if:
1. `v7_full` passes the frozen clean blind task gate;
2. its selected program uses at least one automatically induced macro;
3. the used macro was induced from families different from the holdout family;
4. full and no-library selected implementations are semantically non-equivalent;
5. either no-library fails the clean gate, or full has >=1.25x harmonic speedup over no-library at equal validity/retries;
6. an exact macro-removal replay or semantic-equivalence-class ablation removes the qualifying advantage;
7. equal execution/evaluator budgets were maintained.

## Default clean blind task gate
- 100/100 valid outputs
- harmonic speedup >=1.50x over the source-faithful reference
- minimum per-record speedup >=1.05x
- zero invalid-output retries

## Real-pilot success gate
All are required:
- at least 3/6 clean unseen wins for `v7_full`,
- wins span at least 3 distinct holdout families,
- at least 2 causal-transfer wins,
- causal wins span at least 2 holdout families,
- at least 2 distinct automatically induced macro IDs receive causal credit,
- `v7_full` beats `v7_no_library` by at least 2 task wins,
- `v7_full` beats `v7_random_library` by at least 2 task wins,
- median task-specific human solver contribution is zero,
- no selected task swap/drop/reclassification after source access,
- no post-result threshold change.

Failure of any required condition fails this real pilot. Thresholds may not be relaxed after selection.

## Claim boundary
Passing this six-task pilot would justify constructing the full V7 apprenticeship/generalization campaign. It would still not establish a world-level breakthrough. A breakthrough candidate additionally requires a larger frozen denominator, external-frontier improvement, independent reproduction, and confound checks.