# LEXIGEN V7 — Real Mechanism-Genesis Pilot R2

## Mission
Test whether algorithmic abstractions induced from previously sealed real Lexigen apprenticeship evidence can causally improve autonomous solver discovery on fresh real AlgoTune tasks from semantically distant families.

R2 is a clean selection repair after R1 workflow run `32700863128` established, using names/metadata only, that six eligible distinct families do not exist after the frozen contamination and semantic-distance restrictions. R1 opened no task source, description, manifest, payload, report, solver, or leaderboard information. The five R1-revealed task identities are permanently excluded from R2.

## Frozen external snapshots
- AlgoTune source commit: `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`
- AlgoTune dataset revision: `bb02811fa47ca1c833baaa344949bcd8fb307ac8`
- Hosted runner: Ubuntu 24.04 / Python 3.12

## Frozen learned library
The preholdout learned library was induced before any R1/R2 holdout identity was known in workflow run `32700707971` and is sealed by `LIBRARY_R1_RESULT.json` / SHA-256 `0cd5c3d0078237e395d18b21ba193ef9928ac98e8a3d6557188b5b9570884241`. R2 may not change this library.

## Apprenticeship boundary
The initial apprenticeship families remain:
- `graph_discrete`
- `numerical_optimization`
- `linear_algebra`
- `signal_processing`

The learner receives only the already-frozen low-level typed traces. No R2 holdout information may alter the induced macros.

## Strong semantic-distance rule
R2 holdouts may not come from any apprenticeship-source family. `graph_discrete`, `numerical_optimization`, `linear_algebra`, and `signal_processing` remain ineligible.

## Contamination boundary
All benchmark tasks exposed through V5 plus the five identities revealed by the infeasible R1 selector are excluded before R2 selection. A selected R2 task is never swapped because it is hard or fails.

## R2 holdout selection
After this R2 protocol, updated exclusion set, family classifier, selector, frozen learned library, comparison arms, budgets, and success criteria are committed:
- deterministically select **5 tasks**,
- require **5 distinct eligible families**,
- maximum 1 task per family,
- selection seed `LEXIGEN-V7-REAL-MECHANISM-GENESIS-2026-08-24-B`,
- selection may use only task names and repository/dataset metadata,
- selected task source/description/manifests/payloads remain closed until the selection transcript is sealed.

The change from 6 to 5 is not a post-result scientific threshold relaxation: the R1 scientific experiment never began. R1 failed at name-only denominator feasibility, and all names it revealed are removed from R2.

## Comparison arms
Every R2 holdout receives equal candidate/evaluator/training budgets:
1. `v7_full` — low-level primitives plus the frozen automatically induced macros.
2. `v7_no_library` — identical primitives/search, learned macros unavailable.
3. `v7_random_library` — equal-size/arity deterministic random macro library frozen with the learned library.
4. `v5_compatible` — frozen predecessor behavior where scientifically executable.

## Per-task causal credit
A task earns `causal_transfer_win=true` only if:
1. `v7_full` passes the clean blind task gate;
2. its selected program uses at least one frozen automatically induced macro;
3. the used macro's source families differ from the holdout family;
4. full/no-library selected implementations are semantically non-equivalent;
5. either no-library fails the clean gate, or full harmonic speedup is >=1.25x no-library at equal validity/retries;
6. exact macro-removal or semantic-equivalence-class ablation removes the qualifying advantage;
7. equal budgets were maintained.

## Default clean blind task gate
- 100/100 valid outputs
- harmonic speedup >=1.50x over source-faithful reference
- minimum per-record speedup >=1.05x
- zero invalid-output retries

## R2 real-pilot success gate
All are required:
- at least **3/5** clean unseen wins for `v7_full`,
- wins span at least 3 distinct holdout families,
- at least 2 causal-transfer wins,
- causal wins span at least 2 holdout families,
- at least 2 distinct learned macro IDs receive causal credit,
- `v7_full` beats `v7_no_library` by at least 2 task wins,
- `v7_full` beats `v7_random_library` by at least 2 task wins,
- median task-specific human solver contribution is zero,
- no selected task swap/drop/reclassification after source access,
- no post-result threshold change.

Failure of any required condition fails the real pilot.

## Claim boundary
Passing this five-family real pilot would justify the full V7 apprenticeship/generalization campaign. It is not itself a world-level breakthrough. A breakthrough candidate still requires a larger frozen denominator, an external-frontier improvement, independent reproduction, and confound checks.
