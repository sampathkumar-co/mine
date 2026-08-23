# LEXIGEN v5 — Causal Transfer Generalization Protocol

## Mission

Determine whether abstract mechanisms learned before holdout selection can causally improve autonomous solver discovery on fresh unrelated unseen tasks, rather than merely expanding a static operator vocabulary.

v5 is a successor experiment. It does not alter, reopen, or reinterpret the frozen v4 campaign.

## Frozen benchmark snapshot

- AlgoTune source commit: `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`
- AlgoTune dataset revision: `bb02811fa47ca1c833baaa344949bcd8fb307ac8`
- Python: 3.12
- Hosted reference environment: Ubuntu 24.04

Using the same public snapshot preserves comparability while fresh holdout identities are selected from tasks never used by the prior Lexigen benchmark research.

## Contamination boundary

Before holdout selection, commit a complete exclusion set containing every task previously selected, inspected, benchmarked, or deliberately excluded because of prior exposure.

No selected v5 holdout task source, description, train manifest, train payload, test manifest, test payload, public solver, benchmark report, or leaderboard information may be opened before its protocol stage permits it.

Selected difficult or failed tasks may not be replaced.

## Transfer memory

`TRANSFER_MEMORY.json` is frozen before v5 holdout selection.

It contains only abstract reusable recipes and negative lessons distilled from the completed v4 campaign. It contains no prior payloads, answers, task-specific constants, or solver source.

The critical causal change from v4 is structural:

- `v5_full` may instantiate frozen learned recipe IDs from transfer memory when name-independent source features satisfy their committed preconditions.
- `v5_no_transfer` is forbidden from those learned recipe compositions and causal IDs. It receives the same compute budget and generic base operators but no learned recipes.
- If both arms nevertheless produce semantically equivalent implementations, the task earns zero transfer credit.

This asymmetry is the intended ablation: the only capability removed from `v5_no_transfer` is knowledge learned from prior tasks.

## Comparison arms

Every holdout uses equal task-stage budgets for:

1. `v5_full` — generic base operators plus frozen learned transfer recipes.
2. `v5_no_transfer` — generic base operators only; learned transfer recipes unavailable.
3. `random_search` — deterministic random search over the generic base-operator space only.
4. `static_template` — static generic templates frozen independently of transfer memory.
5. `v4_compatible` — predecessor architecture behavior frozen from v4.

No arm may receive hidden task-specific human solver ideas.

## Holdout selection

After the v5 engine, transfer memory, selector, exclusions and success gates are committed and locked:

- deterministically select 10 tasks,
- require at least 8 distinct name-derived task families,
- permit at most 2 tasks from any family,
- use seed `LEXIGEN-V5-CAUSAL-TRANSFER-2026-08-23-A`,
- use only repository/dataset path metadata and task names during selection,
- commit the eligible-inventory hash, selected names/families/scores and selection transcript before opening any selected task source.

## Per-task staged boundary

For every selected task:

1. Commit `TASK_START.json` before opening source or description.
2. Open only source/description and generate bounded proposals with the frozen engine.
3. Freeze concrete candidate implementations and provenance before official train access.
4. Run independent synthetic correctness checks where possible without official train/test payloads.
5. Open official training only after the training package and hashes are locked.
6. Permit at most 3 official training revisions exactly as preregistered. Scientific failures remain preserved.
7. Freeze one candidate per arm before test access.
8. Open blind/test data only after a committed blind lock.
9. Preserve all failed runs, infrastructure incidents, reruns and causal-equivalence findings.

## Proposal budget

- Maximum 6 revision-1 proposals per arm per task.
- Generic failure taxonomy may guide at most 3 official training revisions.
- No post-blind candidate revision is allowed.
- Equal official data access and execution budgets apply to all arms.

## Default clean blind task gate

Unless a stricter task-specific gate is committed before official training:

- 100 / 100 valid blind outputs,
- harmonic speedup >= 1.50x over the official reference,
- minimum per-record speedup >= 1.05x,
- zero invalid-output retries.

## Causal transfer credit for one task

A task earns `causal_transfer_win = true` only if all of the following were frozen before blind access and then observed:

1. `v5_full` passes the clean blind task gate.
2. Its selected candidate uses at least one `TRANSFER_MEMORY.json` learned-template causal ID.
3. `v5_full` and `v5_no_transfer` selected implementations are semantically non-equivalent.
4. The transferred recipe was learned from a different source family than the current holdout family.
5. At least one causal-separation condition holds:
   - `v5_no_transfer` fails the clean blind task gate while `v5_full` passes; or
   - both pass, but `v5_full` harmonic speedup is at least 1.25x the `v5_no_transfer` harmonic speedup with equal validity and retry count.
6. A frozen recipe-removal replay or equivalent preregistered ablation confirms that removing the transferred recipe eliminates the qualifying advantage.

Timing differences between implementation-equivalent programs never earn transfer credit.

## Full v5 campaign success gate

All conditions are required:

- at least 6 of 10 clean autonomous unseen-task wins for `v5_full`,
- wins span at least 6 unrelated families,
- `v5_full` beats `v5_no_transfer` by at least 2 task wins and at least 20 percentage points of task success rate,
- `v5_full` beats `random_search` by at least 2 task wins,
- `v5_full` beats `static_template` by at least 2 task wins,
- `v5_full` beats `v4_compatible` by at least 2 task wins,
- at least 2 causal transfer wins,
- causal transfer wins span at least 2 current holdout families,
- causal transfer wins use at least 2 distinct learned-template causal IDs,
- median human task-specific solver contribution is zero,
- no selected task is swapped, dropped or reclassified after source access,
- no post-result threshold change occurs.

If any required condition fails, the v5 full campaign gate fails. Thresholds may not be relaxed afterward.

## Claim boundary

Passing v5 would be a substantially stronger internal AI-research result than v4 because it would directly support causal cross-domain mechanism reuse rather than static mechanism breadth.

It would still not automatically be a world-level AI breakthrough. That stronger claim requires independent implementation, external reproduction, artifact/benchmark confound checks, and evidence that the advantage is not a known task-specific rediscovery or measurement artifact.
