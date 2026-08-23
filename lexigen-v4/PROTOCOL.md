# Lexigen v4 — Frozen Generalization Experiment

## Mission

Determine whether a single frozen discovery architecture can outperform the v3 discovery process across unrelated unseen tasks, with less intervention and with mechanisms that transfer between domains.

This campaign is independent of all covering-design, Mini-ORIGIN, and Language Genesis branches. It must never read or modify their files, triggers, locks, results, or workflows.

## Frozen benchmark snapshots

- AlgoTune task-source repository: `oripress/AlgoTune` commit `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`.
- AlgoTune dataset revision: `bb02811fa47ca1c833baaa344949bcd8fb307ac8`.
- Python runner: 3.12 on `ubuntu-24.04`.

## Holdout exclusions

The selector must permanently exclude every task whose specification, generator, verifier, training data, test data, public solver, or leaderboard information was previously accessed by any Lexigen benchmark campaign:

`numerical_integration`, `water_filling`, `polynomial_mixed`, `vector_quantization`, `integer_factorization`, `chacha_encryption`, `outer_product`, `base64_encoding`, `articulation_points`, `cvar_projection`, `kmeans`, `procrustes`, and `sha256_hashing`.

A selected task is never replaced because it is difficult or fails. A task may be marked infrastructure-ineligible only if the frozen snapshot lacks the required source or both train and test objects; that determination must be made before its contents are opened and remains part of the denominator.

## Selection commitment

- Selection seed: `LEXIGEN-V4-GENERALIZATION-2026-07-28-A`.
- Task count: 8.
- Minimum distinct families: 6.
- Maximum tasks from one family: 2.
- Family classification uses only the frozen task name and the precommitted keyword rules in `selector.py`; source contents, reports, solvers and data contents are forbidden during selection.
- Within each family, candidates are ordered by SHA-256 of `seed + NUL + task_name`; the globally lowest admissible sequence satisfying the diversity rules is selected.
- The selector must commit names, families, snapshot identities, inventory hash and selection transcript before any selected task source is opened.

## Frozen discovery architecture

The v4 engine is the exact code and data hash-bound in `ENGINE_LOCK.json` before selection. For every task it:

1. Parses only the permitted task specification, generator and verifier into a name-independent structural fingerprint.
2. Builds a mechanism graph from generic operators.
3. Ranks bounded compositions using frozen structural compatibility, predicted speed benefit, correctness risk and transfer evidence.
4. Generates a maximum of six revision-1 proposals.
5. Uses official training feedback only to update the frozen generic failure taxonomy, not to add task-specific rules.
6. Allows at most three official training revisions.
7. Selects one candidate before blind access.

Transfer memory contains only abstract mechanisms and negative lessons from earlier campaigns. It contains no holdout names, values, payloads, answer tables or solver source.

## Comparison arms

Each selected task receives equal wall-clock, CPU, candidate-count and revision budgets under four frozen arms:

1. `v4_full`: fingerprint, mechanism graph, transfer memory and risk-aware staging.
2. `v4_no_transfer`: identical engine with transfer evidence zeroed.
3. `random_search`: uniformly samples from the same legal proposal graph with the same candidate count.
4. `template_synthesis`: deterministic enumeration of single generic operators, with no compositions or learned ranking.

A fifth `v3_compatible` arm uses only wrapper/backend substitution, direct vectorisation and one shallow structure observation. It has no transfer memory, failure taxonomy or risk-aware composition. This is a reproducible approximation of the v3 discovery architecture and is frozen before holdout selection.

Where a task has a documented strong task-specific human baseline, it may be compared only after Lexigen blind evidence is sealed. It cannot influence proposals or gates.

## Per-task protocol

- Training and blind splits remain strictly separate.
- Candidate source, dependency lock, runner, verifier, thresholds, revision, transcript hash and execution budget are committed before blind access.
- One selected candidate and one reference execution per blind record.
- Zero invalid-output retries.
- Successful blind records are never rerun.
- Infrastructure failures are preserved separately and do not permit scientific changes.
- Exact output validity is mandatory unless the frozen official verifier explicitly permits approximation.
- Existing task-specific benchmark thresholds are used when preregistered; otherwise the campaign threshold is 100% validity, at least 1.50x harmonic speedup and at least 1.05x minimum speedup.

## Campaign success gate

All conditions are required:

- At least 5 of 8 clean autonomous unseen-task wins.
- Wins span at least 4 unrelated families.
- `v4_full` exceeds `v3_compatible` by at least 2 task wins and by at least 20 percentage points of task success rate.
- `v4_full` exceeds both `random_search` and `template_synthesis` by at least 2 task wins.
- `v4_full` exceeds `v4_no_transfer` by at least 1 task win or reduces median discovery cost by at least 35% with no fewer wins.
- At least one abstract mechanism learned before the campaign is selected and succeeds in two different families.
- Median human task-specific solver contribution is zero; no task with substantial human solver design counts as a clean win.
- No regression invalidates previously valid v3 solvers when rerun on scientifically reusable records under an identical environment.

Failure of any condition is a `failed gate`; no threshold may be changed after selection.

## Claim boundary

Even a campaign pass is an `internal research breakthrough` or `generalization milestone`, not automatically an AI breakthrough. A `candidate AI breakthrough` additionally requires independent implementation, external evaluation, reproducibility outside the original repository, and evidence that the transferable mechanism—not a faster library, hardware feature, benchmark artefact or known algorithm rediscovery—caused the advantage.
