# LEXIGEN v6 — Applicability-Conditioned Causal Transfer Replication Protocol

## Mission

Test whether the frozen Lexigen transfer architecture can **repeat causal mechanism selection across multiple genuinely fresh tasks**, while explicitly controlling for the weak-reference confound exposed by the v5 Task-10 audit.

v6 does not reopen, rescore, reinterpret, or repair v4/v5. The completed v5 result remains 2/10 clean wins and 1 causal-transfer win, with the full v5 gate failed.

## Scientific hypothesis

A reusable autonomous discovery system should do more than occasionally beat a weak benchmark reference. On fresh tasks where its frozen structural fingerprint says transfer memory is applicable, it should:

1. instantiate a learned mechanism that is materially different from the no-transfer arm;
2. preserve exact/certified correctness;
3. produce a blind performance advantage that disappears when the learned mechanism is removed; and
4. remain competitive with a separately frozen strong-baseline arm, so a weak official reference cannot by itself create a causal/discovery claim.

## Frozen benchmark snapshot

- AlgoTune source commit: `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`
- AlgoTune dataset revision: `bb02811fa47ca1c833baaa344949bcd8fb307ac8`
- Python: 3.12
- Hosted reference environment: Ubuntu 24.04
- Source fingerprint/proposal engine: exact v5 frozen engine blob pinned by `ENGINE_LOCK.json`.

Using the same public snapshot is acceptable only because every task exposed in v4/v5 is excluded before v6 selection.

## Contamination boundary

Before any v6 task source/description is opened, commit:

- this protocol;
- transfer memory;
- complete contamination exclusions;
- engine/design lock;
- screening-pool selector and seed;
- strong-baseline catalog/policy;
- all success/failure gates.

No v6-selected task source, description, train manifest, train payload, test manifest, test payload, benchmark report, leaderboard entry, or public task-specific solver may be opened before its permitted stage.

No screened or evaluated task may be swapped, dropped, reclassified, or replaced after source access. Screened-out tasks remain part of the denominator for the applicability/coverage claim.

## Two-stage holdout design

### Stage A — name/metadata-only screening-pool selection

Deterministically select **24 fresh tasks** from the common frozen source/dataset inventory using only task names and repository/dataset path metadata.

Frozen selection constraints:

- seed: `LEXIGEN-V6-TRANSFER-REPLICATION-2026-08-24-A`;
- 24 tasks;
- at least 8 name-derived families;
- at most 4 tasks per family;
- all tasks in `CONTAMINATION_EXCLUSIONS.json` are ineligible;
- no task contents or manifests may be opened.

### Stage B — source-fingerprint applicability screening

After the 24 identities are committed, open **only source/description** for all 24 under a single frozen workflow. Run the exact hash-pinned v5 fingerprint/proposal engine. Do not open train/test manifests, reports, leaderboards, or public solvers.

A screening-pool task is `transfer_applicable=true` iff the frozen engine emits at least one learned-template causal ID from `TRANSFER_MEMORY.json` whose source family differs from the task's current name-derived family.

Final evaluation identities are chosen deterministically from applicable tasks by `(applicability_score, frozen selection score, task name)` with these constraints:

- exactly **8 evaluated tasks**;
- at least **5 current families**;
- at most **2 evaluated tasks per current family**;
- no human preference, task difficulty, runtime, training data, or solver result may affect selection.

If fewer than 8 applicable tasks exist, or the applicable set cannot supply 5 families under the frozen cap, **v6 fails the applicability/coverage gate immediately**. No replacements are allowed.

## Applicability/coverage gate

Before any official training access, all must hold:

- at least 8 / 24 screening tasks are transfer-applicable;
- applicable tasks span at least 5 current name-derived families;
- at least 2 distinct learned causal IDs appear across the applicable screening set;
- the deterministic final selector successfully freezes 8 evaluation tasks spanning at least 5 families.

This gate prevents v6 from hiding a narrow transfer memory behind cherry-picked applicable tasks.

## Arms

Every evaluated task receives equal source/train/test access and bounded proposal budgets:

1. `v6_full` — exact v5 frozen proposal engine with learned transfer recipes enabled.
2. `v6_no_transfer` — same generic base operators and budget, but learned recipe compositions/causal IDs forbidden.
3. `random_search` — deterministic random generic-operator compositions only.
4. `static_template` — independently frozen static generic templates only.
5. `v5_compatible` — exact predecessor behavior for regression comparison.
6. `strong_baseline` — independent confound-control arm drawn only from `STRONG_BASELINE_CATALOG.json`; it receives no transfer-memory causal IDs and may not inspect public task-specific solvers.

`strong_baseline` is not allowed to influence proposal generation or candidate selection in the other arms.

## Strong-baseline policy

The v5 Task-10 audit showed that a 14x speedup over a benchmark reference can coexist with a known-style exact baseline that is roughly 4.5x faster than Lexigen's candidate. Therefore speed against the official reference alone is insufficient for a v6 causal claim.

The strong-baseline arm is generated only from a precommitted generic catalog of broadly reusable exact/native techniques. A catalog entry may be instantiated only when its frozen structural preconditions are satisfied by the same source fingerprint. If no catalog entry applies, the strongest independently available source-equivalent/native-backend baseline is used and the task is flagged `strong_baseline_coverage_limited=true`.

No post-blind human-designed task-specific baseline may be used to rescue a v6 causal win. Post-blind literature review may narrow claims but cannot increase the score.

## Per-task stages

For each of the 8 frozen evaluation tasks:

1. Commit task identity/source hash lock.
2. Generate bounded candidates with the frozen engine from source/description only.
3. Freeze concrete implementation mapping and provenance before official training.
4. Run independent synthetic/adversarial correctness checks.
5. Freeze the strong-baseline implementation from the catalog before official training.
6. Open official training only after all hashes are locked.
7. Permit at most 3 official-training revisions, limited to preregistered harness/environment/schema corrections; scientific candidate failures remain preserved.
8. Freeze one candidate per arm before test access.
9. Execute exactly one blind/test run.
10. Preserve all failures, equivalence findings, retries, and confound flags.

No post-blind candidate revision or timing rerun is allowed.

## Clean blind gate

For a candidate to pass the task clean gate:

- 100 / 100 valid blind outputs;
- harmonic speedup >= 1.50x over the official reference;
- minimum per-record speedup >= 1.05x;
- zero invalid-output retries.

## Baseline-qualified causal-transfer win

A task counts as a v6 causal-transfer win only if **all** conditions hold:

1. `v6_full` passes the clean blind gate.
2. selected full uses at least one learned causal ID from transfer memory.
3. selected full/no-transfer implementations are semantically non-equivalent.
4. the learned recipe source family differs from the current holdout family.
5. causal separation holds: no-transfer fails the clean gate, or full harmonic speedup is >=1.25x no-transfer with equal validity/retries.
6. frozen recipe-removal replay/ablation eliminates the qualifying advantage.
7. `strong_baseline` is valid on the same blind denominator.
8. full harmonic runtime competitiveness is at least **0.80x** the best applicable strong-baseline arm, defined as `strong_baseline_time / full_time >= 0.80` in harmonic aggregate. Equivalently, full may be at most 1.25x slower than the strongest frozen generic baseline.

A task that satisfies 1–6 but fails 7–8 is recorded as `causal_transfer_detected_but_baseline_uncompetitive`; it does **not** count toward the v6 causal-win gate.

## Full v6 replication gate

All conditions are required:

- applicability/coverage gate passes;
- at least **3 / 8 baseline-qualified causal-transfer wins**;
- those causal wins span at least **3 current families**;
- those causal wins use at least **2 distinct learned-template causal IDs**;
- `v6_full` has at least **4 / 8 clean blind wins** overall;
- full exceeds no-transfer by at least **2 task wins**;
- full exceeds random_search by at least **2 task wins**;
- full exceeds static_template by at least **2 task wins**;
- full exceeds v5-compatible by at least **2 task wins**;
- median human task-specific solver contribution is zero;
- zero task swaps/drops/reclassification after source access;
- zero post-result threshold changes.

If any required condition fails, the v6 replication gate fails.

## Novel-discovery gate — separate from v6 replication

Passing v6 does **not** imply a new algorithm or world-level breakthrough.

A task may receive the additional label `novel_discovery_candidate=true` only if:

- it is already a baseline-qualified causal-transfer win;
- full is >=1.25x faster than the best applicable frozen strong baseline on harmonic blind runtime while preserving equal correctness/retries;
- post-blind prior-art search finds no clearly equivalent established method or public implementation;
- an independent reimplementation reproduces the advantage.

This optional label does not affect whether v6 replication passes. It only governs stronger discovery claims.

## Claim boundary

A v6 pass would support a stronger statement than v5: learned abstract mechanisms repeatedly and causally guide solver construction on fresh, structurally applicable tasks, and the resulting advantages are not explained merely by weak benchmark references.

Even a v6 pass is not automatically a world-level AI breakthrough. External reproduction, independent benchmark construction, stronger baselines, and adversarial contamination/measurement review remain required.
