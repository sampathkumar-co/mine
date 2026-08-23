# LEXIGEN v4 — G8 Independent Validation Protocol

## Purpose

G8 tests the claims that remain supportable after the frozen eight-task campaign. It must not be used to rescue the failed G4 transfer or G5 baseline-superiority gates.

The claims eligible for independent validation are:

1. The frozen v4 architecture produced clean unseen-task wins on 5 of 8 preregistered tasks across 5 task families.
2. The v4 arm achieved three blind task-win advantages over the restricted v3-compatible arm (Tasks 3, 5 and 6).
3. The campaign provides negative evidence for the transfer thesis: no task earned causal transfer credit and the no-transfer arm matched the v4 task-win count.

World-level AI breakthrough, superior learned transfer, and superiority over random/template search are explicitly **not** eligible claims.

## Independence requirements

An evaluation counts for G8 only if the evaluator is outside the original experiment execution path and records its identity, environment and implementation provenance.

The independent evaluator must use a fresh workspace and must not import, copy, or mechanically translate `lexigen-v4/tasks/*/candidates.py` when performing the implementation-independence layer.

The evaluator may inspect the frozen protocol, task identities, public benchmark source at the committed AlgoTune snapshot, frozen gate thresholds, and the final G6/G7 audit only after agreeing to the reproduction procedure.

## Layer A — Sealed artifact cold replay

This layer verifies custody and measurement reproducibility. It is necessary but is not by itself an independent implementation.

Required checks:

- Verify the frozen engine/protocol and holdout-selection locks.
- Verify the AlgoTune source commit `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`.
- Verify the dataset revision `bb02811fa47ca1c833baaa344949bcd8fb307ac8`.
- Verify all eight selected task names and families against the committed selection transcript.
- Re-run the sealed measurement packages in a clean Python 3.12 / Ubuntu 24.04 environment without using prior caches or result files as inputs.
- Compare validity, harmonic speedup, minimum speedup, retry counts, manifest hashes and selected candidates with the committed task result files.
- Preserve all deviations rather than retrying until agreement.

Key final artifacts include:

- Task 5 blind result: `lexigen-v4/tasks/05-tensor-completion-3d/BLIND_R1_RESULT.json`
- Task 6 blind result: `lexigen-v4/tasks/06-unit-simplex-projection/BLIND_R1_RESULT.json`
- Task 7 synthetic failure evidence: `lexigen-v4/tasks/07-ode-fitzhughnagumo/SYNTHETIC_R1_FAILURE.json`
- Task 8 blind result: `lexigen-v4/tasks/08-dst-type-II-scipy-fftpack/BLIND_R1_RESULT.json`
- Final campaign audit: `lexigen-v4/CAMPAIGN_FINAL_RESULT.json`
- G7 causal audit: `lexigen-v4/G7_CAUSAL_AUDIT.json`

## Layer B — Fresh implementation reproduction

This layer is the actual implementation-independence test.

The evaluator must independently implement candidates from the **abstract mechanism descriptions**, not from Lexigen implementation source, for at least the three tasks supporting the v4-over-v3 claim:

- Task 3 `max_common_subgraph`: word/bit-parallel graph representation plus sparse/frontier restriction.
- Task 5 `tensor_completion_3d`: structurally reduced completion formulation with bounded/refined numerical solution.
- Task 6 `unit_simplex_projection`: active-set decomposition with an independently derived correctness certificate.

For each task the evaluator must also implement or use the preregistered restricted v3-compatible comparator under the same resource and validity rules.

The evaluator must freeze its implementation hashes before opening the corresponding blind/test payloads.

A reproduction succeeds for a task only if:

- all 100 outputs are valid,
- harmonic speedup is at least 1.50x,
- minimum speedup is at least 1.05x,
- there are zero invalid-output retries,
- the independently implemented newer mechanism passes while the restricted v3-compatible comparator fails the task gate or otherwise reproduces the preregistered task-win advantage.

## Negative-transfer reproduction

The evaluator should separately test the causal conclusion rather than assume it.

For Tasks 3, 5, 6 and 8, compare an implementation selected with no cross-task transfer memory against the corresponding transferable-memory description. If no-transfer reproduces the same winning mechanism/task outcome, this supports the campaign's negative transfer conclusion.

No timing difference between implementation-equivalent programs may be counted as transfer evidence.

## G8 decision rule

`G8_PASS_PARTIAL_CLAIM` requires:

- Layer A cold replay has no unexplained material contradiction; and
- Layer B independently reproduces at least two of the three v4-over-v3 task-win mechanisms on unseen benchmark payloads; and
- no independent result contradicts the 5/8 broad-competence classification; and
- the evaluator confirms that the transfer-superiority claim remains unsupported unless genuinely new causal evidence appears.

`G8_FAIL` is required if the independent implementation cannot reproduce at least two of Tasks 3/5/6, if custody/hash checks materially disagree, or if the claimed validity/speed gates do not reproduce.

## Evidence package required from the evaluator

The evaluator must return:

- evaluator identity or organization,
- repository/commit for the fresh implementation,
- environment lock and dependency hashes,
- pre-blind implementation commitment,
- per-record validity/timing evidence,
- aggregate metrics,
- rerun count,
- any deviations from this protocol,
- signed or immutable final conclusion.

## Current status

G8 is **prepared but not completed**. Execution by the original ChatGPT/repository workflow does not satisfy implementation independence. An outside evaluator or independently controlled implementation is required before G8 can pass.
