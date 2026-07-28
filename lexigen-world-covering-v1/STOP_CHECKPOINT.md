# LEXIGEN World Covering Record v1 — Safe Stop Checkpoint

Stopped on 2026-07-28 after preserving the protocol, dependency set, generic solver, independent verifier, lock, trigger, workflow, and prior campaign evidence.

## Completed before stop

- Campaign v1 is preserved with two clean blind wins.
- Campaign v2 is preserved with two clean blind wins:
  - `chacha_encryption`: 100/100 exact, 2.2562322328721613x harmonic speedup, 2.099163393912871x minimum speedup.
  - `outer_product`: 100/100 exact, 1.644959340831312x harmonic speedup, 1.405224910635449x minimum speedup.
- Campaign v2 honestly remains a failed campaign-level gate because it required three blind wins and achieved two.
- The stronger mathematical-record experiment was frozen on branch `lexigen/world-covering-record-v1`.
- It uses the committed April 24, 2026 La Jolla Combinatorics Repository snapshot and one generic engine combining randomized greedy construction, redundancy deletion, local replacements, CP-SAT feasibility search, and an independent exhaustive covering verifier.

## Actual safe-stop state

- Draft research PR #42 was closed without merging.
- The PR is marked not to reopen until the user explicitly requests continuation.
- GitHub Actions run `30359329280` had already started before the PR was closed and had entered the target-selection/solver step.
- The available GitHub connector did not expose a workflow-cancel action, so the already-started isolated runner could not be cancelled directly.
- The workflow has read-only repository permissions, no credentials, no write-back step, and a hard 90-minute timeout. It can only compute and upload an Actions artifact; it cannot modify the repository, either laptop, or prior evidence.
- Closing PR #42 prevents further pull-request runs from this branch unless it is explicitly reopened.
- No laptop computation is running for this experiment. Sampath's laptop was offline, and Yaswanth's device was not used.
- Existing evidence and prior campaign branches remain unchanged.
- Any artifact produced by the already-started run must be treated as sealed, unreviewed evidence until continuation is explicitly requested.

## Exact continuation point

1. Check the final status of Actions run `30359329280` without rerunning it.
2. If an artifact exists, download it once and verify its hashes and independent-verifier result.
3. Preserve a failure as a failure; do not change the selected targets, time budgets, engine, or success gate.
4. Compare any claimed smaller covering against the latest external literature and obtain independent expert verification before making a public record claim.
5. Keep PR #42 closed unless continuation is explicitly requested.

## Claim boundary

No world-level breakthrough has been established yet. A smaller independently verified covering than the frozen repository upper bound would be a genuine combinatorial record candidate, but it would still require comparison with post-snapshot updates and independent expert review before being called a world record.
