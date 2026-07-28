# LEXIGEN World Covering Record v1 — Safe Stop Checkpoint

Stopped safely on 2026-07-28 after freezing the protocol, dependency set, generic solver, independent verifier, lock, and GitHub Actions workflow.

## Completed before stop

- Campaign v1 preserved with two clean blind wins.
- Campaign v2 preserved with two clean blind wins:
  - `chacha_encryption`: 100/100 exact, 2.2562322328721613x harmonic speedup, 2.099163393912871x minimum speedup.
  - `outer_product`: 100/100 exact, 1.644959340831312x harmonic speedup, 1.405224910635449x minimum speedup.
- Campaign v2 honestly remains a failed campaign-level gate because it required three blind wins and achieved two.
- A new mathematical-record experiment was started on branch `lexigen/world-covering-record-v1`.
- The new experiment uses the committed April 24, 2026 La Jolla Combinatorics Repository snapshot commitment and selects three untouched unsolved covering-design targets only after the lock verifies.
- The generic engine combines randomized greedy construction, redundancy deletion, local replacements, CP-SAT feasibility search, and an independent exhaustive covering verifier.

## Safe-stop state

- No target-selection run has executed.
- No snapshot data has been opened by the workflow.
- No pull request has been opened for this branch.
- `TRIGGER_ONCE` has not been created.
- No GitHub Actions workflow is active for the latest frozen commit.
- No laptop computation is required or running for this experiment.
- Existing evidence and prior campaign branches remain unchanged.

## Exact continuation point

1. Audit `PROTOCOL.md`, `selector_solver.py`, `verify_results.py`, `requirements.txt`, `LOCK.json`, and `.github/workflows/lexigen-world-covering-v1.yml`.
2. Confirm all hashes in `LOCK.json` match.
3. Create `TRIGGER_ONCE` as the final commit on this branch.
4. Open a draft, non-mergeable research PR against `main`.
5. Allow the workflow to select the three targets and execute once.
6. Preserve all successes and failures; do not change targets, time budgets, or gates after seeing results.

## Claim boundary

No world-level breakthrough has been established yet. A smaller independently verified covering than the frozen repository upper bound would be a genuine new combinatorial record candidate, but it would still require comparison with the latest external literature and independent expert verification before public breakthrough claims.
