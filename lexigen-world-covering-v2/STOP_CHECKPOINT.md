# LEXIGEN World Covering Record v2 — Frozen Checkpoint

Prepared and committed as an isolated successor while sealed v1 run `30359329280` remained active.

## Preserved state

- Branch: `lexigen/world-covering-record-v2`.
- The original pre-run audit found that the first draft did not exactly reproduce v1 target reservation because v1 and v2 used different ranking rules.
- This was corrected **before any v2 snapshot access or target reveal**.
- The corrected selector now reproduces the exact frozen v1 eligibility, score, seed, tie breaker, diversity cap, and first-three selection; it excludes those exact names before applying the v2 selector.
- Evidence normalization now distinguishes `generic_greedy`, `stochastic_fixed_budget`, and `cp_sat_minimization` without changing blocks, validity, or solver outcomes.
- The protocol, dependency lock, generic engines, independent verifier, and trigger-gated workflow are committed.
- Every Git blob SHA listed in `LOCK.json` must match GitHub before activation.
- No v2 snapshot access has occurred.
- No v2 target identity is known.
- `TRIGGER_ONCE` does not exist.
- No v2 research workflow has run.
- v1 files, PR #42, and run `30359329280` were not changed or rerun.
- Neither laptop was used.

## Validation requirement

A fresh isolated snapshot-free validation must pass after this correction. It must verify:

- exact frozen hashes;
- exact v1 reservation behavior against a reference implementation on synthetic metadata;
- no overlap between reserved and v2 slices;
- acceptance of a known Fano-plane covering;
- rejection of incomplete and impossible cases;
- truthful method attribution.

## Next safe action

After validation passes and v1 evidence is preserved, add `TRIGGER_ONCE` as the final v2 commit and permit exactly one research run.

Failure remains failure. A valid smaller covering is only a verified world-record candidate until checked against current records and independent review.
