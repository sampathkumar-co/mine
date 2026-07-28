# Lexigen v11 External Campaign

Frozen parent language commit: `c0b32603fc367506bc1721a6e8a480fd8fe882b2`

Status: campaign protocol committed before task selection. No task identity, demonstration, test input, sealed output or score was known when this manifest was written.

## Gate identities

The campaign contains exactly ten gates:

- `v11-campaign-01`
- `v11-campaign-02`
- `v11-campaign-03`
- `v11-campaign-04`
- `v11-campaign-05`
- `v11-campaign-06`
- `v11-campaign-07`
- `v11-campaign-08`
- `v11-campaign-09`
- `v11-campaign-10`

Every gate uses the same frozen campaign engine commit and pinned ARC-GEN commit. The gate identity only changes the deterministic task-selection digest.

## Immutable protocol

For each gate:

1. Select one task deterministically from the same eligible ARC-GEN snapshot.
2. Commit the selected identity before generation.
3. Generate six demonstrations and twenty hidden tests.
4. Commit the redacted package and SHA-256 commitment before solving.
5. Run v6, v7, v8, v9, v10 and v11 once under frozen budgets.
6. If no v11 candidate exists, preserve a permanent pre-score result.
7. If v11 produces predictions, commit its artifact and all predictions before hidden scoring.
8. Score exactly once and preserve the aggregate result.
9. Never revise the language, retry the task or replace a selected gate inside this campaign.

## Campaign interpretation

- A v6–v10 solution is a baseline-dominated gate, not Language Genesis evidence.
- A v11 training fit with failed hidden score is a permanent negative result.
- One exact v11 hidden win while all prior baselines fail is a credible external candidate, not a world-level result.
- A stronger claim requires repeated wins, extension reuse on another family and independent reproduction.
- All ten gates count in the denominator; unsuccessful or irrelevant gates cannot be discarded.
