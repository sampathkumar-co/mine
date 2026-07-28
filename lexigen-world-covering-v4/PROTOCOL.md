# LEXIGEN World Covering Record v4

## Objective

Run one frozen generic covering-design discovery engine on three fresh blind targets after exactly reproducing and excluding the deterministic v1, corrected-v2, and v3 lineages.

## Frozen lineage and snapshot

- Snapshot: Zenodo `10.5281/zenodo.19735294`, `coverdata.json`.
- Required MD5: `b2c626b07f216aac830d344eff5ad523`.
- Exact excluded lineage:
  - v1: `C(15,8,5)`, `C(11,6,5)`, `C(14,5,3)`.
  - corrected v2: `C(12,7,5)`, `C(14,8,5)`, `C(16,7,4)`.
  - v3: `C(15,6,4)`, `C(17,9,5)`, `C(16,8,5)`.
- The selector aborts unless it reproduces those nine targets in that exact order before selecting positions ten through twelve.
- No replacement is allowed after target reveal.

## Generic engine

The same engine is applied to all three targets: deterministic and seeded randomized greedy construction, redundancy pruning, multi-block exact destroy-and-repair, a full CP-SAT set-cover search at `upper-1`, and independent reconstruction of every required subset from serialized blocks.

No selected-target formulas, manual constructions, live-record lookup, retries, target substitutions, weakened gates, or post-reveal tuning are permitted.

## Frozen budget

- 12 deterministic and 36 randomized greedy attempts.
- 42 exact-repair rounds within 360 seconds per target.
- 1,050 seconds full CP-SAT per target.
- Four workers, three targets, one workflow execution, 90-minute hard timeout.

## Acceptance

At least one independently verified covering strictly below the frozen upper bound. Any success remains a verified world-record candidate pending current authoritative record comparison and external review.
