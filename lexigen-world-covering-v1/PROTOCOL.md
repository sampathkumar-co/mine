# LEXIGEN World Covering Record v1

## Objective

Test whether one frozen, generic discovery engine can produce a strictly smaller valid covering design than the best value recorded in the April 24, 2026 La Jolla Coverings Repository snapshot.

A success is a **world-record candidate**, not merely a runtime benchmark: for a selected `(v,k,t)` instance the engine must output at most `recorded_upper_bound - 1` distinct `k`-subsets and independently verify that every `t`-subset of the `v`-set is contained in at least one output block.

## Frozen external snapshot

- Zenodo record: `10.5281/zenodo.19735294`
- File: `coverdata.json`
- Expected MD5: `b2c626b07f216aac830d344eff5ad523`
- Direct file URL: `https://zenodo.org/records/19735294/files/coverdata.json?download=1`

The workflow must abort if the downloaded bytes do not match the committed MD5.

## Target selection before inspection

The solver and this protocol are committed before the workflow downloads the snapshot. The selector parses all entries and retains only unsolved instances satisfying every condition:

- `10 <= v <= 22`
- `4 <= k <= min(10, v-2)`
- `3 <= t <= min(5, k-1)`
- recorded upper bound exceeds the lower bound by at least 2
- recorded upper bound is at most 100
- `C(v,k) <= 50,000`
- `C(v,t) <= 5,000`
- `C(v,k) * C(k,t) <= 3,000,000`
- `recorded_upper_bound - 1 >= lower_bound`

Eligible instances are ranked by a frozen opportunity score that rewards a larger open gap, older last improvement, and lower incidence complexity. The first three instances are selected, with at most two sharing the same `(k,t)` pair. No selected target may be replaced after its identity is printed.

Selection seed material is fixed as:

`32c897005c91865319f1b7da264b6162fc1ff4de|b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1`

The seed breaks only exact score ties.

## Generic engine

Exactly the same code and parameter schedule is used for every selected target:

1. Enumerate all `t`-subsets and all candidate `k`-blocks.
2. Build exact incidence lists.
3. Run deterministic randomized-weighted greedy construction attempts.
4. Remove redundant blocks and perform one-for-one and two-for-one local replacements.
5. Build a CP-SAT feasibility model with one Boolean variable per candidate block, one coverage constraint per `t`-subset, and a fixed cardinality limit of `recorded_upper_bound - 1`.
6. Apply only universal symmetry breaking: require the lexicographically first `k`-block, which is valid because any nonempty covering can be relabelled so one selected block becomes that block.
7. Supply the best generic heuristic construction as a solver hint.
8. Independently verify any returned design without trusting CP-SAT status.

No task-specific formula, manually chosen block, web search for the selected parameters, existing `covers.json` construction, or post-selection code change is permitted.

## Compute budget

- Three selected targets.
- At most 1,200 seconds of CP-SAT time per target.
- Eight deterministic greedy attempts per target, plus 24 seeded randomized attempts.
- CP-SAT uses at most four workers on GitHub Ubuntu 24.04.
- One workflow execution. No target substitution or threshold weakening.

## Acceptance

A target is a record candidate only when all are true:

- output blocks are distinct and each contains exactly `k` valid points;
- output block count is strictly below the frozen snapshot upper bound;
- every `t`-subset is covered;
- an independent verifier recomputes coverage from the serialized result;
- the result includes snapshot hash, code hashes, selected parameters, prior upper/lower bounds, solver status and complete block list.

Any candidate must then be checked against the live repository for improvements published after April 24, 2026. Until external review or repository acceptance, the wording is **verified world-record candidate**, not established theorem or peer-reviewed world record.

## Claim boundary

Failure is preserved. Success on one target would demonstrate that the frozen engine generated a new combinatorial construction relative to the committed global snapshot. It would not by itself prove AGI, general self-improvement, mathematical optimality, or that every Lexigen-generated idea is novel.