# LEXIGEN v5 Task 10 causal audit R1

## Purpose

Attack the sealed Task-10 `vertex_cover` causal-transfer result without changing, reopening, or rescoring the v5 campaign.

The sealed v5 result remains immutable. This audit is post-hoc robustness/novelty analysis only.

## Frozen evidence under audit

- Task-10 result checkpoint: `4ae98e50a9e4393f5f0a1498ea99423255f56552`
- Blind run: `32696527576`
- Blind artifact: `9509000357`
- Learned causal ID: `TM-BFR-01`
- Learned implementation: `learned_bit_frontier_exact`
- Sealed blind result: 100/100 valid, harmonic speedup 14.081539448936624x, minimum speedup 8.640538448084365x.

## Questions

1. **Timing robustness** — does the advantage persist under repeated median timings and rotated execution order?
2. **Reference weakness** — does the 14x headline largely disappear against a stronger known exact graph-search baseline rather than the benchmark's repeated-SAT reference?
3. **Independent reimplementation** — can the abstract frozen TM-BFR recipe be reimplemented separately and recover correctness/performance without copying the sealed Task-10 implementation body?
4. **Distribution robustness** — does the mechanism remain exact and useful on deterministic graph stress cases outside the official 100 records?
5. **Novelty boundary** — is the concrete algorithmic family already represented in longstanding maximum-clique / maximum-independent-set branch-and-bound literature?

## Audit implementations

The audit freezes these implementations before execution:

- `sealed_bfr`: import the exact sealed v5 Task-10 learned implementation.
- `reproduced_bfr`: separate implementation derived from the abstract TM-BFR recipe, not copied from the sealed function body.
- `color_bound_clique`: post-hoc strong known-style bitset maximum-clique branch-and-bound on the complement graph, using greedy coloring as an upper bound. This is a confound baseline, not a v5 arm and not eligible for v5 credit.
- `pysat_reference`: exact sealed benchmark-compatible reference.
- `rc2_exact`: exact MaxSAT alternative already present in the sealed candidate package.

## Measurement

Official test audit:
- reuse the already-open sealed Task-10 test manifest identity only;
- execute all algorithms on all 100 official test records;
- verify every returned cover is valid and has the same optimum cardinality as the exact reference;
- use 5 timed repetitions per algorithm per record;
- rotate algorithm execution order deterministically by record and repetition;
- aggregate per-record medians, harmonic ratios, minima and medians.

Stress audit:
- deterministic Erdős-Rényi graphs with n in {24, 32, 40}, density in {0.10, 0.30, 0.50, 0.70, 0.90}, two seeds each;
- deterministic structural cases: empty, complete, star, matching, cycle, complete bipartite, disconnected-with-isolates;
- require agreement of exact optimum cardinality across `sealed_bfr`, `reproduced_bfr`, and `color_bound_clique`; use PySAT reference on the bounded stress suite as an additional oracle.

## Interpretation rules

- The original v5 causal-transfer credit is not revoked merely because a stronger post-hoc human baseline exists; v5 credit is defined against the preregistered frozen arms.
- A stronger known baseline matching or beating TM-BFR means the 14x result is primarily evidence of **useful mechanism selection / rediscovery**, not a novel vertex-cover algorithm.
- If `reproduced_bfr` fails exactness or loses the effect completely, confidence in reusable recipe transfer is reduced.
- If the effect disappears under repeated timings or order rotation, classify the original timing claim as fragile.
- No result from this audit may be back-propagated into the v5 campaign score.

## Prior-art boundary

Before execution, the audit records that exact maximum clique / independent set branch-and-bound and bitset variants are established prior art (including Tomita-family algorithms and BBMC-style bitset encodings). Therefore no algorithmic-novelty or world-level breakthrough claim is permitted from Task 10 alone even if this audit reproduces the speed advantage.
