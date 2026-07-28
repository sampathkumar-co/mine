# RIFT-0 benchmark specification

## Purpose

RIFT-0 is a minimal executable benchmark for distinguishing:

- additional search inside a fixed straight-line language; from
- invention of a reusable control/semantic primitive that changes what the language can express compactly and execute reliably.

It is deliberately small enough for reproducible GitHub Actions. Passing RIFT-0 alone is not a world-level result.

## Hidden structure

All worlds implement monotone propagation over a finite state space. Starting from an initial active set, rules repeatedly add consequences until no state changes.

The surface encodings differ:

1. **edge world** — directed edges and seed vertices;
2. **rule world** — named facts and implication rules;
3. **grid world** — open cells and directional propagation.

A shallow fixed-language program may apply the one-step transition a bounded number of times. That succeeds on small-diameter training cases and fails on longer hidden-scale cases.

A least-fixed-point primitive applies the transition until convergence and transfers across all surfaces.

## Public synthetic stage

The initial CI stage is public and validates benchmark behaviour:

- training-style cases require at most three propagation rounds;
- transfer cases require four to twelve rounds;
- the `bounded_unroll_3` baseline must fail materially on transfer cases;
- the independently implemented fixed-point oracle must achieve exact correctness;
- all cases must terminate because propagation is monotone over finite sets.

The oracle exists only to validate the benchmark. It must never be reported as autonomous invention.

## Future blind stage

Before a research claim, the following must be frozen before hidden task access:

- starting language and interpreter hash;
- invention-engine hash;
- compute, memory, wall-clock, and candidate budgets;
- allowed training worlds;
- language-artifact schema;
- evaluator and independent-verifier hashes;
- hidden generator commitment;
- ablation and fixed-language comparison plan.

A future hidden seed should be derived from public randomness after all above commitments.

## Candidate language artifact

An invented primitive must be serializable and contain:

- `name`;
- typed signature;
- operational semantics or executable implementation;
- termination conditions;
- applicability/activation rule;
- provenance from observed failures;
- dependencies on prior primitives;
- claimed compression or capability benefit.

A separate interpreter must execute the artifact without importing the invention engine.

## Metrics

1. exact hidden-task accuracy;
2. number of search expansions;
3. execution steps;
4. language description length;
5. transfer accuracy by surface domain;
6. ablation delta;
7. equal-budget fixed-language gap;
8. portability to a second interpreter;
9. verifier agreement;
10. reuse on later tasks without modification.

## RIFT-0 pass conditions

The synthetic benchmark itself is valid only when:

- oracle accuracy is 1.0;
- bounded baseline transfer accuracy is below 0.80;
- bounded baseline training-style accuracy is at least 0.95;
- all independent-verifier checks agree;
- deterministic reruns produce byte-identical reports.

## Research success conditions

An autonomous engine passes the first research gate only when it:

1. begins without a fixed-point/recursion/loop-until-stable primitive;
2. sees training examples and its own failures;
3. emits an executable primitive without a human naming or coding it;
4. reaches exact hidden transfer accuracy;
5. beats expanded fixed-language search under the same budget;
6. survives primitive ablation and paraphrased surface encodings;
7. executes in a separately implemented interpreter.

RIFT-0 is a mechanism probe. Later RIFT versions must include several unrelated missing primitives so that the system cannot be hard-coded to invent fixed points.
