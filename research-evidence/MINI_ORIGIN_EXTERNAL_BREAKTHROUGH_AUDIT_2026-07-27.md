# Mini-ORIGIN External-Breakthrough Audit

Date: 2026-07-27

## Verdict

**No externally defensible breakthrough has been achieved yet.**

Mini-ORIGIN produced several replicated internal milestones and one replicated external-breakthrough candidate, but the candidate was rejected by a stronger fresh-seed Pareto audit. This report preserves both the positive evidence and the falsifications.

An external breakthrough is not defined here as “a private benchmark passed.” It requires a central mechanism that survives strong controls, comparison with the closest outside methods, independent implementation, and external review.

## Research progression

### v0.4 — competitive damage-tolerant routing

Five of five searches evolved robust local signal routing through damaged grids. A transparent max-flood control remained stronger, and high-performing candidates were already common in the redesigned search space.

**Verdict:** replicated internal milestone; not externally novel.

### v0.5 — within-lifetime local plasticity

Five of five searches evolved a dimension-agnostic feedback rule that learned hidden linear mappings during its lifetime and retained them after deleting up to 65% of redundant memory cells.

A first score above 0.99 was rejected because isotropic examples permitted a Hebbian covariance shortcut. The hardened benchmark used ill-conditioned examples and isotropic queries. The accepted median hidden score was 0.8544 with 99.68% median retention.

The evolved mechanism was a high-gain delta/error-correction rule, an established learning principle.

**Verdict:** strong project-level learning breakthrough; not an external scientific breakthrough.

### v0.6 — self-expanding plasticity language

Task-specific programs were mined for reusable closed macros, then reused under a shallow syntax budget.

- median hidden score: 0.7170
- median library advantage: 0.0658
- passing seeds: 0/3

**Verdict:** failed. Closed macros were tied too closely to source signal names.

### v0.7 — parameterized operator transfer

Source expressions were anti-unified into parameterized templates and instantiated with new temporal signal roles. A raw-MSE metric was rejected after a zero predictor received a deceptively high score; it was replaced by skill beyond the mean predictor.

The system repeatedly abstracted the generic residual-credit operator, but its worst-case temporal skill remained only about 0.17–0.28 under repeated switches.

**Verdict:** abstraction worked, but the learning operator was too weak; no external candidate.

### v0.8 — counterexample-driven state invention

A one-state learner failed recurrent contexts, so the design language was expanded with conditional state creation and contextual addressing.

- median hidden score: 0.4443
- median advantage over one-state control: 0.1488
- median advantage over hand dynamic memory: 0.0044

One seed reached 0.546 on the hardest case versus 0.384 for the hand control, but this did not replicate.

**Verdict:** useful internal state invention; not replicated externally.

### v0.9 — adversarial state repair

The search added state initialization, replacement reset, and survivor-based reconstruction. All three searches selected zero initialization and no reset, making their scores identical to their no-repair ablations.

The failure exposed an information-theoretic issue: independently generated mappings cannot be reconstructed from unrelated surviving states unless redundancy is encoded before damage.

**Verdict:** rejected. The apparent recovery was relearning, not repair.

### v0.10 — sparse coded plastic memory

Mappings were encoded across physical cells before damage; learning was frozen after targeted deletion.

- median frozen recovery: 0.9258
- median dense gap: -0.0030
- median advantage over equal-memory replication: 0.1688
- median retention: 0.9626
- median write density: 0.4667

The result approached dense recovery but exceeded the fixed 40% write budget.

**Verdict:** strong lead; failed the efficiency gate.

### v0.11 — hard-constrained static sparse codes

Any candidate exceeding 40% writes or losing logical rank was discarded before ranking. No search seed found a feasible code under 55–62% targeted deletion.

This is consistent with a support lower bound: a static representation cannot guarantee survival after deleting a larger fraction of cells than the fraction that contains each logical component.

**Verdict:** infeasible static target; rejected without lowering the budget.

### v0.12 — temporally rotating sparse sufficient statistics

Every example wrote to a changing sparse subset of cells. Knowledge accumulated as online sufficient statistics, and learning was frozen after targeted deletion.

Official aggregate result:

- passing searches: 2/3
- median frozen recovery: 0.9076
- median dense gap: -0.0192
- median advantage over static sparse writes: 0.5047
- median advantage over the initial hand rotating schedule: 0.0973
- median write fraction: 0.20
- median retention: 0.9616

This crossed the predefined external-candidate gate.

**Provisional verdict:** replicated external-breakthrough candidate pending stronger controls.

### v0.13 — fresh-seed Pareto audit

The v0.12 candidate was retested with no search privilege against:

- one-cell random sharding;
- iid writes at 5%, 10%, 15%, and 20%;
- antithetic writes;
- balanced rotating writes;
- static sparse writes;
- dense writes.

All three fresh audit families rejected the candidate. Untuned iid 10–20% writes matched or Pareto-dominated the searched schedules, while the v0.12 candidates fell below their recovery or retention gates on the harder distribution.

- surviving audits: 0/3

**Verdict:** v0.12 external candidate rejected. The mechanism reduced to ordinary randomized sharding.

### v0.14 — online information-balanced routing

Examples were routed to cells missing the current feature direction, using diagonal information geometry, load penalties, optional randomness, and several selection modes.

Aggregate result:

- passing searches: 0/3
- median recovery relative to dense: 0.9135
- median gap versus iid-20%: -0.0293
- median gap versus hand information routing: 0.0049
- median retention: 0.9135
- median write fraction: 0.1667

**Verdict:** failed. Data-dependent deterministic routing created high-value cells that the targeted deleter could exploit; it did not beat ordinary iid routing.

## What was learned

The strongest confirmed conclusions are negative and methodological:

1. Passing a benchmark is insufficient when a simple control has not been tested.
2. Static sparse encoding cannot satisfy an erasure target that exceeds its logical support density.
3. Random cell deletion can make replicated memories look self-healing even when no repair occurs.
4. Isotropic learning examples can make Hebbian correlation look like feedback learning.
5. Raw MSE can make a zero predictor look successful when target variance is small.
6. Deterministic information concentration can help a post-hoc targeted attacker.
7. Temporally rotating sparse writes are useful, but ordinary iid sharding already captures much of the benefit.

## Current scientific boundary

Mini-ORIGIN has demonstrated:

- automated cloud execution and replication;
- within-lifetime learning;
- damage-tolerant distributed memory;
- executable plasticity-program search;
- operator abstraction;
- conditional state creation;
- pre-damage coding and frozen-learning recovery;
- counterexample-driven benchmark hardening.

It has **not** demonstrated:

- invention of a fundamentally new learning principle;
- invention of a new computational substrate;
- a robust advantage over the closest specialist methods;
- independent external reproduction;
- peer-reviewed acceptance.

## Required gate for any future external claim

A future claim must simultaneously provide:

1. a mechanism not reducible to a standard named baseline;
2. a preregistered acceptance threshold;
3. fresh hidden distributions and seeds;
4. causal ablations;
5. equal-memory and equal-operation accounting;
6. specialist baselines from the closest field;
7. a separately implemented verifier;
8. independent external reproduction or peer review.

Until those conditions are met, the repository must use the wording **internal milestone**, **research lead**, or **external-breakthrough candidate**, never **external breakthrough**.
