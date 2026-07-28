# Lexigen Language Genesis

Status: **frontier benchmark scaffold; no breakthrough claim**

Frozen on: 2026-07-28

## Research objective

Build a system that can detect when its current representational language is inadequate, autonomously invent a new executable reasoning primitive or language, and demonstrate that the invention creates a transferable capability unavailable to the frozen starting system under equal-resource comparison.

This track is separate from:

- the existing Lexigen solver-discovery campaigns;
- the isolated covering-design record branches;
- any Oraphim/Yaswanth-device work.

Those branches must not be modified, merged into, or used as evidence for Language Genesis.

## Why this track exists

The current Lexigen campaigns mainly search for better solvers inside human-provided representations. That work supplies useful blind-testing and verification infrastructure, but it does not yet test the original language-invention claim.

The public frontier already includes latent reasoning, emergent compact symbolic protocols, predicate invention, abstraction learning, and library-learning systems. Therefore, success cannot mean merely producing shorter symbols, hidden vectors, macros, predicates, or a task-specific DSL.

## Required capability loop

1. Observe repeated failures under a frozen language and budget.
2. diagnose a representation failure rather than a search failure;
3. invent a primitive with explicit executable semantics;
4. build or extend an interpreter safely;
5. solve hidden tasks that the original language cannot solve within the same resource budget;
6. transfer the primitive across unrelated surface domains;
7. preserve it for future tasks;
8. pass ablation, portability, and independent-verification gates.

## Frontier levels

- **L0 — Fixed representation:** solve inside a supplied language.
- **L1 — Compressed/latent reasoning:** change the channel, not the executable ontology.
- **L2 — Library or macro learning:** add reusable compositions inside fixed semantics.
- **L3 — Predicate/symbol invention:** add named concepts inside a supplied logic or protocol.
- **L4 — Executable language growth:** invent new operators with explicit semantics and cross-task reuse.
- **L5 — Verified capability genesis:** detect the need, invent the semantics, extend the verifier, and unlock a transferable capability that equal-budget fixed-language systems cannot reproduce.

The initial research target is a defensible **L4**, with L5 as the breakthrough gate.

## First benchmark: RIFT-0

RIFT means **Representation-Invention Frontier Test**.

RIFT-0 creates several differently encoded monotone-propagation worlds. Small training cases can be solved by shallow unrolling. Hidden-scale cases require the reusable concept of least-fixed-point iteration.

The repository initially contains:

- a fixed-language bounded-unrolling baseline;
- an oracle least-fixed-point language artifact used only to validate the benchmark;
- an independent reference verifier;
- public synthetic CI checks.

The oracle is not a research result. The next engine must infer and emit an equivalent executable primitive without being told its identity.

## Non-negotiable breakthrough gates

A future claim requires all of the following:

- autonomous primitive invention;
- explicit executable semantics;
- hidden cross-domain transfer;
- equal compute and data budgets;
- failure of stronger fixed-language search baselines;
- ablation showing the new primitive is necessary;
- portability to a separately implemented interpreter or model;
- independent verification;
- immutable negative results and complete provenance;
- adversarial comparison against known abstractions and published systems.

See `FRONTIER_2026-07-28.md` and `benchmark/SPEC.md`.
