# LEXIGEN Ω Architecture Contract

## Purpose

LEXIGEN Ω is an AGI-oriented algorithm/scientific-discovery subsystem. The implementation goal is not merely to search a larger fixed DSL. Ω must be able to invent new executable semantics, test whether those semantics transfer prospectively, and evolve the discovery machinery itself during development.

## Three nested evolutionary loops

### A. Program loop
Generate and mutate executable candidate programs/patches under a hard evaluator. Preserve diverse successful lineages rather than only the current maximum score.

### B. Language loop
From successful lower-level programs, induce candidate semantic genes. A gene may summarize, parameterize, factor, or introduce a separately executable semantic primitive. A gene is speculative until prospective causal transfer is demonstrated.

### C. Searcher loop
Generate candidate changes to proposal/decomposition/mutation/scheduling/memory policies. Searcher changes are evaluated on meta-holdouts that were not used to propose the change. During development, multiple searcher lineages may coexist in an open-ended archive.

## Semantic-gene lifecycle

1. **proposed** — induced from development traces;
2. **executable** — has deterministic semantics or a verified lowering to the substrate;
3. **attacked** — counterexamples and metamorphic tests generated;
4. **prospective** — frozen before target tasks are opened;
5. **causal candidate** — full arm earns qualifying wins where controls do not;
6. **ablated** — exact removal destroys the qualifying advantage;
7. **admitted** — diversity and source/target separation policy passes;
8. **composable** — may be used to induce higher-order genes.

No stage may be skipped for long-term causal memory.

## Universal substrate boundary

The substrate may contain general computational machinery such as arithmetic, comparisons, collections, indexing, bounded control flow, state, functions, graph/tensor structural access, constraints, and resource accounting.

It must not contain a finished operator named for a benchmark task, repository issue, hidden solution, or observed holdout-specific semantic rule. Higher-level convenience operations are allowed only when induced during development or independently justified as generic substrate operations before holdout selection.

## Causal memory boundary

A learned mechanism does not receive causal credit because it appears in a winning candidate. Credit requires equal-budget comparison against:

- the same searcher with learned semantic memory unavailable;
- the same searcher with an equal-capacity frozen random memory;
- exact removal or semantic-equivalence-class ablation of the credited gene.

The full candidate must remain mechanistically distinct from controls and no post-result revision may occur.

## Self-improvement boundary

Ω may modify its search policy during development, but it cannot use final blind results to select or edit the final searcher. Searcher variants must be evaluated on development/meta-holdout evidence before Ω9. At Ω9 the selected archive, router, budgets and policy are frozen.

## Global campaign boundary

A benchmark-specific adapter may expose observations, build/run hooks, verification, score and resource usage. It may not encode task-specific solution logic. The same frozen Ω core must operate across heterogeneous ecosystems.

## Claim boundary

Even a successful internal campaign is not automatically AGI. A credible global breakthrough claim requires strong heterogeneous blind evidence, causal transfer, external-frontier novelty and independent reproduction. AGI is a broader capability claim than algorithm discovery and requires separate evidence outside this subsystem.
