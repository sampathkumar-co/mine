# Lexigen Language Genesis v16 — verifier co-synthesis

v16 starts from frozen v15 and targets a missing requirement: an induced executable language must carry an independently executable verifier contract rather than being accepted only because it fits demonstrations.

## Research gate

1. Treat the v15 IR program as immutable input.
2. Generate candidate verifier predicates from a lower-level verifier grammar.
3. Select a smallest contract that accepts every demonstrated execution.
4. Require the contract to reject independently generated semantic mutations.
5. Bind the contract cryptographically to the exact program AST.
6. Reproduce contract evaluation in a separate portable verifier.
7. Preserve contracts that are vacuous, incomplete, or non-transferable as negative evidence.

## Claim boundary

This stage may establish generic verifier co-synthesis over existing v15 programs. It cannot establish autonomous primitive invention because the scene IR atoms and verifier predicate grammar are still human supplied.

## Preserved checkpoints

The initial frozen checkpoint at commit `e2dfd6f` remains immutable:

- 900 correct fresh outputs;
- 6,969 valid mutant outputs;
- four contracts strengthened by fresh counterexamples;
- zero learned screens using exact-output equality.

The later full-scale freeze extends—rather than replaces—that evidence:

- 9,000 correct fresh outputs;
- 69,841 valid mutant outputs;
- zero screening failures in either verifier runtime;
- zero mandatory-soundness failures in either verifier runtime;
- six of nine contracts strengthened by counterexample-guided revisions;
- zero learned screens using the exact digest.

Every final verifier also carries a separate mandatory exact-digest soundness anchor. The learned contract is evaluated independently as a screening layer; it must reject every tested mutant without relying on that anchor.
