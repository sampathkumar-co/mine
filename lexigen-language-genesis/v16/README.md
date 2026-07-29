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
