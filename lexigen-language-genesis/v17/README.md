# Lexigen Language Genesis v17 - constructive primitive substrate

v17 begins from frozen v16 commit `851e6117b64ea74f5eaaa631f1611ed6205464a8`.

## Objective

Construct executable programs for three structurally different frozen behaviours using only a generic coordinate/dataflow grammar:

- scalar arithmetic and Boolean composition;
- input-grid sampling and coordinate clamping;
- palette mode and unique-point reductions;
- pair arithmetic and unit-direction calculation;
- conditional output-cell construction.

Named v14/v15 scene operators are forbidden from generated programs. Synthesis receives demonstrations only, never task identifiers.

## Frozen result

- 30,000 accepted fresh cases across three families.
- 30,000/30,000 exact in the primary runtime.
- 30,000/30,000 exact in the independent portable runtime.
- 30,000/30,000 runtime agreement.
- 3,000 correct outputs accepted by both verifier implementations.
- 24,000/24,000 mutant outputs rejected by both learned screens and both mandatory soundness anchors.
- Two of three verifier contracts required preserved CEGIS revision.
- Zero learned contracts used exact-output digest equality.
- Nineteen adversarial tests passed.
- Evidence SHA-256: `5ecedba17a6fdacfe1e9a6ec0ca1938e60b7474d65502f05d0876fbee6f7f771`.

## Claim boundary

v17 demonstrates construction of composite executable programs from generic low-level coordinate/dataflow operations, plus verifier co-synthesis and independent execution.

The low-level grammar, the three candidate-construction schemas, and the verifier predicate grammar remain human supplied. This is not autonomous invention of the semantic substrate, not unrestricted language invention, and not a world-level breakthrough.
