# Lexigen Language Genesis v17 — constructive primitive substrate

v17 begins from frozen v16 commit `851e6117b64ea74f5eaaa631f1611ed6205464a8`.

## Milestone A

Construct executable programs for three unrelated frozen behaviours using only a generic coordinate/dataflow grammar:

- scalar arithmetic and Boolean composition;
- input-grid sampling and coordinate clamping;
- palette mode and unique-point reductions;
- pair arithmetic and unit-direction calculation;
- conditional output-cell construction.

Named v14/v15 scene operators are forbidden from generated programs.

## Required evidence

1. Synthesis uses demonstrations only, never task identifiers.
2. Generated programs replay every demonstration exactly.
3. Fresh accepted ARC-GEN cases pass in two separately implemented interpreters.
4. Program and grammar hashes are frozen.
5. Forbidden-opcode scans and adversarial tests pass.

## Claim boundary

The low-level grammar and candidate-construction schemas remain human supplied. Passing this milestone would demonstrate autonomous construction of composite executable programs, not autonomous invention of the semantic substrate and not a world-level breakthrough.

## Frozen v17 result

- 30,000 accepted fresh cases across three families.
- 30,000/30,000 exact in the primary runtime.
- 30,000/30,000 exact in the independent portable runtime.
- 30,000/30,000 runtime agreement.
- 3,000 correct outputs accepted by both verifier implementations.
- 24,000/24,000 mutant outputs rejected by both learned screens and both mandatory soundness anchors.
- Two of three verifier contracts required preserved CEGIS revision.
- Zero learned contracts used exact-output digest equality.
- Evidence SHA-256: `5ecedba17a6fdacfe1e9a6ec0ca1938e60b7474d65502f05d0876fbee6f7f771`.

This remains mechanism evidence, not a world-level breakthrough. The low-level grammar, search-family schemas, and verifier predicate grammar are human supplied.
