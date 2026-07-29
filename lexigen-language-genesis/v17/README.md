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
