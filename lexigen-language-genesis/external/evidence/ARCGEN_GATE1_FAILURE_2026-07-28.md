# ARC-GEN external gate 1 — preserved failure

Date: 2026-07-28

Status: **failed before hidden scoring; hidden outputs remain unopened**

## Frozen identities

- protocol: `arcgen-gate-v1`
- Lexigen pre-access commit: `3ff1d046399bff37e5c4bcea50c43d260affe7d5`
- ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- eligible ARC-GEN tasks: `900`
- selection digest: `e15cbf29f31be0eb9d2f2627e61eb66fecbb8e16a832ca7e00cb38da762b8c1b`
- selection index: `675`
- selected external task: `bf89d739`

## Sealed package

- demonstrations: `6`
- hidden pairs: `20`
- redacted package SHA-256: `416378ede673947f6891eaa378e462e8a1dfcb905e072b06523115686d80dc29`
- sealed outputs SHA-256: `555dd62613792e1c8181a2edd571b818be3fccbf7a0ee4065c764ba4a7fedbb9`

Only demonstrations and hidden inputs were viewed. The selected ARC-GEN task module and sealed outputs were not opened.

## Frozen solver result

- primitive inventory size: `57`
- maximum program depth: `3`
- candidate budget: `75,000`
- candidates actually evaluated: `20,235`
- distinct semantic signatures: `3,147`
- one-primitive baseline found: `false`
- compositional language program found: `false`

The demonstrations require drawing a new-colour segment between same-colour anchor cells sharing a row or column. The frozen language had colour transforms, geometric transforms, cropping, tiling and component extraction, but no executable relation for connecting aligned anchors.

## Interpretation

This is a useful external representation-failure diagnosis, not a benchmark success. No task prediction was committed, so hidden scoring was not performed. Any language extension learned from this failure is prohibited from being evaluated on gate 1. It must be frozen and tested on a newly selected untouched external task.
