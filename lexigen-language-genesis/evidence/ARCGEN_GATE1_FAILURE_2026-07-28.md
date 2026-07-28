# ARC-GEN external gate 1 — preserved failure

Date: 2026-07-28

Status: **blind external attempt failed before scoring; no breakthrough claim**.

## Frozen commitments

- Lexigen commit used for task selection: `3ff1d046399bff37e5c4bcea50c43d260affe7d5`
- ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- Eligible generator count: `900`
- Selection digest: `e15cbf29f31be0eb9d2f2627e61eb66fecbb8e16a832ca7e00cb38da762b8c1b`
- Selected external task: `bf89d739`
- Demonstrations exposed: `6`
- Hidden test inputs exposed: `20`
- Hidden outputs remained sealed.

## Evidence hashes

- Redacted task SHA-256: `416378ede673947f6891eaa378e462e8a1dfcb905e072b06523115686d80dc29`
- Sealed outputs SHA-256: `555dd62613792e1c8181a2edd571b818be3fccbf7a0ee4065c764ba4a7fedbb9`

## Frozen solver outcome

- Candidate budget: `75,000`
- Maximum program depth: `3`
- Primitive inventory size: `57`
- Candidates evaluated: `20,235`
- Distinct execution signatures: `3,147`
- Single-primitive baseline found: `false`
- Composed language program found: `false`

Because no candidate program was found, no hidden predictions were produced and the sealed outputs were not scored or inspected. This is a permanent negative result for this frozen engine and selected task. The same task must not be retried as a blind claim after modifying the language.

## Post-failure diagnosis boundary

Only after the failure was fixed did development analysis inspect the six already-exposed demonstrations. They show a missing relational drawing operation: connect pairs of equal-colour markers that share a row or column, preserving endpoints and filling interior cells with a derived colour. Any later addition of such a primitive is therefore task-informed development and cannot validate gate 1. It may only be evaluated on a newly selected task derived from a later frozen commit.
