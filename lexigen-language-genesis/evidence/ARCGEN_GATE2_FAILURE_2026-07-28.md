# ARC-GEN external gate 2 — preserved failure

Date: 2026-07-28

Status: **second blind external attempt failed before scoring; no breakthrough claim**.

## Frozen commitments

- Lexigen commit used for selection: `aef25446282206ac5f9fd537eb895f71edea8e50`
- ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- Protocol: `arcgen-gate-v2`
- Eligible generator count: `900`
- Selection digest: `1331cedfdff0bea710ebba50656959ece410aefa8217dfeadaa38133cbdb399c`
- Selected external task: `a3df8b1e`
- Demonstrations exposed: `6`
- Hidden test inputs exposed: `20`
- Hidden outputs remained sealed.

## Evidence hashes

- Redacted task SHA-256: `05b78a5d01fca8ab1aef92eac24135d8d95bf5b72c0456611469ea6e5f8b951c`
- Sealed outputs SHA-256: `571267ed46455cc6f93290d4aa0eab71bec5b6b4009073410688ada868a5ef1a`

## Frozen solver outcome

- Candidate budget: `75,000`
- Maximum program depth: `3`
- Primitive inventory size: `45`
- Candidates evaluated: `11,475`
- Distinct execution signatures: `1,986`
- Single-primitive baseline found: `false`
- Composed language program found: `false`

No predictions were produced, so the sealed outputs were neither opened nor scored. This is a permanent negative result for gate 2.

## Post-failure diagnosis boundary

After the attempt was irreversibly recorded as a failure, analysis of the six exposed demonstrations showed another missing representational family: trajectory drawing with boundary reflection. A single marker begins at the lower-left corner and advances diagonally upward, reflecting horizontally at grid boundaries to form a deterministic zig-zag path. The current language contains static geometry, cropping, colour transforms, and aligned-segment construction, but no stateful ray/trajectory operator with reflection.

Any future reflected-trajectory primitive is informed by this failed task and cannot be used to reclassify gate 2. It may only be evaluated on a newly selected external task after a new engine commit.
