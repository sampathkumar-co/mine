# ARC-GEN external gate 3 — preserved failure

Date: 2026-07-28

Status: **third blind external attempt exhausted its frozen search budget before scoring**.

## Frozen commitments

- Lexigen commit used for selection: `94aa1f4f23dad16292aa75963381e8496118680f`
- ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- protocol: `arcgen-gate-v4`
- eligible generator count: `900`
- selection digest: `8b1a0c70c1cc1f4927a5e09c24863e686301092d8dd673c857e3fd8f093c1afe`
- selected external task: `305b1341`
- demonstrations exposed: `6`
- hidden test inputs exposed: `20`
- hidden outputs remained sealed.

## Evidence hashes

- redacted task SHA-256: `592e0d1826cfea0956b0091866cc45a5a45a9b3c3f0be8bee3aca441cc42bd2c`
- sealed outputs SHA-256: `c447aeaf83ea99b5ec4efa6210a741a416885b1018253a2846a196c4120d7152`

## Frozen solver outcome

- primitive inventory size: `459`
- maximum program depth: `3`
- candidate budget: `75,000`
- candidates evaluated: `75,000`
- distinct execution signatures: `41,628`
- one-primitive baseline found: `false`
- composed language program found: `false`

No predictions were produced. The hidden outputs were not opened or scored.

## Interpretation boundary

This result shows that adding anchor-spine construction and reflected trajectories did not collapse the broader ARC representation problem. The task contains several multi-cell coloured structures and single-pixel control colours whose outputs expand into large structured regions. Post-failure analysis may identify and encode a new general semantic family, but that extension is permanently ineligible for gate 3 and must be evaluated on a newly selected untouched task.
