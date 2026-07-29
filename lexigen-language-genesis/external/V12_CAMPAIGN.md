# Lexigen v12 Blind External Campaign

## Frozen language

- Parent language commit: `8584f5600c6466185806a7940e9e00b42c004c7e`
- ARC-GEN commit: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- Gate count: 20

## Precommitted gate identities

- `v12-campaign-01`
- `v12-campaign-02`
- `v12-campaign-03`
- `v12-campaign-04`
- `v12-campaign-05`
- `v12-campaign-06`
- `v12-campaign-07`
- `v12-campaign-08`
- `v12-campaign-09`
- `v12-campaign-10`
- `v12-campaign-11`
- `v12-campaign-12`
- `v12-campaign-13`
- `v12-campaign-14`
- `v12-campaign-15`
- `v12-campaign-16`
- `v12-campaign-17`
- `v12-campaign-18`
- `v12-campaign-19`
- `v12-campaign-20`

## Immutable rules

1. All task selections must be committed together before any demonstrations are generated.
2. Every gate counts in the denominator, including baseline wins, invalid generators and infrastructure failures.
3. The v7-v12 language implementation trees must remain byte-identical to the frozen parent commit.
4. Each task receives six demonstrations and twenty hidden tests.
5. Hidden outputs are stored outside Git and only their SHA-256 commitments may enter the repository before scoring.
6. A candidate must be remotely committed before one-shot hidden scoring.
7. No candidate retry, task replacement, grammar edit or budget increase is permitted inside the campaign.
8. A v12 language-growth win requires v6-v11 to fail, v12 primary and portable runtimes to agree, and all twenty hidden predictions to score exact.
9. A world-level claim requires multiple blind wins across distinct task families plus independent reproduction; one isolated win is only a breakthrough candidate.
