# Lexigen v13 Fresh Blind Campaign

## Immutable commitments

- Frozen Lexigen engine: `18885982a880eb903f15752be04f7fd7ab2e8c2f`
- Pinned ARC-GEN snapshot: `a15cbdb44c776610aeeb9f487a06af875d3d0878`
- Campaign size: 40 deterministic gates
- Per valid gate: 6 demonstrations and 20 hidden tests
- No retries, replacements, task cherry-picking, or language edits inside the campaign

## Strong freshness boundary

The selector permanently excludes all 36 ARC-GEN task identities exposed by earlier Lexigen external gates, the v11 campaign, the v12 campaign, and v13 post-failure development. Selection is deterministic over the remaining eligible tasks from the pinned repository snapshot. All 40 identities are committed together before any demonstration or hidden output is generated. Deterministic collisions remain in the denominator and are not replaced.

## Sealing boundary

Hidden outputs are generated only on the Sampath device and immediately moved into a non-Git vault. Git stores only task identities, demonstrations, test inputs, hashes, generator-invalid records, solver checkpoints, generated programs and predictions. Generator failures remain in the denominator and are never replaced.

No hidden output may be read or scored until the corresponding prediction checkpoint has been committed remotely. Scoring verifies the committed manifest hash, sealed-output hash, prediction hash, task identity and prediction count.

## Candidate rule

A gate is a v13-only blind candidate only when:

1. v6 through v12 fail to synthesize an exact demonstration program;
2. v13 synthesizes an exact program from demonstrations;
3. the independent portable interpreter exactly agrees on demonstrations and all 20 predictions;
4. predictions are committed before hidden scoring; and
5. all 20 hidden outputs match exactly.

One success is reported as a blind candidate. A campaign-level breakthrough requires at least two exact v13-only successes on distinct task identities and distinct selected latent-operator families, with all failures and invalid generators preserved.
