# Preselection engine amendment

The first snapshot-free validation run (`30370956491`, artifact `8692836324`) passed its mechanical assertions but exposed two scientific flaws during evidence review:

1. Substring matching treated unrelated identifiers such as `reshape` as containing the token `sha`.
2. One weak feature was enough to admit operators whose mechanisms were structurally unrelated to the synthetic task.

No benchmark inventory, holdout name, task source, manifest, payload, public solver or report had been accessed. Therefore the architecture was not yet locked and could be corrected without contaminating a campaign result.

`engine_v2.py` preserves the original engine as negative design history and changes only generic preselection architecture:

- exact lexical atoms replace substring matching;
- operator-specific minimum structural evidence is required;
- mixed precision requires matrix, decomposition and tolerance evidence;
- native one-shot backends require bytes plus crypto/hash/encoding evidence;
- dynamic programming requires recurrence evidence;
- structural initialisation and active-set mechanisms require at least two supporting features;
- correctness controls require an iterative or approximate-verifier context.

The corrected engine must pass stronger semantic validation before any holdout inventory access. The original PR #83 remains preserved and cannot be used as the campaign engine.
