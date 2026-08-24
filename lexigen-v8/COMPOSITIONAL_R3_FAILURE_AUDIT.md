# LEXIGEN V8 IPWM Compositional R3 — Frozen Failure Audit

## Classification

R3 is a **failed frozen synthetic falsification**, not a pass and not real transfer evidence.

The authoritative source is workflow run `32759420540`, job `97534409709`, executing source head `de235a61bfe0d82bd248bb19c8380eda624c900f`. Artifact `9532335796` was uploaded with ZIP SHA-256 `42b75d11b25dfa98470786a31e8ad0dfc42d6f663d62884dcb826fed1ad6d062`.

No real intervention corpus was opened.

## Why GitHub showed green

The workflow step was:

```bash
python lexigen-v8/check_ipwm_r3_result.py --evaluation evidence-r3/evaluation-r3.json | tee evidence-r3/gate-result.json
```

The shell did not enable `pipefail`. The Python process raised `AssertionError`, but `tee` exited successfully, so the workflow step and overall job appeared successful. This is a CI plumbing defect only. It cannot override the frozen scientific checker.

Future scientific gate workflows must use one of:

```bash
set -o pipefail
python ... | tee ...
```

or write gate output without a success-masking pipeline.

## Frozen gates that failed

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| Repository validity null AUROC | <= 0.62 | 0.744827 | FAIL |
| Held-out intervention-family AUROC | >= 0.70 | 0.599724 | FAIL |
| Held-out intervention-family Spearman | >= 0.50 | 0.298699 | FAIL |
| Full - no-alignment AUROC | >= 0.05 | -0.000458 | FAIL |
| Full - no-alignment Spearman | >= 0.10 | 0.040409 | FAIL |
| Full - stratified-null AUROC | >= 0.10 | 0.068027 | FAIL |
| Full - stratified-null Spearman | >= 0.20 | 0.134644 | FAIL |

## What did work diagnostically

The representation retained strong predictive signal when holding out repositories, repository families, or languages:

- repository holdout: AUROC `0.886837`, Spearman `0.827993`;
- repository-family holdout: AUROC `0.867948`, Spearman `0.825062`;
- language holdout: AUROC `0.879751`, Spearman `0.847758`;
- repository speedup global-null maxima were acceptable: AUROC `0.537696`, |Spearman| `0.062976`.

Those diagnostics do **not** rescue the hypothesis. The central R3 target was extrapolation to an intervention family never seen during training. On that test the full model reached AUROC `0.599724` and the no-alignment control reached `0.600182`, so the aligned primitive interaction did not provide the required causal/compositional advantage.

## Scientific interpretation

R3 shows that primitive-aligned program x action features can help ordinary cross-repository prediction, but the current representation does not establish compositional transfer to a genuinely unseen action family. The remaining high validity-null score also indicates unresolved confounding / evaluative leakage in the synthetic setup.

## Frozen consequence

The preregistered R3 protocol states:

1. any frozen gate failure blocks opening real intervention data;
2. if intervention-family transfer fails, kill the current IPWM compositional route rather than create an R3.1 with weaker gates;
3. if validity null remains above `0.62`, real-data execution remains blocked.

Therefore the current IPWM compositional route is **killed at R3**.

## Next step

Return to the pre-real-data architecture tournament. Any successor must be materially different in mechanism, not a threshold-relaxed or generator-tuned R3 variant. Preserve the 20% eventual real search-gain gate and the requirement for prospective causal benefit over strong transfer/world-model controls.
