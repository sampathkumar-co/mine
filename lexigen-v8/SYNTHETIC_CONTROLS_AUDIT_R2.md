# LEXIGEN V8 IPWM — Synthetic Control Hardening R2 Audit

## Verdict

**FAIL — do not open the real intervention corpus yet.**

The R2 run was locked before execution at `7d2d0d945fb2228233542a3656964ff4c039440b`. The R1 synthetic generator and the separately frozen 20% real top-k development threshold were unchanged.

The authoritative workflow was `32757803096`. Protocol lock and compilation passed. The frozen assertion stage failed on the completely held-out intervention-family stress test.

## Decisive failure

Frozen requirement:

- held-out intervention-family full positive-speedup AUROC >= **0.62**
- held-out intervention-family full Spearman >= **0.30**

Observed:

- AUROC = **0.5849387840**
- Spearman = **0.2790798890**

The no-cross model reached AUROC **0.5907994253**, slightly above the full model. Therefore the full program×intervention interaction representation did not provide the intended discriminative advantage when the intervention family itself was unseen.

This is exactly the extrapolation capability IPWM was supposed to demonstrate before real data was opened, so the failure is scientifically meaningful rather than plumbing.

## What did work

Repository holdout remained strong: full AUROC **0.8859812572**, Spearman **0.8488541600**. Full beat no-cross by about **+0.1091 AUROC** and **+0.2416 Spearman**. Repository-family and language holdouts were also strong.

The hardened repository speedup null behaved much better than R1: global-null max AUROC **0.5281** and max absolute Spearman **0.0722**. The stratified speedup null also remained well below the full model.

However, repository top-k gain over the frequency baseline was only **17.26%** in R2. The 20% real-data development gate was not applied to this synthetic test, but importantly it was not lowered.

## Additional unresolved warning

The validity null is still suspicious. On repository holdout, the global-null validity AUROC averaged about **0.730** and reached **0.762**. That means the speedup null problem was substantially repaired, but the validity-side null design/model still contains structure that must be understood before any real-data claim.

## Duplicate legacy workflow

PR creation also triggered old workflow run `32757803010`. It expected obsolete file `lexigen-v8/R2_EXECUTION_LOCK.json`, failed in its lock step, never compiled/evaluated the model, and produced no evidence. It is infrastructure-only and non-authoritative. The authoritative R2 run is `32757803096`.

## Claim boundary

R2 is synthetic falsification only. It establishes no real transfer, no causal search benefit, and no breakthrough.

The correct response is **not** to loosen the 0.62/0.30 intervention-family thresholds or modify the synthetic generator. If Lexigen continues, R3 must be a separately preregistered representation/architecture change designed to generalize compositionally to intervention families it never observed during fitting.
