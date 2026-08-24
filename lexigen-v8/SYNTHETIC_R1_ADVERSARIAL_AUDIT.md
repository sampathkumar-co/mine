# LEXIGEN V8 IPWM Synthetic R1 — Adversarial Audit

The R1 pipeline sanity test passed, but it exposed weaknesses that must be repaired **before any real intervention corpus is opened**. R1 evidence is immutable; this audit does not reinterpret it as real transfer.

## Attack 1 — static features are already strong

On leave-one-repository-out synthetic evaluation:
- full positive-speedup AUROC: 0.8788
- static-only AUROC: 0.7890
- full Spearman: 0.8576
- static-only Spearman: 0.6257

The full model is better, but the static baseline alone captures a large part of the constructed signal. A real V8 claim therefore cannot be based merely on full-vs-frequency improvement. It must show **incremental value from the intervention × program-state interaction beyond program-state-only prediction**.

R2 requirement: report and gate full-minus-static AUROC/Spearman/top-k improvement, plus a `no_cross_terms` ablation that contains both feature blocks but cannot represent interactions.

## Attack 2 — shuffled validity null is too predictive

R1 shuffled validity AUROC remained approximately:
- repository holdout: 0.690
- family holdout: 0.666
- language holdout: 0.703

That is far too high for a convincing null. The current implementation independently shuffles labels over the full training fold. With group imbalance and correlated validity/speedup structure, a single global permutation is not a sufficiently diagnostic null; it may also produce unstable class distributions and chance correlations.

R2 requirement:
- use multiple deterministic null permutations;
- shuffle **within intervention-family × repository-family strata where possible**, preserving base rates while breaking context/effect pairing;
- report null distributions rather than one lucky seed;
- compare full score to the high quantile of the null distribution;
- separately permute validity, positive-speedup and log-speedup targets.

## Attack 3 — synthetic top-k is below the real gate

Repository-holdout relative top-k gain over frequency was ~19.20%, below the already-frozen 20% real development threshold. Family/language results were also around 19%.

This is **not a reason to lower the real threshold or tune the synthetic generator**. The synthetic test had a weaker >5% plumbing requirement and correctly passed only that requirement.

R2 action: leave the real 20% threshold unchanged. Do not modify the R1 generator to manufacture a pass.

## Attack 4 — intervention identity alone is weak, but cross terms need a direct test

Intervention-only is near frequency baseline, which is encouraging. But R1 does not distinguish:
- useful program-state × intervention conditional reasoning,
from
- additive use of two independently predictive feature blocks.

R2 requirement: add a `no_cross_terms` model and require full to outperform it prospectively. If no-cross performs similarly, IPWM's claimed conditional-effect representation is unsupported.

## Attack 5 — no held-out intervention family

A model can generalize across repositories while memorizing intervention-family behavior. This is useful but weaker than predicting a transformation family absent from training.

R2 requirement: where at least 3 intervention families exist, run leave-one-intervention-family-out evaluation. This is a development stress test, not necessarily a future confirmatory gate for every task, because unseen intervention schemas may lack meaningful feature support.

## Attack 6 — a real corpus must avoid expert-success selection bias

Training only on expert patches would make the model mostly learn which successful interventions experts chose. V8 requires null/negative intervention outcomes.

Before real data collection, freeze a corpus protocol that includes:
- controlled automatically generated interventions from a fixed transformation catalogue;
- failed correctness interventions;
- speed-neutral and slowing interventions;
- successful interventions;
- repeated measurements;
- identical attempted intervention distributions across repositories where applicable.

Expert patches may be included only in an apprenticeship partition and must not dominate the intervention distribution.

## R1 verdict

**Pipeline sanity: PASS.**

**Real IPWM transfer hypothesis: NOT TESTED.**

**Ready to collect real data: NOT YET.** R2 null controls and incremental-interaction tests must pass first.
