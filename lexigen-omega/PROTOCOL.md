# LEXIGEN Ω Development Protocol

## Development vs blind evidence

All tasks used before Ω9 are development-contaminated and may be inspected, attacked and reused for engineering. They can demonstrate mechanism function but cannot count as final blind evidence.

At Ω9 the following are frozen before final holdout identity selection:

- engine source and portable runtime;
- semantic-gene memory and all fingerprints;
- searcher archive/router;
- model/provider configuration where applicable;
- benchmark adapters;
- task selectors and exclusion/contamination registry;
- per-arm compute, model, evaluator and wall-clock budgets;
- random seeds or seed derivation rules;
- success thresholds;
- causal-credit rules;
- anti-gaming rules;
- evidence schema.

## Final comparison arms

Each blind task must receive equal budgets across at least:

1. `omega_full` — frozen Ω with admitted semantic memory;
2. `omega_no_memory` — identical Ω with admitted learned genes unavailable;
3. `omega_random_memory` — identical Ω with frozen equal-capacity random semantic memory;
4. `omega_static_search` — frozen non-self-evolved search baseline where technically meaningful.

Relevant strong external/domain baselines should also be measured, but they do not replace the internal causal controls.

## Causal-transfer credit

A task can receive causal credit for a semantic gene only if:

- full earns a clean task success;
- the gene is actually used in the selected mechanism;
- full and no-memory are not semantically/mechanistically equivalent;
- no-memory does not reproduce the qualifying advantage;
- random-memory does not reproduce the qualifying advantage;
- exact gene removal or semantic-equivalence-class ablation removes the qualifying advantage;
- budgets are equal;
- the credited target is prospectively separated from the gene's source evidence;
- no post-result revision occurred.

## Integrity

Infrastructure failures before candidate evaluation may be repaired only if the incident is preserved and the repair changes no scientific variable. A repair after observing candidate correctness/performance must be treated as a new preregistration or as contamination, depending on stage.

## Breakthrough boundary

Ω development may produce strong mechanism results without a global breakthrough. The final claim requires the later Ω10/Ω11 preregistered campaign and outside-verifiable frontier evidence. This file intentionally does not retroactively declare the final numeric gate; those exact values must be locked before final holdout selection.
