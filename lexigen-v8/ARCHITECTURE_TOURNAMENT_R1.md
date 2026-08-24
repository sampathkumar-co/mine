# LEXIGEN V8 Architecture Tournament R1

## Purpose

V7 is sealed as a failed confirmatory GSO causal-transfer pilot. V8 must not be an incremental rescue of V7 and must not alter V7 evidence. This tournament attacks candidate successor architectures before implementation.

The V7 failure to fix is specific: V7 could optimize, but the equally budgeted no-library arm repeatedly reconstructed the same useful mechanism classes. The learned macro library therefore failed to provide a unique or sufficiently concentrated causal capability.

## Prior-art attack snapshot

The following families are treated as established prior art and are not acceptable as the core V8 breakthrough claim:

1. **Bigger learned macro/library alone — REJECTED.** DreamCoder-style library learning, LAPS, Stitch and abstraction-learning systems already learn reusable program abstractions and/or search guidance.
2. **Retrieval memory over slow/fast examples — REJECTED.** Retrieval Augmented Search (RAS) already retrieves optimization examples to guide LLM program optimization; AEGIS decomposes examples into atomic edits.
3. **Generic LLM + evolutionary evaluator loop — REJECTED.** FunSearch and AlphaEvolve already establish this pattern at high capability and scale.
4. **Learned search heuristic alone — REJECTED.** Neural-guided synthesis, LAPS, RL/value-guided synthesis and decades of meta-learning already cover this family.
5. **Generic learned compiler cost model alone — REJECTED.** MLGO, ProGraML, learned compiler heuristics, Meta LLM Compiler and related autotuning work already learn program representations and optimization decisions.
6. **Equality-saturation/rewrite engine alone — REJECTED.** E-graph/equality-saturation optimization is established.

These components may be used as baselines or infrastructure, but reproducing them is not a V8 breakthrough.

## Surviving candidate: Interventional Performance World Model (IPWM)

### Core hypothesis

A system can learn, from *measured interventions on prior repositories*, a transferable model of the conditional performance effect of semantic code transformations. On a genuinely unseen repository, before candidate timing feedback, the frozen model should predict which interventions are likely to help, how much they may help, when they are unsafe, and which intervention chains deserve scarce evaluator budget.

This is deliberately stronger and more falsifiable than V7's static macro library.

### What is learned

The learned object is not a list of recipes. It is a conditional intervention-effect model:

`P(delta_metrics, validity | program_state, intervention, environment)`

where:
- `program_state` is a typed performance graph capturing control/data flow, allocation, representation, dispatch, loop structure, external calls and cheap runtime counters;
- `intervention` is a typed semantic-preserving transformation schema with preconditions and correctness obligations;
- `environment` includes language/runtime/compiler/hardware descriptors;
- `delta_metrics` includes runtime, memory, allocations, compilation cost and other preregistered metrics.

The model must output uncertainty, predicted direction, approximate magnitude, applicability and likely failure modes.

### Apprenticeship data

Training data is generated from many repositories by controlled interventions and measured outcomes, not only successful human/expert patches. It must preserve negative and null interventions as first-class evidence.

Each record contains:
- pre-intervention performance graph;
- intervention schema and concrete edit;
- correctness result;
- before/after metrics with repeated measurements;
- environment descriptor;
- provenance and contamination status;
- whether the intervention was randomized, searched, or naturally observed.

Randomized or deliberately diversified interventions are important because purely observational successful patches create severe selection bias.

### Holdout-time cycle

For a frozen unseen task:
1. construct the permitted program/performance state;
2. IPWM predicts a ranked distribution over intervention schemas **before any candidate timing feedback**;
3. a deterministic search compiler converts the top predictions into bounded edit-program proposals;
4. the same LLM/compiler used by controls instantiates code edits;
5. correctness and timing evaluators score candidates;
6. feedback may update only the task-local candidate queue within the frozen rule, never the IPWM weights;
7. final patches are frozen and evaluated authoritatively.

### Why this directly targets V7's failure

V7 stored a tiny coarse macro vocabulary that controls could trivially reconstruct. IPWM instead attempts to transfer *conditional effect knowledge*: which transformations are valuable in which structural contexts and which are not. The learned information is tested prospectively by pre-feedback prediction accuracy and search efficiency.

The final patch is **not required to be mechanistically unique**. If a no-model control could eventually discover the same mechanism with much larger search, that does not erase causal transfer. The confirmatory question is whether the frozen learned world model changes discovery probability / evaluator sample complexity at the preregistered equal budget, and whether ablation removes that advantage.

## Mandatory controls

Every confirmatory holdout gets at least:

1. `v8_full_ipwm`: frozen IPWM + frozen search compiler.
2. `v8_no_world_model`: same intervention vocabulary, LLM, evaluator, search depth and budget; no learned effect model.
3. `v8_shuffled_world_model`: same model architecture/checkpoint statistics but intervention-effect associations shuffled/frequency-matched.
4. `retrieval_baseline`: RAS-style retrieval from the same apprenticeship corpus.
5. `evolution_baseline`: generic evaluator-guided evolutionary search with equal evaluator budget.
6. `v7_baseline`: frozen V7-style learned macro search where feasible.

## Prospective transfer gates before optimization credit

V8 cannot claim transfer merely because it later finds a fast patch. Before timing feedback on each holdout, the frozen IPWM predictions are committed.

Required predictive gates across unseen repositories:
- intervention validity AUROC / calibrated probability materially above frequency and static-feature baselines;
- positive-vs-nonpositive speedup direction prediction materially above baselines;
- rank correlation between predicted and measured intervention value above preregistered threshold;
- prediction advantage survives repository-family and language holdouts;
- shuffled-label/world-model control collapses the predictive advantage;
- uncertainty is calibrated enough that high-confidence predictions outperform low-confidence predictions.

If these fail, optimization wins cannot be called learned-world-model transfer.

## Causal optimization credit

A task can earn `causal_transfer_win=true` only if all hold:

1. full earns a clean authoritative optimization win;
2. the winning intervention chain was in the frozen pre-feedback IPWM-ranked search distribution;
3. at equal evaluator/model budget, no-world-model and shuffled-model controls fail the native success gate or are materially worse under preregistered sample-efficiency/performance criteria;
4. exact IPWM removal or intervention-effect permutation removes the advantage;
5. retrieval and generic evolution baselines do not explain the same advantage at equal budget;
6. prediction and search logs show no hidden expert/test leakage;
7. anti-gaming and correctness audits pass.

Final-patch semantic equivalence with a control is allowed; what matters is frozen-budget discovery causality and ablation.

## Kill attacks against IPWM

IPWM is killed before a large campaign if any of these occur in development:

- it cannot predict intervention effect direction on held-out repositories better than simple frequency/static-feature baselines;
- gains disappear under repository-family or language holdout;
- shuffled intervention labels perform similarly;
- a retrieval baseline matches the same predictive/search benefit, showing the world model adds no new capability;
- its best predictions are dominated by generic folklore such as caching/preallocation and do not condition meaningfully on program structure;
- environment shift makes predictions unusably unstable;
- model uncertainty is uncalibrated and search collapses to brute-force evaluation;
- the no-world-model control matches full at equal evaluator budget on development holdouts;
- training requires expert target patches from future benchmark distributions in a way that destroys a clean holdout boundary.

## Breakthrough boundary

Even a successful V8 internal campaign would not itself justify a world-level AI breakthrough claim. A strong claim additionally requires:
- larger preregistered denominator across independent repository families;
- independent clean-room reproduction;
- at least one external frontier result not merely matching a hidden expert patch;
- evidence that the learned intervention model transfers to a domain/repository family absent from apprenticeship;
- clear comparison to AlphaEvolve-style evolution, retrieval-based optimization, learned compiler models and library-learning systems.

## Tournament verdict

**Do not implement a bigger V7 macro library, retrieval-only system, generic evolutionary system, or generic cost model as V8.**

**Proceed only with a small falsification prototype of IPWM.** Its first milestone is not code optimization. Its first milestone is prospective intervention-effect prediction on untouched repositories. If that prediction gate fails, kill IPWM before spending on a full autonomous optimization campaign.
