# LEXIGEN V8 Architecture Tournament R2 — Capability Pivot

## Status

This document is written after the frozen R3 IPWM compositional route failed and before any successor holdout campaign begins.

R3 is not being rescued. Its preregistered kill rule is binding: no R3.1, no weaker unseen-intervention threshold, no synthetic-generator tuning, and no opening of the blocked real IPWM corpus.

## What R1/R2/R3 taught us

The IPWM sequence produced a useful negative result:

- models could predict intervention outcomes across unseen repositories, repository families and languages;
- stronger controls exposed that this did **not** imply extrapolation to a completely unseen intervention family;
- even a deliberately constrained aligned primitive representation failed to earn the required advantage over a no-alignment control;
- validity-null behavior remained suspicious enough to block real-data execution;
- therefore the route does not justify spending a large campaign budget.

The R3 false-green CI incident also establishes a process requirement: repository status badges are never scientific evidence. Gate-checker exit status and committed result JSON are authoritative.

## Prior-art conclusion

A second prior-art attack makes an architecture-first novelty hunt increasingly unproductive. The following mechanisms already have substantial precedent and may be used only as components or baselines:

- learned compiler cost/performance models and world models;
- causal/feature-aware autotuning and design-of-experiments probing;
- retrieval-guided program optimization;
- LLM/evolution/evaluator loops;
- library/abstraction learning;
- CEGIS, invariant synthesis and counterexample-guided repair;
- equality saturation and semantics-preserving rewrite systems;
- automatic mathematical reformulation and equivalence checking;
- learned or agentic reduction discovery/routing.

V8 must therefore stop treating a renamed combination of these mechanisms as the desired breakthrough.

## Strategic pivot

### Primary target: capability, not architecture novelty

The successor target is a **Frozen Verified Discovery system (FVD)**.

FVD is allowed to use established components. The scientific question is instead:

> Can one frozen autonomous discovery system use experience learned before a holdout campaign to causally increase discovery success on genuinely unseen, unrelated algorithmic tasks, under equal resource budgets, and then reproduce that advantage on an external frontier problem?

A positive answer would be interesting even if individual components are not architecturally novel. A negative answer is equally useful and must terminate the claim.

## What must be frozen

Before official holdout selection/access, commit and hash:

1. discovery engine source;
2. model/runtime identity and prompts where applicable;
3. learned experience artifact;
4. task-independent search grammar / proposal operators;
5. verifier and correctness rules;
6. candidate/evaluator/model-call budgets;
7. control-arm definitions;
8. success thresholds;
9. task-selection algorithm and candidate inventory commitment;
10. anti-contamination rules;
11. result schema and strict gate checker.

No hidden-result-dependent code or threshold update is permitted after this freeze.

## Learned experience artifact

FVD may learn during apprenticeship, but the resulting artifact must be a generic task-independent object with auditable provenance. It may contain abstractions, failure boundaries, search priors, retrieval indices, learned proposal policy state, compressed trajectories or other reusable experience.

It must **not** contain future holdout identities, holdout solutions, benchmark-specific patches, or manually injected task-specific solver logic.

The causal object is the whole frozen experience artifact. The confirmatory question is whether removing or destroying that artifact removes discovery advantage.

## Official control arms

Every holdout receives equal resource ceilings:

1. `fvd_full` — frozen engine + frozen learned experience artifact.
2. `fvd_no_experience` — same engine and budgets with learned experience physically removed/zeroed.
3. `fvd_shuffled_experience` — same artifact size/interface, but task-to-experience associations or reusable-state semantics are deterministically permuted where meaningful.
4. `retrieval_only` — retrieval from the same apprenticeship corpus without the full learned controller/abstraction machinery.
5. `evolution_only` — generic evaluator-guided evolutionary search with the same evaluator budget.
6. `predecessor_v7` — frozen predecessor where technically applicable.

A task cannot earn causal-transfer credit merely because full finds a good answer.

## Development gates before an official campaign

The engine must first survive non-claiming development tests that verify:

- the same frozen interface can express all selected task families;
- every candidate has independent correctness verification;
- no-experience and shuffled-experience controls are physically effective rather than cosmetic;
- budget accounting is exact;
- hidden task data cannot be accessed before the commit barrier;
- failed gate processes propagate nonzero exit codes to CI;
- result artifacts are deterministic or explicitly record allowed randomness/seeds.

These tests are plumbing evidence only.

## Confirmatory campaign shape

The first FVD campaign must contain at least **6 untouched tasks spanning at least 4 unrelated families**. Prefer 8–10 tasks if resource cost is acceptable.

Holdouts must be selected only after the full engine/artifact/protocol freeze from a committed inventory that was not used during apprenticeship.

The same frozen engine is used for every task. Human task-specific solver contribution must be zero.

## Frozen internal success gates

A campaign passes only if **all** hold:

- `fvd_full` clean authoritative wins on at least 4 of 6 tasks (or >= 2/3 if denominator is larger);
- clean wins span at least 4 unrelated families;
- full beats `fvd_no_experience` by at least 2 task wins and at least 20 percentage points;
- full beats `fvd_shuffled_experience` by at least 2 task wins;
- full beats `retrieval_only` by at least 2 task wins;
- full beats `evolution_only` by at least 2 task wins;
- full beats `predecessor_v7` by at least 2 task wins where the comparison is applicable;
- at least 2 tasks qualify as causal transfer wins;
- causal wins span at least 2 unrelated holdout families;
- at least 2 distinct pieces/classes of learned experience are causally implicated;
- exact removal/permutation replay eliminates the qualifying advantage on each causal win;
- median human task-specific solver contribution is zero;
- no task swap/drop/reclassification after result access;
- no threshold or resource-budget change after freeze;
- every authoritative verifier accepts the final outputs.

Any failed conjunct means the campaign failed. Partial positives are reported as partial positives, never promoted to a pass.

## Causal-transfer win definition

A holdout earns `causal_transfer_win=true` only if:

1. full produces an independently verified qualifying result;
2. no-experience fails the native gate or is materially worse under the preregistered equal-budget metric;
3. shuffled-experience does not explain the gain;
4. at least one nontrivial learned artifact element used by full has apprenticeship provenance from a different task family;
5. removing/permuting that element or the complete artifact removes the qualifying advantage in replay;
6. retrieval-only and evolution-only do not match the result within the frozen budget;
7. all proposal/evaluation logs respect the holdout access boundary.

## External frontier gate

Passing the internal campaign is **not** a world-level breakthrough.

Only after an internal pass may FVD attempt a frozen external frontier campaign. A strong final claim requires all of:

- at least one objectively new, independently verifiable frontier result, preferably two in unrelated families;
- evidence the result was not present in the frozen literature/data snapshot;
- the same frozen FVD system, not a frontier-specific rescue branch;
- causal dependence on pre-frontier learned experience under a clean ablation/control;
- independent clean-room reproduction by a separate implementation/environment;
- public or otherwise inspectable evidence sufficient for third-party verification.

Until then the correct label is an internal autonomous-discovery result, not a world breakthrough.

## Tournament R2 verdict

**Kill the IPWM architecture route. Do not search for another novelty-sounding module just to create V8.**

Proceed with FVD as an empirical capability program. Components may be established; the burden of proof moves to frozen cross-domain causal transfer, strict controls, external novelty and independent reproduction.
