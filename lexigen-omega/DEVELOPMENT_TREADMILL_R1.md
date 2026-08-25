# LEXIGEN Ω exposed development treadmill R1

Status: development policy; not a breakthrough claim.

## Purpose

LEXIGEN Ω may iterate aggressively on tasks whose scientific/blind status has already been consumed by V6, V7, earlier Lexigen campaigns, public development benchmarks, synthetic tasks, or deliberate debugging. These tasks form the **exposed development treadmill**.

The treadmill exists so Ω can continue improving after negative experiments without corrupting the future global blind campaign.

## Allowed on exposed development tasks

Ω may repeatedly:

- change its semantic representation;
- invent, mutate, compose and delete semantic genes;
- change proposal/search policy;
- evolve searcher implementations and maintain competing lineages;
- use hindsight from correctness/performance failures;
- generate counterexamples and adversarial cases;
- change development-only models and hyperparameters;
- inspect known solutions after a task has been formally moved to the exposed pool;
- rerun development evaluations as often as useful.

These iterations are engineering/research development, not prospective evidence.

## Permanent prohibition

No observation derived from an exposed treadmill task may be relabelled as final blind evidence, causal-transfer evidence, or external-frontier evidence later.

Development memory and final causal memory are separate evidence classes. There is no automatic promotion path between them.

## Infrastructure incidents

Install failures, runner failures, unavailable dependencies, timeouts caused by infrastructure, evaluator crashes, and partially evaluated rows carry zero scientific reward and zero scientific penalty. They may be used to harden infrastructure only.

## Readiness gate before a new blind campaign

Development continues until one frozen Ω candidate satisfies all of the following on the exposed treadmill:

1. multi-task success across at least 5 distinct task families or ecosystems;
2. leave-one-family-out or prospective-development prediction materially above controls;
3. at least 3 reusable semantic genes with positive ablation evidence on multiple exposed tasks;
4. search-policy improvement survives held-out exposed tasks instead of only its training tasks;
5. no single task contributes enough repeated comparisons to dominate mechanism ranking;
6. independent runtime/evaluator checks remain green;
7. a final candidate, budgets, adapters, semantic memory and search policy can be frozen without task-specific rescue.

These are **development readiness conditions only**. Passing them does not establish a breakthrough.

## Final scientific boundary

After readiness, choose a fresh untouched global denominator and freeze Ω before opening task identities/content beyond preregistered metadata. The blind result is recorded whatever happens.

If the blind campaign fails, the result remains failed. Those tasks may subsequently become exposed development tasks for a new research cycle, but the next campaign must use a new untouched denominator and must report the existence of prior failed blind campaigns. We do not keep drawing blind sets until one happens to pass and then describe that as a first-shot breakthrough.

## Current migration

V6 and completed/failed V7 evidence may be used for development once its original scientific result is preserved unchanged. Current live V7 jobs remain read-only until their source campaign seals or terminates them.
