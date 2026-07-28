# Frozen comparison arms

## v4_full

Uses structural fingerprinting, compositional mechanism graph, prior abstract transfer evidence, correctness-risk scoring and the generic failure taxonomy.

## v4_no_transfer

Identical implementation and proposal graph, but all transfer weights are zero. This is the primary ablation for whether prior-task abstractions help.

## random_search

Samples uniformly from the same legal composition graph with the same proposal count and deterministic campaign seed. It cannot use ranking scores.

## template_synthesis

Enumerates only single generic operators. It cannot compose mechanisms or use transfer evidence.

## v3_compatible

Restricted to wrapper/backend substitution, representation changes, direct vectorisation and dtype specialisation. It cannot use transfer memory, multi-operator composition, the failure taxonomy or risk-aware staging.

All arms receive identical candidate-count, training-revision, wall-clock and CPU budgets. A task is scored for every arm; selective omission is forbidden.
