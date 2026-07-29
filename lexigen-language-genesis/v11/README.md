# Lexigen v11 — Unified Compositional Language

Status: initialized after gate-9 permanent failure; no breakthrough claim.

## Architectural correction

Versions 7–10 each proved that one typed language family could be synthesized and independently executed, but fresh blind tasks repeatedly fell outside the newest family. v11 replaces the sequence of isolated grammars with a common typed pipeline and shared search protocol.

Every candidate program is serialized as stages over explicit types:

1. `Grid -> Scene` — extract cells, objects, regions, graphs, markers or seeds;
2. `Scene -> Relations` — construct adjacency, containment, correspondence or sensor relations;
3. `Relations -> Plan` — aggregate, select, derive vectors, layouts or transitions;
4. `Plan -> Grid` — transform and render;
5. optional `State -> State` loop — execute bounded state machines.

The registry may expose generic substrates learned by earlier versions, but a candidate must still synthesize all task parameters and stage composition from demonstrations. Finished task names and generator identities are forbidden.

## v11 development target

Permanent gate-9 task `33067df9` requires a new pipeline:

- parse a sparse regular symbol lattice;
- allocate a larger uniform tile layout;
- expand nonzero symbols into rectangular regions;
- construct equality-labelled horizontal and vertical adjacency edges;
- render horizontal edges first;
- render vertical edges only when neither endpoint already participates in a horizontal edge.

The pipeline is expressed as generic lattice, layout, relation, precedence and rendering productions. It must transfer over fresh official-rendered cases and reproduce in a separately implemented runtime.

## Blind gate requirement

After v11 and all prior regressions pass independently, the exact engine commit is frozen before an untouched task is selected. A credible external candidate requires:

- v6–v10 fixed baselines fail;
- a v11 compositional program is generated without post-selection edits;
- primary and portable predictions agree;
- candidate and predictions are committed before scoring;
- one-shot hidden accuracy is exact;
- ablation removes the capability;
- the generated production later transfers to another untouched family.
