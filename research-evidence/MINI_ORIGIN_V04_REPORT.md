# Mini-ORIGIN v0.4 Research Report

Date: 2026-07-27

## Verdict

**Replicated internal project milestone. Not a world-level unconventional-computing breakthrough.**

Mini-ORIGIN v0.4 replaced smooth neighbour blending with a competitive signed-source substrate. Each cell locally compares its current signal with north, south, west, and east sources, then applies an evolved soft selection, gain, scale, and inertia.

## Why the architecture changed

Earlier experiments exposed three failures:

1. The v0.1 toroidal relay benchmark made opposite edges adjacent, invalidating the original relay result.
2. Corrected smooth local laws could transmit signals on clean grids but were brittle at hidden distances.
3. Domain-randomized smooth laws improved damaged-grid performance but collapsed on some larger unseen layouts.

The failure suggested a representational limitation rather than a lack of evolutionary compute: averaging neighbouring signals destroys polarity and amplitude when a route must choose between alternate paths.

## Experiment

Five independent searches used seeds 41–45. Each search:

- started from random competitive-genome parameters;
- trained on changing grid sizes, aspect ratios, damage fractions, and obstacle layouts;
- used both positive and negative signals;
- was re-ranked against a fixed validation suite;
- was evaluated on hidden grids up to 72×37 with 37% damage;
- was compared with a transparent signed max-flood control.

## Results

| Metric | Result |
|---|---:|
| Runs | 5 |
| Passing runs | 5 |
| Success rate | 100% |
| Hidden worst-case minimum | 0.8631 |
| Hidden worst-case median | 0.8946 |
| Hidden worst-case mean | 0.9091 |
| Hidden worst-case maximum | 0.9558 |
| Hand-control worst case | 0.9842 |
| Median fraction of hand control | 90.9% |

Strongest run, seed 45:

| Hidden environment | Score |
|---|---:|
| 48×25, 33% damage | 0.9558 |
| 60×31, 35% damage | 0.9991 |
| 72×37, 37% damage | 0.9852 |

## What was actually learned

The strongest rule evolved:

- a high competitive temperature, producing near-discrete local source selection;
- low inertia, allowing the signal wave to move rapidly;
- a signal scale close to bipolar-amplitude preservation;
- a strong preference for the west source, supporting eastward progress;
- enough north/south competitiveness to route around blocked paths.

This is a coherent local routing law and generalizes to hidden obstacle distributions.

## Why it is not yet a major breakthrough

- The competitive operation was introduced by us rather than invented by the system.
- A transparent max-flood rule remains a stronger and simpler control.
- Generation-zero populations already contained high-performing candidates, showing that the redesigned search space carries much of the solution.
- No within-lifetime learning, structural regrowth, primitive invention, or efficiency advantage has been demonstrated.
- Flood-like signal propagation and competitive neighbour selection are known computational ideas.

## Next scientific gate

The project should now stop optimizing relay accuracy. The next defensible target is:

> Can one local competitive substrate change its behaviour from examples during its own lifetime, learn multiple unseen mappings, and recover those learned mappings after damage without evolving a new genome?

That would move Mini-ORIGIN from discovering a robust transport primitive toward discovering a local learning law.

## Evidence

- `research-evidence/mini-origin-competitive-v4-summary.json`
- `research-evidence/mini-origin-competitive-v4-seed45.json`
- GitHub Actions run `30272710239`
- Pull request `#22`
