# Mini-ORIGIN

**Autonomous search for local, damage-tolerant computational substrates.**

Mini-ORIGIN is the first research scaffold for ORIGIN COMPUTE: an attempt to search below conventional neural-network architecture design and discover local computational laws that can preserve information, transmit signals, adapt, and recover after structural damage.

> Status: early, falsifiable research prototype. It is not yet evidence of a new form of computation or a replacement for neural networks.

## What v0.1 implements

- A toroidal two-dimensional cell universe.
- One shared local rule executed independently by every cell.
- Cells observe only their own state, neighbour mean, neighbour maximum, and local input.
- A genome describing the local update law.
- Mutation and crossover over local laws.
- Three measurable tasks: noisy memory, signal relay, and post-damage recovery.
- Complexity-penalized evolutionary selection.
- Deterministic replay through explicit random seeds.
- CPU-only tests and a GitHub Actions experiment.

The candidate substrate has no global controller while it runs and receives no gradient updates. Evolution occurs between complete evaluations.

## Architecture

```text
Genome population
      |
      v
Local-rule cellular universes
      |
      +--> noisy memory test
      +--> signal relay test
      +--> damage/recovery test
      |
      v
Fitness + complexity penalty
      |
      v
Elitism, crossover, mutation
```

## Run in GitHub Codespaces or locally

```bash
cd mini-origin
python -m pip install -e ".[dev]"
pytest -q
mini-origin demo --generations 12 --population 24 --output results/demo.json
```

The GitHub Actions workflow runs the tests and a compact reproducible discovery experiment automatically. The resulting JSON is uploaded as a workflow artifact.

## Research claim being tested

> Can search discover a shared local computational rule that performs multiple information-processing tasks and recovers after damage without being given a neural-network architecture?

The repository does **not** assume that the answer is yes. Negative results are useful: they indicate whether the substrate, search space, fitness design, or central hypothesis must change.

## Scientific safeguards

A future candidate will not be called a discovery unless it survives:

1. held-out seeds and grid sizes;
2. unseen noise and damage distributions;
3. ablation of each claimed mechanism;
4. equal-compute neural, reservoir, cellular-automata, and random-search baselines;
5. complexity and operation-count controls;
6. transfer to tasks excluded from search;
7. independent reruns.

## Roadmap

### v0.2 — Primitive archive

Add novelty search and quality-diversity archives for memory, routing, copying, and logic-like primitives.

### v0.3 — Local learning

Separate inherited update laws from within-lifetime plasticity. A substrate must improve from examples without evolving a new genome for each task.

### v0.4 — Regeneration

Introduce structural cell death, connection damage, regrowth, and recovery measurements against nontrivial controls.

### v0.5 — Multi-task substrate

Require one substrate to learn multiple hidden tasks and generalize to larger worlds.

### v1.0 — Research evaluation

Add rigorous baselines, hidden benchmarks, ablations, efficiency accounting, and a reproducible paper-ready experiment suite.

## Repository layout

```text
src/mini_origin/genome.py      evolvable local law
src/mini_origin/substrate.py   cell-universe simulator
src/mini_origin/tasks.py       falsifiable task definitions
src/mini_origin/search.py      evolutionary discovery loop
src/mini_origin/cli.py         experiment command
tests/                         deterministic tests
```

## Current limitations

- The substrate family is still human-selected.
- The update law is a small differentiable equation, not an invented compiler or physical medium.
- Search is simple elitist evolution, not open-ended discovery.
- The recovery task tests state restoration, not rebuilding arbitrary structures.
- Fitness design can create shortcuts and must be attacked continuously.

These limitations are explicit so later results cannot be overstated.
