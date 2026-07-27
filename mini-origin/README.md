# Mini-ORIGIN

**Autonomous search for local, damage-tolerant computational substrates.**

Mini-ORIGIN is a CPU-only research scaffold for ORIGIN COMPUTE: an attempt to search below conventional neural-network architecture design and discover local computational laws that preserve information, transmit signals, learn, and recover after damage.

> Current status: replicated within-lifetime local-learning breakthrough inside the project. Mini-ORIGIN has not discovered a universal replacement for neural networks or conventional hardware.

## Current breakthrough — v0.5 within-lifetime feedback learning

One inherited, dimension-agnostic plasticity law now learns a newly generated linear mapping from examples during its own lifetime. The genome is fixed before the mapping exists and is not changed between tasks.

After learning, up to 65% of the distributed memory cells are deleted. The surviving cells retain almost all learned performance.

### Accepted benchmark

The first apparent v0.5 success was rejected because isotropic examples allowed a Hebbian correlation shortcut. The accepted benchmark therefore uses:

- strongly correlated, ill-conditioned training examples;
- isotropic test queries;
- hidden orthogonal mappings generated inside each lifetime;
- evolution only on dimensions 3–4;
- hidden evaluation on dimensions 5, 6, and 8;
- hidden condition numbers up to 36;
- observation noise and up to 65% cell deletion;
- frozen-memory, feedback-ablation, hand-Hebbian, and hand-delta controls.

Five independent GitHub cloud searches used seeds 51–55. All five crossed the predefined breakthrough gate.

| Metric | Result |
|---|---:|
| Independent searches | 5 |
| Passing searches | 5 |
| Success rate | 100% |
| Hidden worst-case range | 0.8527–0.8599 |
| Hidden median | 0.8544 |
| Minimum post-damage retention | 99.44% |
| Median post-damage retention | 99.68% |
| Median advantage over correlation-only control | 0.7074 |
| Median fraction of hand-delta performance | 104.10% |

The strongest evolved law, seed 55, scored:

- `0.9877` on 5D mappings, condition number 16, after 45% cell death;
- `0.9529` on 6D mappings, condition number 24, after 55% cell death;
- `0.8599` on 8D mappings, condition number 36, after 65% cell death.

Its aggregate controls scored:

| Control | Score |
|---|---:|
| No learning | 0.1212 |
| Prediction feedback removed | 0.0515 |
| Hand Hebbian learning | 0.1462 |
| Hand delta learning | 0.8207 |
| Evolved law strict hidden worst case | 0.8599 |

### Learned mechanism

The strongest rule evolved an effective teacher coefficient of `+3.6864` and an effective prediction-feedback coefficient of `-3.5718`.

Those near-opposite coefficients implement a high-gain local error-correction rule:

```text
local memory change ≈ gain × (target - prediction) ⊗ input
```

Each cell receives its own noisy observations and updates only its own memory. Roughly 29.5% of examples are hidden from each cell, so cells develop overlapping but non-identical memories. Mean/median consensus across survivors preserves the mapping after severe deletion.

### Honest interpretation

This is a **real replicated project breakthrough** because the mapping is generated after the genome is fixed, learning occurs during the substrate's lifetime, larger dimensions and covariance structures are hidden, most cells are removed after learning, shortcut controls fail, and five independent searches reproduce the result.

It is not yet a world-level learning breakthrough. The memory matrix, supervised target signal, and candidate plasticity operators remain human-designed.

Permanent evidence:

```text
research-evidence/mini-origin-plasticity-v5-summary.json
research-evidence/mini-origin-plasticity-v5-seed55.json
research-evidence/MINI_ORIGIN_V05_REPORT.md
```

## Previous milestone — v0.4 competitive local routing

Mini-ORIGIN v0.4 discovered robust parameters for a competitive signed-neighbour routing substrate. Five of five searches transmitted bipolar signals across unseen grids with persistent dead cells and partial walls, reaching a hidden median of `0.8946` and about 90.9% of a transparent max-flood control.

The earlier v0.1 toroidal relay result was explicitly rejected because opposite edges were accidentally adjacent. That failure led to fixed boundaries, anti-leakage checks, hidden sizes, damage distributions, and explicit controls.

## Implemented systems

- Fixed-boundary and legacy toroidal cellular universes.
- Smooth gated directional local laws.
- Competitive signed-neighbour routing laws.
- Distributed local plastic memories.
- Dimension-agnostic feedback-plasticity genomes.
- Noisy memory, bounded relay, repair, persistent-damage routing, and lifetime-learning tasks.
- Correlated-training/isotropic-query anti-shortcut benchmarks.
- Frozen, identity, random, Hebbian, delta, feedback-ablation, and explicit routing controls.
- Curriculum evolution, worst-case selection, hidden dimensions, and independent cloud replication.

## Run in GitHub Codespaces or locally

```bash
cd mini-origin
python -m pip install -e ".[dev]"
pytest -q

python -m mini_origin.plasticity_v5 \
  --seed 55 \
  --population 52 \
  --generations 30 \
  --output results/plasticity-v5-seed-55.json
```

GitHub Actions runs the tests, independent search matrix, and aggregate evaluation without requiring a personal computer.

## Research progression

### v0.1 — Initial scaffold

Implemented genomes, cellular worlds, memory, relay, repair, evolution, deterministic replay, tests, and cloud execution. Its toroidal relay claim was later rejected.

### v0.2 — Strict bounded relay

Added fixed boundaries, directional features, sign-balanced evaluation, hidden distances, leakage penalties, and controls. Four of fifteen runs crossed an internal threshold, but a hand east-copy rule was stronger.

### v0.3 — Damage and domain randomization

Added dead cells, partial walls, larger unseen grids, changing layouts, lower-tail selection, and fixed validation. Smooth weighted-sum laws remained brittle.

### v0.4 — Competitive routing

Changed the primitive from neighbour blending to competitive signed source selection. Five of five searches approached an explicit max-flood control across unseen damaged grids.

### v0.5 — Within-lifetime feedback learning

Added distributed local memories and a dimension-agnostic plasticity law. Five of five searches learned hidden mappings from ill-conditioned examples, transferred to larger unseen dimensions and isotropic queries, survived up to 65% cell deletion, defeated correlation-only controls, and slightly exceeded the fixed hand-delta control on the aggregate hidden benchmark.

## Next major gate

The next result should remove another major human assumption rather than merely increase v0.5 scores:

1. **Sparse-reward or self-supervised learning:** learn without receiving the full target vector at every update.
2. **Nonlinear and temporal tasks:** learn stateful programs, sequences, or nonlinear transformations.
3. **Structural regeneration:** regrow deleted memory cells and reconstruct learned knowledge from neighbours.
4. **Plasticity-language invention:** allow search to create or compose its own local update operators.
5. **Equal-compute baselines:** compare against neural, reservoir, and established associative-learning methods under matched memory and operations.

## Scientific safeguards

No result will be promoted beyond an internal breakthrough unless it survives:

1. held-out seeds, dimensions, layouts, mappings, and distributions;
2. explicit hand-designed controls;
3. mechanism ablations and shortcut attacks;
4. equal-compute neural, reservoir, and cellular baselines;
5. operation, memory, and complexity accounting;
6. tasks excluded from evolution;
7. independent reruns;
8. external review and literature comparison.

## Repository layout

```text
src/mini_origin/genome.py          smooth evolvable local law
src/mini_origin/substrate.py       initial cellular simulator
src/mini_origin/tasks.py           memory, relay, and repair tasks
src/mini_origin/search.py          initial evolutionary loop
src/mini_origin/research_v2.py     strict fixed-boundary relay study
src/mini_origin/resilience_v2.py   persistent-damage adaptation
src/mini_origin/resilience_v3.py   worst-case domain randomization
src/mini_origin/competitive_v4.py  competitive signed routing
src/mini_origin/plasticity_v5.py   within-lifetime feedback learning
tests/                             deterministic anti-shortcut tests
```

## Current limitations

- Substrate families and operator bases are still human-designed.
- The v0.5 mapping tasks are linear and supervised.
- Examples are delivered externally to many cells.
- Damage tolerance currently relies on redundancy rather than regrowth.
- The substrate does not invent objectives or decide what to learn.
- External literature review and third-party replication remain necessary.
- No matched-compute advantage over strong modern learning systems has been demonstrated.

These limitations are recorded explicitly so a successful internal benchmark cannot be mistaken for a universal-computing claim.
