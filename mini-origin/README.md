# Mini-ORIGIN

**Autonomous search for local, damage-tolerant computational substrates.**

Mini-ORIGIN is a CPU-only research scaffold for ORIGIN COMPUTE: an attempt to search below conventional neural-network architecture design and discover local computational laws that preserve information, transmit signals, adapt, and recover after damage.

> Current status: replicated internal research milestone. Mini-ORIGIN has not discovered a new universal form of computation and is not a replacement for neural networks or conventional hardware.

## Current milestone — v0.4 competitive local routing

The original v0.1 relay benchmark used a toroidal grid, which accidentally made opposite edges adjacent. That invalidated its relay claim. The benchmark was replaced with fixed boundaries, bipolar signals, early-leakage checks, hidden grid sizes, persistent dead cells, partial walls, and explicit controls.

Smooth weighted-sum local laws improved on damaged grids but failed robust transfer. Their median hidden score remained below 0.30. Mini-ORIGIN v0.4 therefore tests a different substrate primitive: local sources compete, and each cell uses a learned soft selection over the strongest signed neighbouring signal.

Five independent cloud searches were run with seeds 41–45. All five passed the internal milestone gate:

| Metric | Result |
|---|---:|
| Independent runs | 5 |
| Passing runs | 5 |
| Success rate | 100% |
| Hidden worst-case range | 0.8631–0.9558 |
| Hidden median | 0.8946 |
| Hidden mean | 0.9091 |
| Transparent hand-control score | 0.9842 |
| Median fraction of hand control | 90.9% |

The strongest evolved genome scored:

- `0.9558` on 48×25 grids with 33% damage;
- `0.9991` on 60×31 grids with 35% damage;
- `0.9852` on 72×37 grids with 37% damage.

These grids, damage levels, obstacle layouts, and random seeds were excluded from search.

### Honest interpretation

This is a **replicated project-level milestone**, not a world-level computing breakthrough. The competitive primitive strongly shapes the solution space, and a transparent signed max-flood control still performs better in the strict worst case. The experiment demonstrates that evolution can repeatedly find robust parameters within this substrate family; it does not demonstrate autonomous invention of the substrate family itself.

Permanent evidence is stored in:

```text
research-evidence/mini-origin-competitive-v4-summary.json
research-evidence/mini-origin-competitive-v4-seed45.json
```

## Implemented systems

- Fixed-boundary and legacy toroidal cellular universes.
- Shared smooth gated local laws with directional perception.
- Competitive signed-neighbour local laws.
- Noisy memory, bounded relay, state repair, and persistent-damage routing tasks.
- Counterexample-driven benchmark corrections.
- Random, identity, clean-shift, base-genome, and explicit max-flood controls.
- Curriculum evolution and worst-case domain randomization.
- Held-out sizes, damage levels, layouts, and random seeds.
- Independent GitHub Actions experiment matrices and aggregate evidence.

## Run in GitHub Codespaces or locally

```bash
cd mini-origin
python -m pip install -e ".[dev]"
pytest -q

python -m mini_origin.competitive_v4 \
  --seed 45 \
  --population 64 \
  --generations 55 \
  --output results/competitive-v4-seed-45.json
```

GitHub Actions runs the tests and independent search matrix without requiring a personal computer.

## Research progression

### v0.1 — Initial scaffold

Implemented genomes, cellular worlds, memory, relay, repair, evolution, deterministic replay, tests, and cloud execution. Its toroidal relay result was later rejected.

### v0.2 — Strict bounded relay

Added fixed boundaries, directional features, sign-balanced evaluation, hidden distances, leakage penalties, and controls. Four of fifteen runs crossed the internal threshold, but an explicit east-copy rule outperformed them and exposed the task as a known transport primitive.

### v0.3 — Damage and domain randomization

Added dead cells, partial walls, larger unseen grids, changing obstacle layouts, lower-tail selection, and fixed validation. Smooth weighted-sum laws improved over their base genome but remained brittle on the largest hidden environments.

### v0.4 — Competitive routing

Changed the computational primitive from blending neighbours to competitive signed source selection. Five of five searches robustly approached an explicit max-flood control across unseen damaged grids.

## What qualifies as the next major advance

The next result will not count merely for improving routing accuracy. Mini-ORIGIN must demonstrate at least one of these:

1. **Within-lifetime local learning:** one inherited law learns new tasks from examples without evolving a new genome.
2. **Structural regeneration:** deleted computational cells or connections regrow and restore multiple capabilities.
3. **Task transfer:** a substrate evolved without a hidden task learns or performs that task better than equal-compute controls.
4. **Primitive invention:** search expands its own update-language or substrate operations instead of choosing parameters inside a human-fixed family.
5. **Efficiency advantage:** a discovered law beats strong neural, reservoir, or cellular baselines under equal operation and memory budgets.

## Scientific safeguards

No result will be promoted beyond an internal milestone unless it survives:

1. held-out seeds, sizes, layouts, and distributions;
2. explicit hand-designed controls;
3. ablation of every claimed mechanism;
4. equal-compute neural, reservoir, and cellular baselines;
5. operation, memory, and complexity accounting;
6. task transfer excluded from search;
7. independent reruns;
8. external review and literature comparison.

## Repository layout

```text
src/mini_origin/genome.py          smooth evolvable local law
src/mini_origin/substrate.py       initial cell-universe simulator
src/mini_origin/tasks.py           memory, relay, and repair tasks
src/mini_origin/search.py          initial evolutionary loop
src/mini_origin/research_v2.py     strict fixed-boundary relay study
src/mini_origin/resilience_v2.py   persistent-damage adaptation
src/mini_origin/resilience_v3.py   worst-case domain randomization
src/mini_origin/competitive_v4.py  competitive signed-routing substrate
tests/                             deterministic and anti-shortcut tests
```

## Current limitations

- The substrate families and operators are still human-designed.
- Search optimizes compact parameter sets rather than inventing a compiler or physical medium.
- No within-lifetime learning has been demonstrated.
- No structural regrowth has been demonstrated.
- The v0.4 result rediscovers a known flood-like local routing mechanism.
- No equal-compute advantage over established unconventional-computing baselines has been demonstrated.

The project records these limits explicitly so a successful benchmark cannot be mistaken for a world breakthrough.
