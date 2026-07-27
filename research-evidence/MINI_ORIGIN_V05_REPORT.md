# Mini-ORIGIN v0.5 Research Report

Date: 2026-07-27

## Verdict

**Replicated within-lifetime local-learning breakthrough inside the Mini-ORIGIN project.**

This is materially stronger than the v0.4 routing milestone. A single inherited, dimension-agnostic plasticity law now learns a newly generated mapping from examples during its own lifetime. The genome is not changed between mappings. After learning, up to 65% of the distributed memory cells are deleted, yet the learned mapping remains available.

This remains an internal research breakthrough rather than a world-level learning breakthrough because the candidate plasticity operator basis was human-designed.

## Benchmark correction before acceptance

The first v0.5 experiment appeared to pass with a hidden score above 0.99. It was rejected because isotropic training examples allowed a Hebbian correlation rule to imitate learning the mapping.

The accepted benchmark uses:

- strongly correlated, ill-conditioned training examples;
- isotropic test queries;
- hidden orthogonal mappings generated independently inside each lifetime;
- dimensions 5, 6, and 8 excluded from evolution;
- condition numbers up to 36 excluded from evolution;
- observation noise excluded from the easiest controls;
- distributed cell deletion up to 65%;
- frozen-memory, feedback-ablation, hand-Hebbian, and hand-delta controls.

A covariance-weighted Hebbian memory cannot solve the accepted benchmark because the test-query distribution differs from the training covariance.

## Experiment

Five independent GitHub Actions searches used seeds 51–55. Evolution only saw dimensions 3 and 4 with milder conditioning and damage. Each evolved genome defines the coefficients of one local update law shared by all memory cells.

Every cell receives its own noisy observation, computes its own prediction and error, and updates only its own memory matrix. Cells do not share their memory parameters. The final output combines predictions from surviving cells.

The effective local update has the form:

```text
memory change =
    teacher coefficient × target ⊗ input
  + feedback coefficient × prediction ⊗ input
  - decay × memory
```

The inherited coefficients remain fixed while the substrate learns different mappings during its lifetime.

## Replicated results

| Metric | Result |
|---|---:|
| Independent searches | 5 |
| Passing searches | 5 |
| Success rate | 100% |
| Hidden worst-case minimum | 0.8527 |
| Hidden worst-case median | 0.8544 |
| Hidden worst-case maximum | 0.8599 |
| Minimum post-damage retention | 99.44% |
| Median post-damage retention | 99.68% |
| Median advantage over correlation-only control | 0.7074 |
| Median fraction of hand-delta performance | 104.10% |

The strongest evolved law, seed 55, scored:

| Hidden environment | Score |
|---|---:|
| 5 dimensions, condition 16, 45% cell death | 0.9877 |
| 6 dimensions, condition 24, 55% cell death | 0.9529 |
| 8 dimensions, condition 36, 65% cell death | 0.8599 |

Its controls scored:

| Control | Aggregate hidden score |
|---|---:|
| Frozen memory / no learning | 0.1212 |
| Feedback removed | 0.0515 |
| Hand Hebbian learning | 0.1462 |
| Hand delta learning | 0.8207 |
| Evolved feedback law | 0.8599 strict worst case |

## Learned mechanism

The strongest law evolved:

- learning rate: `0.3989`;
- effective teacher coefficient: `+3.6864`;
- effective prediction-feedback coefficient: `-3.5718`;
- memory decay: `0`;
- observation dropout: `29.53%`;
- approximately equal mean/median consensus.

The near-opposite teacher and prediction coefficients form a high-gain local error-correction rule. The system therefore does not merely accumulate target correlations. It repeatedly subtracts what it already predicts and writes the remaining local error.

Observation dropout forces different cells to experience different subsets and noise realisations. Their redundant memories remain sufficiently consistent that deleting most cells has little effect on the consensus.

## Why this is a real project breakthrough

The accepted result demonstrates all of the following together:

1. The mapping is generated after the genome is fixed.
2. Learning happens during the substrate's lifetime.
3. The same coefficients work at larger unseen dimensions.
4. The training and query distributions differ.
5. Most cells can be deleted after learning.
6. Frozen and correlation-only controls fail.
7. Five independent evolutionary searches reproduce the result.
8. The evolved rule slightly exceeds the fixed hand-delta control on the aggregate hidden benchmark.

## Scientific limits

This result does not establish a new general theory of learning.

- The local memory is a matrix, a familiar computational object.
- The operator basis includes supervised target and prediction terms chosen by us.
- The benchmark uses linear orthogonal mappings.
- Examples are externally delivered to many cells rather than discovered autonomously.
- The substrate does not choose its own objectives.
- It has not yet learned nonlinear programs, temporal tasks, or tasks without target labels.
- External literature comparison and independent third-party replication are still required.

## Next major gate

The next advance must remove one of the strongest remaining human assumptions. The best target is:

> Evolve a local plasticity language that learns nonlinear and temporal tasks from sparse reward or self-supervised signals, without receiving the target vector at every update.

A second route is structural regeneration: allow dead memory cells to regrow, reconstruct the learned mapping from neighbours, and recover capacity rather than merely relying on surviving redundancy.

## Evidence

- `research-evidence/mini-origin-plasticity-v5-summary.json`
- `research-evidence/mini-origin-plasticity-v5-seed55.json`
- GitHub Actions run `30275086582`
- Pull request `#22`
