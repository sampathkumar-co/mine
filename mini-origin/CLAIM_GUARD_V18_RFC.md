# ClaimGuard v0.18 — Executable Scientific Claim Contract RFC

Status: **research candidate; not an externally accepted breakthrough**

## 1. Purpose

ClaimGuard prevents an autonomous research system from emitting a `breakthrough` status unless a frozen scientific claim contract and all required evidence obligations verify successfully.

It is prospective enforcement, not merely post-hoc paper auditing.

## 2. Threat model

The verifier must reject at least these failure classes:

- candidate selection after hidden evaluation;
- hidden-set access before candidate freeze;
- threshold changes after results arrive;
- seed cherry-picking or duplication;
- benchmark shortcuts and violated assumptions;
- impossible thresholds above an oracle ceiling;
- unequal candidate/control operation budgets;
- missing or stronger controls and ablations;
- candidate, manifest or contract substitution;
- malformed contracts, metrics, budgets, seeds and check schemas;
- a breakthrough label inconsistent with the evidence.

## 3. Contract fields

A contract contains:

- `claim_id` — non-empty identifier;
- `candidate_hash` — lowercase SHA-256 of the frozen candidate;
- `required_runs` and `min_successes`;
- per-run and aggregate score thresholds;
- minimum control and ablation gaps;
- oracle ceiling;
- operation budget;
- mandatory benchmark-assumption checks.

The contract itself is content-addressed.

## 4. Cross-language canonical contract digest

Contract digests must be identical across implementations.

1. Convert each contract float field to `float:<canonical-number>`.
2. Canonical numbers use the shortest round-trippable 17-significant-digit representation, with trailing decimal zeros removed and zero encoded as `0`.
3. Sort object keys lexicographically.
4. Encode compact UTF-8 JSON.
5. Compute SHA-256.

Float fields:

```text
score_threshold
median_threshold
min_control_gap
median_control_gap
min_ablation_gap
oracle_ceiling
operation_budget
```

The Python and JavaScript implementations must produce the same digest for every conformance vector.

## 5. Sealed evaluation manifest

Only after the candidate and contract are frozen may an evaluator derive hidden seeds from:

```text
HMAC-SHA256(evaluator_secret, contract_digest || namespace || run_index)
```

The public manifest contains:

- contract digest;
- candidate hash;
- positive unique hidden seeds;
- seed commitment;
- an explicit assertion that the manifest was issued after candidate freeze.

## 6. Run evidence obligations

Every run must include:

- the exact hidden seed;
- candidate score;
- strongest control score;
- causal ablation score;
- candidate and control operation budgets;
- threshold actually used;
- contract, candidate and manifest bindings;
- exactly one hidden candidate evaluated;
- no post-holdout candidate selection;
- zero holdout-policy violations;
- all mandatory benchmark checks.

Scores, controls and ablations must be finite and within the oracle range. Budgets must be non-negative, within the contract limit and equal between candidate and control.

## 7. Aggregate obligations

Certification requires:

- exact declared run count;
- positive, unique, committed seed set;
- minimum number of successful runs;
- median score threshold;
- median control-gap threshold;
- minimum ablation gap;
- zero schema, provenance, holdout or budget violations.

A valid evidence bundle that is not labelled `breakthrough` is not certified. An invalid bundle labelled `breakthrough` is explicitly rejected as a false claim.

## 8. Implementations

Reference implementation:

```text
src/mini_origin/claim_contract_v18.py
```

Independent JavaScript verifier:

```text
independent/claim_guard_verifier_v18.mjs
```

Independent JavaScript mutation generator:

```text
independent/claim_guard_external_v18.mjs
```

Neutral JSON adapter:

```text
src/mini_origin/claim_guard_adapter_v18.py
```

## 9. Current evidence

First-party five-seed study:

- 14 integrity mutation families;
- 224 invalid trials and 64 valid trials per seed;
- 100% detection;
- 0% false rejection;
- perfect replay of 14 historical failures plus one valid case;
- at least 85.7 percentage-point detection advantage over threshold-only, replication-only and provenance-only gates.

Independent cross-language study:

- five evaluator seeds;
- 18 frozen cases per seed;
- 90 total bundles;
- Python accuracy: 100%;
- JavaScript accuracy: 100%;
- verifier disagreements: zero.

The independent suite initially broke the verifier on 15/18 cases and exposed a cross-language digest mismatch. Those failures were preserved and converted into permanent regression tests before the clean rerun.

## 10. Novelty boundary

ClaimGuard must not currently be described as an externally established breakthrough.

Adjacent systems already demonstrate portions of the idea, including executable preregistration, content-addressed evidence, sealed contract-first evaluation, protocol auditing and runtime-verifiable agent contracts. ClaimGuard's candidate contribution is the combined prospective enforcement surface plus adversarial conformance benchmark, not the invention of preregistration or sealed testing.

## 11. External acceptance gate

No external-breakthrough claim is permitted until all are complete:

1. an implementation written by an outside contributor without importing either verifier;
2. an independently authored mutation suite;
3. agreement on public conformance vectors;
4. broader comparison with executable preregistration and sealed-evaluation systems;
5. adversarial security review;
6. peer-reviewed or otherwise public external scrutiny.

Negative replications are first-class results and must remain visible.
