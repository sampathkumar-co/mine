# Mini-ORIGIN v0.24 — Semantic Alphabet Theorem Report

Status: **formal internal result; empirical candidate gate rejected; not a world breakthrough**

## 1. Problem

Consider a Gaussian causal chain whose arrows point away from an unknown root. Every root orientation induces the same observational AR(1) covariance. Passive observations therefore cannot identify the root.

After clamping a queried node to one, the immediate left and right neighbour responses can be inactive, active, or physically absent. A controller must convert those continuous responses into one of three actions:

- keep the candidate interval left of the query;
- accept the query as the root;
- keep the candidate interval right of the query.

v0.23 synthesized a three-bit response alphabet that worked extremely well with a lower-midpoint query policy, but its statistical advantage over every two-bit alphabet did not replicate.

v0.24 asks the stronger question: what is the smallest response alphabet that remains correct independently of the optimal midpoint tie-breaking policy?

## 2. Seven required semantic states

The exhaustive state set is:

| State | Left response | Right response | Required action |
|---|---:|---:|---|
| Left boundary, root equals query | absent | active | accept |
| Left boundary, root is right | absent | inactive | keep right |
| Right boundary, root equals query | active | absent | accept |
| Right boundary, root is left | inactive | absent | keep left |
| Interior, root is left | inactive | active | keep left |
| Interior, root equals query | active | active | accept |
| Interior, root is right | active | inactive | keep right |

The candidate Boolean feature grammar contains:

```text
left_exists
right_exists
left_active
right_active
left_greater
```

All 31 non-empty feature subsets were evaluated exhaustively.

## 3. Exact lower-bound result

The unique minimum exact alphabet is:

```text
left_exists + right_exists + left_active + right_active
```

It classifies all seven semantic states correctly.

No one-, two-, or three-feature subset is exact. The strongest smaller alphabet classifies at most six of the seven states.

The necessity is structural:

- without `left_exists`, left-boundary equality collides with the interior root-left state;
- without `right_exists`, right-boundary equality collides with the interior root-right state;
- without `left_active`, right-boundary equality collides with right-boundary root-left;
- without `right_active`, left-boundary equality collides with left-boundary root-right.

Thus four independent distinctions are required by this feature grammar for policy-independent semantics.

## 4. Policy-specific compression

The v0.23 three-bit alphabet was:

```text
left_exists + left_active + right_active
```

It omits `right_exists` and therefore cannot distinguish:

```text
right-boundary equality
interior root-right
```

A lower-midpoint binary-search controller almost never needs the ambiguous right-boundary equality transition: when two candidates remain, it queries the lower one and can terminate at the upper endpoint without measuring it.

Therefore the three-bit alphabet is a valid policy-specific compression but not a policy-independent measurement language.

## 5. Cloud replication

Five independent seeds reproduced the exact certificate:

- selected feature set: identical in 5/5 runs;
- semantic accuracy: 100% in 5/5 runs;
- perfect smaller alphabets: zero in 5/5 runs;
- best smaller semantic accuracy: 6/7 in 5/5 runs;
- observational covariance discrepancy: at most `2.22e-16`.

The exact alphabet was then tested on unseen chain sizes 10, 15, 26, 41 and 70 using lower-midpoint, upper-midpoint and alternating midpoint policies.

- candidate hidden accuracy: 99.06%–99.25%;
- worst policy-family accuracy: at least 98.83%;
- invalid-transition rate: at most 0.083%;
- gap over random codebooks: at least 74.27 percentage points.

## 6. Why the preregistered study failed

The candidate gate also required a six-percentage-point hidden accuracy advantage over the strongest three-bit controller.

Observed gaps were only:

```text
4.75% to 5.44%
```

Therefore zero of five runs passed the full locked gate. The six-point threshold was not lowered after observing the results.

This does not invalidate the exhaustive semantic lower bound. It rejects the additional claim that the lower bound must always produce at least a six-point average closed-loop advantage under the chosen hidden distribution.

## 7. Correct scientific claim

Supported:

> Within the declared Boolean feature grammar, policy-independent interpretation of all seven ideal local intervention states requires four features, while a lower-midpoint policy admits a three-feature compression by making one boundary state unreachable or unnecessary.

Not supported:

> Four features always improve average noisy closed-loop root-identification accuracy by at least six percentage points.

## 8. External boundary

This result concerns a deliberately controlled causal-chain and finite-controller benchmark. It connects established ideas from causal intervention design, decision trees, automata minimization and binary search. It is not evidence of a new universal form of intelligence or a world-level scientific breakthrough.

The next credible external direction must leave this hand-designed chain family and test a general compiler that receives arbitrary executable model classes, proves observational equivalence, and synthesizes a minimal distinguishing interface without receiving the relevant semantic predicates in advance.
