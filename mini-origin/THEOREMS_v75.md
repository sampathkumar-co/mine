# Mini-ORIGIN v0.75 — theorem package

This document states the mathematical claims implemented by the response-cost Pareto quotient and the exact lower-bound planner. It deliberately separates theorem assumptions from empirical evidence and from novelty claims.

## 1. Model

Let:

- `H` be a finite set of hypotheses;
- `label : H → L` be the diagnosis label;
- `μ : H → ℝ_{>0}` be a strictly positive hypothesis mass;
- `Q` be a finite set of tests;
- `ρ_q : H → Ω_q` be the deterministic response of test `q`;
- `κ_q : Ω_q → ℝ_{≥0}` be the static response-dependent cost of `q`.

A search state is `(S, A)`, where `S ⊆ H` is the current nonempty hypothesis set and `A ⊆ Q` is the set of tests still available. Querying `q ∈ A` creates the nonempty response cells

`P_S(q) = { {h ∈ S : ρ_q(h) = ω} : ω ∈ Ω_q } \ {∅}`.

The test is then removed. Thus every child receives the available set `A \ {q}`. Test identity has no other effect on future feasibility or costs.

A state is pure when all hypotheses in it have the same diagnosis label. A plan may fail to diagnose an impure state if no informative test remains.

For a plan `T` at `(S, A)`, define:

- `M(T)`: diagnosed hypothesis mass;
- `E(T)`: mass-weighted sum of every test cost incurred before diagnosis or abandonment;
- `W(T)`: maximum incurred path cost;
- `r(T)`: the root test identifier, or `+∞` for no root.

Mini-ORIGIN maximizes the lexicographic score

`Score(T) = (M(T), -E(T), -W(T), -r(T))`.

Only the current root identifier is part of this score. Child-root identifiers are not recursively embedded in the parent score.

## 2. Local response-partition equivalence

### Definition 2.1

Tests `q` and `r` are locally equivalent at `S`, written `q ≡_S r`, when

`P_S(q) = P_S(r)`

as unordered families of hypothesis subsets.

The response symbols themselves may differ. Equality is by the induced cells.

### Lemma 2.2 — hereditary equivalence

If `q ≡_S r`, then `q ≡_T r` for every nonempty `T ⊆ S`, after empty restricted cells are removed.

#### Proof

For any `h, h' ∈ S`, local partition equality means

`ρ_q(h) = ρ_q(h')` if and only if `ρ_r(h) = ρ_r(h')`.

Restricting both equivalence relations to `T × T` preserves this biconditional. Their equivalence classes on `T` are therefore identical. ∎

### Lemma 2.3 — local redundancy after use

If `q ≡_S r`, then after querying either test at `S`, the other test is noninformative in every resulting child.

#### Proof

Every child is one common cell `C ∈ P_S(q) = P_S(r)`. Both `q` and `r` are constant on `C`, so neither splits `C`. ∎

## 3. Response-cost vectors

For an informative test `q` at `S`, define its cell-aligned cost vector

`c_S(q) = (κ_q(ρ_q(h_C)))_{C ∈ P_S(q)}`,

where `h_C` is any member of cell `C`. The value is well-defined because `ρ_q` is constant on a response cell. The implementation orders cells by their hypothesis bit masks, making alignment deterministic.

### Definition 3.1 — equivalent-test dominance

For `q ≡_S r`, test `q` dominates `r` at `S` when either:

1. `c_S(q) ≤ c_S(r)` componentwise and the inequality is strict in at least one cell; or
2. `c_S(q) = c_S(r)` and `id(q) < id(r)`.

A local equivalence class retains every nondominated cost vector. Equal vectors retain only the lowest test identifier.

### Lemma 3.2 — hereditary weak cost dominance

If `c_S(q) ≤ c_S(r)` componentwise and `q ≡_S r`, then for every nonempty `T ⊆ S`, the restricted vectors satisfy

`c_T(q) ≤ c_T(r)`

on their common restricted cells.

Strict inequality may disappear when all strictly improved cells are removed.

#### Proof

By Lemma 2.2, the cells at `T` are nonempty intersections of cells at `S`. Restriction deletes coordinates but changes no surviving response cost. Componentwise weak inequality is therefore preserved. ∎

## 4. Equivalent-test substitution theorem

### Theorem 4.1 — metric-preserving substitution

Let `q ≡_S r` and `c_S(q) ≤ c_S(r)` componentwise. For every feasible plan rooted at `r` in state `(S, A)` with `q, r ∈ A`, there exists a feasible plan rooted at `q` with:

- the same diagnosed mass;
- expected cost no greater;
- worst cost no greater.

#### Proof

The root response cells are identical, so align the `r`-branches and `q`-branches by their common hypothesis subsets.

Take each child subtree of the original `r`-rooted plan. In the original subtree, `r` is unavailable and `q` is available. By Lemma 2.3, `q` is noninformative throughout that child at the moment immediately after the root. Delete any occurrence of `q`; deleting a noninformative nonnegative-cost test cannot reduce diagnosed mass and cannot increase either cost objective.

Attach the resulting child subtree below root `q`. In the transformed tree, `q` is unavailable and `r` remains available, but Lemma 2.3 makes `r` noninformative in the same child, so no deleted or retained decision requires it.

All other tests have identical availability and behaviour. The transformed plan is feasible and diagnoses exactly the same hypotheses.

For every hypothesis, the root cost under `q` is no greater than the aligned root cost under `r`; all retained downstream costs are unchanged or reduced. Hence both aggregate expected cost and every path cost are no greater. ∎

### Corollary 4.2 — strict dominance exclusion

Under strictly positive hypothesis masses, if `q` strictly dominates `r` in at least one current cell, no optimal plan at `(S, A)` is rooted at `r`.

#### Proof

Theorem 4.1 gives a plan with the same diagnosed mass and no greater worst cost. At least one nonempty cell has strictly lower root cost, and that cell has positive total mass, so expected cost is strictly lower. The transformed plan has a strictly better lexicographic score. ∎

### Corollary 4.3 — equal-vector identifier exclusion

If `q ≡_S r`, their local cost vectors are equal, and `id(q) < id(r)`, no optimal plan at `(S, A)` is rooted at `r`.

#### Proof

Theorem 4.1 gives identical objective metrics. The lower current root identifier breaks the tie in favour of `q`. ∎

### Theorem 4.4 — local Pareto quotient preserves the value function

Let `Canon(S, A)` retain all nondominated response-cost vectors in every local equivalence class, breaking equal-vector ties by lower identifier. Then the optimum score at `(S, A)` equals the optimum score obtained by considering only roots in `Canon(S, A)`.

#### Proof

Every removed root is excluded by Corollary 4.2 or Corollary 4.3. Every retained root remains feasible. Removing only roots that cannot be optimal leaves the maximum score unchanged. ∎

## 5. Why incomparable vectors must remain

### Proposition 5.1

There is no prior-independent scalar representative for an equivalence class containing incomparable cost vectors.

#### Witness

Consider two equivalent binary tests with vectors `(1, 9)` and `(9, 1)`. Under cell masses `(9, 1)`, the first test has lower expected cost; under masses `(1, 9)`, the second does. Neither can be universally removed.

This is why Mini-ORIGIN retains the componentwise Pareto frontier rather than choosing a weighted-average winner.

## 6. Descendant-local recomputation

### Proposition 6.1

Tests that are not equivalent at `S` may become equivalent at a proper descendant `T ⊂ S`.

#### Example

Let three hypotheses have responses

- `q = (0, 0, 1)`;
- `r = (0, 1, 1)`.

They induce different partitions on all three hypotheses. On descendant `{h₁, h₃}`, both induce the two singleton cells and are equivalent.

Therefore one-time global duplicate removal cannot in general recover the descendant-local quotient.

## 7. Canonical recursion and its tie-breaking boundary

The implementation recursively passes the canonical remaining-test set to children. A test removed at an ancestor remains weakly dominated at every descendant by Lemmas 2.2 and 3.2, so deleting it cannot worsen diagnosed mass, expected cost, or worst cost.

Strict cost dominance at an ancestor can become equality in a descendant if the only strictly improved cell disappears. If the removed test has a lower identifier, a separately solved descendant with the original full test set could choose that identifier under a local tie.

Accordingly, the proved scope is:

- preservation of all three objective metrics throughout recursion;
- preservation of the selected root query at the state where the quotient is computed;
- preservation of the implementation's returned `Plan` value.

It does **not** claim preservation of a globally canonical full tree under a recursive node-by-node identifier ordering that is absent from the implemented objective.

This distinction is part of the theorem, not an implementation footnote.

## 8. Full-diagnosis lower bound

For candidate root `q` at state `(S, A)`, let its child cells be `C₁, …, C_k`. For every impure child `C_i`, any full-diagnosis continuation must next use at least one informative test from `A \ {q}`.

Define:

- `mE(C_i)` as the minimum immediate mass-weighted cost of an informative remaining test on `C_i`;
- `mW(C_i)` as the minimum immediate worst cost among tests attaining `mE(C_i)`;
- an impure child with no informative remaining test as impossible to diagnose fully.

Define the candidate lower bounds

`LB_E(q) = ImmediateExpected(q, S) + Σ_i mE(C_i)`

and

`LB_W(q) = max_i [ Cost(q, C_i) + mW(C_i) ]`.

Pure children contribute zero continuation cost.

### Lemma 8.1 — first-step lower-bound validity

Every full-diagnosis plan rooted at `q` has expected cost at least `LB_E(q)`. Among plans whose expected cost equals `LB_E(q)`, worst cost is at least `LB_W(q)`.

#### Proof

Every impure child must select an informative next test. Its immediate expected cost is at least the minimum `mE(C_i)`. Summing over disjoint children and adding the root immediate cost proves the expected bound.

If equality holds in expected cost, every child must choose a test attaining its local minimum expected immediate cost. Its immediate worst cost is therefore at least `mW(C_i)`. Later costs are nonnegative, so the full path worst cost cannot be below the maximum root-plus-child first-step bound. ∎

### Lemma 8.2 — impossibility certificate

If an impure child has no informative remaining test, no plan rooted at `q` can diagnose the full mass of `S`.

#### Proof

All remaining tests are constant on that impure child. No future response can separate hypotheses with different labels. ∎

### Theorem 8.3 — incumbent pruning safety

Suppose an incumbent diagnoses the full mass of `S`. Candidate `q` may be pruned when:

1. a child is impossible by Lemma 8.2;
2. `LB_E(q)` exceeds incumbent expected cost;
3. expected bounds tie and `LB_W(q)` exceeds incumbent worst cost; or
4. both cost bounds tie and `id(q)` exceeds the incumbent root identifier.

No pruned candidate can improve the implemented lexicographic score.

#### Proof

Full diagnosed mass is already achieved, so a challenger can improve only the remaining lexicographic coordinates. Lemma 8.1 lower-bounds those coordinates in the same order. Lemma 8.2 excludes full-mass equality. Each pruning condition therefore proves that the candidate cannot beat the incumbent. ∎

## 9. Assumptions required by the proofs

The proofs require all of the following:

1. finite hypothesis and test sets;
2. deterministic, persistent test responses;
3. strictly positive hypothesis masses for strict expected-cost exclusion;
4. nonnegative static costs determined only by test identity and observed response;
5. inherited availability: a chosen test is removed and no other future feasibility changes;
6. no test-specific side effects, setup costs, group acquisition discounts, or state-dependent future prices;
7. exact equality of response cells, not approximate similarity;
8. the implemented lexicographic score, whose identifier tie-break is current-root only.

The theorem does not cover negative costs, stochastic responses, time-varying prices, exogenous test failure, test interactions, or a recursively canonical full-tree tie-break.

## 10. Evidence versus proof

The mathematical arguments above establish the claimed transformations under the stated model. Repository certificates serve different purposes:

- v0.56 checks hereditary quotient behaviour and exact-plan agreement on generated and real-data states;
- v0.57 independently reproduces the solver in Rust;
- v0.62 exhaustively compares complete finite-model frontiers;
- v0.65/v0.66 validate and independently reproduce the lower-bound planner;
- v0.71 verifies label independence of benchmark state selection.

Those computations test implementations. They are not substitutes for the proofs, and the proofs are not evidence of empirical significance.

## 11. Novelty boundary

These theorems are not presented as proof that every ingredient is new. Blackwell informativeness, optimal identification trees, expected/worst testing-cost objectives, and value-dependent costs are established prior work. The candidate contribution must remain the narrow descendant-local algorithmic combination identified in `RESEARCH_CLAIM_BOUNDARY_v73.md`.
