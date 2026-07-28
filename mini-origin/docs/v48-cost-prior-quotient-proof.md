# State-local response-partition quotient with test costs and hypothesis priors

## Model

Let `S` be the current nonempty set of hypotheses and `R` the tests that remain available. Each hypothesis has a positive mass `p(h)` and every test `t` has a positive, response-independent cost `c(t)`.

A test induces the unlabeled partition

`P_S(t) = { {h in S : t(h) = a} : a is a nonempty response class }`.

A terminal state is successful when all hypotheses in `S` have the same target label. Plans are compared lexicographically by:

1. maximum diagnosed prior mass;
2. minimum prior-weighted total test cost;
3. minimum worst-case test cost;
4. deterministic test-index tie-breaking.

The remaining-test mask is inherited by descendants except that a selected test is removed. Costs and hypothesis masses are static. Response-dependent costs, state-dependent future costs, and exogenous changes to descendant availability are outside the theorem.

## Lemma 1: hereditary equivalence

If `P_S(t) = P_S(u)`, then `P_A(t) = P_A(u)` for every subset `A` of `S`.

### Proof

Every cell of `P_A(t)` is the nonempty intersection of `A` with a cell of `P_S(t)`. Because the two partitions of `S` contain exactly the same cells, their nonempty intersections with `A` are also exactly the same. Therefore the induced unlabeled partitions agree on every descendant candidate subset. QED.

## Lemma 2: cheapest representative dominance

Suppose `P_S(t) = P_S(u)` and `c(t) <= c(u)`. Any feasible decision subtree rooted at `u` can be transformed into a feasible subtree rooted at `t` with:

- the same diagnosed hypotheses;
- no greater expected cost;
- no greater worst-case cost.

### Proof

Replace the root test `u` with `t`. The root partitions are equal, so their children can be paired by identical candidate subsets. Attach to every paired child the original descendant subtree. Lemma 1 ensures that all future tests have exactly the same behavior on those descendants.

Every hypothesis follows a path whose suffix is unchanged. Its root cost changes from `c(u)` to `c(t)`, so its individual cost decreases by `c(u)-c(t) >= 0`. Consequently the prior-weighted total cost and worst-case cost cannot increase, while diagnosis behavior is unchanged. QED.

## Lemma 3: discarded equivalent tests are useless below the chosen representative

After selecting a test `t` at state `S`, any test `u` with `P_S(u) = P_S(t)` is constant on every child of `t` and therefore cannot separate any descendant hypotheses.

### Proof

Every child of `t` is one cell of the shared partition. Test `u` has that same cell as one of its response classes, so `u` is constant on the child. The statement remains true on every subset of that child. QED.

## Theorem: exact quotient preservation

At every state, group remaining tests by `P_S(t)` and keep the test minimizing `(c(t), index(t))` in each nontrivial class. Exact dynamic programming over only these representatives returns the same optimal objective triple as dynamic programming over all remaining tests.

### Proof

Consider any optimal full-search plan at state `(S,R)`. If its root test is already the canonical representative of its partition class, retain it. Otherwise replace it by that class's canonical representative. Lemma 2 proves that the transformed plan is lexicographically no worse. Lemma 3 proves that the discarded members of the class carry no information in any resulting child.

Apply the same argument recursively to each child. Because each recursive call has fewer informative remaining tests, the transformation terminates and produces an equally good or better plan containing only canonical representatives at every state. Thus the quotient search space contains an optimum of the full search space. Since every quotient plan is also a valid full-search plan, neither space can have a strictly better optimum than the other. Their optimal objective triples are equal. QED.

## Practical consequence

Global duplicate-test preprocessing is only the root instance of this theorem. Tests that differ on the full dataset can become equivalent after earlier outcomes restrict the candidate set. Recomputing equivalence on each descendant can therefore remove branches that no one-time preprocessing pass can detect.
