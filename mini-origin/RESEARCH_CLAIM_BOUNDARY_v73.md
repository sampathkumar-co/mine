# Mini-ORIGIN v0.73 — research claim boundary

This note prevents Mini-ORIGIN's internal evidence from being described more broadly than the literature currently allows.

## Established foundations

The finite-hypothesis optimal identification-tree problem is classical. Garey (1972) already models object priors, fixed test costs, and dynamic programming for minimum expected testing cost.

Joint expected- and worst-cost diagnosis trees are also established. Cicalese, Laber, and Saettler (ICML 2014) give simultaneous approximation guarantees, while Saettler, Laber, and Cicalese (arXiv:1406.3655) explicitly study costs that depend on the observed value. Mini-ORIGIN therefore must not claim to have introduced response-dependent testing costs.

Blackwell's comparison of experiments (1953) is a deeper boundary. For deterministic tests, a finer response partition is an experiment from which the coarser response can be recovered by merging outcomes. Thus strict partition refinement is a special deterministic case of greater statistical informativeness. This is a mathematical connection, not a claim that Blackwell formulated Mini-ORIGIN's exact recursion. It means the broad rule “a more informative, no-more-expensive test dominates a coarser test” is not a credible standalone novelty claim.

Deb and Stewart (2018) further apply informativeness ordering to adaptive test choice in a strategic model. Their objective differs from exact identification, but the paper reinforces that adaptive selection based on informativeness is established theory.

## Narrow candidate contribution

The candidate contribution is not the surrounding problem model. It is the combination of:

1. recomputing exact response-partition equivalence classes at every descendant hypothesis state;
2. retaining the componentwise Pareto frontier of response-dependent cost vectors inside each local equivalence class;
3. using the resulting canonical state in exact dynamic programming;
4. adding safe full-diagnosis lower bounds without changing the deterministic lexicographic optimum; and
5. demonstrating an exact-search frontier expansion with separately written compiled implementations matching plans and operation counters.

The v0.46 pinned source audit found no equivalent descendant-local implementation in accessible snapshots of PySTreeD, DL8.5, OSDT, or GOSDT. The closest implementation was PySTreeD's one-time global duplicate-feature preprocessing.

That finding is evidence of implementation distinctness, not proof of universal novelty.

## MurTree limitation

The MurTree paper has been reviewed, but the pinned Bitbucket implementation at commit `86e439ef537e40cb93afa641aafe2078` has not been obtained reproducibly through the available transport. The source-level novelty audit therefore remains incomplete. Any manuscript must either resolve this gap or state it plainly.

## Language allowed today

Defensible:

> Mini-ORIGIN contains an internally and cross-language reproduced exact-search mechanism whose descendant-local response-cost quotient and lower-bound combination appear distinct from the accessible pinned implementations audited so far.

Not defensible:

- world-first;
- world-level breakthrough;
- universal novelty;
- first use of response-dependent test costs;
- first recognition that finer tests can be more informative;
- independent external reproduction;
- peer-reviewed acceptance.

## Publication gate

A novelty claim requires, at minimum:

- a broader backward/forward citation review;
- terminology searches around garbling, Blackwell dominance, test subsumption, redundant or indistinguishable queries, partition lattices, and sufficient experiments;
- resolution or explicit disclosure of the MurTree source gap;
- an outside technical reproduction or review; and
- a manuscript that claims only the narrow mechanism actually supported by evidence.

## Primary references

- Blackwell, D. “Equivalent Comparisons of Experiments.” *Annals of Mathematical Statistics* 24(2), 1953. DOI: `10.1214/aoms/1177729032`.
- Garey, M. R. “Optimal Binary Identification Procedures.” *SIAM Journal on Applied Mathematics* 23(2), 1972. DOI: `10.1137/0123019`.
- Cicalese, F.; Laber, E.; Saettler, A. M. “Diagnosis determination: decision trees optimizing simultaneously worst and expected testing cost.” ICML 2014, PMLR 32(1):414–422.
- Saettler, A.; Laber, E.; Cicalese, F. “Trading off Worst and Expected Cost in Decision Tree Problems and a Value Dependent Model.” `arXiv:1406.3655`.
- Deb, R.; Stewart, C. “Optimal adaptive testing: Informativeness and incentives.” *Theoretical Economics* 13(3), 2018. DOI: `10.3982/TE2914`.
- Jia, S.; Navidi, F.; Nagarajan, V.; Ravi, R. “Optimal Decision Tree and Adaptive Submodular Ranking with Noisy Outcomes.” *JMLR* 25(382), 2024.
