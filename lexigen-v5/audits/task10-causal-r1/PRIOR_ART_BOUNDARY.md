# Task 10 prior-art boundary

This audit treats the v5 Task-10 result as an **internal causal mechanism-selection success**, not as a novel graph algorithm.

Established prior art predates Lexigen by many years:

- Tomita/Seki maximum-clique branch-and-bound (2003) and later Tomita-family refinements use exact branch-and-bound with ordering/bounding.
- Tomita/Kameda and Tomita/Sutani et al. further improve exact maximum-clique branch-and-bound.
- San Segundo / BBMC-style implementations use bitset encodings for exact maximum clique.
- Maximum independent set is equivalent to maximum clique on the complement graph; minimum vertex cover is the complement of maximum independent set.

Relevant public implementation/documentation references inspected before the audit:

- https://github.com/darrenstrash/open-mcs
- https://github.com/pprosser/maxClique
- https://doi.org/10.1007/3-540-45066-1_22
- https://doi.org/10.1007/s10898-006-9039-7

Therefore the strongest claim permitted if the audit succeeds is:

> Lexigen's frozen transfer memory causally selected/instantiated a useful known-style exact graph-search mechanism on a fresh holdout where its frozen no-transfer controls did not.

The audit does **not** permit these claims:

- new maximum-independent-set algorithm;
- new minimum-vertex-cover algorithm;
- world-record exact solver;
- world-level AI breakthrough;
- general self-improving AI.
