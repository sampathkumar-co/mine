# Mini-ORIGIN v0.27–v0.30 Audit

Date: 2026-07-28

## Campaign verdict

Four distinct preregistered versions were implemented and evaluated with five fresh seeds each.

| Version | Gate | Verdict | Main result |
|---|---:|---|---|
| v0.27 | 4/5 required | PASS, 4/5 | Empirically fitted decoder thresholds improved weak-signal performance over fixed 0.20, but one seed was unstable. |
| v0.28 | 4/5 required | PASS, 5/5 | Weighted sum-distance querying plus normalized decoding achieved 100% hidden accuracy on heterogeneous weighted trees up to 255 nodes with exact half-mass certificates. |
| v0.29 | 4/5 required | FAIL, 0/5 | Adaptive sampling retained 100% accuracy but saved only about 22–25% mean observations, below the locked 40% target. |
| v0.30 | 4/5 required | FAIL, 3/5 | Cross-domain holdouts achieved 100% accuracy, but representation was not unique: three seeds selected minimax buckets and two selected Gini, while the gate required exact minimax identity. |

## Strongest new result

v0.28 is the strongest clean result in this campaign:

- five of five independent seeds passed;
- hidden accuracy was 100% in every seed;
- candidate priors were nonuniform and edge strengths heterogeneous;
- recursive half-mass certificates had zero violations;
- normalized decoding beat the unnormalized control by at least 26.17 accuracy points;
- weighted querying reduced mean query count and remaining posterior mass relative to unweighted querying.

This remains an internal transfer milestone around weighted tree medians, not a new external theory.

## Important negative results

v0.29 demonstrates that preserving accuracy is not enough: its sequential schedule failed the promised efficiency improvement. The efficiency threshold was preserved unchanged.

v0.30 demonstrates a different issue: behavior transferred perfectly across domains, but the symbolic representation was not identifiable. Minimax-bucket and Gini objectives were behaviorally equivalent on the tested tasks. A future study must quotient programs by behavior before requiring symbolic replication, then face new holdouts.

## Current scientific status

Mini-ORIGIN has become substantially broader and more rigorous:

1. raw relational predicates rather than only named features;
2. topology transfer from chains to trees;
3. weighted candidate priors and heterogeneous measurement scales;
4. empirical calibration under weak signals;
5. abstract partition-based transfer to unseen diagnosis domains;
6. permanent preservation of rejected gates.

However, it has **not** produced a world-class external breakthrough. The strongest mechanisms map to established ideas: decision stumps, weighted medians, sequential testing, generalized binary search and active diagnosis.

## Next highest-value direction

The most promising next study is not another brute-force version. It should construct a behavioral quotient over synthesized programs, merge provably equivalent objectives such as minimax bucket and Gini on binary partitions, freeze the quotient-class selector, and then test on new multi-outcome domains where those objectives diverge. That would directly address v0.30’s failure rather than hiding it.
