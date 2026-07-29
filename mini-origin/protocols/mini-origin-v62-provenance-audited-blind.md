# Mini-ORIGIN v0.62 — provenance-audited blind gate

## Motivation

v0.61 is preserved as provenance-contaminated because three of seven supposedly unseen datasets had appeared in earlier Mini-ORIGIN work. v0.62 repairs the protocol rather than reinterpreting the result.

## Immutable baseline

The complete pre-v0.62 repository baseline is commit:

`fece4308846badfc4257a5285cdc0a46c86bf725`

Every provenance check must scan the full Git history reachable from this baseline, not only the baseline working tree.

## Candidate suite

The initial candidate pool contains official UCI CC BY 4.0 classification datasets not found by the preliminary repository search:

1. Dry Bean — UCI 602 — DOI 10.24432/C50S4B
2. Rice (Cammeo and Osmancik) — UCI 545 — DOI 10.24432/C5MW4Z
3. Maternal Health Risk — UCI 863
4. Raisin — UCI 850 — DOI 10.24432/C5660T
5. Predict Students' Dropout and Academic Success — UCI 697
6. Obesity Levels Based on Eating Habits and Physical Condition — UCI 544
7. CDC Diabetes Health Indicators — UCI 891

No archive may be opened until the audit and hash lock both pass.

## Stage 0 — full-history provenance audit

For every candidate, inspect all commits reachable from the immutable baseline for:

- canonical name and normalized aliases;
- UCI ID;
- DOI;
- official archive filename and URL slug;
- archive SHA-256 after hash-only download;
- known internal filenames;
- derivative/split naming patterns;
- references in JSON, YAML, Markdown, source, workflow logs committed as evidence, and benchmark manifests.

A candidate is excluded before Stage A if any exact dataset, derivative, split, archive hash, or named source appears anywhere in the pre-v0.62 history. Exclusions are preserved; replacements must come from the predeclared reserve pool and pass the same audit.

Reserve pool:

- Estimation of Obesity Levels — UCI 544 alias handling
- HCV Data — UCI 571
- Heart Failure Clinical Records — UCI 519
- Audit Data — UCI 475
- Drug Consumption — UCI 373
- Online Shoppers Purchasing Intention — UCI 468
- Room Occupancy Estimation — UCI 864

At least seven audited-clean datasets are required before Stage A.

## Stage A — hash-only archive lock

The lock workflow may download official archive bytes and record only URL, byte length, SHA-256, UCI ID, DOI and license. It may not list archive members, decompress, decode records, inspect labels, or compute statistics.

The archive hash is compared against every 64-hex token present anywhere in the immutable baseline history. Any match invalidates that candidate.

## Stage B — unchanged scientific protocol

The passing v0.60 conditioned-cell generator and all v0.61 scientific thresholds are copied unchanged:

- at most 384 distinct records by SHA-256 rank;
- deterministic response-cell paths to depth three;
- at most six hash-ranked separating queries per cell;
- at most 96 hash-ranked cells per depth;
- candidate cells of size 8–24 retained directly;
- larger cells sampled deterministically to 24, 20, 16 and 12;
- 6–16 exact local partition classes;
- 10–64 raw tests;
- at least four redundant tests;
- at most 12 states per dataset;
- response-cost seeds 6001, 6002 and 6003;
- equal 500,000-expansion budgets;
- budget ladder 10k, 50k, 250k and 500k;
- independent Rust replay of every profiled state.

## Locked gate

- seven provenance-clean archive locks;
- at least five datasets contribute states;
- at least 50 base states and 150 profiled states;
- Pareto solves at least 90%;
- at least 40 shared exact solves;
- zero plain/Pareto optimum mismatches;
- at least 25 Pareto-only solves;
- Rust matches every Python status, plan and operation counter;
- at least 1,000 dominated equivalent tests removed;
- at least one incomparable root Pareto class;
- median expansion ratio at least 10x;
- p90 at least 30x;
- at 50k, Pareto solves at least 20 more states than plain.

No gate, seed, budget, selector or dataset may be changed after the seven clean archive hashes are committed.

## Claim boundary

A pass would restore a rigorous preregistered fresh-data result with independent implementation reproduction. It would still not establish universal novelty, independent human reproduction, peer review, world-first status, or a world-class breakthrough.
