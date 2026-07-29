# Mini-ORIGIN v0.76 — outside replication challenge

## What this challenge can establish

This kit lets an outside researcher test whether a separately written implementation exactly reproduces Mini-ORIGIN's descendant-local response-cost Pareto quotient on 195 frozen compact states.

Publishing the kit is **not** independent reproduction. A reproduction exists only when an outside party supplies:

1. a matching output file;
2. separately written solver source;
3. build and run instructions;
4. an independence attestation;
5. enough environment information for another person to rerun it.

## Permitted inputs

A reproducer may use:

- this mathematical document;
- `THEOREMS_v75.md`;
- `states.txt` from the challenge artifact;
- `input-manifest.json`;
- `challenge.json`;
- the output schema below.

A qualifying implementation must not import, call, translate line-by-line, or wrap Mini-ORIGIN's Python, C++, or Rust planner implementations.

Reading the existing implementation after completing an independent solver does not automatically invalidate a result, but the timing and extent of any exposure must be disclosed.

## Input format

The public state file uses `mini-origin-response-cost-state-v1`.

It begins with:

```text
COUNT <number of states>
```

Each state contains:

```text
STATE <digest> <profile-seed> <hypothesis-count> <query-count>
LABELS <one integer per hypothesis>
MASSES <one positive integer per hypothesis>
QUERY_IDS <one original query id per compact query>
RESPONSES <one response id per compact query>   # repeated for each hypothesis
COSTS <one response-dependent cost per query>   # repeated for each hypothesis
END
```

Costs are constant among hypotheses sharing the same response to the same query.

## Required objective

At each state, tests are deterministic and may be used at most once.

The plan objective is lexicographic:

1. maximize diagnosed mass;
2. minimize mass-weighted expected cost numerator;
3. minimize worst path cost.

The challenge's `plan` field is the three-element list:

```text
[diagnosed_mass, expected_cost_numerator, worst_cost]
```

## Required quotient

At every descendant state:

1. partition the current hypotheses by each informative remaining query;
2. group queries inducing exactly the same unordered hypothesis-cell partition;
3. align each query's response costs by those common cells;
4. remove only componentwise-dominated cost vectors;
5. for equal vectors, retain the lower original query id;
6. retain every incomparable vector;
7. recurse using the resulting canonical remaining-query set.

## Required output

Submit UTF-8 JSON:

```json
{
  "schema": "mini-origin-response-cost-replication-output-v1",
  "rows": [
    {
      "digest": "...",
      "solved": true,
      "plan": [0, 0, 0],
      "query_expansions": 0,
      "calls": 0,
      "memo_entries": 0,
      "memo_hits": 0,
      "raw_queries_considered": 0,
      "representative_queries_considered": 0,
      "dominated_queries_removed": 0
    }
  ]
}
```

All 195 state digests must occur exactly once. Extra fields are rejected.

## Validation

Run:

```bash
python -m mini_origin.validate_external_replication_v76 \
  --challenge challenge.json \
  --submission your-output.json \
  --output validation.json
```

The validator canonicalizes the required rows and compares their SHA-256 with the committed expected-output digest. The expected rows themselves are not included in the kit.

A hash match proves exact equality of the frozen outputs. It does not by itself prove implementation independence.

## Independence attestation

A submission should state:

- author names and affiliations, if they wish to disclose them;
- repository and immutable commit of the independent solver;
- programming language and compiler/interpreter versions;
- operating system and hardware;
- whether the author saw Mini-ORIGIN solver code before implementation;
- any reused third-party decision-tree code;
- exact command used;
- output and validation SHA-256 values;
- all failures or deviations.

Negative reproductions are welcome. Do not tune the challenge, omit states, or replace the expected commitment after seeing results.

## What a successful outside submission would mean

It would provide credible outside-human reproduction of the frozen core quotient implementation result. It would **not** alone establish universal novelty, broad practical usefulness, peer-review acceptance, or a world-level breakthrough.
