# LEXIGEN Ω Engineering Roadmap

This roadmap is development guidance. Only a later preregistration before final holdout selection may define the authoritative Ω10 success gate.

| Gate | Objective | Estimated calendar days | Current state |
|---|---|---:|---|
| Ω0 | Scientific lock, isolation, contamination and claim boundaries | 1–2 | locked |
| Ω1 | Universal low-level executable substrate + independent interpreter | 4–6 | in progress |
| Ω2 | Semantic genesis: automatically invent executable primitives | 5–8 | not started |
| Ω3 | Evaluator-guided program evolution + diverse archive | 3–5 | not started |
| Ω4 | Searcher evolution on meta-holdouts | 5–7 | not started |
| Ω5 | Prospective causal-memory admission and ablation | 3–5 | foundation started |
| Ω6 | Counterexample generation + verifier co-synthesis | 4–6 | not started |
| Ω7 | At least five heterogeneous benchmark adapters | 5–8 | not started |
| Ω8 | Exposed-task development kill campaign | 7–10 | not started |
| Ω9 | Final immutable freeze | 2–3 | not started |
| Ω10 | 36–60 task global blind campaign | 5–10 | not started |
| Ω11 | Frontier validation + independent reproduction package | 5–10 internal | not started |

Serial engineering total: approximately 49–80 days. With parallel CI, independent-runtime development, adapter work and evaluator execution, working target is approximately 35–55 calendar days to an internal global verdict. External third-party reproduction has no guaranteed schedule.

## Rule after Ω9

After Ω9, architecture/search/memory/adapters/thresholds receive **zero** blind-result-driven revisions. A failed Ω10 is preserved as a negative global result; it is not rescued by creating an immediate Ω+1 against the same holdouts.

## Immediate Ω1 checklist

- [x] deterministic SSA-like core representation
- [x] canonical serialization and SHA-256 program identity
- [x] executable task-agnostic core operations
- [x] semantic gene object with executable expansion
- [x] strict causal admission object
- [x] unit-test and CI foundation
- [ ] explicit state/memory operations
- [ ] bounded loops/control-flow blocks
- [ ] typed function definitions/calls
- [ ] graph/tensor structural values
- [ ] resource accounting
- [ ] independently written portable interpreter
- [ ] differential tests between both runtimes
- [ ] canonical lowering format for invented semantic genes

Ω1 is complete only when the two runtimes agree on a preregistered generated-program corpus and all substrate operations remain task-agnostic.
