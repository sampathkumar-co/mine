# LEXIGEN V7 GSO — Frozen Candidate-Generation Contract R1

This contract is frozen before any selected GSO `prob_script` or correctness/performance `tests` value is opened.

## Research agent
The same autonomous research agent generates candidates for every arm from the permitted task specification and base-repository source. The human user supplies no task-specific solver or optimization idea.

## Primitive mechanism vocabulary
The common low-level vocabulary available to every arm is:
- `REPRESENT`: alter data representation without changing semantics.
- `RESTRICT`: avoid work proven irrelevant by a task-local predicate.
- `REDUCE`: transform the input/work domain to a smaller equivalent domain.
- `EXECUTE`: replace or restructure the core execution path.
- `LIFT`: map a reduced-domain result back to the original domain.
- `REFINE`: improve an approximate/intermediate result while preserving the contract.
- `CERTIFY`: check a precondition, invariant, or output-equivalence condition.
- `RECOVER`: use a safe fallback/recovery path when a fast-path condition does not hold.
- `SPECIALIZE`: exploit a source-visible type/shape/value regime without using hidden benchmark values.

These primitive meanings are generic. They contain no selected-task code, constants, benchmark values, expert patch information, or task-specific mechanism.

## Learned V7 macros
Only `v7_full` may invoke these frozen learned compositions as atomic search operators:
- `V7M-001 = EXECUTE -> CERTIFY`
- `V7M-002 = EXECUTE -> CERTIFY -> RECOVER`
- `V7M-003 = CERTIFY -> RECOVER`
- `V7M-004 = REDUCE -> EXECUTE`

A macro invocation must declare which concrete patch regions instantiate each role. Merely attaching a macro ID to an unrelated patch is invalid provenance.

## Random-library macros
Only `v7_random_library` may invoke these frozen equal-size controls:
- `V7R-001 = RECOVER -> REFINE`
- `V7R-002 = REPRESENT -> CERTIFY -> RESTRICT`
- `V7R-003 = RECOVER -> RECOVER`
- `V7R-004 = REPRESENT -> REDUCE`

## `v7_no_library`
`v7_no_library` receives the same task source/specification, primitive vocabulary, proposal count, revision budget and evaluator access, but no learned or random macro may be used as an atomic search operator.

It may independently compose primitives during ordinary candidate construction. If it independently reconstructs an implementation semantically/mechanistically equivalent to a learned-macro candidate, the task cannot earn causal-transfer credit from that macro.

## Source-only proposal budget
Before official correctness/performance feedback, each arm may create at most **6** distinct proposals.

Each proposal must record:
- arm,
- proposed primitive sequence,
- learned/random macro IDs used (if any),
- source-visible preconditions,
- correctness risk,
- expected performance mechanism,
- files/functions intended to change.

No proposal may use expert commit/diff, hints, leaderboard solutions, prior GSO model trajectories, or hidden evaluator results.

## Candidate execution budget
After source-only proposals are frozen:
- at most **3** concrete candidate patches per arm may enter task-local preflight/correctness evaluation;
- at most **2** official feedback revisions per arm are permitted;
- every arm receives the same evaluation count and wall-clock cap at a given stage;
- failed candidates remain in the transcript and cannot be erased;
- final selection freezes exactly one patch per arm before authoritative final evaluation.

## Candidate selection
Within each arm, select candidates lexicographically by:
1. correctness / task-contract validity,
2. anti-gaming/hack audit eligibility,
3. GSO-native performance score,
4. lower patch complexity (changed executable lines),
5. deterministic candidate ID.

Performance feedback may choose among already lawful mechanism families or use one of the two frozen revision slots. It cannot add a new task-specific primitive or alter the learned/random libraries.

## Causal-equivalence rule
A learned macro can receive causal credit only when the selected `v7_full` patch is mechanistically non-equivalent to the selected/no-library-accessible implementation class.

Equivalence is judged from executable behavior and mechanism, not identifier names. Timing noise between equivalent patches never earns causal credit.

## Macro-removal replay
For a provisional causal win, replay the final `v7_full` search with the credited learned macro and its semantic-equivalence class removed while preserving the same evaluator/model budget. The qualifying advantage must disappear.

## Human contribution boundary
Human task-specific solver contribution must remain zero. Infrastructure repair is allowed only when it does not change candidate mechanism families, thresholds, selected task, learned/random libraries, or arm budgets.

## Kill rule
If the preregistered six-task GSO campaign gate becomes mathematically impossible, finish only the minimum already-committed evidence needed to diagnose the failure, seal the negative result, and do not rescue the campaign by changing thresholds or swapping tasks.
