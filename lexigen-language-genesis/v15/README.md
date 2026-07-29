# Lexigen Language Genesis v15 — induced executable scene language

v15 starts from the frozen v14 evidence commit and addresses its central limitation: v14's generic scene operations were still written by humans after inspecting visible failures.

The v15 research objective is to represent scene reasoning as a low-level executable AST, compile known solutions into that substrate, and automatically induce reusable macro productions by anti-unification and minimum-description-length scoring.

## Required progression

1. Execute v14-equivalent programs entirely through the low-level AST runtime.
2. Reproduce them in a separately implemented portable runtime.
3. Automatically discover parameterized repeated subtrees.
4. Rewrite programs to use the induced language and demonstrate real compression.
5. Freeze the substrate, macro learner, verifier and search budgets.
6. On a fresh sealed task, invent a macro absent from the frozen library, commit it and predictions before scoring, and transfer the macro to another untouched task.

Steps 1–4 are mechanism work. Steps 5–6 are the external breakthrough gate.

## Current result

The v15 compiler translates all nine frozen v14 programs into a lower-level executable AST and replays all 54 published demonstrations exactly.

The anti-unification miner automatically induces three parameterised macros without task IDs or human-written macro definitions:

- rectangle-object extraction parameterised by object mode;
- rectangle precedence plus concentric rendering parameterised by object mode;
- singleton selection parameterised by colour.

The induced language preserves exact macro expansion and passed 9,000 fresh official ARC-GEN cases with zero IR failures, zero portable-runtime failures and zero cross-runtime disagreements.

A safe baseline-plus-induced portfolio reduces aggregate candidates from 328 to 216 (1.52x), with 22.3x and 55x improvements on the two shared rectangle families and a worst unrelated slowdown of 6.45%. A strict shape-typed benchmark shows no gain, which is preserved as an important negative result.

This is not a world-level breakthrough. The atomic scene operations remain human-authored v14 semantics; v15 only induces reusable executable macros over them.
## Frozen evidence

The completed v15 freeze strengthens the earlier mechanism result:

- 90,000 accepted fresh ARC-GEN cases across nine source families;
- exact agreement among the primary IR, an independently implemented portable IR, and the frozen v14 portable runtime;
- zero semantic failures and zero three-runtime disagreements;
- seven adversarial and mechanism tests passing;
- a reproducible scan of 898 other public ARC-GEN validation families finding zero exact matches for the induced rectangle macro.

The zero-of-898 scan means the current induced library has not demonstrated held-out semantic transfer. Its search improvement is therefore treated as in-corpus reuse evidence only. See `V15_EVIDENCE.json` and `EVIDENCE.md` for the frozen hashes and claim boundary.
