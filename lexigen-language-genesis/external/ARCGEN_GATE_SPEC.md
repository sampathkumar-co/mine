# ARC-GEN external blind gate

Frozen protocol version: `arcgen-gate-v1`

## Purpose

Move Lexigen Language Genesis beyond project-authored RIFT worlds by using the independently maintained Google ARC-GEN procedural benchmark generator.

ARC-GEN source is treated as evaluator infrastructure. The selected task module must not be opened, searched, imported interactively, or used to guide solver development. The solver sees only generated input/output demonstrations and generated test inputs.

## One-shot sequence

1. Commit the generic solver, DSL, selection logic, budgets, and scorer.
2. Record the frozen Lexigen commit and ARC-GEN source commit.
3. Parse eligible task IDs only from `task_list.py` import declarations.
4. Select one task with SHA-256 of:
   `arcgen-gate-v1 | lexigen_commit | arcgen_commit`.
5. Generate 6 demonstration pairs and 20 sealed test pairs with ARC-GEN's deterministic seeds.
6. Write a redacted package containing demonstrations and test inputs only.
7. Preserve hidden outputs in a separate sealed file and expose only its SHA-256.
8. Run the frozen solver once on the redacted package.
9. Commit the emitted language artifact and predictions before scoring.
10. Score once. Preserve success or failure permanently.

## Fixed budgets

- maximum primitive sequence depth: 3;
- maximum evaluated candidate programs: 75,000;
- exact-match demonstrations only;
- deterministic candidate ordering;
- no model/API calls after task reveal;
- one final prediction per test input;
- no revisions after hidden scoring.

## Comparison

The fixed-language baseline may select one primitive only.

The Language Genesis system may compose up to three primitives and serialize the composition as a new executable macro with:

- explicit operational semantics;
- demonstration-evidence hash;
- independent interpreter execution;
- ablation against the single-primitive baseline;
- exact hidden-pair scoring.

A successful task is evidence of external executable-language transfer, not by itself a world breakthrough. A stronger claim requires multiple untouched task families, competitive external baselines, and a discovery unavailable to known library-learning or program-synthesis systems.
