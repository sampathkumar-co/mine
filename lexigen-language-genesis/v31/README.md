# Lexigen v31 validated motif recurrence

v31 tests whether the v30 program validated on task `9565186b` recurs on 64 identities never used by v21-v30.

The only candidates are the same background-preserving foreground-recolor program with color values `0` through `9`, in ascending order. Every task receives six deterministic demonstrations.

A task qualifies only when exactly one color matches all six demonstrations. That candidate must immediately pass 100 fixed fresh cases using the frozen AST runtime, an independent direct runtime and the relational verifier. No task or case replacement is allowed.

One new fresh-passed task would establish repeated public task-level transfer across the v30 source identity and the new identity. This still would not constitute outside-human reproduction or a world-level breakthrough.
