# Lexigen v4 Generalization Engine

This branch contains only the frozen architecture, selection protocol, baselines, validation and future benchmark evidence for the v4 architecture-vs-v3 experiment.

It is independent of:

- `lexigen/world-covering-record-*`
- `lexigen/language-genesis-*`
- Mini-ORIGIN branches
- all frozen v3 task branches

Do not merge this branch into those tracks. Do not copy target identities, solver ideas, triggers or evidence between them.

The engine validation is snapshot-free. Holdout selection occurs only after an `ENGINE_LOCK.json` binds the validated engine and selector.
