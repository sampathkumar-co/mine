# Candidate materialization R1 incident

Run `32743868213` was a no-execution/no-timing construction gate. Tasks 3–6 materialized all nine selected candidates successfully. Task 2 materialized F1, then F2 failed because an implementation-only replacement marker was not scoped to `Llama.eval` and occurred twice in `llama.py`.

This is not candidate performance/correctness evidence. R2 may only scope the already-frozen F2/N1/N3 transforms to the intended `eval` region; Task-2 proposals, selected candidate IDs, semantics, budgets, tests, thresholds, and libraries cannot change.
