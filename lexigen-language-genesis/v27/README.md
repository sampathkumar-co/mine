# Lexigen v27 — compact guided frontier

v27 keeps the exact v25 typed language, the v26 support-first order, the same task identities, demonstrations, depth, and five-million raw-candidate ceiling.

It adds deterministic target-derived beams for ObjectSet and PointSet expressions. Support candidates are ranked against changed cells, source and target foreground, and target colour regions. Grid candidates keep the v26 target-distance beam. Exact outputs are never pruned.

The first controlled gate must finish depth 5 with at most 200,000 retained expressions and no more raw evaluations than v26, or find an exact program. Held-out generators remain unopened until discovery evidence is frozen.
