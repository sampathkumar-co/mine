# Lexigen World Covering v3 — Snapshot-Free Validation

This branch is created from `main` and contains no v3 research workflow or research trigger. The validation workflow checks out sealed v3 commit `7ca690ba471da0aadea2b34b8f7563da7fb59024` read-only into a subdirectory.

It validates locked hashes, independently reproduces v1 and corrected-v2 target lineage on synthetic metadata, confirms v3 exclusion, verifies a known Fano-plane construction, exercises restricted CP-SAT, proves an impossible six-block case infeasible, and runs the generic v3 solver on the synthetic feasible case.

This is validation evidence only and must not be merged.
