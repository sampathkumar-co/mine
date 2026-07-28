# Third preselection amendment

PR #85 failed snapshot-free validation with:

`v4 omitted expected mechanisms for grouped_numeric: ['bounded_exact_refinement']`

The failure exposed a mechanism-graph construction bug. The engine initially discarded every operator not immediately legal from raw task features. Therefore an operator that becomes legal only after another mechanism—such as exact refinement after a warm-start operator—could never appear in a composition.

`engine_v4.py` computes the generic reachability closure of operator outputs before enumerating compositions. Single-operator template and v3-compatible baselines still use only immediately legal operators. No ranking threshold, campaign gate or synthetic expectation was weakened.

No benchmark inventory, holdout name, task source, manifest, payload, public solver or report had been accessed when this correction was committed.
