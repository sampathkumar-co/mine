#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path
row = json.loads(
    Path("campaigns/v79-small-query-coverage.json").read_text(encoding="utf-8")
)
assert row["status"] == "opened_data_development_preregistration"
assert row["parent_v78_commit"] == "1171b81240069e2880aec29dbaacae65b30ea26f"
assert row["parent_v78_first_attempt_run"] == 30459481449
assert row["fresh_blind_claim"] is False
assert row["exact_solver_revisions"] == 0
assert row["compiler_revisions"] == 0
assert row["selector_revisions"] == 1
assert row["adaptive_selector"]["dataset_specific_exceptions"] is False
PY

git diff --exit-code 1171b81240069e2880aec29dbaacae65b30ea26f -- \
  src/mini_origin/response_cost_pareto_v56.py \
  src/mini_origin/response_cost_lower_bound_v65.py \
  compiled/response_cost_lower_bound_v66.rs \
  src/mini_origin/conditioned_cell_frontier_v60.py \
  src/mini_origin/numeric_threshold_frontier_v70.py \
  src/mini_origin/numeric_threshold_repaired_v70.py \
  src/mini_origin/label_free_selector_certificate_v71.py \
  src/mini_origin/label_free_frontier_v72.py \
  src/mini_origin/openml_blind_v78.py \
  ../research-evidence/mini-origin-v77-openml-hash-lock.json

pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_openml_blind_v78.py \
  tests/test_small_query_coverage_v79.py

mkdir -p results/v79
python -m mini_origin.small_query_coverage_v79 reference \
  --states results/v79/states.txt \
  --reference results/v79/python-reference.json
rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v79/response_cost_lower_bound_v79

results/v79/response_cost_lower_bound_v79 \
  --input results/v79/states.txt \
  --output results/v79/rust-results.json

python -m mini_origin.small_query_coverage_v79 validate \
  --reference results/v79/python-reference.json \
  --rust results/v79/rust-results.json \
  --output results/v79/evidence.json
