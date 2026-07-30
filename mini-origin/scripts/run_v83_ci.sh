#!/usr/bin/env bash
set -euo pipefail

PARENT=19b5ae7fd97d92c75451269e78a032d0f298c8d7
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

git diff --exit-code "$PARENT" -- \
  mini-origin/src/mini_origin/response_cost_pareto_v56.py \
  mini-origin/src/mini_origin/response_cost_lower_bound_v65.py \
  mini-origin/compiled/response_cost_lower_bound_v66.rs \
  mini-origin/src/mini_origin/conditioned_cell_frontier_v60.py \
  mini-origin/src/mini_origin/numeric_threshold_frontier_v70.py \
  mini-origin/src/mini_origin/numeric_threshold_repaired_v70.py \
  mini-origin/src/mini_origin/label_free_selector_certificate_v71.py \
  mini-origin/src/mini_origin/label_free_frontier_v72.py \
  mini-origin/src/mini_origin/openml_blind_v78.py \
  mini-origin/src/mini_origin/small_query_coverage_v79.py \
  mini-origin/src/mini_origin/pmlb_blind_v82.py \
  research-evidence/mini-origin-v79-small-query-coverage-pass.json \
  research-evidence/mini-origin-v81-pmlb-hash-lock.json \
  research-evidence/mini-origin-v82-pmlb-blind-rejected.json

cd mini-origin
python -m pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_small_query_coverage_v79.py \
  tests/test_pmlb_blind_v82.py \
  tests/test_medium_small_query_coverage_v83.py

rm -rf results/v83
mkdir -p results/v83
python -m mini_origin.medium_small_query_coverage_v83 reference \
  --states results/v83/states.txt \
  --reference results/v83/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v83/response_cost_lower_bound_v83

results/v83/response_cost_lower_bound_v83 \
  --input results/v83/states.txt \
  --output results/v83/rust-results.json

python -m mini_origin.medium_small_query_coverage_v83 validate \
  --reference results/v83/python-reference.json \
  --rust results/v83/rust-results.json \
  --output results/v83/evidence.json
