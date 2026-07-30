#!/usr/bin/env bash
set -euo pipefail

PARENT=35e5fb5a425731e9f55361a1911fe4e9eb51d617
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
  mini-origin/src/mini_origin/near_small_query_coverage_v83.py \
  research-evidence/mini-origin-v82-pmlb-blind-rejection.json \
  research-evidence/mini-origin-v83-near-small-query-coverage-rejection.json

cd mini-origin
python -m pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_small_query_coverage_v79.py \
  tests/test_pmlb_blind_v82.py \
  tests/test_partition_signature_coverage_v84.py

rm -rf results/v84
mkdir -p results/v84
python -m mini_origin.pmlb_development_v84 reference \
  --states results/v84/states.txt \
  --reference results/v84/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v84/response_cost_lower_bound_v84

results/v84/response_cost_lower_bound_v84 \
  --input results/v84/states.txt \
  --output results/v84/rust-results.json

python -m mini_origin.pmlb_development_v84 validate \
  --reference results/v84/python-reference.json \
  --rust results/v84/rust-results.json \
  --output results/v84/evidence.json
