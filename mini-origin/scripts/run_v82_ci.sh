#!/usr/bin/env bash
# Activation note: ordinary Mini-ORIGIN CI passed on preregistered head aea1fb3125f48c8457d1cddce456157bb3bd019e.
# This annotation changes no campaign input, adapter, solver, budget, or scientific threshold.
set -euo pipefail

PARENT=d04379797d5aa3da5328bc8c1f51bfc6d4204f4f
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
  research-evidence/mini-origin-v79-small-query-coverage-pass.json \
  research-evidence/mini-origin-v80-pmlb-preblind-registry.json \
  research-evidence/mini-origin-v81-pmlb-hash-lock.json

cd mini-origin
python -m pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_small_query_coverage_v79.py \
  tests/test_pmlb_blind_v82.py

rm -rf results/v82
mkdir -p results/v82
python -m mini_origin.pmlb_blind_v82 reference \
  --states results/v82/states.txt \
  --reference results/v82/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v82/response_cost_lower_bound_v82

results/v82/response_cost_lower_bound_v82 \
  --input results/v82/states.txt \
  --output results/v82/rust-results.json

python -m mini_origin.pmlb_blind_v82 validate \
  --reference results/v82/python-reference.json \
  --rust results/v82/rust-results.json \
  --output results/v82/evidence.json
