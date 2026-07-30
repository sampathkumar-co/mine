#!/usr/bin/env bash
set -euo pipefail

PARENT=66a100bead80d486591ca9fa16f470ad595f1b2e
V82_REJECTION_SHA256=69ccc610acb28d7d5881a48ae926ba35226f7dc0523994ad8fe5fcd990468747
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
  research-evidence/mini-origin-v80-pmlb-preblind-registry.json \
  research-evidence/mini-origin-v81-pmlb-hash-lock.json

# The v0.82 rejection record was committed only after the frozen v0.82 parent.
# Verify its exact bytes instead of incorrectly requiring it to exist in PARENT.
test -f research-evidence/mini-origin-v82-pmlb-blind-rejection.json
printf '%s  %s\n' \
  "$V82_REJECTION_SHA256" \
  research-evidence/mini-origin-v82-pmlb-blind-rejection.json | sha256sum --check --strict

cd mini-origin
python -m pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_small_query_coverage_v79.py \
  tests/test_pmlb_blind_v82.py

rm -rf results/v83
mkdir -p results/v83
python -m mini_origin.pmlb_development_v83 reference \
  --states results/v83/states.txt \
  --reference results/v83/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v83/response_cost_lower_bound_v83

results/v83/response_cost_lower_bound_v83 \
  --input results/v83/states.txt \
  --output results/v83/rust-results.json

python -m mini_origin.pmlb_development_v83 validate \
  --reference results/v83/python-reference.json \
  --rust results/v83/rust-results.json \
  --output results/v83/evidence.json
