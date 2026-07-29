#!/usr/bin/env bash
set -euo pipefail

PARENT=555c3146111a7726702bb98e0a72f3b214d07190

git diff --exit-code "$PARENT" -- \
  mini-origin/src/mini_origin/small_query_coverage_v79.py \
  mini-origin/src/mini_origin/openml_blind_v78.py \
  mini-origin/src/mini_origin/label_free_frontier_v72.py \
  mini-origin/src/mini_origin/numeric_threshold_frontier_v70.py \
  mini-origin/src/mini_origin/clean_lower_bound_conditioned_v68.py \
  mini-origin/compiled/response_cost_lower_bound_v66.rs \
  research-evidence/mini-origin-v79-small-query-coverage-pass.json

python -m pytest -q \
  tests/test_pmlb_preblind_audit_v80.py \
  tests/test_openml_preblind_audit_v76.py \
  tests/test_repository_dataset_audit_v63.py

rm -rf results/v80
mkdir -p results/v80
python -m mini_origin.pmlb_preblind_audit_v80 \
  --output results/v80/registry.json
