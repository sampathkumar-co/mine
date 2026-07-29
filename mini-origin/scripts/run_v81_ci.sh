#!/usr/bin/env bash
set -euo pipefail

PARENT=22cc53c9b22b1d6e15190e769462846992e28149
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

git diff --exit-code "$PARENT" -- \
  mini-origin/src/mini_origin/small_query_coverage_v79.py \
  mini-origin/src/mini_origin/openml_blind_v78.py \
  mini-origin/src/mini_origin/label_free_frontier_v72.py \
  mini-origin/src/mini_origin/numeric_threshold_frontier_v70.py \
  mini-origin/src/mini_origin/clean_lower_bound_conditioned_v68.py \
  mini-origin/compiled/response_cost_lower_bound_v66.rs \
  research-evidence/mini-origin-v79-small-query-coverage-pass.json \
  research-evidence/mini-origin-v80-pmlb-preblind-registry.json

cd mini-origin
python -m pytest -q \
  tests/test_pmlb_hash_lock_v81.py \
  tests/test_pmlb_preblind_audit_v80.py

rm -rf results/v81
mkdir -p results/v81
python -m mini_origin.pmlb_hash_lock_v81 \
  --output results/v81/manifest.json
