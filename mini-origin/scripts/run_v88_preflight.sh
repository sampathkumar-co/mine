#!/usr/bin/env bash
set -euo pipefail

V85=912c3ebd933ae39eb05e10467f1ecad56e326b03
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

git diff --exit-code "$V85" HEAD -- \
  mini-origin/src/mini_origin/response_lattice_closure_v85.py \
  mini-origin/src/mini_origin/pmlb_blind_v82.py \
  mini-origin/src/mini_origin/openml_blind_v78.py \
  mini-origin/src/mini_origin/numeric_threshold_frontier_v70.py \
  mini-origin/src/mini_origin/conditioned_cell_frontier_v60.py \
  mini-origin/src/mini_origin/response_cost_lower_bound_v65.py \
  mini-origin/compiled/response_cost_lower_bound_v66.rs \
  mini-origin/campaigns/v85-response-lattice-closure.json \
  mini-origin/campaigns/v85-response-lattice-implementation-amendment.json \
  research-evidence/mini-origin-v85-authoritative-opened-data-development-pass.json

cd mini-origin
pytest -q \
  tests/test_v88_final_gate_boundary.py \
  tests/test_openml_hash_lock_v77.py \
  tests/test_openml_blind_v78.py \
  tests/test_response_lattice_closure_v85.py \
  tests/test_numeric_threshold_frontier_v70.py
