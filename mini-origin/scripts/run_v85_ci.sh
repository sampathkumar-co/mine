#!/usr/bin/env bash
set -euo pipefail

PARENT=43d3311c15fb6b1dbde8dddafd749f1ef623ded4
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Preserve the complete v0.84 scientific parent and all prior rejections.
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
  mini-origin/src/mini_origin/partition_signature_coverage_v84.py \
  research-evidence/mini-origin-v82-pmlb-blind-rejection.json \
  research-evidence/mini-origin-v83-near-small-query-coverage-rejection.json \
  research-evidence/mini-origin-v84-partition-signature-coverage-rejection.json

# Freeze both pre-access protocol records byte-for-byte.
echo "cf8407d58c87542ed34c59239191aeaf11d2576382b0e0165e7371c6e429b969  mini-origin/campaigns/v85-response-lattice-closure.json" | sha256sum -c -
echo "3c582716a2f4b152b1c87959af99f8ff3847b79fff17f0f8e12756cdbf437766  mini-origin/campaigns/v85-response-lattice-implementation-amendment.json" | sha256sum -c -

cd mini-origin
python -m pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_small_query_coverage_v79.py \
  tests/test_pmlb_blind_v82.py \
  tests/test_partition_signature_coverage_v84.py \
  tests/test_response_lattice_closure_v85.py \
  tests/test_pmlb_development_v85.py

rm -rf results/v85
mkdir -p results/v85
python -m mini_origin.pmlb_development_v85 reference \
  --states results/v85/states.txt \
  --reference results/v85/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v85/response_cost_lower_bound_v85

results/v85/response_cost_lower_bound_v85 \
  --input results/v85/states.txt \
  --output results/v85/rust-results.json

python -m mini_origin.pmlb_development_v85 validate \
  --reference results/v85/python-reference.json \
  --rust results/v85/rust-results.json \
  --output results/v85/evidence.json
