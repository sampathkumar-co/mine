#!/usr/bin/env bash
set -euo pipefail

PARENT=912c3ebd933ae39eb05e10467f1ecad56e326b03
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

git diff --exit-code "$PARENT" -- \
  mini-origin/src/mini_origin/response_lattice_closure_v85.py \
  mini-origin/src/mini_origin/pmlb_development_v85.py \
  mini-origin/src/mini_origin/partition_signature_coverage_v84.py \
  mini-origin/src/mini_origin/near_small_query_coverage_v83.py \
  mini-origin/src/mini_origin/pmlb_blind_v82.py \
  mini-origin/src/mini_origin/small_query_coverage_v79.py \
  mini-origin/src/mini_origin/label_free_selector_certificate_v71.py \
  mini-origin/src/mini_origin/numeric_threshold_frontier_v70.py \
  mini-origin/compiled/response_cost_lower_bound_v66.rs \
  mini-origin/campaigns/v85-response-lattice-closure.json \
  mini-origin/campaigns/v85-response-lattice-implementation-amendment.json \
  mini-origin/campaigns/v82-pmlb-blind.json \
  research-evidence/mini-origin-v82-pmlb-blind-rejection.json \
  research-evidence/mini-origin-v83-near-small-query-coverage-rejection.json \
  research-evidence/mini-origin-v84-partition-signature-coverage-rejection.json \
  research-evidence/mini-origin-v85-authoritative-opened-data-development-pass.json \
  research-evidence/mini-origin-v85-exact-rerun-reproducibility.json \
  research-evidence/mini-origin-v80-pmlb-preblind-registry.json

git fetch --force --prune origin '+refs/heads/*:refs/remotes/origin/*'

cd mini-origin
python -m pytest -q \
  tests/test_v87_preaccess_evidence_record.py \
  tests/test_ucr_preblind_audit_v86.py \
  tests/test_ucr_catalogue_protocol_v87.py \
  tests/test_pmlb_preblind_audit_v80.py \
  tests/test_pmlb_development_v85.py \
  tests/test_response_lattice_closure_v85.py

cd "$ROOT"
rm -rf mini-origin/results/v86
mkdir -p mini-origin/results/v86
python -m mini_origin.ucr_preblind_audit_v86 \
  --output mini-origin/results/v86/registry.json
