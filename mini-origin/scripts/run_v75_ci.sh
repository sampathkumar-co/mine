#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path

row = json.loads(
    Path("campaigns/v75-fresh-external-blind.json")
    .read_text(encoding="utf-8-sig")
)
assert row["status"] == "preregistered_before_record_access"
assert row["parent_v74_commit"] == "0d03af492972000f794528b1bf6927d65aee3ada"
assert row["parent_v74_lock_digest"] == "6730d5029294a753236f77cef1be1334885a5e8cc8d114a285637987eae5fbaf"
assert row["parent_v74_manifest_sha256"] == "d0d6b849f4cb07a6ee2ae42b6dfc92319579368c91480dd074522dda7d475019"
assert row["frozen_v72_commit"] == "dae02829efc4819935a4ec87c31ea5eee3305d83"
assert row["parent_v72_evidence_digest"] == "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
assert row["record_or_label_access_before_preregistration"] is False
assert row["solver_execution_before_preregistration"] is False
assert row["algorithm_revisions_after_record_access"] == 0
assert row["scientific_threshold_revisions_after_record_access"] == 0
PY

git diff --exit-code 0d03af492972000f794528b1bf6927d65aee3ada -- \
  external-data/uci-v74/manifest.json \
  ../research-evidence/mini-origin-v74-fresh-external-hash-lock.json \
  ../research-evidence/mini-origin-v73-frozen-dataset-registry.json \
  ../research-evidence/mini-origin-v72-label-free-frontier-pass.json \
  campaigns/v72-label-free-frontier-revalidation.json \
  src/mini_origin/label_free_frontier_v72.py \
  src/mini_origin/numeric_threshold_frontier_v70.py \
  src/mini_origin/numeric_threshold_repaired_v70.py \
  src/mini_origin/label_free_selector_certificate_v71.py \
  src/mini_origin/conditioned_cell_frontier_v60.py \
  src/mini_origin/response_cost_pareto_v56.py \
  src/mini_origin/response_cost_lower_bound_v65.py \
  compiled/response_cost_lower_bound_v66.rs

pytest -q \
  tests/test_numeric_threshold_frontier_v70.py \
  tests/test_label_free_selector_v71.py \
  tests/test_label_free_frontier_v72.py \
  tests/test_fresh_external_blind_v75.py

mkdir -p results/v75
python -m mini_origin.fresh_external_blind_v75 reference \
  --states results/v75/states.txt \
  --reference results/v75/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v75/response_cost_lower_bound_v75

results/v75/response_cost_lower_bound_v75 \
  --input results/v75/states.txt \
  --output results/v75/rust-results.json

python -m mini_origin.fresh_external_blind_v75 validate \
  --reference results/v75/python-reference.json \
  --rust results/v75/rust-results.json \
  --output results/v75/evidence.json
