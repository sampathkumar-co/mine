#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import hashlib
import json
from pathlib import Path

row = json.loads(
    Path("campaigns/v78-openml-blind.json")
    .read_text(encoding="utf-8-sig")
)
manifest = Path("../research-evidence/mini-origin-v77-openml-hash-lock.json")
assert row["status"] == "preregistered_before_record_access"
assert row["parent_v77_commit"] == "8168664e4068aa3a8b8736dc3ff13b35ecf67981"
assert row["parent_v77_lock_digest"] == "03a64c8c5928070fb41b15d4892c2f720a909fc39c3a7f5b9597cd79f1879590"
assert row["parent_v77_manifest_sha256"] == "e9e500a6441720feff3455cfea183248b03b4fb991a5e4e8448840463c845284"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == row["parent_v77_manifest_sha256"]
assert row["frozen_v75_commit"] == "d8aa4153b69b82ccb714cfbb50d12c5137186047"
assert row["parent_v75_evidence_digest"] == "db379850b2a517e16d5ea442047ac4933ad06fdcf4d6838d91fc36d72e75bc47"
assert row["record_or_label_access_before_preregistration"] is False
assert row["solver_execution_before_preregistration"] is False
assert row["algorithm_revisions_after_record_access"] == 0
assert row["scientific_threshold_revisions_after_record_access"] == 0
PY

git diff --exit-code 8168664e4068aa3a8b8736dc3ff13b35ecf67981 -- \
  ../research-evidence/mini-origin-v77-openml-hash-lock.json \
  ../research-evidence/mini-origin-v75-fresh-external-blind-pass.json \
  campaigns/v75-fresh-external-blind.json \
  src/mini_origin/fresh_external_blind_v75.py \
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
  tests/test_fresh_external_blind_v75.py \
  tests/test_openml_blind_v78.py

mkdir -p results/v78
python -m mini_origin.openml_blind_v78 reference \
  --states results/v78/states.txt \
  --reference results/v78/python-reference.json

rustc -C opt-level=3 -C debuginfo=0 \
  compiled/response_cost_lower_bound_v66.rs \
  -o results/v78/response_cost_lower_bound_v78

results/v78/response_cost_lower_bound_v78 \
  --input results/v78/states.txt \
  --output results/v78/rust-results.json

python -m mini_origin.openml_blind_v78 validate \
  --reference results/v78/python-reference.json \
  --rust results/v78/rust-results.json \
  --output results/v78/evidence.json
