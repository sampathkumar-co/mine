#!/usr/bin/env bash
set -euo pipefail

PARENT=f90a77bace939036781c57e249d7cc2a85a96d5c
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

allowed() {
  case "$1" in
    mini-origin/src/mini_origin/ucr_byte_lock_v87.py) return 0 ;;
    mini-origin/tests/test_ucr_byte_lock_v87.py) return 0 ;;
    mini-origin/scripts/run_v87_lock_preflight.sh) return 0 ;;
    .github/workflows/mini-origin-v87-ucr-byte-lock.yml) return 0 ;;
    research-evidence/mini-origin-v87-byte-lock-executor-preregistration.json) return 0 ;;
    mini-origin/.v87-live-lock-trigger) return 0 ;;
    *) return 1 ;;
  esac
}

while IFS= read -r path; do
  if ! allowed "$path"; then
    echo "forbidden post-v87 change before live lock: $path" >&2
    exit 1
  fi
done < <(git diff --name-only "$PARENT" HEAD)

git diff --exit-code "$PARENT" HEAD -- \
  mini-origin/campaigns/v86-ucr-preblind-audit.json \
  mini-origin/src/mini_origin/ucr_catalogue_protocol_v87.py \
  mini-origin/src/mini_origin/pmlb_blind_v82.py \
  mini-origin/src/mini_origin/response_lattice_closure_v85.py \
  research-evidence/mini-origin-v82-pmlb-blind-rejection.json \
  research-evidence/mini-origin-v85-authoritative-opened-data-development-pass.json \
  research-evidence/mini-origin-v87-authoritative-preaccess-record.json

cd mini-origin
python -m pytest -q \
  tests/test_ucr_byte_lock_v87.py \
  tests/test_ucr_catalogue_protocol_v87.py \
  tests/test_v87_preaccess_evidence_record.py
