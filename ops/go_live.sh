#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${1:-.env}"
BASE_URL="${2:-}"
DIST_DIR="${DIRECTOR_RELEASE_DIST_DIR:-dist}"

mkdir -p "$DIST_DIR"

DOCTOR_ARGS=(
  --env-file "$ENV_FILE"
  --json-output "$DIST_DIR/release-doctor.json"
  --markdown-output "$DIST_DIR/release-doctor.md"
)

if [[ -n "$BASE_URL" ]]; then
  DOCTOR_ARGS+=(--base-url "$BASE_URL")
fi

python3 ops/release_doctor.py "${DOCTOR_ARGS[@]}"
docker compose --env-file "$ENV_FILE" -f compose.production.yml config >/dev/null
python3 ops/release_manifest.py \
  --output "$DIST_DIR/release-manifest.json" \
  --checksums "$DIST_DIR/CHECKSUMS.sha256"

if [[ -n "$BASE_URL" ]]; then
  METRICS_TOKEN="$(
    python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import sys

for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if raw.strip().startswith("DIRECTOR_METRICS_TOKEN="):
        print(raw.split("=", 1)[1].strip().strip('"').strip("'"))
        break
PY
  )"
  DIRECTOR_METRICS_TOKEN="$METRICS_TOKEN" bash ops/smoke.sh "$BASE_URL"
fi

echo "Director OS release gate passed."
echo "Artifacts: $DIST_DIR/release-doctor.json, $DIST_DIR/release-doctor.md, $DIST_DIR/release-manifest.json, $DIST_DIR/CHECKSUMS.sha256"
