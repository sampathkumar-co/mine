#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="${1:-${DIRECTOR_SMOKE_BASE_URL:-http://localhost:8000}}"
BASE_URL="${BASE_URL%/}"
API_URL="${BASE_URL}/api/v1"
CURL_ARGS=(--fail --silent --show-error --connect-timeout 8 --max-time 30)

if [[ "${DIRECTOR_SMOKE_INSECURE_TLS:-false}" == "true" ]]; then
  CURL_ARGS+=(--insecure)
fi

json_field() {
  python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

echo "Checking liveness..."
LIVE="$(curl "${CURL_ARGS[@]}" "$API_URL/health/live")"
[[ "$(printf '%s' "$LIVE" | json_field status)" == "alive" ]]

echo "Checking dependency readiness..."
READY="$(curl "${CURL_ARGS[@]}" "$API_URL/health/ready")"
READINESS="$(printf '%s' "$READY" | json_field status)"
[[ "$READINESS" == "ready" || "$READINESS" == "degraded" ]]

if [[ -n "${DIRECTOR_METRICS_TOKEN:-}" ]]; then
  echo "Checking protected metrics..."
  METRICS="$(curl "${CURL_ARGS[@]}" -H "Authorization: Bearer ${DIRECTOR_METRICS_TOKEN}" "$API_URL/metrics")"
  grep -q "director_build_info" <<<"$METRICS"
fi

if [[ -n "${DIRECTOR_SMOKE_EMAIL:-}" && -n "${DIRECTOR_SMOKE_PASSWORD:-}" ]]; then
  echo "Checking authenticated account flow..."
  LOGIN="$(curl "${CURL_ARGS[@]}" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,os; print(json.dumps({"email":os.environ["DIRECTOR_SMOKE_EMAIL"],"password":os.environ["DIRECTOR_SMOKE_PASSWORD"]}))')" \
    "$API_URL/auth/login")"
  ACCESS_TOKEN="$(printf '%s' "$LOGIN" | json_field access_token)"
  curl "${CURL_ARGS[@]}" -H "Authorization: Bearer ${ACCESS_TOKEN}" "$API_URL/auth/account" >/dev/null
fi

echo "Smoke verification passed for $BASE_URL"
