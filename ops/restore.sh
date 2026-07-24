#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 <backup-directory> --confirm"
  exit 2
}

[[ $# -eq 2 && "$2" == "--confirm" ]] || usage
BACKUP_DIR="$(cd "$1" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_FILE="${DIRECTOR_ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

COMPOSE_FILE="${DIRECTOR_COMPOSE_FILE:-compose.production.yml}"
POSTGRES_USER="${DIRECTOR_POSTGRES_USER:-director}"
POSTGRES_DATABASE="${DIRECTOR_POSTGRES_DATABASE:-director}"

for required in SHA256SUMS postgres.dump director-data.tar.gz redis-data.tar.gz; do
  [[ -f "$BACKUP_DIR/$required" ]] || { echo "Missing $required" >&2; exit 1; }
done
(
  cd "$BACKUP_DIR"
  sha256sum -c SHA256SUMS
)

echo "Stopping application services..."
docker compose -f "$COMPOSE_FILE" stop caddy frontend api worker beat || true
docker compose -f "$COMPOSE_FILE" up -d postgres redis

echo "Restoring PostgreSQL..."
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  dropdb --if-exists -U "$POSTGRES_USER" "$POSTGRES_DATABASE"
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  createdb -U "$POSTGRES_USER" "$POSTGRES_DATABASE"
cat "$BACKUP_DIR/postgres.dump" | docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_restore --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE"

echo "Restoring Director data volume..."
cat "$BACKUP_DIR/director-data.tar.gz" | docker compose -f "$COMPOSE_FILE" run \
  --rm -T --no-deps --entrypoint sh api \
  -c 'find /data -mindepth 1 -maxdepth 1 -exec rm -rf {} + && tar -C /data -xzf -'

echo "Restoring Redis state..."
docker compose -f "$COMPOSE_FILE" stop redis
cat "$BACKUP_DIR/redis-data.tar.gz" | docker compose -f "$COMPOSE_FILE" run \
  --rm -T --no-deps --entrypoint sh redis \
  -c 'find /data -mindepth 1 -maxdepth 1 -exec rm -rf {} + && tar -C /data -xzf -'
docker compose -f "$COMPOSE_FILE" up -d redis

echo "Applying current migrations and restarting services..."
docker compose -f "$COMPOSE_FILE" run --rm migrate
docker compose -f "$COMPOSE_FILE" up -d

echo "Restore complete. Run: ./ops/smoke.sh https://${DIRECTOR_DOMAIN:-localhost}"
