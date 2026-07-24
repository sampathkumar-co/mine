#!/usr/bin/env bash
set -Eeuo pipefail

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
BACKUP_ROOT="${DIRECTOR_BACKUP_DIR:-./backups}"
RETENTION_DAYS="${DIRECTOR_BACKUP_RETENTION_DAYS:-14}"
POSTGRES_USER="${DIRECTOR_POSTGRES_USER:-director}"
POSTGRES_DATABASE="${DIRECTOR_POSTGRES_DATABASE:-director}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DESTINATION="${BACKUP_ROOT%/}/${STAMP}"

mkdir -p "$DESTINATION"
chmod 700 "$DESTINATION"

echo "Creating PostgreSQL backup..."
docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_dump --format=custom --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" \
  > "$DESTINATION/postgres.dump"

echo "Creating Director data backup..."
docker compose -f "$COMPOSE_FILE" run --rm -T --no-deps --entrypoint sh api \
  -c 'tar -C /data -czf - .' > "$DESTINATION/director-data.tar.gz"

echo "Creating Redis state backup..."
docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli SAVE >/dev/null
docker compose -f "$COMPOSE_FILE" run --rm -T --no-deps --entrypoint sh redis \
  -c 'tar -C /data -czf - .' > "$DESTINATION/redis-data.tar.gz"

cat > "$DESTINATION/manifest.txt" <<EOF
created_at=${STAMP}
compose_file=${COMPOSE_FILE}
postgres_database=${POSTGRES_DATABASE}
contents=postgres.dump,director-data.tar.gz,redis-data.tar.gz
EOF
(
  cd "$DESTINATION"
  sha256sum postgres.dump director-data.tar.gz redis-data.tar.gz manifest.txt > SHA256SUMS
)

find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$RETENTION_DAYS" -print -exec rm -rf {} +

echo "Backup complete: $DESTINATION"
echo "Verify with: (cd '$DESTINATION' && sha256sum -c SHA256SUMS)"
