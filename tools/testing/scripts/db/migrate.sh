#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run Go-based migrations
# Usage: scripts/db/migrate.sh <service> [command]
#   service: credits | shadow | cdefnet | all
#   command: up (default) | down | status | force <version>

if [[ ${1:-} == "" ]]; then
  echo "Usage: $0 <service> [command]" >&2
  exit 1
fi

SERVICE="$1"; shift || true
COMMAND="${1:-up}"; shift || true

# DB env (override as needed)
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export DB_USER="${DB_USER:-postgres}"
export PGPASSWORD="${PGPASSWORD:-}"
export DB_NAME="${DB_NAME:-fortress}"
export DB_SSL_MODE="${DB_SSL_MODE:-disable}"

echo "Running migrations: service=$SERVICE command=$COMMAND DB=$DB_NAME@$DB_HOST:$DB_PORT"

# Execute via go run to avoid requiring a prior build
if [[ "$COMMAND" == "force" ]]; then
  VERSION="$1" || { echo "force requires <version>"; exit 2; }
  go run ./cmd/migrate-db "$SERVICE" force "$VERSION"
else
  go run ./cmd/migrate-db "$SERVICE" "$COMMAND"
fi

echo "Done."
