#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/db/restore.sh <dumpfile> <dbname>
DUMP=${1:?dump file required}
DB_NAME=${2:?target database name required}

echo "Restoring $DUMP into $DB_NAME"
PGPASSWORD="${PGPASSWORD:-}" pg_restore \
  -h "${DB_HOST:-localhost}" \
  -p "${DB_PORT:-5432}" \
  -U "${DB_USER:-postgres}" \
  -d "$DB_NAME" \
  --clean --if-exists --no-owner --no-privileges \
  -j "${PG_JOBS:-4}" \
  "$DUMP"
echo "Restore complete"
