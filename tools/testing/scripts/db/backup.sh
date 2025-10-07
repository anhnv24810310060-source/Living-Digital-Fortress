#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/db/backup.sh <dbname> [outdir]
DB_NAME=${1:-fortress}
OUT_DIR=${2:-./backups}

mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FILE="$OUT_DIR/${DB_NAME}_${STAMP}.dump"

echo "Creating backup: $FILE"
PGPASSWORD="${PGPASSWORD:-}" pg_dump -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" -d "$DB_NAME" -Fc -j "${PG_JOBS:-4}" -f "$FILE"
echo "Backup complete: $FILE"
