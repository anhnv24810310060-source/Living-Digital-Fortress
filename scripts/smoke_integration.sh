#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[smoke] building integration harness"
go build -o integration-harness ./cmd/integration-harness

echo "[smoke] running harness"
OUTPUT=$(./integration-harness 2>&1 | tee /dev/stderr)

if echo "$OUTPUT" | grep -q "analysis_results_total 1"; then
  echo "[smoke] PASS: analysis_results_total observed"
else
  echo "[smoke] FAIL: expected analysis_results_total 1 not found" >&2
  exit 1
fi
