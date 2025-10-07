#!/usr/bin/env bash
set -euo pipefail

echo "=== ShieldX Core Healthcheck ==="

declare -A endpoints=(
  [orchestrator]="http://localhost:8080/health"
  [ingress]="http://localhost:8081/healthz"
  [guardian]="http://localhost:9090/healthz"
  [credits]="http://localhost:5004/health"
  [contauth]="http://localhost:5002/health"
  [shadow]="http://localhost:5005/health"
)

ok=0
total=0
for name in "${!endpoints[@]}"; do
  total=$((total+1))
  url=${endpoints[$name]}
  if curl -fsS "$url" >/dev/null; then
    printf "  ✅ %-12s %s\n" "$name" "$url"
    ok=$((ok+1))
  else
    printf "  ⚠️  %-12s %s (DOWN)\n" "$name" "$url"
  fi
done

echo ""
echo "Summary: $ok/$total services healthy"
exit 0
