#!/usr/bin/env bash
set -euo pipefail

# Simple metrics verification: ensure endpoints return Prometheus text with HELP/TYPE lines.
# Usage: ./tools/testing/metrics-check.sh

ADM_SECRET="dev-secret-12345"
ADM_MIN=$(( $(date +%s) / 60 ))
ADM_TOKEN=$(printf "%s|%s" "$ADM_MIN" "ingress" | openssl dgst -sha256 -hmac "$ADM_SECRET" -binary | xxd -p -c 256)

curl_do() {
  local url="$1"; shift || true
  if [[ "$url" == https://localhost:8081/* ]]; then
    curl -skf --connect-timeout 2 --max-time 3 -H "X-ShieldX-Admission: $ADM_TOKEN" "$url" "$@"
  else
    curl -sf --connect-timeout 2 --max-time 3 "$url" "$@"
  fi
}

declare -a NAMES=(
  auth-service locator shieldx-gateway ml-orchestrator verifier-pool contauth policy-rollout
)
declare -a URLS=(
  http://localhost:8084/metrics \
  http://localhost:8083/metrics \
  http://localhost:8082/metrics \
  http://localhost:8087/metrics \
  http://localhost:8090/metrics \
  http://localhost:5002/metrics \
  http://localhost:8099/metrics \
)

fail=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; url="${URLS[$i]}"
  if ! body=$(curl_do "$url" 2>/dev/null); then
    echo "❌ $name metrics unreachable: $url"
    fail=$((fail+1))
    continue
  fi
  if echo "$body" | grep -q "^# HELP "; then
    echo "✅ $name metrics OK"
  else
    echo "❌ $name metrics missing HELP/TYPE lines"
    fail=$((fail+1))
  fi
done

if [[ $fail -gt 0 ]]; then
  echo "=== Metrics check: $fail failures" >&2
  exit 1
fi
echo "=== Metrics check: all OK"
