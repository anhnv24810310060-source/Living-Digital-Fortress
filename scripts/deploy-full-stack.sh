#!/usr/bin/env bash

set -euo pipefail

PROJECT_NAME="shieldx"
STACK_FILE="docker-compose.full.yml"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_PATH="${ROOT_DIR}/${STACK_FILE}"
SETUP_SCRIPT="${ROOT_DIR}/scripts/setup-dev-p0.sh"

declare -a required_bins=("docker" "curl")

log() {
  local level="$1"; shift
  local color
  case "$level" in
    INFO) color='\033[0;34m';;
    OK) color='\033[0;32m';;
    WARN) color='\033[0;33m';;
    ERR) color='\033[0;31m';;
    *) color='\033[0m';;
  esac
  printf "${color}[%s]\033[0m %s\n" "$level" "$*"
}

usage() {
  cat <<'EOF'
Usage: deploy-full-stack.sh [options]

Options:
  --skip-build       Skip docker compose build step
  --skip-setup       Skip dev asset bootstrap (certs, policies)
  --no-down          Do not stop existing stack before deploying
  --smoke-test       Run quick HTTP smoke tests after startup
  --help             Show this help message
EOF
}

SKIP_BUILD=false
SKIP_SETUP=false
NO_DOWN=false
RUN_SMOKE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true ;;
    --skip-setup) SKIP_SETUP=true ;;
    --no-down) NO_DOWN=true ;;
    --smoke-test) RUN_SMOKE=true ;;
    --help|-h) usage; exit 0 ;;
    *) log ERR "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

start_ts=$(date +%s)

for bin in "${required_bins[@]}"; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    log ERR "Missing required binary: $bin"
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  log ERR "Docker Compose V2 plugin is not available"
  exit 1
fi

if [[ ! -f "$STACK_PATH" ]]; then
  log ERR "Cannot find compose file at $STACK_PATH"
  exit 1
fi

log INFO "Working inside $ROOT_DIR"
cd "$ROOT_DIR"

if [[ "$SKIP_SETUP" == false ]]; then
  if [[ ! -x "$SETUP_SCRIPT" ]]; then
    log WARN "Bootstrap script $SETUP_SCRIPT not executable; attempting to set +x"
    chmod +x "$SETUP_SCRIPT" || true
  fi
  if [[ -x "$SETUP_SCRIPT" ]]; then
    log INFO "Bootstrapping dev assets via $SETUP_SCRIPT"
    "$SETUP_SCRIPT" >/tmp/shieldx-setup.log 2>&1 || {
      log WARN "Bootstrap script returned non-zero, continuing regardless"
    }
  else
    log WARN "Bootstrap script missing, skipping asset setup"
  fi
fi

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

if [[ "$NO_DOWN" == false ]]; then
  log INFO "Stopping any previous stack (if running)"
  docker compose -p "$PROJECT_NAME" -f "$STACK_FILE" down --remove-orphans >/dev/null 2>&1 || true
fi

if [[ "$SKIP_BUILD" == false ]]; then
  log INFO "Building container images (parallel)"
  docker compose -p "$PROJECT_NAME" -f "$STACK_FILE" build --pull --parallel
else
  log INFO "Skipping image build per flag"
fi

log INFO "Starting ShieldX stack"
docker compose -p "$PROJECT_NAME" -f "$STACK_FILE" up -d

log INFO "Waiting for core services to become responsive"

wait_for() {
  local name="$1" url="$2" timeout="${3:-120}"
  local attempt=0
  until curl -sf --connect-timeout 1 "$url" >/dev/null 2>&1; do
    if (( attempt >= timeout )); then
      log ERR "Service $name not ready after ${timeout}s ($url)"
      return 1
    fi
    ((attempt++))
    sleep 1
  done
  log OK "$name is ready ($url)"
}

set +e
failures=0
wait_for "orchestrator" "http://localhost:8080/health" || ((failures++))
wait_for "locator" "http://localhost:8083/healthz" || ((failures++))
wait_for "guardian" "http://localhost:9090/healthz" || wait_for "guardian" "http://localhost:9090/health" || ((failures++))
wait_for "ingress" "http://localhost:8081/health" || ((failures++))
wait_for "ml-orchestrator" "http://localhost:8087/health" || ((failures++))
wait_for "verifier-pool" "http://localhost:8090/health" || ((failures++))
wait_for "shieldx-gateway" "http://localhost:8082/health" || ((failures++))
wait_for "auth-service" "http://localhost:8084/health" || ((failures++))
wait_for "contauth" "http://localhost:5002/health" || ((failures++))
wait_for "policy-rollout" "http://localhost:8099/health" || ((failures++))
wait_for "prometheus" "http://localhost:9090/-/healthy" || ((failures++))
wait_for "grafana" "http://localhost:3000/api/health" || ((failures++))
set -e

if (( failures > 0 )); then
  log ERR "One or more services failed health checks ($failures). See 'docker compose logs' for details."
  exit 1
fi

if [[ "$RUN_SMOKE" == true ]]; then
  log INFO "Running ingress -> gateway smoke test"
  curl -sf -H "X-Admin-Token: test" http://localhost:8082/health >/dev/null && \
    log OK "Gateway responded" || log WARN "Gateway smoke test responded with unexpected status"
fi

duration=$(( $(date +%s) - start_ts ))
log OK "ShieldX stack is up in ${duration}s"

cat <<'SUMMARY'
Services:
  - Orchestrator:   http://localhost:8080/health
  - Ingress:        http://localhost:8081/health
  - ShieldX Gateway:http://localhost:8082/health
  - Locator:        http://localhost:8083/healthz
  - Auth Service:   http://localhost:8084/health
  - ML Orchestrator:http://localhost:8087/health
  - Verifier Pool:  http://localhost:8090/health
  - ContAuth:       http://localhost:5002/health
  - Policy Rollout: http://localhost:8099/health
  - Guardian:       http://localhost:9090/healthz
  - Prometheus:     http://localhost:9090
  - Grafana:        http://localhost:3000 (admin/fortress123)
SUMMARY

exit 0
