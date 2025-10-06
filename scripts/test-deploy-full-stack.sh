#!/usr/bin/env bash
# shellcheck disable=SC2155

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_SCRIPT="${PROJECT_ROOT}/scripts/deploy-full-stack.sh"
ORIG_PATH="$PATH"

PASS_COUNT=0
FAIL_COUNT=0

declare -a CLEANUP_PATHS=()

register_cleanup() {
  CLEANUP_PATHS+=("$1")
}

cleanup_all() {
  local path
  for path in "${CLEANUP_PATHS[@]}"; do
    if [[ -n "${path:-}" ]] && [[ -e "$path" || -L "$path" ]]; then
      rm -rf "$path"
    fi
  done
}

trap cleanup_all EXIT

create_fixture() {
  local with_stack="${1:-1}"
  local fixture
  fixture="$(mktemp -d)"
  register_cleanup "$fixture"

  mkdir -p "$fixture/scripts" "$fixture/stubs" "$fixture/.state"
  cp "$DEPLOY_SCRIPT" "$fixture/scripts/deploy-full-stack.sh"
  chmod +x "$fixture/scripts/deploy-full-stack.sh"

  cat <<'EOF' >"$fixture/scripts/setup-dev-p0.sh"
#!/usr/bin/env bash
set -euo pipefail
log_dir="${SHIELDX_STUB_LOG_DIR:?}"
mkdir -p "$log_dir"
printf '%s\n' "setup $*" >>"$log_dir/setup.log"
exit "${SHIELDX_SETUP_EXIT_CODE:-0}"
EOF
  chmod +x "$fixture/scripts/setup-dev-p0.sh"

  cat <<'EOF' >"$fixture/stubs/docker"
#!/usr/bin/env bash
set -euo pipefail
log_dir="${SHIELDX_STUB_LOG_DIR:?}"
mkdir -p "$log_dir"
printf '%s\n' "docker $*" >>"$log_dir/docker.log"
if [[ "${1:-}" == "compose" ]]; then
  shift
  if [[ "${1:-}" == "version" ]]; then
    exit "${SHIELDX_DOCKER_VERSION_EXIT:-0}"
  fi
  if [[ "${SHIELDX_DOCKER_FAIL_PARALLEL_ONCE:-0}" != "0" ]]; then
    for arg in "$@"; do
      if [[ "$arg" == "--parallel" ]]; then
        flag="$log_dir/parallel-failed.flag"
        if [[ ! -f "$flag" ]]; then
          touch "$flag"
          exit "${SHIELDX_DOCKER_PARALLEL_EXIT_CODE:-1}"
        fi
      fi
    done
  fi
fi
exit 0
EOF
  chmod +x "$fixture/stubs/docker"

  cat <<'EOF' >"$fixture/stubs/curl"
#!/usr/bin/env bash
set -euo pipefail
log_dir="${SHIELDX_STUB_LOG_DIR:?}"
mkdir -p "$log_dir"
printf '%s\n' "curl $*" >>"$log_dir/curl.log"
url=""
for arg in "$@"; do
  if [[ "$arg" == http://* || "$arg" == https://* ]]; then
    url="$arg"
  fi
done
pattern="${SHIELDX_CURL_FAIL_PATTERN:-}"
if [[ -n "$pattern" && -n "$url" && "$url" == *"$pattern"* ]]; then
  if [[ "${SHIELDX_CURL_FAIL_ALWAYS:-0}" != "0" ]]; then
    exit "${SHIELDX_CURL_EXIT_CODE:-56}"
  fi
  limit="${SHIELDX_CURL_FAIL_UNTIL:-0}"
  state_file="$log_dir/curl-fail-count"
  count=0
  if [[ -f "$state_file" ]]; then
    read -r count <"$state_file"
  fi
  if (( count < limit )); then
    echo $((count + 1)) >"$state_file"
    exit "${SHIELDX_CURL_EXIT_CODE:-56}"
  fi
fi
exit 0
EOF
  chmod +x "$fixture/stubs/curl"

  cat <<'EOF' >"$fixture/stubs/sleep"
#!/usr/bin/env bash
exit 0
EOF
  chmod +x "$fixture/stubs/sleep"

  if [[ "$with_stack" == "1" ]]; then
    cat <<'EOF' >"$fixture/docker-compose.full.yml"
version: "3.8"
services: {}
EOF
  fi

  printf '%s' "$fixture"
}

create_minimal_toolchain() {
  local toolchain
  toolchain="$(mktemp -d)"
  register_cleanup "$toolchain"
  for cmd in bash cat chmod date dirname env grep printf pwd read tail; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ln -s "$(command -v "$cmd")" "$toolchain/$cmd"
    fi
  done
  printf '%s' "$toolchain"
}

run_test() {
  local name="$1"
  shift
  printf 'Running %s... ' "$name"
  if ( set -euo pipefail; "$@" ); then
    echo "ok"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "fail"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

test_help_option() {
  local output
  output="$(bash "$DEPLOY_SCRIPT" --help 2>&1)"
  grep -q "Usage: deploy-full-stack.sh" <<<"$output"
}

test_missing_required_binary() {
  local toolchain output status
  toolchain="$(create_minimal_toolchain)"
  set +e
  output="$(env -i PATH="$toolchain" HOME="$HOME" TERM="${TERM:-dumb}" bash "$DEPLOY_SCRIPT" 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]]
  grep -q "Missing required binary: docker" <<<"$output"
}

test_missing_stack_file() {
  local fixture output status
  fixture="$(create_fixture 0)"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]]
  grep -q "Cannot find compose file" <<<"$output"
}

test_successful_deploy_flow() {
  local fixture output status docker_log setup_log
  fixture="$(create_fixture 1)"
  docker_log="$fixture/.state/docker.log"
  setup_log="$fixture/.state/setup.log"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]]
  [[ -f "$setup_log" ]]
  grep -q 'setup ' "$setup_log"
  [[ -f "$docker_log" ]]
  grep -q 'down --remove-orphans' "$docker_log"
  grep -q 'build --pull --parallel' "$docker_log"
  grep -q 'build --pull$' "$docker_log"
  grep -q 'up -d' "$docker_log"
  grep -q 'ShieldX stack is up' <<<"$output"
}

test_skip_flags_prevent_actions() {
  local fixture output status docker_log setup_log
  fixture="$(create_fixture 1)"
  docker_log="$fixture/.state/docker.log"
  setup_log="$fixture/.state/setup.log"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]]
  [[ ! -f "$setup_log" ]]
  if [[ -f "$docker_log" ]]; then
    ! grep -q 'down --remove-orphans' "$docker_log"
    ! grep -q 'build --pull' "$docker_log"
  fi
  grep -q 'Skipping image build per flag' <<<"$output"
}

test_parallel_build_retry() {
  local fixture output status docker_log
  fixture="$(create_fixture 1)"
  docker_log="$fixture/.state/docker.log"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" SHIELDX_DOCKER_FAIL_PARALLEL_ONCE=1 PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --no-down 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]]
  grep -q 'Parallel build failed, retrying sequential build' <<<"$output"
  grep -q 'build --pull --parallel' "$docker_log"
  grep -q 'build --pull$' "$docker_log"
}

test_smoke_test_invocation() {
  local fixture output status curl_log
  fixture="$(create_fixture 1)"
  curl_log="$fixture/.state/curl.log"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down --smoke-test 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]]
  [[ -f "$curl_log" ]]
  grep -q 'curl -sf -H X-Admin-Token: test http://localhost:8082/health' "$curl_log"
  grep -q 'Gateway responded' <<<"$output"
}

test_health_check_failure() {
  local fixture output status
  fixture="$(create_fixture 1)"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" SHIELDX_CURL_FAIL_PATTERN=8083/healthz SHIELDX_CURL_FAIL_ALWAYS=1 PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]]
  grep -Eq 'Service locator not ready after|One or more services failed health checks' <<<"$output"
}

main() {
  run_test "help option prints usage" test_help_option
  run_test "missing docker binary aborts" test_missing_required_binary
  run_test "missing compose file aborts" test_missing_stack_file
  run_test "full deployment flow succeeds" test_successful_deploy_flow
  run_test "skip flags suppress actions" test_skip_flags_prevent_actions
  run_test "parallel build retry works" test_parallel_build_retry
  run_test "smoke test issues admin request" test_smoke_test_invocation
  run_test "health check failure propagates" test_health_check_failure

  printf '\n%d tests passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
  if (( FAIL_COUNT == 0 )); then
    return 0
  fi
  return 1
}

main "$@"
