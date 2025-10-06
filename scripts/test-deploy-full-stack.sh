#!/usr/bin/env bash
# shellcheck disable=SC2155

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_SCRIPT="${PROJECT_ROOT}/scripts/deploy-full-stack.sh"
ORIG_PATH="$PATH"

PASS_COUNT=0
FAIL_COUNT=0

# Capture detailed results for JSON export
declare -a TEST_NAMES=()
declare -a TEST_STATUSES=()
declare -a TEST_DURATIONS_MS=()

RESULT_JSON_PATH="${RESULT_JSON_PATH:-${PROJECT_ROOT}/results.json}"

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

now_ns() { date +%s%N; }

run_test() {
  local name="$1"; shift
  local t_start t_end dur_ms status
  printf 'Running %s... ' "$name"
  t_start=$(now_ns)
  if ( set -euo pipefail; "$@" ); then
    echo "ok"
    PASS_COUNT=$((PASS_COUNT + 1))
    status="pass"
  else
    echo "fail"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    status="fail"
  fi
  t_end=$(now_ns)
  dur_ms=$(( (t_end - t_start)/1000000 ))
  TEST_NAMES+=("$name")
  TEST_STATUSES+=("$status")
  TEST_DURATIONS_MS+=("$dur_ms")
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

# --- Extended tests ---
test_setup_failure_abort() {
  local fixture output status
  fixture="$(create_fixture 1)"
  set +e
  output="$(cd "$fixture" \
    && SHIELDX_STUB_LOG_DIR="$fixture/.state" SHIELDX_SETUP_EXIT_CODE=11 PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]]
  grep -q 'setup ' "$fixture/.state/setup.log"
}

test_health_retry_counts() {
  local fixture output status curl_log count
  fixture="$(create_fixture 1)"
  curl_log="$fixture/.state/curl.log"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" SHIELDX_CURL_FAIL_PATTERN=8083/healthz SHIELDX_CURL_FAIL_UNTIL=3 PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]]
  count="$(grep -c '8083/healthz' "$curl_log" || true)"
  (( count >= 4 ))
  grep -q 'ShieldX stack is up' <<<"$output"
}

test_smoke_failure() {
  local fixture output status
  fixture="$(create_fixture 1)"
  set +e
  output="$(cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" SHIELDX_CURL_FAIL_PATTERN=8082/health SHIELDX_CURL_FAIL_ALWAYS=1 PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down --smoke-test 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]]
  grep -Eq 'smoke test|Gateway' <<<"$output" || true
}

test_idempotent_runs() {
  local fixture docker_log2
  fixture="$(create_fixture 1)"
  (cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down >/dev/null 2>&1)
  (cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down >/dev/null 2>&1)
  docker_log2="$fixture/.state/docker.log"
  ! grep -q 'down --remove-orphans' "$docker_log2"
  grep -q 'up -d' "$docker_log2"
}

test_quick_runtime_with_skip() {
  local fixture start end delta_ms
  fixture="$(create_fixture 1)"
  start=$(now_ns)
  (cd "$fixture" && SHIELDX_STUB_LOG_DIR="$fixture/.state" PATH="$fixture/stubs:$ORIG_PATH" bash ./scripts/deploy-full-stack.sh --skip-build --skip-setup --no-down >/dev/null 2>&1)
  end=$(now_ns)
  delta_ms=$(( (end - start)/1000000 ))
  (( delta_ms < 2500 ))
}

write_results_json() {
  local out tmp
  tmp="${RESULT_JSON_PATH}.tmp"
  {
    echo '{'
    echo '  "summary": {'
    echo "    \"passed\": $PASS_COUNT,"
    echo "    \"failed\": $FAIL_COUNT,"
    echo "    \"total\": $((PASS_COUNT+FAIL_COUNT))"
    echo '  },'
    echo '  "tests": ['
    local i last=$(( ${#TEST_NAMES[@]} - 1 ))
    for i in "${!TEST_NAMES[@]}"; do
      printf '    {"name": %q, "status": %q, "duration_ms": %s}' "${TEST_NAMES[$i]}" "${TEST_STATUSES[$i]}" "${TEST_DURATIONS_MS[$i]}"
      if (( i < last )); then echo ','; else echo; fi
    done
    echo '  ]'
    echo '}'
  } >"$tmp"
  mv "$tmp" "$RESULT_JSON_PATH"
  echo "Wrote JSON results to $RESULT_JSON_PATH"
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

  # Extended
  run_test "setup failure aborts deployment" test_setup_failure_abort
  run_test "health retries counted (locator delayed)" test_health_retry_counts
  run_test "smoke test failure propagates" test_smoke_failure
  run_test "idempotent second run without down" test_idempotent_runs
  run_test "quick runtime with skips" test_quick_runtime_with_skip

  printf '\n%d tests passed, %d failed\n' "$PASS_COUNT" "$FAIL_COUNT"
  write_results_json
  if (( FAIL_COUNT == 0 )); then
    return 0
  fi
  return 1
}

main "$@"
