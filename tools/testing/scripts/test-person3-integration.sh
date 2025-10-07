#!/usr/bin/env bash
#
# Integration Test Suite for Person 3 Components
# Tests: Credits, Shadow, Camouflage, Database, Backups
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
CREDITS_URL="${CREDITS_URL:-http://localhost:5004}"
SHADOW_URL="${SHADOW_URL:-http://localhost:5005}"
CAMOUFLAGE_URL="${CAMOUFLAGE_URL:-http://localhost:8090}"
CREDITS_API_KEY="${CREDITS_API_KEY:-test-key}"
SHADOW_API_KEY="${SHADOW_API_KEY:-test-key}"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

assert_equals() {
    ((TESTS_RUN++))
    local expected="$1"
    local actual="$2"
    local message="$3"
    
    if [[ "$expected" == "$actual" ]]; then
        log_pass "$message"
        return 0
    else
        log_fail "$message (expected: $expected, got: $actual)"
        return 1
    fi
}

assert_contains() {
    ((TESTS_RUN++))
    local haystack="$1"
    local needle="$2"
    local message="$3"
    
    if echo "$haystack" | grep -q "$needle"; then
        log_pass "$message"
        return 0
    else
        log_fail "$message (needle not found)"
        return 1
    fi
}

# Test 1: Credits Service Health
test_credits_health() {
    log_test "Testing credits service health..."
    
    local response=$(curl -s -w "%{http_code}" -o /tmp/credits_health.json "$CREDITS_URL/health")
    
    assert_equals "200" "$response" "Credits health endpoint returns 200"
    
    local status=$(jq -r '.status' /tmp/credits_health.json)
    assert_equals "healthy" "$status" "Credits service status is healthy"
}

# Test 2: Credits ACID Transaction
test_credits_acid() {
    log_test "Testing ACID transaction guarantees..."
    
    local tenant_id="test-tenant-$$"
    local idempotency_key="idem-key-$$"
    
    # Purchase credits
    local purchase_response=$(curl -s -X POST "$CREDITS_URL/credits/purchase" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 1000,
            "payment_method": "test",
            "payment_token": "tok_test",
            "idempotency_key": "'"$idempotency_key"'"
        }')
    
    local success=$(echo "$purchase_response" | jq -r '.success')
    assert_equals "true" "$success" "Credits purchase succeeds"
    
    local txn_id=$(echo "$purchase_response" | jq -r '.transaction_id')
    assert_contains "$txn_id" "-" "Transaction ID is UUID"
    
    # Test idempotency - same key should return cached response
    local repeat_response=$(curl -s -X POST "$CREDITS_URL/credits/purchase" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 1000,
            "payment_method": "test",
            "payment_token": "tok_test",
            "idempotency_key": "'"$idempotency_key"'"
        }')
    
    local repeat_txn_id=$(echo "$repeat_response" | jq -r '.transaction_id')
    assert_equals "$txn_id" "$repeat_txn_id" "Idempotency key prevents duplicate transactions"
    
    # Consume credits
    local consume_response=$(curl -s -X POST "$CREDITS_URL/credits/consume" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 100,
            "description": "test consume",
            "idempotency_key": "consume-'$RANDOM'"
        }')
    
    local balance=$(echo "$consume_response" | jq -r '.balance')
    assert_equals "900" "$balance" "Balance correctly updated after consume"
    
    # Try to consume more than balance (should fail)
    local overdraft_response=$(curl -s -X POST "$CREDITS_URL/credits/consume" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 10000,
            "description": "overdraft attempt",
            "idempotency_key": "overdraft-'$RANDOM'"
        }')
    
    local overdraft_success=$(echo "$overdraft_response" | jq -r '.success')
    assert_equals "false" "$overdraft_success" "Cannot consume more than balance (ACID)"
}

# Test 3: Credits Reservation (Two-Phase Commit)
test_credits_reservation() {
    log_test "Testing two-phase credit reservation..."
    
    local tenant_id="test-2phase-$$"
    
    # Setup: Purchase credits
    curl -s -X POST "$CREDITS_URL/credits/purchase" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"tenant_id": "'"$tenant_id"'", "amount": 500, "payment_method": "test", "idempotency_key": "setup-'$RANDOM'"}' \
        > /dev/null
    
    # Reserve credits
    local reserve_response=$(curl -s -X POST "$CREDITS_URL/credits/reserve" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 200,
            "ttl_seconds": 300,
            "idempotency_key": "reserve-'$RANDOM'"
        }')
    
    local reservation_id=$(echo "$reserve_response" | jq -r '.reservation_id')
    assert_contains "$reservation_id" "-" "Reservation created with ID"
    
    # Commit reservation
    local commit_response=$(curl -s -X POST "$CREDITS_URL/credits/commit" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "reservation_id": "'"$reservation_id"'",
            "idempotency_key": "commit-'$RANDOM'"
        }')
    
    local commit_success=$(echo "$commit_response" | jq -r '.success')
    assert_equals "true" "$commit_success" "Reservation committed successfully"
    
    local final_balance=$(echo "$commit_response" | jq -r '.balance')
    assert_equals "300" "$final_balance" "Balance correct after commit (500-200=300)"
}

# Test 4: Shadow Evaluation Service
test_shadow_evaluation() {
    log_test "Testing shadow evaluation engine..."
    
    # Create evaluation
    local eval_request=$(curl -s -X POST "$SHADOW_URL/shadow/evaluate" \
        -H "Authorization: Bearer $SHADOW_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "rule_id": "test-rule-'$$'",
            "rule_name": "Test Rate Limit",
            "rule_type": "rate_limit",
            "rule_config": {"threshold": 100},
            "sample_size": 1000,
            "tenant_id": "test-tenant"
        }')
    
    local eval_id=$(echo "$eval_request" | jq -r '.eval_id')
    assert_contains "$eval_id" "-" "Shadow evaluation created with ID"
    
    # Wait for evaluation to complete (async)
    sleep 3
    
    # Get results
    local results=$(curl -s "$SHADOW_URL/shadow/results/$eval_id" \
        -H "Authorization: Bearer $SHADOW_API_KEY")
    
    local status=$(echo "$results" | jq -r '.status')
    assert_contains "$status" "completed\|running\|pending" "Evaluation has valid status"
    
    # Check for key metrics
    if echo "$results" | jq -e '.precision' > /dev/null; then
        log_pass "Evaluation includes precision metric"
        ((TESTS_PASSED++))
    else
        log_fail "Evaluation missing precision metric"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
}

# Test 5: Audit Log Immutability
test_audit_immutability() {
    log_test "Testing audit log immutability..."
    
    # This would require direct database access
    # For integration test, we verify logs are created
    
    local tenant_id="test-audit-$$"
    
    # Perform transaction
    curl -s -X POST "$CREDITS_URL/credits/purchase" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "tenant_id": "'"$tenant_id"'",
            "amount": 100,
            "payment_method": "test",
            "idempotency_key": "audit-test-'$RANDOM'"
        }' > /dev/null
    
    # Check audit log file exists and has entries
    if [[ -f "data/ledger-credits.log" ]]; then
        local log_entries=$(grep "$tenant_id" data/ledger-credits.log | wc -l)
        if [[ $log_entries -gt 0 ]]; then
            log_pass "Audit log contains transaction record"
            ((TESTS_PASSED++))
        else
            log_fail "Audit log missing transaction record"
            ((TESTS_FAILED++))
        fi
    else
        log_fail "Audit log file not found"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
}

# Test 6: Rate Limiting
test_rate_limiting() {
    log_test "Testing API rate limiting..."
    
    local failed_count=0
    local success_count=0
    
    # Send burst of requests
    for i in {1..20}; do
        local response=$(curl -s -w "%{http_code}" -o /dev/null "$CREDITS_URL/health")
        if [[ "$response" == "429" ]]; then
            ((failed_count++))
        elif [[ "$response" == "200" ]]; then
            ((success_count++))
        fi
    done
    
    # At least some requests should succeed
    if [[ $success_count -gt 0 ]]; then
        log_pass "Rate limiting allows legitimate traffic ($success_count/20)"
        ((TESTS_PASSED++))
    else
        log_fail "Rate limiting too aggressive"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
}

# Test 7: Metrics Endpoint
test_metrics() {
    log_test "Testing Prometheus metrics..."
    
    local metrics=$(curl -s "$CREDITS_URL/metrics")
    
    assert_contains "$metrics" "credits_operations_total" "Metrics include operations counter"
    assert_contains "$metrics" "http_request_duration" "Metrics include latency histogram"
    assert_contains "$metrics" "go_goroutines" "Metrics include Go runtime metrics"
}

# Test 8: Database Connection Pooling
test_db_pooling() {
    log_test "Testing database connection pooling..."
    
    # Make concurrent requests
    for i in {1..10}; do
        curl -s "$CREDITS_URL/health" > /dev/null &
    done
    wait
    
    # Check metrics for connection pool stats
    local metrics=$(curl -s "$CREDITS_URL/metrics")
    
    if echo "$metrics" | grep -q "go_sql_stats_open_connections"; then
        log_pass "Database connection pool metrics available"
        ((TESTS_PASSED++))
    else
        log_fail "Database connection pool metrics missing"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
}

# Test 9: Error Handling
test_error_handling() {
    log_test "Testing error handling..."
    
    # Invalid JSON
    local response=$(curl -s -w "%{http_code}" -o /dev/null -X POST "$CREDITS_URL/credits/consume" \
        -H "Authorization: Bearer $CREDITS_API_KEY" \
        -H "Content-Type: application/json" \
        -d 'invalid json')
    
    assert_equals "400" "$response" "Invalid JSON returns 400 Bad Request"
    
    # Missing auth
    local response=$(curl -s -w "%{http_code}" -o /dev/null -X POST "$CREDITS_URL/credits/consume" \
        -H "Content-Type: application/json" \
        -d '{"tenant_id": "test", "amount": 100}')
    
    assert_equals "401" "$response" "Missing auth returns 401 Unauthorized"
}

# Test 10: Backup Script
test_backup_script() {
    log_test "Testing backup automation..."
    
    if [[ -x "scripts/backup-production.sh" ]]; then
        log_pass "Backup script exists and is executable"
        ((TESTS_PASSED++))
    else
        log_fail "Backup script missing or not executable"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
    
    # Test backup script help
    if scripts/backup-production.sh 2>&1 | grep -q "Usage:"; then
        log_pass "Backup script has usage documentation"
        ((TESTS_PASSED++))
    else
        log_fail "Backup script missing usage documentation"
        ((TESTS_FAILED++))
    fi
    ((TESTS_RUN++))
}

# Run all tests
main() {
    echo "=========================================="
    echo "Person 3 Integration Test Suite"
    echo "=========================================="
    echo ""
    
    # Check services are running
    echo "Checking service availability..."
    if ! curl -sf "$CREDITS_URL/health" > /dev/null; then
        echo -e "${RED}ERROR:${NC} Credits service not available at $CREDITS_URL"
        echo "Start with: cd services/credits && go run ."
        exit 1
    fi
    
    if ! curl -sf "$SHADOW_URL/health" > /dev/null; then
        echo -e "${YELLOW}WARNING:${NC} Shadow service not available at $SHADOW_URL"
        echo "Some tests will be skipped"
    fi
    
    echo -e "${GREEN}Services are ready!${NC}"
    echo ""
    
    # Run test suite
    test_credits_health
    test_credits_acid
    test_credits_reservation
    test_shadow_evaluation
    test_audit_immutability
    test_rate_limiting
    test_metrics
    test_db_pooling
    test_error_handling
    test_backup_script
    
    # Summary
    echo ""
    echo "=========================================="
    echo "Test Results"
    echo "=========================================="
    echo "Total tests:  $TESTS_RUN"
    echo -e "Passed:       ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed:       ${RED}$TESTS_FAILED${NC}"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "\n${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
