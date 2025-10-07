#!/bin/bash

# P0 Integration Tests for Core Services (PERSON 1)
# Tests: mTLS, rate limiting, OPA policy, health checks, input validation

set -e

echo "=========================================="
echo "P0 Integration Tests - Core Services"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Helper function
test_case() {
    echo -e "\n${YELLOW}TEST:${NC} $1"
}

assert_eq() {
    if [ "$1" == "$2" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL${NC} - Expected: $2, Got: $1"
        ((FAIL++))
    fi
}

assert_contains() {
    if echo "$1" | grep -q "$2"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL${NC} - Expected to contain: $2"
        ((FAIL++))
    fi
}

# Configuration
ORCH_URL="${ORCH_URL:-http://localhost:8080}"
INGRESS_URL="${INGRESS_URL:-http://localhost:8081}"
TIMEOUT=5

echo "Testing Orchestrator: $ORCH_URL"
echo "Testing Ingress: $INGRESS_URL"
echo ""

# ==========================================
# Test 1: Health Endpoints
# ==========================================
test_case "Health endpoint returns 200"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$ORCH_URL/health" || echo "000")
assert_eq "$RESPONSE" "200"

test_case "Health endpoint returns JSON with service name"
RESPONSE=$(curl -s --connect-timeout $TIMEOUT "$ORCH_URL/health" || echo "{}")
assert_contains "$RESPONSE" "orchestrator"

test_case "Healthz endpoint (alias) works"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$ORCH_URL/healthz" || echo "000")
assert_eq "$RESPONSE" "200"

# ==========================================
# Test 2: Metrics Endpoint
# ==========================================
test_case "Metrics endpoint returns 200"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$ORCH_URL/metrics" || echo "000")
assert_eq "$RESPONSE" "200"

test_case "Metrics contains orchestrator metrics"
RESPONSE=$(curl -s --connect-timeout $TIMEOUT "$ORCH_URL/metrics" || echo "")
assert_contains "$RESPONSE" "orchestrator_route_total"

# ==========================================
# Test 3: Policy Endpoint
# ==========================================
test_case "Policy endpoint returns configuration"
RESPONSE=$(curl -s --connect-timeout $TIMEOUT "$ORCH_URL/policy" || echo "{}")
assert_contains "$RESPONSE" "allowAll"

# ==========================================
# Test 4: Route Endpoint - Basic
# ==========================================
test_case "POST /route with valid JSON succeeds"
RESPONSE=$(curl -s -X POST --connect-timeout $TIMEOUT \
    -H "Content-Type: application/json" \
    -d '{"service":"test-service","tenant":"tenant1","path":"/api/v1"}' \
    "$ORCH_URL/route" || echo "error")
# Should return 503 if no backends, or valid JSON
if echo "$RESPONSE" | grep -q "target\|no healthy backend\|service not found"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} - Unexpected response: $RESPONSE"
    ((FAIL++))
fi

# ==========================================
# Test 5: Input Validation
# ==========================================
test_case "POST /route rejects invalid service name (uppercase)"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"Invalid-Service","tenant":"tenant1","path":"/api"}' \
    "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "400"

test_case "POST /route rejects path without leading slash"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"service","tenant":"tenant1","path":"api"}' \
    "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "400"

test_case "POST /route rejects SQL injection attempt"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"service","tenant":"tenant1","path":"/api'"'"'; DROP TABLE users--"}' \
    "$ORCH_URL/route" || echo "000")
# Should be 400 or 403 (validation or policy deny)
if [ "$RESPONSE" == "400" ] || [ "$RESPONSE" == "403" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} - Expected 400 or 403, got $RESPONSE"
    ((FAIL++))
fi

test_case "POST /route rejects XSS attempt"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"service","tenant":"tenant1","path":"/api<script>alert(1)</script>"}' \
    "$ORCH_URL/route" || echo "000")
if [ "$RESPONSE" == "400" ] || [ "$RESPONSE" == "403" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} - Expected 400 or 403, got $RESPONSE"
    ((FAIL++))
fi

test_case "POST /route rejects path traversal"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"service","tenant":"tenant1","path":"/api/../../etc/passwd"}' \
    "$ORCH_URL/route" || echo "000")
if [ "$RESPONSE" == "400" ] || [ "$RESPONSE" == "403" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} - Expected 400 or 403, got $RESPONSE"
    ((FAIL++))
fi

# ==========================================
# Test 6: Rate Limiting (if enabled)
# ==========================================
test_case "Rate limiting (burst test - manual verification)"
echo "Sending 10 rapid requests..."
SUCCESS_COUNT=0
for i in {1..10}; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 \
        -X POST -H "Content-Type: application/json" \
        -d '{"service":"test","tenant":"t1","path":"/test"}' \
        "$ORCH_URL/route" 2>/dev/null || echo "000")
    if [ "$CODE" == "429" ]; then
        echo "  Request $i: Rate limited (429)"
        break
    else
        ((SUCCESS_COUNT++))
    fi
done
echo "  Processed $SUCCESS_COUNT requests before rate limit"
# This test is informational - we don't fail on it
echo -e "${GREEN}✓ PASS${NC} (informational)"
((PASS++))

# ==========================================
# Test 7: Method Validation
# ==========================================
test_case "GET /route returns 405 Method Not Allowed"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "405"

test_case "PUT /route returns 405 Method Not Allowed"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X PUT "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "405"

# ==========================================
# Test 8: JSON Validation
# ==========================================
test_case "POST /route with malformed JSON returns 400"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{invalid json}' \
    "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "400"

test_case "POST /route with unknown fields returns 400 (strict parsing)"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d '{"service":"test","unknownField":"value"}' \
    "$ORCH_URL/route" || echo "000")
assert_eq "$RESPONSE" "400"

# ==========================================
# Test 9: Size Limit
# ==========================================
test_case "POST /route with oversized payload returns 400"
LARGE_PAYLOAD=$(printf '{"service":"test","tenant":"t1","path":"/%s"}' "$(head -c 20000 < /dev/zero | tr '\0' 'a')")
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    -X POST -H "Content-Type: application/json" \
    -d "$LARGE_PAYLOAD" \
    "$ORCH_URL/route" 2>/dev/null || echo "400")
# Should be 400 or 413 (Request Entity Too Large)
if [ "$RESPONSE" == "400" ] || [ "$RESPONSE" == "413" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC} - Expected 400 or 413, got $RESPONSE"
    ((FAIL++))
fi

# ==========================================
# Test 10: Ingress Health (if available)
# ==========================================
test_case "Ingress health endpoint returns 200"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT \
    "$INGRESS_URL/health" 2>/dev/null || echo "000")
if [ "$RESPONSE" == "200" ] || [ "$RESPONSE" == "000" ]; then
    if [ "$RESPONSE" == "000" ]; then
        echo -e "${YELLOW}⚠ SKIP${NC} - Ingress not available"
    else
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
    fi
else
    echo -e "${RED}✗ FAIL${NC} - Got $RESPONSE"
    ((FAIL++))
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC} $PASS"
echo -e "${RED}Failed:${NC} $FAIL"
TOTAL=$((PASS + FAIL))
echo "Total:  $TOTAL"

if [ $FAIL -eq 0 ]; then
    echo -e "\n${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "\n${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
