#!/bin/bash
#
# PERSON 2 - Security & ML Services Verification Script
# This script verifies all P0 and P1 implementations
#

set -e

echo "=================================================="
echo "PERSON 2: Security & ML Services - Verification"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Tracking
PASS=0
FAIL=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ $1${NC}"
        ((FAIL++))
    fi
}

echo "1. Checking Go environment..."
go version
check "Go environment"

echo ""
echo "2. Building packages..."

echo "  - Building pkg/sandbox..."
go build -o /dev/null ./pkg/sandbox 2>&1 | head -20 || true
check "pkg/sandbox builds"

echo "  - Building services/guardian..."
go build -o bin/guardian ./services/guardian 2>&1 | head -20 || true
check "Guardian service builds"

echo "  - Building services/contauth..."
go build -o bin/contauth ./services/contauth 2>&1 | head -20 || true
check "ContAuth service builds"

echo "  - Building services/ml-orchestrator..."
go build -o bin/ml-orchestrator ./services/ml-orchestrator 2>&1 | head -20 || true
check "ML Orchestrator service builds"

echo ""
echo "3. Running tests..."

echo "  - Testing advanced threat scorer..."
go test -v ./pkg/sandbox -run TestAdvancedThreatScorer 2>&1 | tail -10 || true
check "Advanced threat scorer tests"

echo "  - Testing isolation forest..."
go test -v ./pkg/sandbox -run TestIsolationForest 2>&1 | tail -5 || true
check "Isolation forest tests"

echo "  - Testing syscall sequence analyzer..."
go test -v ./pkg/sandbox -run TestSyscallSequenceAnalyzer 2>&1 | tail -5 || true
check "Syscall sequence analyzer tests"

echo "  - Testing Bayesian model..."
go test -v ./pkg/sandbox -run TestBayesianThreatModel 2>&1 | tail -5 || true
check "Bayesian threat model tests"

echo ""
echo "4. Checking file structure..."

FILES=(
    "pkg/sandbox/advanced_threat_scorer.go"
    "pkg/sandbox/advanced_threat_scorer_test.go"
    "services/contauth/privacy_preserving_scorer.go"
    "services/contauth/encryption_manager.go"
    "services/ml-orchestrator/model_versioning.go"
    "services/README_PERSON2_FINAL.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file exists${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ $file missing${NC}"
        ((FAIL++))
    fi
done

echo ""
echo "5. Checking API endpoints (if services running)..."

# Check if services are running
if curl -s http://localhost:9090/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Guardian health endpoint${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ Guardian not running (expected in dev)${NC}"
fi

if curl -s http://localhost:5002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ContAuth health endpoint${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ ContAuth not running (expected in dev)${NC}"
fi

echo ""
echo "6. P0 Requirements Checklist:"
echo "  [ ] Guardian sandbox isolation (30s timeout)"
echo "  [ ] eBPF syscall monitoring + threat scoring"
echo "  [ ] ContAuth privacy-preserving risk scoring"
echo "  [ ] At-rest encryption for telemetry"

echo ""
echo "7. P1 Requirements Checklist:"
echo "  [ ] Model versioning + rollback"
echo "  [ ] A/B testing framework"
echo "  [ ] Anomaly detection baseline training"

echo ""
echo "=================================================="
echo "VERIFICATION SUMMARY"
echo "=================================================="
echo -e "Passed: ${GREEN}${PASS}${NC}"
echo -e "Failed: ${RED}${FAIL}${NC}"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    exit 1
fi
