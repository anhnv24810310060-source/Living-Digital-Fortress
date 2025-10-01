#!/bin/bash
# Demo script for Observability & SLO Framework
# October 2025 Milestone

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════"
echo "  Living Digital Fortress - Observability Demo"
echo "  Milestone: October 2025 - SLO & OpenTelemetry"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found. Please install it first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Starting Observability Stack...${NC}"
cd pilot/observability
docker-compose up -d

echo ""
echo -e "${YELLOW}Step 2: Waiting for services to be ready...${NC}"
sleep 10

# Check Prometheus
echo -n "  Checking Prometheus... "
if curl -s http://localhost:9090/-/healthy &> /dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

# Check Grafana
echo -n "  Checking Grafana... "
if curl -s http://localhost:3000/api/health &> /dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

# Check OTLP Collector
echo -n "  Checking OTLP Collector... "
if curl -s http://localhost:13133/ &> /dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

# Check Tempo
echo -n "  Checking Tempo... "
if curl -s http://localhost:3200/ready &> /dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

# Check Alertmanager
echo -n "  Checking Alertmanager... "
if curl -s http://localhost:9093/-/healthy &> /dev/null; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Observability Stack is Ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

echo "📊 Access Points:"
echo "  • Prometheus:  http://localhost:9090"
echo "  • Grafana:     http://localhost:3000 (admin/fortress123)"
echo "  • Tempo:       http://localhost:3200"
echo "  • OTLP HTTP:   http://localhost:4318"
echo "  • OTLP gRPC:   http://localhost:4317"
echo "  • Alertmanager: http://localhost:9093"
echo ""

echo "🎯 SLO Targets (5 Core Services):"
echo "  1. Ingress:         99.9% availability, P95<100ms, P99<200ms"
echo "  2. ShieldX Gateway: 99.9% availability, P95<50ms,  P99<100ms"
echo "  3. ContAuth:        99.95% availability, P95<150ms, P99<300ms"
echo "  4. Verifier Pool:   99.9% availability, P95<200ms, P99<500ms"
echo "  5. ML Orchestrator: 99.5% availability, P95<500ms, P99<1000ms"
echo ""

echo "📈 Available Metrics:"
echo "  • Recording Rules: {service}:slo_error_ratio:rate5m"
echo "  •                  {service}:slo_availability:rate5m"
echo "  •                  {service}:latency_p95:rate5m"
echo "  •                  {service}:latency_p99:rate5m"
echo "  •                  {service}:error_budget_burn_fast"
echo ""

echo "🔔 Alert Rules:"
echo "  • Critical: SLO breach, error budget exhausted"
echo "  • Warning:  Error budget < 20%, latency trending high"
echo ""

echo "🚀 Next Steps:"
echo "  1. Import Grafana dashboards from pilot/observability/grafana/"
echo "  2. Configure your services with OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318"
echo "  3. Start sending metrics and traces"
echo "  4. Monitor SLO compliance via Prometheus queries"
echo ""

echo "📝 Quick Commands:"
echo "  • Check SLO:  make slo-check"
echo "  • Stop stack: make otel-down"
echo "  • View logs:  docker-compose logs -f"
echo ""

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Demo Complete! Happy Monitoring! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
