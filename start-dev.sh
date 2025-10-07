#!/bin/bash
# Quick start script for development

# Load environment
export $(cat .env.dev | grep -v '^#' | xargs)

echo "Starting Orchestrator on port $ORCH_PORT..."
./bin/orchestrator &
ORCH_PID=$!

echo "Orchestrator PID: $ORCH_PID"
echo "Logs: tail -f data/ledger-orchestrator.log"
echo ""
echo "Test with:"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/metrics"
echo ""
echo "Stop with: kill $ORCH_PID"
