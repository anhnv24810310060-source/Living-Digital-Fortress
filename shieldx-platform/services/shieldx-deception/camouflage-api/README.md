# Camouflage API

High-performance deception selection service. Provides optimal decoy selection using UCB1 multi-armed bandit and accepts feedback for continuous adaptation.

Endpoints:
- POST/GET /select -> returns a decoy template and a one-time token
- POST /feedback { node_id, reward } -> reward in [-1,1]
- GET /graph -> JSON metrics of current nodes and effectiveness
- GET /health, GET /metrics

Env:
- PORT (default 8089)
- CAMOUFLAGE_API_KEY (optional bearer token; health/metrics are public)

Run:
- go run .

Security notes:
- Input validated, metrics and auth middleware included.
- Decoy selection is stateful but in-memory; for HA, back it with Redis/DB.# Camouflage API Service

## Overview

Production-ready API service that provides adaptive camouflage templates for Edge Workers and other components in the ShieldX ecosystem.

## Features

- **Template Management**: Load and serve JSON-based camouflage templates
- **Session Management**: Create and track camouflage sessions
- **Reconnaissance Logging**: Log and analyze reconnaissance attempts
- **Authentication**: Bearer token authentication
- **Metrics**: Prometheus-compatible metrics
- **Health Checks**: Built-in health monitoring

## API Endpoints

### Get Template
```
GET /v1/camouflage/template/{name}
Authorization: Bearer <token>
X-Client-IP: <client_ip>
X-Recon-Type: <recon_type>
```

### List Templates
```
GET /v1/camouflage/templates
Authorization: Bearer <token>
```

### Create Session
```
POST /v1/camouflage/session
Authorization: Bearer <token>
Content-Type: application/json

{
  "template_type": "apache",
  "client_ip": "192.168.1.100",
  "user_agent": "nmap scanner",
  "recon_type": "nmap"
}
```

### Log Reconnaissance
```
POST /v1/camouflage/log
Authorization: Bearer <token>
Content-Type: application/json

{
  "timestamp": "2024-01-01T12:00:00Z",
  "client_ip": "192.168.1.100",
  "user_agent": "nmap scanner",
  "pathname": "/admin",
  "recon_type": "nmap",
  "cf_ray": "abc123",
  "country": "US"
}
```

## Environment Variables

- `CAMOUFLAGE_API_PORT`: Server port (default: 8091)
- `TEMPLATES_PATH`: Path to template directory (default: ./core/maze_engine/templates)
- `CAMOUFLAGE_API_KEY`: API authentication key

## Template Format

Templates are JSON files with the following structure:

```json
{
  "name": "apache",
  "version": "2.4.54",
  "fingerprint_id": "apache_2454_ubuntu",
  "headers": {
    "Server": "Apache/2.4.54 (Ubuntu)",
    "X-Powered-By": "PHP/8.1.2"
  },
  "error_pages": {
    "404": {
      "title": "404 Not Found",
      "body": "<html>...</html>",
      "content_type": "text/html"
    }
  },
  "behavioral_patterns": {
    "response_timing": {
      "min_ms": 50,
      "max_ms": 200,
      "distribution": "normal",
      "jitter_factor": 0.1
    }
  },
  "vulnerability_simulation": {
    "cve_2021_41773": {
      "enabled": false,
      "paths": ["/cgi-bin/.%2e/.%2e/etc/passwd"],
      "response": "403 Forbidden"
    }
  }
}
```

## Deployment

### Docker Compose
```bash
# Set environment variables
export CAMOUFLAGE_API_KEY="your_secure_api_key"
export GRAFANA_PASSWORD="your_grafana_password"

# Start services
docker-compose up -d

# Check health
curl http://localhost:8091/health
```

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=camouflage-api
```

## Monitoring

### Metrics
- `camouflage_api_template_requests_total`: Total template requests
- `camouflage_api_sessions_created_total`: Total sessions created
- `camouflage_api_log_requests_total`: Total log requests
- `camouflage_api_errors_total`: Total API errors

### Grafana Dashboard
Access Grafana at http://localhost:3000 (admin/admin) to view:
- Request rates and latency
- Template usage statistics
- Reconnaissance attempt patterns
- Error rates and health status

## Security

### Authentication
All endpoints (except /health and /metrics) require Bearer token authentication:
```
Authorization: Bearer <CAMOUFLAGE_API_KEY>
```

### Rate Limiting
- Implemented at Edge Worker level
- API service focuses on template serving performance

### Audit Logging
All requests are logged to:
- `data/ledger-camouflage-api.log`: API access logs
- `data/ledger-reconnaissance.log`: Reconnaissance attempt logs

## Testing

### Unit Tests
```bash
go test -v ./...
```

### Integration Tests
```bash
# Start service
docker-compose up -d camouflage-api

# Run tests
./test_api.sh
```

### Load Testing
```bash
# Install hey
go install github.com/rakyll/hey@latest

# Test template endpoint
hey -n 1000 -c 10 -H "Authorization: Bearer test_key" \
  http://localhost:8091/v1/camouflage/template/apache
```

## Week 2 Acceptance Criteria ✅

- [x] **Edge Worker can request template**: Cloudflare Worker fetches templates via API
- [x] **Orchestrator applies headers**: API serves templates with proper headers
- [x] **Session management**: Create and track camouflage sessions
- [x] **Reconnaissance detection**: Log and analyze reconnaissance attempts
- [x] **Production ready**: Docker, monitoring, health checks, authentication
- [x] **Template system**: JSON-based templates with behavioral patterns
- [x] **Vulnerability simulation**: Configurable vulnerability responses

## Performance Targets

- **Template Request Latency**: < 50ms p95
- **Throughput**: > 1000 requests/second
- **Availability**: > 99.9%
- **Fingerprint Mismatch Rate**: > 90% (attackers fooled)

## Integration with ShieldX

1. **Edge Workers**: Fetch templates for adaptive responses
2. **ML Orchestrator**: Analyze reconnaissance patterns
3. **Audit System**: Log all camouflage activities
4. **Metrics System**: Monitor engagement and effectiveness