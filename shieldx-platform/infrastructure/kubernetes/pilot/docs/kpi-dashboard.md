# ShieldX KPI Dashboard & Metrics

> **Lưu ý:** Tài liệu này giữ lại như tư liệu lịch sử. Bảng điều khiển SLO hiện tại được mô tả trong `pilot/docs/slo-dashboard.md`.

## Executive Summary Dashboard

### Business KPIs

#### Revenue Metrics

#### Customer Metrics

### Technical KPIs

#### System Performance

#### Security Effectiveness

## Operational Metrics

### Infrastructure Health

#### Kubernetes Cluster
```
Cluster Status: ✅ Healthy
Nodes: 12/12 Ready
CPU Utilization: 68%
Memory Utilization: 72%
Storage Utilization: 45%
Network Throughput: 2.3 Gbps
```

#### Service Health Matrix
| Service | Status | Uptime | Response Time | Error Rate | Replicas |
|---------|--------|--------|---------------|------------|----------|
| Orchestrator | ✅ | 99.99% | 45ms | 0.01% | 3/3 |
| Credits | ✅ | 99.98% | 23ms | 0.02% | 2/2 |
| ContAuth | ✅ | 99.97% | 156ms | 0.05% | 2/2 |
| Shadow | ✅ | 99.95% | 234ms | 0.08% | 2/2 |
| Digital Twin | ✅ | 99.94% | 1.2s | 0.12% | 1/1 |
| WebAPI | ✅ | 99.99% | 67ms | 0.01% | 2/2 |

### Database Performance

#### PostgreSQL Clusters
```
Credits DB:
  - Connections: 45/100
  - Query Time (P95): 12ms
  - Cache Hit Ratio: 98.7%
  - Replication Lag: 0.2s

ContAuth DB:
  - Connections: 23/100
  - Query Time (P95): 8ms
  - Cache Hit Ratio: 99.1%
  - Replication Lag: 0.1s

Shadow DB:
  - Connections: 12/100
  - Query Time (P95): 45ms
  - Cache Hit Ratio: 96.8%
  - Replication Lag: 0.3s
```

### Security Metrics

#### Threat Intelligence

#### Attack Statistics (Last 24h)
```
Total Attacks Detected: 15,847
├── SQL Injection: 4,234 (26.7%)
├── XSS Attempts: 3,567 (22.5%)
├── Brute Force: 2,891 (18.2%)
├── DDoS: 1,456 (9.2%)
├── Malware: 1,234 (7.8%)
├── Phishing: 987 (6.2%)
└── Other: 1,478 (9.3%)

Blocked by Component:
├── WAF Rules: 8,234 (52.0%)
├── Behavioral Analysis: 3,456 (21.8%)
├── IP Reputation: 2,345 (14.8%)
├── Rate Limiting: 1,234 (7.8%)
└── Manual Rules: 578 (3.6%)
```

## Feature Adoption Metrics

### Credits System
  - Plugin Executions: 18.9M credits (44.9%)
  - Digital Twin Simulations: 12.3M credits (29.2%)
  - ML Model Training: 8.7M credits (20.7%)
  - Shadow Evaluations: 2.2M credits (5.2%)

### Plugin Marketplace
  1. Malware Detector: 456K executions
  2. Behavioral Analyzer: 234K executions
  3. Network Scanner: 189K executions
  4. Threat Hunter: 167K executions
  5. Vulnerability Scanner: 145K executions

### Shadow Evaluation Usage

### Continuous Authentication

## Performance Benchmarks

### Latency Targets vs Actual

| Component | Target (P95) | Actual (P95) | Status |
|-----------|--------------|--------------|---------|
| API Gateway | <100ms | 67ms | ✅ |
| Orchestrator | <200ms | 187ms | ✅ |
| Credits Service | <50ms | 23ms | ✅ |
| ContAuth | <300ms | 156ms | ✅ |
| Shadow Eval | <5s | 1.2s | ✅ |
| Plugin Execution | <10s | 3.4s | ✅ |

### Throughput Metrics

```
Peak Traffic Handling:
├── Requests/Second: 50,000
├── Concurrent Users: 125,000
├── Data Processed: 2.3 TB/day
├── Events Analyzed: 15.6M/hour
└── ML Predictions: 890K/hour

Resource Utilization:
├── CPU: 68% average, 89% peak
├── Memory: 72% average, 91% peak
├── Network: 2.3 Gbps average, 8.1 Gbps peak
├── Storage IOPS: 15K average, 45K peak
└── Database Connections: 180/500 used
```

## Financial Metrics

### Cost Analysis

#### Infrastructure Costs (Monthly)
```
Cloud Infrastructure: $145,000
├── Compute (EC2): $89,000 (61.4%)
├── Storage (EBS/S3): $23,000 (15.9%)
├── Network (Data Transfer): $18,000 (12.4%)
├── Database (RDS): $12,000 (8.3%)
└── Other Services: $3,000 (2.1%)

Software Licenses: $67,000
├── Kubernetes Platform: $25,000
├── Monitoring Tools: $18,000
├── Security Tools: $15,000
└── Development Tools: $9,000

Total Monthly OpEx: $212,000
```

#### Revenue per Customer Segment
```
Enterprise (156 customers):
├── Average Contract Value: $180,000/year
├── Total ARR: $28.08M (97.5%)
└── Gross Margin: 87%

SMB (1,091 customers):
├── Average Contract Value: $6,600/year
├── Total ARR: $7.2M (25.0%)
└── Gross Margin: 78%

Total Blended Gross Margin: 84.2%
```

### Unit Economics

## Quality Metrics

### Code Quality

### Deployment Metrics

### Customer Support

## Compliance & Security KPIs

### Compliance Status

### Security Posture

### Data Protection

## Growth Metrics

### User Engagement

### Market Expansion

## Alerting Thresholds

### Critical Alerts (PagerDuty)

### Warning Alerts (Slack)

### Business Alerts (Email)

## Reporting Schedule

### Daily Reports (Automated)

### Weekly Reports (Automated)

### Monthly Reports (Manual)

### Quarterly Reports (Manual)


**Dashboard Version**: 1.0  
**Last Updated**: 2024-01-15  
**Data Refresh**: Real-time (5-minute intervals)  
**Next Review**: 2024-02-15


Operational note (2025-09-30):
  - Ingress: http://<ingress-host>:<port>/metrics
  - Locator: http://<locator-host>:<port>/metrics
  - Guardian (loopback): http://127.0.0.1:<GUARDIAN_PORT>/metrics
  - ML Orchestrator: http://<ml-orchestrator-host>:<port>/metrics
  - ML Service (feature_store.py): http://<ml-service-host>:5000/metrics
  - ContAuth: http://<contauth-host>:<port>/metrics
  - Verifier Pool: http://<verifier-pool-host>:<port>/metrics
 - Per-endpoint metrics (for debugging): *_http_requests_by_path_total, *_http_request_duration_by_path_seconds; dùng có kiểm soát để tránh cardinality bùng nổ.
 - Tracing: optional OpenTelemetry OTLP/HTTP collector at http://localhost:4318; set OTEL_EXPORTER_OTLP_ENDPOINT to the collector address in services to enable.