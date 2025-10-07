# ShieldX Production Runbook

## System Overview

ShieldX is a production-grade cybersecurity platform providing real-time threat detection, behavioral analysis, and automated response capabilities.

### Core Components

- **Orchestrator**: Central routing and decision engine
- **Credits Service**: Resource management and billing
- **Continuous Auth**: Behavioral authentication
- **Shadow Evaluator**: Rule testing and validation
- **Digital Twin**: Attack simulation
- **Web Console**: Management interface

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Console   │    │   Monitoring    │
│   (HAProxy)     │    │   (React)       │    │   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Orchestrator │  │   Credits   │  │  ContAuth   │            │
│  │   (3 pods)  │  │  (2 pods)   │  │  (2 pods)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Shadow    │  │Digital Twin │  │   WebAPI    │            │
│  │  (2 pods)   │  │  (1 pod)    │  │  (2 pods)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │     MinIO       │
│   (HA Cluster)  │    │   (Cluster)     │    │   (Cluster)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] Kubernetes cluster (v1.25+) with 3+ nodes
- [ ] 16 CPU cores, 32GB RAM minimum per node
- [ ] 500GB SSD storage per node
- [ ] Network policies enabled
- [ ] RBAC configured
- [ ] Image registry access configured

### Security Requirements
- [ ] TLS certificates installed
- [ ] Image signing keys configured
- [ ] Seccomp profiles deployed
- [ ] Network policies applied
- [ ] RBAC policies configured
- [ ] Secrets management configured

### Database Setup
- [ ] PostgreSQL clusters deployed (Credits, ContAuth, Shadow)
- [ ] Database migrations completed
- [ ] Backup procedures configured
- [ ] Monitoring configured

## Deployment Procedures

### 1. Initial Deployment

```bash
# Create namespace
kubectl create namespace shieldx-system

# Deploy security policies
kubectl apply -f pilot/hardening/image-signing.yml
kubectl apply -f pilot/hardening/seccomp-profiles.yml

# Deploy databases
kubectl apply -f services/credits/docker-compose.yml
kubectl apply -f services/contauth/docker-compose.yml
kubectl apply -f services/shadow/docker-compose.yml

# Deploy core services
kubectl apply -f deployments/orchestrator.yml
kubectl apply -f deployments/credits.yml
kubectl apply -f deployments/contauth.yml
kubectl apply -f deployments/shadow.yml
kubectl apply -f deployments/webapi.yml

# Verify deployment
kubectl get pods -n shieldx-system
kubectl get services -n shieldx-system
```

### 2. Rolling Updates

```bash
# Update image tags
kubectl set image deployment/shieldx-orchestrator orchestrator=registry.shieldx.io/shieldx/orchestrator:v1.1.0 -n shieldx-system

# Monitor rollout
kubectl rollout status deployment/shieldx-orchestrator -n shieldx-system

# Rollback if needed
kubectl rollout undo deployment/shieldx-orchestrator -n shieldx-system
```

### 3. Configuration Updates

```bash
# Update ConfigMap
kubectl patch configmap shieldx-config -n shieldx-system --patch '{"data":{"new_setting":"value"}}'

# Restart affected pods
kubectl rollout restart deployment/shieldx-orchestrator -n shieldx-system
```

## Monitoring and Alerting

### Key Metrics

#### System Health
- Pod availability: >99.9%
- Response time: <200ms p95
- Error rate: <0.1%
- CPU utilization: <70%
- Memory utilization: <80%

#### Security Metrics
- Attack detection rate: >95%
- False positive rate: <5%
- Credit consumption rate: monitored
- Authentication challenges: tracked

#### Business Metrics
- Active tenants: tracked
- Plugin executions: tracked
- Shadow evaluations: tracked
- Revenue metrics: tracked

### Prometheus Queries

```promql
# Service availability
up{job="shieldx-orchestrator"} == 1

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Credit balance alerts
shieldx_credits_balance < 100

# Attack detection rate
rate(shieldx_attacks_detected[5m])
```

### Alert Rules

```yaml
groups:
- name: shieldx.rules
  rules:
  - alert: ServiceDown
    expr: up{job=~"shieldx-.*"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "ShieldX service {{ $labels.job }} is down"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"

  - alert: LowCreditBalance
    expr: shieldx_credits_balance < 100
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Low credit balance for tenant {{ $labels.tenant_id }}"
```

## Troubleshooting Guide

### Common Issues

#### 1. Service Unavailable (503)
**Symptoms**: HTTP 503 responses, pod restarts
**Diagnosis**:
```bash
kubectl describe pod <pod-name> -n shieldx-system
kubectl logs <pod-name> -n shieldx-system --previous
```
**Resolution**:
- Check resource limits
- Verify database connectivity
- Check network policies
- Review recent configuration changes

#### 2. Database Connection Errors
**Symptoms**: Connection timeouts, authentication failures
**Diagnosis**:
```bash
kubectl exec -it <app-pod> -n shieldx-system -- nc -zv <db-host> 5432
kubectl logs <db-pod> -n shieldx-system
```
**Resolution**:
- Verify database credentials
- Check network connectivity
- Review connection pool settings
- Check database resource usage

#### 3. High Memory Usage
**Symptoms**: OOMKilled pods, slow responses
**Diagnosis**:
```bash
kubectl top pods -n shieldx-system
kubectl describe node <node-name>
```
**Resolution**:
- Increase memory limits
- Check for memory leaks
- Review garbage collection settings
- Scale horizontally if needed

#### 4. Authentication Failures
**Symptoms**: 401/403 responses, login issues
**Diagnosis**:
```bash
kubectl logs deployment/shieldx-contauth -n shieldx-system
kubectl get secrets -n shieldx-system
```
**Resolution**:
- Verify JWT configuration
- Check certificate validity
- Review RBAC policies
- Validate user credentials

### Emergency Procedures

#### 1. Complete System Outage
1. Check cluster health: `kubectl get nodes`
2. Verify core services: `kubectl get pods -n shieldx-system`
3. Check ingress controller: `kubectl get pods -n ingress-nginx`
4. Review recent changes in Git history
5. Execute rollback if needed
6. Notify stakeholders via incident channel

#### 2. Database Corruption
1. Stop all application pods
2. Restore from latest backup
3. Verify data integrity
4. Restart application pods
5. Monitor for consistency issues

#### 3. Security Breach
1. Isolate affected components
2. Preserve logs and evidence
3. Rotate all credentials
4. Apply security patches
5. Conduct forensic analysis
6. Update security policies

## Backup and Recovery

### Database Backups
```bash
# Automated daily backups
kubectl create cronjob postgres-backup --image=postgres:15 --schedule="0 2 * * *" \
  -- pg_dump -h postgres-credits -U credits_user credits > /backup/credits-$(date +%Y%m%d).sql

# Manual backup
kubectl exec -it postgres-credits-0 -n shieldx-system -- pg_dump -U credits_user credits > backup.sql

# Restore from backup
kubectl exec -i postgres-credits-0 -n shieldx-system -- psql -U credits_user credits < backup.sql
```

### Configuration Backups
```bash
# Backup all configurations
kubectl get configmaps,secrets -n shieldx-system -o yaml > config-backup.yml

# Restore configurations
kubectl apply -f config-backup.yml
```

## Performance Tuning

### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

### Application Tuning
```yaml
# Resource limits optimization
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"

# JVM tuning (if applicable)
env:
- name: JAVA_OPTS
  value: "-Xmx512m -Xms256m -XX:+UseG1GC"
```

### Network Optimization
```yaml
# Service mesh configuration
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: shieldx-orchestrator
spec:
  host: shieldx-orchestrator
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 10
```

## Security Hardening

### Image Security
- All images signed with Cosign
- Regular vulnerability scanning with Trivy
- Minimal base images (distroless)
- No root users in containers

### Runtime Security
- Seccomp profiles applied
- AppArmor/SELinux enabled
- Network policies enforced
- Pod Security Standards

### Data Protection
- Encryption at rest and in transit
- Regular key rotation
- Secrets management with Vault
- Data classification and handling

## Capacity Planning

### Scaling Guidelines

#### Horizontal Scaling
```bash
# Scale deployment
kubectl scale deployment shieldx-orchestrator --replicas=5 -n shieldx-system

# Auto-scaling
kubectl autoscale deployment shieldx-orchestrator --cpu-percent=70 --min=3 --max=10 -n shieldx-system
```

#### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment shieldx-orchestrator -n shieldx-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"orchestrator","resources":{"limits":{"memory":"1Gi","cpu":"1000m"}}}]}}}}'
```

### Resource Planning
- Plan for 3x peak load capacity
- Monitor growth trends monthly
- Provision additional nodes proactively
- Consider multi-region deployment for DR

## Compliance and Auditing

### Audit Logging
```bash
# Enable audit logging
kubectl patch configmap audit-policy -n kube-system --patch '{"data":{"audit-policy.yaml":"..."}}'

# Review audit logs
kubectl logs kube-apiserver-master -n kube-system | grep audit
```

### Compliance Checks
- SOC 2 Type II compliance
- GDPR data protection requirements
- PCI DSS for payment processing
- Regular security assessments

## Contact Information

### On-Call Rotation
- **Primary**: DevOps Team (+1-555-0123)
- **Secondary**: Security Team (+1-555-0124)
- **Escalation**: Engineering Manager (+1-555-0125)

### Communication Channels
- **Slack**: #shieldx-alerts
- **Email**: ops@shieldx.io
- **PagerDuty**: ShieldX Production

### Vendor Contacts
- **Cloud Provider**: AWS Support
- **Database**: PostgreSQL Support
- **Monitoring**: Datadog Support

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Next Review**: 2024-04-15