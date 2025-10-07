# ShieldX Error Budget Policy

## Overview

Error budgets define the acceptable level of unreliability for each service, balancing innovation velocity with reliability. This document establishes our error budget methodology, policies, and procedures.

## Philosophy

> "100% reliability is the wrong target. We need room to innovate, deploy, and learn. Error budgets give us that room while maintaining user trust."

## Error Budget Fundamentals

### What is an Error Budget?

An **error budget** is the maximum amount of unreliability we can tolerate in a service over a given period without violating our SLO (Service Level Objective).

**Formula**:
```
Error Budget = 100% - SLO Target
```

**Example**:
- SLO: 99.9% availability
- Error Budget: 0.1% (43.2 minutes per month)

### Why Error Budgets Matter

1. **Innovation vs Stability**: Trade-off between new features and reliability
2. **Objective Decision Making**: Data-driven release decisions
3. **Shared Responsibility**: Dev and Ops aligned on reliability
4. **Risk Management**: Quantify and manage reliability risk

## Service Level Objectives (SLOs)

### Core Services SLOs

| Service | Availability | Latency (p95) | Latency (p99) | Error Budget/Month |
|---------|--------------|---------------|---------------|-------------------|
| **ingress** | 99.9% | 200ms | 500ms | 43.2 min |
| **shieldx-gateway** | 99.9% | 150ms | 400ms | 43.2 min |
| **contauth** | 99.5% | 500ms | 1000ms | 3.6 hours |
| **verifier-pool** | 99.5% | 300ms | 800ms | 3.6 hours |
| **ml-orchestrator** | 99.0% | 1000ms | 2000ms | 7.2 hours |
| **locator** | 99.9% | 100ms | 300ms | 43.2 min |
| **policy-rollout** | 99.5% | 200ms | 500ms | 3.6 hours |

### SLO Components

#### Availability SLO
Percentage of time service is available and responding correctly.

**Definition**:
```
Availability = (Total Requests - Failed Requests) / Total Requests × 100%
```

**Failed Request**: Returns 5xx, times out, or fails health check

#### Latency SLO
Percentage of requests completed within target latency.

**Definition**:
```
Latency SLO = (Requests < Target Latency) / Total Requests × 100%
```

#### Error Budget Calculation
```
Monthly Error Budget (minutes) = 
  (100% - SLO%) × Minutes in Month
  
Example (99.9% SLO):
  (100% - 99.9%) × 43,200 min = 43.2 minutes
```

## Error Budget Windows

### Time Windows

1. **Real-time**: Last 1 hour (tactical decisions)
2. **Short-term**: Last 24 hours (operational decisions)
3. **Standard**: Last 30 days (strategic decisions)
4. **Quarterly**: Last 90 days (planning decisions)

### Budget Reset

- **Monthly**: Primary budget window (resets 1st of month)
- **Rolling 30-day**: Continuous tracking
- **Quarterly review**: Strategic adjustments

## Error Budget Policies

### Policy 1: Deployment Freeze

**Trigger**: Error budget < 10% remaining

**Actions**:
1. **Immediate**: Halt non-critical deployments
2. **Communication**: Notify all engineering teams
3. **Focus**: Prioritize reliability improvements
4. **Exception**: Critical security patches only

**Duration**: Until budget recovers to > 25%

### Policy 2: Deployment Slowdown

**Trigger**: Error budget 10-25% remaining

**Actions**:
1. **Reduce velocity**: Deploy only during business hours
2. **Enhanced review**: Require SRE approval for deploys
3. **Increased monitoring**: 2x monitoring during deployments
4. **Canary required**: All deployments must canary first

**Duration**: Until budget recovers to > 40%

### Policy 3: Normal Operations

**Trigger**: Error budget > 25% remaining

**Actions**:
1. **Standard velocity**: Normal deployment cadence
2. **Balanced innovation**: New features and experiments allowed
3. **Regular monitoring**: Standard observability practices

### Policy 4: Over-Budget (Negative)

**Trigger**: Error budget exhausted (< 0%)

**Actions**:
1. **IMMEDIATE FREEZE**: All deployments stopped
2. **Incident declared**: P1 incident automatically created
3. **All-hands**: Engineering leadership involved
4. **Root cause analysis**: Mandatory post-mortem
5. **Remediation plan**: Required before any new work

**Duration**: Until RCA complete and budget positive

## Error Budget Burn Rate

### Burn Rate Definition

Speed at which error budget is consumed.

**Formula**:
```
Burn Rate = (Current Error Rate) / (SLO Error Rate)
```

**Example**:
- SLO: 99.9% (0.1% error rate)
- Current: 1% error rate
- Burn Rate: 10x (burning budget 10x faster than sustainable)

### Alert Thresholds

| Burn Rate | Time Window | Severity | Action |
|-----------|-------------|----------|--------|
| 14.4x | 1 hour | P1 | Page immediately |
| 6x | 6 hours | P2 | Alert on-call |
| 3x | 24 hours | P3 | Create ticket |
| 1x | 30 days | Info | Monitor |

### Multi-Window Alerts

Combine short and long windows to reduce false positives:

```yaml
# Example: ingress availability alert
alert: IngressHighBurnRate
expr: |
  (
    # Short window (1h): 14.4x burn
    sum(rate(http_requests_total{service="ingress",code=~"5.."}[1h]))
    / sum(rate(http_requests_total{service="ingress"}[1h]))
    > 14.4 * 0.001
  )
  and
  (
    # Long window (1d): 1.8x burn  
    sum(rate(http_requests_total{service="ingress",code=~"5.."}[1d]))
    / sum(rate(http_requests_total{service="ingress"}[1d]))
    > 1.8 * 0.001
  )
severity: critical
```

## Budget Allocation

### Planned Consumption

Reserve error budget for known events:

| Activity | Budget Allocation | Frequency |
|----------|------------------|-----------|
| Deployments | 20% | Continuous |
| Maintenance windows | 15% | Monthly |
| Chaos engineering | 10% | Weekly |
| Load testing | 5% | Quarterly |
| **Reserved** | **50%** | - |
| **Available for innovation** | **50%** | - |

### Unplanned Consumption

Incidents consume from available budget:
- Track actual vs planned consumption
- Adjust plans if incidents exceed 30% of budget
- Post-mortem for any single incident > 10% of monthly budget

## Monitoring and Reporting

### Real-Time Dashboard

**Grafana Dashboard**: `ShieldX Error Budget Status`

**Panels**:
1. **Current Budget Remaining** (gauge, per service)
2. **Burn Rate** (graph, 1h/6h/24h/30d)
3. **Budget Consumption Timeline** (area chart)
4. **Top Budget Consumers** (table)
5. **Projected Exhaustion** (time-series forecast)

### Weekly Report

**Recipients**: Engineering leadership, product managers

**Content**:
- Error budget status per service
- Burn rate trends
- Policy status (freeze/slowdown/normal)
- Top incidents by budget impact
- Recommendations

### Monthly Review

**Attendees**: Engineering, Product, SRE

**Agenda**:
1. Review SLO targets (still appropriate?)
2. Budget consumption patterns
3. Incident analysis
4. Policy effectiveness
5. Next month's risk areas
6. SLO adjustments (if needed)

## Incident Classification by Budget Impact

### P0 - Critical (> 25% monthly budget)
- Immediate escalation to leadership
- Mandatory RCA within 24 hours
- Post-mortem with entire engineering
- Prevention plan required

### P1 - High (10-25% monthly budget)
- RCA within 48 hours
- Team post-mortem
- Action items tracked to completion

### P2 - Medium (5-10% monthly budget)
- RCA within 1 week
- Document lessons learned
- Optional improvements

### P3 - Low (< 5% monthly budget)
- Log incident
- Track for patterns
- No formal RCA required

## Budget Recovery Actions

### Immediate (< 24 hours)
1. **Identify root cause**: What's consuming budget?
2. **Mitigate**: Stop the bleeding
3. **Rollback**: If recent deploy, consider rollback
4. **Scale**: Add capacity if needed

### Short-term (1-7 days)
1. **Fix bugs**: Address reliability issues
2. **Improve monitoring**: Better visibility
3. **Update runbooks**: Document solutions
4. **Test**: Verify fixes work

### Long-term (1-4 weeks)
1. **Architectural improvements**: Design for reliability
2. **Automation**: Reduce human error
3. **Capacity planning**: Right-size resources
4. **Training**: Improve team capabilities

## Exception Process

### Requesting Exception

When error budget policy blocks critical work:

1. **Document need**: Why exception is necessary
2. **Risk assessment**: What could go wrong?
3. **Mitigation plan**: How to minimize risk
4. **Approvers**: Director of Engineering + SRE Lead
5. **Monitoring plan**: Extra vigilance during exception

### Exception Criteria

Exceptions granted only for:
- **Security vulnerabilities**: Critical CVEs
- **Data loss prevention**: Backup failures
- **Compliance requirements**: Legal/regulatory
- **Customer-committed features**: Contractual obligations

### Exception Review

- All exceptions logged
- Quarterly review of exceptions
- Pattern analysis (too many exceptions = wrong SLOs)

## SLO Adjustment Process

### When to Adjust

SLOs should be adjusted if:
1. **Consistently exceeding**: > 99% of time above target
2. **Consistently missing**: < 80% of time meeting target
3. **Business changes**: New requirements or expectations
4. **Architecture changes**: Fundamentally different reliability characteristics

### Adjustment Procedure

1. **Proposal**: Engineering proposes new SLO
2. **Analysis**: Review 90 days of historical data
3. **Stakeholder review**: Product, Engineering, Support
4. **Trial period**: 30-day trial with new SLO
5. **Approval**: Director of Engineering approves
6. **Communication**: Announce to all teams
7. **Update**: Dashboards, alerts, documentation

### Adjustment Frequency

- **Minimum**: Quarterly
- **Maximum**: Monthly (avoid thrash)
- **Emergency**: Anytime for critical issues

## Budget Gaming Prevention

### Anti-Patterns to Avoid

1. **Artificially inflating traffic**: To make error % look better
2. **Hiding errors**: Not counting certain failure modes
3. **Cherry-picking windows**: Only reporting good periods
4. **Relaxing SLOs**: Making targets easier instead of improving
5. **Ignoring policy**: Deploying during freeze with workarounds

### Enforcement

- **Automated tracking**: Can't manipulate
- **Regular audits**: SRE team reviews calculations
- **Transparent reporting**: Public dashboards
- **Accountability**: Managers responsible for compliance

## Tools and Automation

### Prometheus Queries

```promql
# Error budget remaining (30d)
1 - (
  sum(rate(http_requests_total{service="ingress",code=~"5.."}[30d]))
  / sum(rate(http_requests_total{service="ingress"}[30d]))
  / 0.001  # 99.9% SLO = 0.1% error rate
)

# Burn rate (1h)
(
  sum(rate(http_requests_total{service="ingress",code=~"5.."}[1h]))
  / sum(rate(http_requests_total{service="ingress"}[1h]))
) / 0.001
```

### API Endpoints

```bash
# Get current budget status
curl http://slo-tracker:8080/api/v1/budget/ingress

# Get burn rate
curl http://slo-tracker:8080/api/v1/burn-rate/ingress?window=1h

# Get deployment decision
curl http://slo-tracker:8080/api/v1/deploy-allowed/ingress
```

### CI/CD Integration

```yaml
# .github/workflows/deploy.yml
- name: Check Error Budget
  run: |
    BUDGET=$(curl -s http://slo-tracker/api/v1/budget/ingress | jq -r '.remaining_percent')
    if [ $(echo "$BUDGET < 10" | bc) -eq 1 ]; then
      echo "Error budget exhausted. Deployment blocked."
      exit 1
    fi
```

## References

- [Google SRE Book - Error Budgets](https://sre.google/sre-book/embracing-risk/)
- [Grafana Dashboard](http://grafana:3000/d/error-budgets)
- [Prometheus Alerts](pilot/observability/alert-rules.yml)
- [SLO Tracker API](http://slo-tracker:8080/docs)

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-01 | ShieldX SRE | Initial policy |

## Approval

**Reviewed by**: Engineering Leadership, SRE Team, Product Management
**Approved by**: CTO
**Effective Date**: 2025-10-01
**Next Review**: 2026-01-01
