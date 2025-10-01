# ShieldX Runbook Specification

## Overview

This document defines the standard structure for operational runbooks at ShieldX. Runbooks provide step-by-step procedures for handling incidents, performing maintenance, and managing operations.

## Runbook Structure

### 1. Header
```markdown
# Runbook: [Title]
**ID**: RB-[YYYY]-[NNN]
**Version**: X.Y.Z
**Last Updated**: YYYY-MM-DD
**Owner**: [Team/Person]
**Severity**: [Critical|High|Medium|Low]
**Estimated Time**: [Duration]
```

### 2. Summary
Brief description of:
- What this runbook covers
- When to use it
- Expected outcome

### 3. Prerequisites
- Required access/permissions
- Tools needed
- Knowledge required
- System state requirements

### 4. Detection/Triggers
How to detect this situation:
- Alerts
- Metrics
- User reports
- Automated triggers

### 5. Impact Assessment
- Affected services
- User impact
- Business impact
- SLA considerations

### 6. Escalation Path
- When to escalate
- Who to contact
- Escalation timeline
- Communication channels

### 7. Procedure Steps
Detailed step-by-step instructions with:
- Command examples
- Expected outputs
- Decision points
- Rollback triggers

### 8. Verification
- How to verify resolution
- Health checks
- Metrics to monitor
- User validation

### 9. Post-Incident
- Cleanup tasks
- Documentation updates
- Post-mortem requirements
- Lessons learned

### 10. References
- Related runbooks
- Documentation links
- Playbook references
- Contact information

## Standard Runbooks

### RB-2025-001: Service Restart
**Path**: `pilot/docs/runbooks/service-restart.md`
**Automates**: `core/autoheal/playbooks/service-restart.yaml`

### RB-2025-002: Memory Leak Investigation
**Path**: `pilot/docs/runbooks/memory-leak-investigation.md`
**Automates**: `core/autoheal/playbooks/memory-leak-mitigation.yaml`

### RB-2025-003: Database Connection Pool Exhaustion
**Path**: `pilot/docs/runbooks/db-connection-pool.md`
**Related**: Database scaling, connection management

### RB-2025-004: Certificate Expiry Emergency
**Path**: `pilot/docs/runbooks/cert-expiry-emergency.md`
**Related**: RA-TLS rotation, PKI management

### RB-2025-005: Node Failure and Migration
**Path**: `pilot/docs/runbooks/node-failure-migration.md`
**Automates**: Kubernetes node drain and pod migration

### RB-2025-006: DDoS Mitigation
**Path**: `pilot/docs/runbooks/ddos-mitigation.md`
**Related**: Rate limiting, WAF rules, Cloudflare

### RB-2025-007: Security Incident Response
**Path**: `pilot/docs/runbooks/security-incident.md`
**Related**: Forensics, audit logs, threat analysis

### RB-2025-008: Backup and Restore
**Path**: `pilot/docs/runbooks/backup-restore.md`
**Related**: Data recovery, disaster recovery

### RB-2025-009: Deployment Rollback
**Path**: `pilot/docs/runbooks/deployment-rollback.md`
**Related**: CI/CD, canary deployments

### RB-2025-010: Performance Degradation
**Path**: `pilot/docs/runbooks/performance-degradation.md`
**Related**: Profiling, optimization, scaling

## Runbook Template

```markdown
# Runbook: [Title]

**ID**: RB-YYYY-NNN
**Version**: 1.0.0
**Last Updated**: 2025-10-01
**Owner**: [Your Team]
**Severity**: [Level]
**Estimated Time**: [Duration]

## Summary
[Brief description of what this runbook addresses]

## Prerequisites
- [ ] Access to [system/tool]
- [ ] [Permission level] permissions
- [ ] Knowledge of [domain]
- [ ] Tools: [list]

## Detection/Triggers

### Automated Alerts
- **Alert Name**: `[alert_name]`
- **Severity**: [level]
- **Source**: Prometheus/Grafana/PagerDuty

### Manual Detection
```bash
# Check [metric/status]
curl http://service:port/metrics | grep [pattern]
```

### Symptoms
- [Symptom 1]
- [Symptom 2]

## Impact Assessment

| Component | Impact | Users Affected | SLA Impact |
|-----------|--------|----------------|------------|
| [Service] | [High/Medium/Low] | [Number/Percentage] | [Yes/No] |

## Escalation Path

1. **Immediate** (0-5 min): On-call engineer
2. **Short-term** (5-15 min): Team lead
3. **Extended** (15+ min): Director of Engineering
4. **Critical**: CTO, CEO (if customer-facing)

**Contacts**:
- On-call: PagerDuty rotation
- Slack: #shieldx-incidents
- Email: incidents@shieldx.io

## Procedure

### Step 1: Initial Assessment
```bash
# Check service health
curl http://service:port/health

# Check metrics
curl http://service:port/metrics | grep [key_metrics]

# Check logs
kubectl logs -n shieldx service-pod --tail=100
```

**Expected Output**: [Description]
**If fails**: [Alternative action]

### Step 2: [Action]
```bash
# Command
[command]
```

**Expected Output**: [Description]
**If fails**: [Alternative action]
**Rollback**: [How to undo]

### Step 3: [Action]
[Instructions]

### Decision Point
- **If [condition]**: Proceed to Step X
- **If [condition]**: Escalate
- **If [condition]**: Rollback

## Verification

### Health Checks
```bash
# Primary health check
curl -f http://service:port/health

# Metrics verification
curl http://service:port/metrics | grep [success_metric]

# End-to-end test
./scripts/e2e-test.sh
```

### Success Criteria
- [ ] Service returns 200 on /health
- [ ] Error rate < 0.01%
- [ ] p95 latency < 500ms
- [ ] No crash loops for 5 minutes

### Monitoring
Monitor these metrics for 15 minutes:
- `http_request_errors_total`
- `http_request_duration_seconds`
- `service_restart_count`

## Rollback Procedure

If verification fails or situation worsens:

1. **Stop current actions immediately**
2. **Restore from backup** (if applicable)
   ```bash
   ./scripts/restore-backup.sh [timestamp]
   ```
3. **Revert configuration changes**
4. **Notify team of rollback**
5. **Document what went wrong**

## Post-Incident Tasks

### Immediate (0-1 hour)
- [ ] Update incident ticket
- [ ] Notify stakeholders of resolution
- [ ] Document timeline in incident channel

### Short-term (1-24 hours)
- [ ] Write incident summary
- [ ] Update runbook with learnings
- [ ] Create follow-up tickets for improvements

### Long-term (1-7 days)
- [ ] Conduct post-mortem meeting
- [ ] Update monitoring/alerting
- [ ] Implement preventive measures
- [ ] Update documentation

## Automation

**Playbook**: `core/autoheal/playbooks/[playbook-name].yaml`

**Auto-trigger conditions**:
- [Condition 1]
- [Condition 2]

**Manual trigger**:
```bash
./bin/playbook-executor run [playbook-name] \
  --param service=[service] \
  --param severity=[level]
```

## References

### Internal Documentation
- [Architecture diagram](link)
- [Service documentation](link)
- [Configuration guide](link)

### External Resources
- [Vendor documentation](link)
- [Community resources](link)

### Related Runbooks
- RB-YYYY-NNN: [Title]
- RB-YYYY-NNN: [Title]

### Playbooks
- [playbook-name].yaml
- [related-playbook].yaml

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-01 | ShieldX SRE | Initial version |

## Approval

**Reviewed by**: [Names]
**Approved by**: [Manager/Lead]
**Next Review**: [Date]
```

## Best Practices

### Writing Runbooks
1. **Be specific**: Use exact commands, not generic instructions
2. **Test regularly**: Verify runbooks work in realistic scenarios
3. **Include screenshots**: Visual aids help during incidents
4. **Assume stress**: Reader may be under pressure
5. **Version control**: Track changes, review updates
6. **Clear language**: Avoid jargon, explain acronyms

### Maintaining Runbooks
1. **Review quarterly**: Ensure accuracy and relevance
2. **Update after incidents**: Incorporate lessons learned
3. **Track usage**: Monitor which runbooks are used most
4. **Gather feedback**: Ask users for improvement suggestions
5. **Archive outdated**: Remove obsolete procedures
6. **Cross-reference**: Link related runbooks and playbooks

### Using Runbooks
1. **Follow exactly**: Don't skip steps
2. **Document actions**: Note what you do and when
3. **Communicate**: Keep team informed of progress
4. **Escalate early**: Don't hesitate if unsure
5. **Update during use**: Note discrepancies or improvements
6. **Provide feedback**: Help improve for next time

## Integration with Auto-heal

### Runbook → Playbook Mapping
Each manual runbook should have:
- Corresponding automated playbook (if possible)
- Clear indication of automation status
- Instructions for triggering automation

### Automation Coverage
- **Fully Automated**: Playbook handles 100%, runbook for reference
- **Partially Automated**: Playbook handles initial steps, manual follow-up
- **Manual Only**: Complex decisions require human judgment

### Escalation from Automation
Playbooks should escalate to runbooks when:
- Prechecks fail
- Actions timeout
- Rollback triggered
- Unknown error states

## Metrics and KPIs

Track runbook effectiveness:
- **MTTR** (Mean Time To Resolution): Target < 30 minutes
- **Success Rate**: Target > 95%
- **Automation Coverage**: Target > 60% of incidents
- **Runbook Usage**: Track most-used runbooks
- **Update Frequency**: Review at least quarterly

## Review Schedule

| Runbook ID | Last Review | Next Review | Owner |
|------------|-------------|-------------|-------|
| RB-2025-001 | 2025-10-01 | 2026-01-01 | SRE Team |
| RB-2025-002 | 2025-10-01 | 2026-01-01 | Platform Team |

## Contact Information

**Runbook Coordinator**: runbooks@shieldx.io
**On-Call Rotation**: PagerDuty
**Incident Channel**: #shieldx-incidents (Slack)
**Documentation**: https://docs.shieldx.io/runbooks
