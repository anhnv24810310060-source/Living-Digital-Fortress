# Playbook Schema Specification v1.0

## Overview
Standardized schema for auto-heal playbooks ensuring consistency, testability, and auditability.

## Schema Structure

```yaml
apiVersion: autoheal.shieldx.io/v1
kind: Playbook
metadata:
  name: <playbook-name>
  version: <semver>
  author: <author-info>
  created: <ISO8601-timestamp>
  tags: [<tag1>, <tag2>, ...]
  
spec:
  # Trigger conditions
  trigger:
    type: <incident-type>          # e.g., node_down, service_unresponsive, memory_leak
    severity: <critical|high|medium|low>
    conditions:
      - metric: <metric-name>
        operator: <gt|lt|eq|ne>    # greater than, less than, equals, not equals
        threshold: <value>
        duration: <time-duration>  # e.g., 5m, 30s
      
  # Pre-checks before execution
  precheck:
    - name: <check-name>
      type: <health_check|metric_check|service_check>
      command: <command-to-run>
      expected: <expected-result>
      timeout: <timeout-duration>
      critical: <true|false>        # If true, stop execution on failure
      
  # Main remediation actions
  actions:
    - name: <action-name>
      type: <restart|scale|migrate|script|api_call>
      target: <target-service-or-node>
      params:
        <key>: <value>
      timeout: <timeout-duration>
      retries: <number>
      retry_delay: <duration>
      on_failure: <continue|rollback|stop>
      
  # Rollback steps (executed in reverse order on failure)
  rollback:
    enabled: <true|false>
    actions:
      - name: <rollback-action-name>
        type: <same-as-actions>
        target: <target>
        params:
          <key>: <value>
        timeout: <timeout-duration>
        
  # Post-execution verification
  postcheck:
    - name: <verification-name>
      type: <health_check|metric_check|service_check>
      command: <command-to-run>
      expected: <expected-result>
      timeout: <timeout-duration>
      critical: <true|false>
      
  # Audit and notifications
  audit:
    enabled: <true|false>
    hashchain: <true|false>         # Record in audit hashchain
    anchor: <true|false>             # Create anchor checkpoint
    
  notifications:
    on_start: [<slack|pagerduty|email>, ...]
    on_success: [<channels>, ...]
    on_failure: [<channels>, ...]
```

## Field Descriptions

### metadata
- **name**: Unique identifier for the playbook
- **version**: Semantic version (MAJOR.MINOR.PATCH)
- **author**: Author or team responsible
- **created**: Creation timestamp (ISO8601)
- **tags**: Searchable tags for categorization

### spec.trigger
Defines when this playbook should be triggered
- **type**: Category of incident
- **severity**: Impact level
- **conditions**: Array of conditions that must be met

### spec.precheck
Safety checks before taking action
- **critical**: If true and check fails, abort execution
- All checks must pass before proceeding to actions

### spec.actions
Remediation steps executed sequentially
- **on_failure**: Behavior when action fails
  - `continue`: Skip and proceed to next action
  - `rollback`: Execute rollback steps
  - `stop`: Halt execution immediately

### spec.rollback
Undo steps for failed actions
- Executed in reverse order
- Should restore system to pre-action state

### spec.postcheck
Verification that remediation succeeded
- Similar to precheck structure
- Determines overall playbook success

### spec.audit
Audit trail configuration
- **hashchain**: Add execution to audit chain
- **anchor**: Create anchor point for compliance

## Examples

### Example 1: Service Restart Playbook

```yaml
apiVersion: autoheal.shieldx.io/v1
kind: Playbook
metadata:
  name: service-restart-simple
  version: 1.0.0
  author: ShieldX SRE Team
  created: 2025-10-01T12:00:00Z
  tags: [restart, service, basic]
  
spec:
  trigger:
    type: service_unresponsive
    severity: high
    conditions:
      - metric: http_request_errors_rate
        operator: gt
        threshold: 0.05
        duration: 5m
        
  precheck:
    - name: verify_service_exists
      type: service_check
      command: "systemctl status {{ .service }}"
      expected: "active|inactive"
      timeout: 10s
      critical: true
      
    - name: check_disk_space
      type: metric_check
      command: "df -h / | tail -1 | awk '{print $5}' | sed 's/%//'"
      expected: "<90"
      timeout: 5s
      critical: true
      
  actions:
    - name: restart_service
      type: restart
      target: "{{ .service }}"
      params:
        graceful: true
        wait_for_ready: true
      timeout: 60s
      retries: 2
      retry_delay: 10s
      on_failure: rollback
      
  rollback:
    enabled: true
    actions:
      - name: restore_previous_version
        type: script
        target: "{{ .service }}"
        params:
          script: "/opt/shieldx/scripts/rollback.sh"
          args: ["{{ .service }}", "{{ .previous_version }}"]
        timeout: 120s
        
  postcheck:
    - name: verify_service_running
      type: health_check
      command: "curl -f http://{{ .service_host }}:{{ .service_port }}/health"
      expected: "200"
      timeout: 30s
      critical: true
      
    - name: verify_metrics_normal
      type: metric_check
      command: "curl -s http://{{ .service_host }}:{{ .service_port }}/metrics | grep error_rate"
      expected: "<0.01"
      timeout: 10s
      critical: false
      
  audit:
    enabled: true
    hashchain: true
    anchor: true
    
  notifications:
    on_start: [slack]
    on_success: [slack]
    on_failure: [slack, pagerduty]
```

### Example 2: Node Recovery Playbook

```yaml
apiVersion: autoheal.shieldx.io/v1
kind: Playbook
metadata:
  name: node-recovery-drain-migrate
  version: 1.1.0
  author: ShieldX Platform Team
  created: 2025-10-01T12:00:00Z
  tags: [node, kubernetes, drain, migrate]
  
spec:
  trigger:
    type: node_down
    severity: critical
    conditions:
      - metric: node_ready_status
        operator: eq
        threshold: "NotReady"
        duration: 2m
        
  precheck:
    - name: verify_node_exists
      type: service_check
      command: "kubectl get node {{ .node_name }}"
      expected: "success"
      timeout: 10s
      critical: true
      
    - name: check_cluster_capacity
      type: metric_check
      command: "kubectl top nodes | grep -v {{ .node_name }} | awk '{sum+=$3} END {print sum}'"
      expected: "<70"
      timeout: 15s
      critical: true
      
  actions:
    - name: cordon_node
      type: api_call
      target: "k8s_api"
      params:
        endpoint: "/api/v1/nodes/{{ .node_name }}"
        method: PATCH
        body: '{"spec":{"unschedulable":true}}'
      timeout: 30s
      retries: 3
      retry_delay: 5s
      on_failure: stop
      
    - name: drain_node
      type: script
      target: "{{ .node_name }}"
      params:
        script: "kubectl drain {{ .node_name }} --ignore-daemonsets --delete-emptydir-data --grace-period=60"
      timeout: 300s
      retries: 1
      retry_delay: 30s
      on_failure: rollback
      
    - name: wait_pods_migrated
      type: health_check
      command: "kubectl get pods --field-selector spec.nodeName={{ .node_name }} -A --no-headers | wc -l"
      expected: "0"
      timeout: 180s
      critical: true
      
  rollback:
    enabled: true
    actions:
      - name: uncordon_node
        type: api_call
        target: "k8s_api"
        params:
          endpoint: "/api/v1/nodes/{{ .node_name }}"
          method: PATCH
          body: '{"spec":{"unschedulable":false}}'
        timeout: 30s
        
  postcheck:
    - name: verify_pods_running
      type: health_check
      command: "kubectl get pods -A -o json | jq '[.items[] | select(.status.phase==\"Running\")] | length'"
      expected: ">0"
      timeout: 60s
      critical: true
      
    - name: verify_no_pending_pods
      type: metric_check
      command: "kubectl get pods -A --field-selector status.phase=Pending --no-headers | wc -l"
      expected: "0"
      timeout: 30s
      critical: false
      
  audit:
    enabled: true
    hashchain: true
    anchor: true
    
  notifications:
    on_start: [slack, pagerduty]
    on_success: [slack]
    on_failure: [slack, pagerduty, email]
```

## Validation Rules

1. **Required fields**: apiVersion, kind, metadata.name, spec.trigger, spec.actions
2. **Version format**: Must follow semver (X.Y.Z)
3. **Timeout format**: Duration string (e.g., "30s", "5m", "1h")
4. **Action types**: Must be one of: restart, scale, migrate, script, api_call
5. **Severity levels**: Must be one of: critical, high, medium, low
6. **Operators**: Must be one of: gt, lt, eq, ne, gte, lte

## Best Practices

1. **Always include prechecks**: Verify system state before taking action
2. **Enable rollback**: For any action that changes system state
3. **Set appropriate timeouts**: Prevent indefinite hangs
4. **Use critical flags wisely**: Only for checks that should abort execution
5. **Include postchecks**: Verify remediation succeeded
6. **Enable audit trail**: For compliance and debugging
7. **Test rollback paths**: Ensure rollback works as expected
8. **Document dependencies**: List required tools, permissions, network access
9. **Version playbooks**: Increment version on any change
10. **Tag appropriately**: Makes searching and categorization easier

## Testing Playbooks

Use the test harness:
```bash
go test -v ./core/autoheal/...
```

Schema validation tooling is planned (playbook-validator). For now, validate by review and unit tests.

## See Also
- [Mesh Controller (Auto-heal core)](../mesh_controller.go)
- [Audit Hashchain](../../../pkg/audit/hashchain.go)
- [Anchor Service](../../../services/anchor/main.go)
