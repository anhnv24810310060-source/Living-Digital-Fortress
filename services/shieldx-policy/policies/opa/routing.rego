# ShieldX Routing Policy (OPA Rego)
# 
# This policy defines advanced routing and security decisions for the Orchestrator service.
# Decisions: allow, deny, divert, tarpit
# 
# Policy Evaluation Order:
# 1. Explicit deny rules (highest priority)
# 2. Tarpit rules (slow down suspicious traffic)
# 3. Divert rules (send to honeypot/Guardian)
# 4. Allow rules (legitimate traffic)
# 5. Default deny (if nothing matches)

package shieldx.authz

import future.keywords.if
import future.keywords.in

# Default decision is deny (fail-secure)
default decision = "deny"

# ============================================================================
# ALLOW RULES - Legitimate Traffic
# ============================================================================

# Allow authenticated tenants with valid scopes
decision = "allow" if {
    input.tenant != ""
    is_known_tenant(input.tenant)
    input.scope in valid_scopes_for_tenant(input.tenant)
    not is_suspicious_ip(input.ip)
    not is_attack_pattern(input.path)
}

# Allow health checks and metrics endpoints (internal traffic)
decision = "allow" if {
    input.path in ["/health", "/healthz", "/metrics", "/ready"]
    is_internal_ip(input.ip)
}

# Allow admin operations from authorized IPs only
decision = "allow" if {
    startswith(input.path, "/admin/")
    is_admin_ip(input.ip)
    input.tenant in admin_tenants
}

# ============================================================================
# DENY RULES - Block Malicious Traffic
# ============================================================================

# Deny known malicious IPs (from threat intel)
decision = "deny" if {
    is_blocklisted_ip(input.ip)
}

# Deny requests with SQL injection patterns
decision = "deny" if {
    contains(lower(input.path), "union select")
}
decision = "deny" if {
    contains(lower(input.path), "' or 1=1")
}
decision = "deny" if {
    contains(lower(input.path), "drop table")
}

# Deny path traversal attempts
decision = "deny" if {
    contains(input.path, "../")
}
decision = "deny" if {
    contains(input.path, "..\\")
}

# Deny access to sensitive paths
decision = "deny" if {
    sensitive_path := [
        "/.env", "/.git", "/.aws", "/.ssh",
        "/admin", "/phpmyadmin", "/wp-admin"
    ]
    input.path in sensitive_path
}

# Deny requests with suspicious user agents
decision = "deny" if {
    suspicious_agent := ["sqlmap", "nmap", "nikto", "masscan"]
    agent_lower := lower(input.metadata.user_agent)
    some scanner in suspicious_agent
    contains(agent_lower, scanner)
}

# Deny rate limit violators (based on metadata)
decision = "deny" if {
    input.metadata.rate_limit_exceeded == true
}

# ============================================================================
# TARPIT RULES - Slow Down Suspicious Traffic
# ============================================================================

# Tarpit medium-reputation IPs (slow them down)
decision = "tarpit" if {
    not is_blocklisted_ip(input.ip)
    ip_reputation(input.ip) < 50
    ip_reputation(input.ip) > 20
}

# Tarpit requests with suspicious patterns but not definitive attacks
decision = "tarpit" if {
    count_suspicious_chars(input.path) > 5
    not is_attack_pattern(input.path)
}

# Tarpit high-frequency requesters (potential scrapers)
decision = "tarpit" if {
    input.metadata.request_rate > 100
    input.metadata.request_rate < 500
    not is_known_tenant(input.tenant)
}

# ============================================================================
# DIVERT RULES - Send to Guardian for Analysis
# ============================================================================

# Divert suspicious requests to sandbox (Guardian)
decision = "divert" if {
    is_suspicious_pattern(input.path)
    not is_attack_pattern(input.path)
    not is_known_tenant(input.tenant)
}

# Divert new/unknown tenants for behavioral analysis
decision = "divert" if {
    input.tenant != ""
    not is_known_tenant(input.tenant)
    tenant_age_hours(input.tenant) < 24
}

# Divert requests with obfuscated payloads
decision = "divert" if {
    input.metadata.has_base64_payload == true
    input.metadata.payload_entropy > 7.5
}

# Divert low-reputation IPs for deeper inspection
decision = "divert" if {
    ip_reputation(input.ip) < 30
    ip_reputation(input.ip) > 0
    not is_blocklisted_ip(input.ip)
}

# ============================================================================
# ROUTING HINTS - Influence Backend Selection
# ============================================================================

# Route high-priority tenants to premium pool
route = {"pool": "premium", "algo": "ewma"} if {
    input.tenant in premium_tenants
}

# Route suspicious traffic to Guardian
route = {"pool": "guardian", "algo": "round_robin"} if {
    decision == "divert"
}

# Route known good traffic to fast pool
route = {"pool": "fast", "algo": "p2c"} if {
    decision == "allow"
    input.tenant in trusted_tenants
}

# Default routing to standard pool
route = {"pool": "standard", "algo": "ewma"}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Check if tenant is in known allowlist
is_known_tenant(tenant) if {
    known_tenants := ["tenant-prod-1", "tenant-prod-2", "tenant-staging"]
    tenant in known_tenants
}

# Valid scopes for each tenant
valid_scopes_for_tenant(tenant) = scopes if {
    scope_map := {
        "tenant-prod-1": ["read", "write", "admin"],
        "tenant-prod-2": ["read", "write"],
        "tenant-staging": ["read"]
    }
    scopes := scope_map[tenant]
}

# Check if IP is blocklisted (from threat intel)
is_blocklisted_ip(ip) if {
    blocklist := ["192.0.2.1", "198.51.100.1", "203.0.113.1"]
    ip in blocklist
}

# Check if IP is internal (RFC 1918)
is_internal_ip(ip) if {
    startswith(ip, "10.")
}
is_internal_ip(ip) if {
    startswith(ip, "192.168.")
}
is_internal_ip(ip) if {
    startswith(ip, "172.16.")
}
is_internal_ip(ip) if {
    ip == "127.0.0.1"
}

# Check if IP is authorized for admin operations
is_admin_ip(ip) if {
    admin_ips := ["10.0.0.100", "10.0.0.101"]
    ip in admin_ips
}

# IP reputation score (0-100, higher is better)
# In production, this would query a real threat intel service
ip_reputation(ip) = score if {
    is_blocklisted_ip(ip)
    score := 0
}
ip_reputation(ip) = score if {
    is_internal_ip(ip)
    score := 100
}
ip_reputation(ip) = score if {
    not is_blocklisted_ip(ip)
    not is_internal_ip(ip)
    score := 50  # Default neutral reputation
}

# Check if path contains attack patterns
is_attack_pattern(path) if {
    attack_patterns := ["union", "select", "drop", "exec", "script", "onerror"]
    path_lower := lower(path)
    some pattern in attack_patterns
    contains(path_lower, pattern)
}

# Check if path has suspicious patterns (not definitive attacks)
is_suspicious_pattern(path) if {
    suspicious_patterns := ["admin", "config", "backup", "test", "debug"]
    path_lower := lower(path)
    some pattern in suspicious_patterns
    contains(path_lower, pattern)
}

# Check if IP is suspicious (basic heuristic)
is_suspicious_ip(ip) if {
    ip_reputation(ip) < 30
}

# Count suspicious characters in path
count_suspicious_chars(path) = count if {
    suspicious_chars := ["<", ">", "'", "\"", ";", "|", "&", "$"]
    matches := [c | c := path[_]; c in suspicious_chars]
    count := count(matches)
}

# Get tenant age in hours (mock - would query database in production)
tenant_age_hours(tenant) = hours if {
    # Mock implementation - new tenants
    not is_known_tenant(tenant)
    hours := 1
}
tenant_age_hours(tenant) = hours if {
    is_known_tenant(tenant)
    hours := 1000
}

# Premium tenants (SLA guarantees)
premium_tenants := ["tenant-prod-1", "tenant-enterprise"]

# Trusted tenants (low latency routing)
trusted_tenants := ["tenant-prod-1", "tenant-prod-2"]

# Admin tenants
admin_tenants := ["tenant-admin", "tenant-ops"]

# ============================================================================
# TESTING - Example Inputs
# ============================================================================

# Example 1: Legitimate request
# input: {"tenant": "tenant-prod-1", "scope": "read", "path": "/api/users", "ip": "10.0.0.50"}
# expected: decision = "allow", route = {"pool": "premium", "algo": "ewma"}

# Example 2: SQL injection attempt
# input: {"tenant": "unknown", "scope": "", "path": "/api?id=1' OR 1=1--", "ip": "203.0.113.1"}
# expected: decision = "deny"

# Example 3: Suspicious new tenant
# input: {"tenant": "new-tenant-123", "scope": "read", "path": "/api/test", "ip": "8.8.8.8"}
# expected: decision = "divert", route = {"pool": "guardian", "algo": "round_robin"}

# Example 4: Low reputation IP
# input: {"tenant": "", "scope": "", "path": "/", "ip": "192.0.2.1"}
# expected: decision = "deny"

# Example 5: Admin request from authorized IP
# input: {"tenant": "tenant-admin", "scope": "admin", "path": "/admin/users", "ip": "10.0.0.100"}
# expected: decision = "allow"
