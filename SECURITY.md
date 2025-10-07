# Security Policy

## Reporting a Vulnerability

**Please DO NOT file a public issue to report a security vulnerability.**

We take the security of ShieldX seriously. If you discover a security vulnerability, please follow these steps:

### Contact Information

- **Email**: security@shieldx.dev (create this email or replace with actual contact)
- **GitHub Security Advisory**: Use GitHub's [private security advisory feature](https://github.com/shieldx-bot/shieldx/security/advisories/new)

### What to Include

When reporting a vulnerability, please provide:

1. **Description**: Clear description of the vulnerability
2. **Impact**: What an attacker could achieve
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Version**: Affected version(s) or commit hash
5. **Environment**: OS, Go version, deployment configuration
6. **Proof of Concept**: Code, commands, or screenshots (if applicable)
7. **Suggested Fix**: If you have one (optional)

### Response Timeline

| Severity | Initial Response | Fix Target |
|----------|-----------------|------------|
| Critical | 24-48 hours | 7 days |
| High | 3-5 days | 30 days |
| Medium | 7 days | 60 days |
| Low | 14 days | Best effort |

### Disclosure Policy

- **Coordinated Disclosure**: We follow responsible disclosure practices
- **Embargo Period**: We request at least 90 days before public disclosure
- **Credit**: Security researchers will be credited in release notes (unless they prefer to remain anonymous)
- **CVE Assignment**: We will request CVE IDs for confirmed vulnerabilities

## Security Hardening Guidelines

### Production Deployment

⚠️ **IMPORTANT**: ShieldX is currently in **ALPHA** status. Do NOT use in production without thorough security review.

#### Required Security Measures

1. **Disable Insecure Modes**
   ```bash
   # NEVER set these in production:
   ORCH_ALLOW_INSECURE=0  # Must be 0 or unset
   GUARDIAN_UNSAFE_MODE=0
   ```

2. **Enable TLS/mTLS**
   ```bash
   # Generate proper certificates (not self-signed for production)
   ORCH_TLS_CERT=/path/to/cert.pem
   ORCH_TLS_KEY=/path/to/key.pem
   ORCH_MTLS_CA=/path/to/ca.pem
   ```

3. **Sandbox Hardening**
   ```bash
   # Use hardware isolation
   GUARDIAN_SANDBOX_BACKEND=firecracker
   FC_KERNEL_PATH=/path/to/vmlinux
   FC_ROOTFS_PATH=/path/to/rootfs.ext4
   
   # Strict timeouts (max 30s per requirement)
   FC_TIMEOUT_SEC=30
   GUARDIAN_MAX_CONCURRENT=32
   ```

4. **Network Isolation**
   - Run services in isolated networks
   - Use firewall rules to restrict inter-service communication
   - Never expose Guardian directly to public internet

5. **Secrets Management**
   - Use secrets manager (Vault, AWS Secrets Manager, etc.)
   - Rotate credentials regularly
   - Never commit secrets to repository

### Known Limitations (Alpha)

| Component | Status | Production Ready? |
|-----------|--------|-------------------|
| Orchestrator Policy Routing | ✅ Stable | Yes (with TLS) |
| Guardian Sandbox | ⚠️ Experimental | No - requires hardening |
| RA-TLS Attestation | 🔬 Research | No - proof of concept |
| Post-Quantum Crypto | 🔬 Research | No - experimental |
| eBPF Monitoring | ⚠️ Beta | Limited (kernel 5.10+) |
| WCH (Whisper Channel) | ⚠️ Beta | Limited testing |

### Security Features

#### Current Protections

- ✅ Circuit breaker for failed backends
- ✅ Rate limiting per IP/tenant
- ✅ Concurrent execution limits
- ✅ Threat scoring (heuristic + eBPF)
- ✅ Sandbox timeout enforcement (30s hard limit)
- ✅ Input validation on all endpoints
- ✅ Health check degradation detection

#### Planned Enhancements

- 🚧 Seccomp profiles for sandbox
- 🚧 AppArmor/SELinux policies
- 🚧 Runtime integrity verification
- 🚧 Audit logging with tamper detection
- 🚧 Zero-trust architecture (mTLS everywhere)

### Secure Configuration Checklist

Before deploying:

- [ ] All services run as non-root users
- [ ] TLS enabled for all external endpoints
- [ ] mTLS enabled for inter-service communication
- [ ] Secrets stored in secure vault (not env vars)
- [ ] Network policies restrict traffic flow
- [ ] Resource limits configured (CPU, memory, file descriptors)
- [ ] Logging captures security events
- [ ] Monitoring alerts on anomalies
- [ ] Regular security scans scheduled (gosec, trivy)
- [ ] Dependency updates automated (Dependabot)
- [ ] Access logs reviewed regularly

### Security Scanning

Run security scans regularly:

```bash
# Go security check
go install github.com/securego/gosec/v2/cmd/gosec@latest
gosec -fmt=json -out=gosec-report.json ./...

# Vulnerability check
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck ./...

# Container scanning
trivy image shieldx/orchestrator:latest
trivy image shieldx/guardian:latest

# SBOM generation
syft dir:. -o cyclonedx-json > sbom.json
```

### Incident Response

If a security incident occurs:

1. **Contain**: Isolate affected systems
2. **Assess**: Determine scope and impact
3. **Eradicate**: Remove threat and patch vulnerability
4. **Recover**: Restore services safely
5. **Learn**: Post-mortem and improve defenses

## Hall of Fame

We recognize security researchers who help improve ShieldX:

<!-- Add contributors here after first vulnerability report -->

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Go Security Best Practices](https://go.dev/doc/security/)
- [Container Security Best Practices](https://kubernetes.io/docs/concepts/security/)

---

**Last Updated**: 2025-10-06
