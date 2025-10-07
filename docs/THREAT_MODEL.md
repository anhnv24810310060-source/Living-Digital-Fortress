# Threat Model (Initial)

S## Representative Threats & Mitigations

### Application Layer Threats

| Threat ID | Threat | Attack Vector | Impact | Mitigation | Status |
|-----------|--------|---------------|--------|------------|--------|
| T-001 | SQL Injection | Malicious input in API requests | Data breach, unauthorized access | Prepared statements, input validation, OPA policies | ✅ Implemented |
| T-002 | XSS (Cross-Site Scripting) | Malicious scripts in user input | Session hijacking, data theft | Content Security Policy, input sanitization, output encoding | ✅ Implemented |
| T-003 | CSRF (Cross-Site Request Forgery) | Forged requests from trusted user | Unauthorized actions | CSRF tokens, SameSite cookies, origin validation | ✅ Implemented |
| T-004 | API Rate Limit Bypass | Distributed requests, IP rotation | Service degradation, cost overrun | Token bucket algorithm, Redis distributed locks, IP reputation | ✅ Implemented |
| T-005 | Authentication Bypass | Weak credentials, session fixation | Unauthorized access | Multi-factor auth, behavioral biometrics, session rotation | ✅ Implemented |

### Infrastructure Threats

| Threat ID | Threat | Attack Vector | Impact | Mitigation | Status |
|-----------|--------|---------------|--------|------------|--------|
| T-101 | Sandbox Escape | Kernel exploit, device passthrough | Host compromise | MicroVM isolation, seccomp-BPF, minimal device surface | ✅ Implemented |
| T-102 | Container Breakout | Privilege escalation, CVE exploit | Cluster compromise | Non-root containers, AppArmor/SELinux, read-only FS | ✅ Implemented |
| T-103 | Network Sniffing | Man-in-the-Middle, ARP spoofing | Data interception | mTLS, network policies, encrypted overlay | ✅ Implemented |
| T-104 | DDoS Attack | Volumetric flood, application-layer DoS | Service unavailability | Cloudflare protection, rate limiting, circuit breakers | ✅ Implemented |
| T-105 | Side-Channel Attack | Timing analysis, cache probing | Information leakage | Constant-time crypto, noise injection, VM isolation | ⚠️ Partial |

### Supply Chain Threats

| Threat ID | Threat | Attack Vector | Impact | Mitigation | Status |
|-----------|--------|---------------|--------|------------|--------|
| T-201 | Modified Policy Bundle | Compromised CI/CD, insider threat | Malicious policy execution | Cosign keyless signing, digest verification, audit log | ✅ Implemented |
| T-202 | Dependency Confusion | Malicious package injection | Code execution, backdoor | Private registry, dependency pinning, SBOM verification | ✅ Implemented |
| T-203 | Compromised Base Image | Backdoored container image | Persistent access | Distroless images, image signing, vulnerability scanning | ✅ Implemented |
| T-204 | Replay of Old Bundle | Captured + replayed signed bundle | Downgrade attack | Version monotonicity check, timestamp validation | ✅ Implemented |
| T-205 | Build System Compromise | Compromised build agent | Malicious artifacts | Hermetic builds, ephemeral agents, SLSA Level 3 | 🚧 In Progress |

### Data Threats

| Threat ID | Threat | Attack Vector | Impact | Mitigation | Status |
|-----------|--------|---------------|--------|------------|--------|
| T-301 | Data Breach | SQL injection, API exploit | PII/PCI exposure | Encryption at rest (AES-256), field-level encryption, access logs | ✅ Implemented |
| T-302 | Credential Theft | Phishing, keylogger, memory dump | Account takeover | Hardware tokens, passwordless auth, vault storage | ⚠️ Partial |
| T-303 | Audit Log Tampering | Direct DB access, log injection | Evidence destruction | Immutable log (hash chain), write-once storage, SIEM integration | ✅ Implemented |
| T-304 | Secrets in Code | Hardcoded credentials, .env in repo | Full system compromise | Secret scanning (pre-commit), Vault integration, rotation policy | ✅ Implemented |
| T-305 | Backup Theft | Stolen backup media, cloud breach | Historical data exposure | Encrypted backups (GPG), separate keys, geographic distribution | ✅ Implemented |

### Insider Threats

| Threat ID | Threat | Attack Vector | Impact | Mitigation | Status |
|-----------|--------|---------------|--------|------------|--------|
| T-401 | Malicious Admin | Privileged account abuse | System compromise | Dual control, audit all admin actions, break-glass procedures | ⚠️ Partial |
| T-402 | Social Engineering | Pretexting, phishing internal staff | Credential theft, unauthorized access | Security awareness training, phishing simulations, MFA | 🚧 Ongoing |
| T-403 | Data Exfiltration | Authorized user stealing data | IP theft, compliance violation | DLP, anomaly detection, egress filtering | ⚠️ Partial |

**Legend**:
- ✅ Implemented - Mitigation deployed in production
- ⚠️ Partial - Some controls in place, gaps remain
- 🚧 In Progress - Active development
- 🔬 Research - Exploring solutionsha draft – will evolve.

## Assets
| Asset | Description | Security Goal |
|-------|-------------|---------------|
| Policy Bundles | Rego rule sets controlling decisions | Integrity / authenticity |
| Execution Environment | MicroVM / WASM sandbox | Isolation / containment |
| Telemetry Data | eBPF + runtime metrics | Integrity / availability |
| Secrets (keys, tokens) | External system credentials | Confidentiality |
| Build Artifacts | Binaries, images | Integrity / provenance |

## Trust Boundaries
* User input -> Ingress
* Orchestrator -> Guardian (task handoff)
* Guardian -> Host kernel (syscalls / network)
* Policy bundle source -> CI pipeline -> runtime cache

## Adversaries
| Actor | Capability |
|-------|-----------|
| Malicious Tenant | Craft requests / payloads, timing attacks |
| Supply Chain Attacker | Attempt to inject dependency or modify artifact |
| Sandbox Escape Attacker | Attempt microVM / WASM breakout |
| Network Eavesdropper | Observe / tamper in transit |

## Representative Threats & Mitigations
| Threat | Mitigation |
|--------|-----------|
| Modified policy bundle | Signed digest + verified before load |
| Sandbox escape | MicroVM isolation, minimal device surface, seccomp (future) |
| Dependency vuln (transitive) | Dependabot + govulncheck + gosec |
| Replay of old bundle | Version pin + digest comparison |
| Resource exhaustion (DoS) | Rate limiting, concurrency caps, timeouts |
| Leakage of secrets | External secret store, no persistent secrets on disk |
| Attestation forgery | Keyless signatures, verifier cross-checks (future) |

## Open Risks / TODO
* Hardening of Firecracker + kernel parameters
* Formal verification of attestation flow
* Seccomp / AppArmor profiles generation
* Audit log integrity chain
