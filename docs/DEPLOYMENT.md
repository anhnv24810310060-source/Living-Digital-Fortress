# Architecture Overview

## Design Goals

### Primary Objectives
* **Deterministic policy evaluation** - Consistent security decisions across all nodes
* **Strong isolation for untrusted execution** - MicroVM and WASM sandboxing
* **Evidence-producing runtime** - Attestation hooks and immutable audit trails
* **Supply chain integrity** - End-to-end verification from source to execution
* **Zero-trust architecture** - Continuous authentication and authorization
* **High availability** - 99.99% uptime with automatic failover

### Design Principles

1. **Defense in Depth** - Multiple layers of security controls
2. **Least Privilege** - Minimal permissions by default
3. **Fail Secure** - System fails to safe state on errors
4. **Separation of Concerns** - Modular, loosely-coupled services
5. **Observable by Default** - Comprehensive metrics and logging

## Component Summary

### Core Services

#### 1. Orchestrator (Port 8080)
**Purpose**: Central coordination and policy enforcement

**Responsibilities**:
- Route requests based on OPA policies
- Enforce rate limits and quotas
- Aggregate threat intelligence
- Coordinate service mesh

**Technology Stack**:
- Go 1.25+ with goroutines for concurrency
- OPA for policy evaluation
- gRPC for inter-service communication
- Prometheus for metrics

**Key Algorithms**:
- Circuit breaker pattern for fault tolerance
- Token bucket rate limiting
- Consistent hashing for load distribution

#### 2. Ingress Gateway (Port 8081)
**Purpose**: Entry point for all external traffic

**Responsibilities**:
- QUIC/HTTP3 protocol handling
- TLS termination (TLS 1.3)
- Request validation and sanitization
- DDoS protection (rate limiting, connection limits)
- Geographic routing

**Technology Stack**:
- Go with quic-go library
- Cloudflare Workers integration
- Redis for distributed rate limiting

**Security Features**:
- Automatic certificate management (Let's Encrypt)
- Request signature verification
- IP reputation checking
- Bot detection (challenge-response)

#### 3. Guardian (Port 9090)
**Purpose**: Secure sandbox execution and threat analysis

**Responsibilities**:
- Execute untrusted code in Firecracker MicroVMs
- WASM runtime for lightweight isolation
- System call monitoring via eBPF
- Memory forensics and behavior analysis
- Malware detonation chamber

**Technology Stack**:
- Firecracker for hardware-level isolation
- Wazero for WASM execution
- eBPF (Cilium library) for kernel monitoring
- Go for orchestration

**Isolation Mechanisms**:
- KVM-based virtualization
- Seccomp-BPF syscall filtering
- Network namespace isolation
- Resource limits (CPU, memory, I/O)

#### 4. Credits Service (Port 5004)
**Purpose**: Resource management and billing

**Responsibilities**:
- Credit allocation and consumption
- Usage tracking and metering
- Billing integration
- Quota enforcement

**Technology Stack**:
- Go with PostgreSQL
- Redis for caching
- Event sourcing for audit trail

**Key Features**:
- ACID transactions (never negative balance)
- Idempotency keys for duplicate prevention
- Real-time balance updates
- Multi-tenant isolation

#### 5. ContAuth (Continuous Authentication) (Port 5002)
**Purpose**: Behavioral biometrics and continuous identity verification

**Responsibilities**:
- Keystroke dynamics analysis
- Mouse movement pattern recognition
- Device fingerprinting
- Risk score calculation
- Session anomaly detection

**Technology Stack**:
- Go for API layer
- Python (scikit-learn, TensorFlow) for ML models
- PostgreSQL for user profiles
- Redis for session state

**ML Models**:
- Random Forest for keystroke classification
- Isolation Forest for anomaly detection
- Neural networks for pattern recognition

#### 6. Shadow Evaluation (Port 5005)
**Purpose**: Safe testing of new security rules

**Responsibilities**:
- Parallel evaluation of candidate rules
- A/B testing with statistical analysis
- Performance impact assessment
- Automated rollout/rollback

**Technology Stack**:
- Go with Docker for rule isolation
- Bayesian A/B testing
- Prometheus for metrics comparison

**Testing Methodology**:
- Champion/Challenger pattern
- Thompson Sampling for traffic allocation
- Statistical significance testing (p < 0.05)
- Automated promotion criteria

#### 7. Policy Rollout (Port 5006)
**Purpose**: Controlled deployment of policy bundles

**Responsibilities**:
- Signed policy bundle distribution
- Version control and rollback
- Canary deployments
- Policy conflict detection

**Technology Stack**:
- Go with Cosign for signing
- Git for version control
- Redis for caching

**Security**:
- Keyless signing with Sigstore
- SHA-256 digest verification
- HMAC-based integrity checks

#### 8. Verifier Pool (Port 5007)
**Purpose**: Remote attestation and integrity verification

**Responsibilities**:
- TPM-based attestation
- Intel SGX enclave verification
- RA-TLS handshake
- Measurement validation

**Technology Stack**:
- Go with go-attestation
- Intel SGX SDK
- gRPC for remote attestation protocol

**Status**: Experimental (not production-ready)

#### 9. Locator (Port 5008)
**Purpose**: Service discovery and health monitoring

**Responsibilities**:
- Service registration
- Health check aggregation
- Load balancing metadata
- Circuit breaker status

**Technology Stack**:
- Go with Consul integration
- DNS-based service discovery
- HTTP health check endpoints

## Data Flows (Simplified)
1. Client request enters Ingress
2. Orchestrator performs policy check (OPA bundle in-memory cache)
3. Approved tasks scheduled to Guardian (microVM or WASM path)
4. Guardian emits execution metadata + telemetry (eBPF events, metrics)
5. Orchestrator scores risk, may adapt rates / isolation level
6. Verifier Pool (optional) validates attestation evidence

## Policy Model
* Rego policies grouped into bundles (namespaced)
* Bundles signed (digest + cosign keyless)
* Promotion pipeline ensures only approved digest used

## Execution Isolation Layers
| Layer | Mechanism | Notes |
|-------|----------|-------|
| Process | Go runtime | Control plane components |
| MicroVM | Firecracker | Strong isolation for native tasks |
| WASM | Wazero runtime | Lightweight deterministic execution |

## Telemetry & Scoring
* eBPF collectors feed event bus
* Scoring engine assigns threat scores (heuristics now, ML future)
* Scores influence circuit breaker thresholds & throttling

## Attestation (Experimental)
* RA-TLS or custom evidence embedded in TLS handshake
* Verifier Pool validates measurements -> issues verdict

## Extensibility
* Pluggable execution backends
* Policy versioning & rollback
* Feature flags via environment / config service (future)

## Future Enhancements
See `docs/ROADMAP.md`.
