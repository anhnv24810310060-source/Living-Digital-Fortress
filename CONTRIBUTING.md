# Contributing to ShieldX

**Thank you for your interest in contributing to ShieldX!** This document provides comprehensive guidelines for contributing to the ShieldX cloud security platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Security Guidelines](#security-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@shieldx.dev.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

**Required:**
- **Go**: 1.21 or later ([Download](https://go.dev/dl/))
- **Git**: 2.30+ ([Download](https://git-scm.com/))
- **Docker**: 24.0+ ([Download](https://docker.com/))
- **Docker Compose**: 2.20+

**Recommended:**
- **Make**: For build automation
- **golangci-lint**: For code linting
- **VS Code** or **GoLand**: IDE with Go support

**Platform-specific:**
- **Linux**: Required for eBPF and Firecracker development
- **Windows/macOS**: Suitable for most development tasks

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space
- **CPU**: 4 cores recommended for full stack development

---

## Development Environment Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/shieldx.git
cd shieldx

# Add upstream remote
git remote add upstream https://github.com/shieldx-bot/shieldx.git

# Verify remotes
git remote -v
```

### 2. Install Dependencies

```bash
# Download Go modules
go mod download

# Verify Go installation
go version

# Install development tools
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/anchore/syft/cmd/syft@latest
```

### 3. Environment Configuration

Create `.env.dev` file:

```bash
# Database
POSTGRES_URL=postgresql://credits_user:credits_pass@localhost:5432/credits
REDIS_URL=redis://localhost:6379

# Services
ORCHESTRATOR_PORT=8080
INGRESS_PORT=8081
GUARDIAN_PORT=9090

# Observability
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_URL=http://localhost:16686

# Development
LOG_LEVEL=debug
ENABLE_PROFILING=true
```

### 4. Start Development Stack

```bash
# Start infrastructure services
docker compose -f docker-compose.full.yml up -d

# Verify services are running
docker ps

# Check service health
make demo-health
```

### 5. Verify Setup

```bash
# Run tests
go test ./...

# Build services
make build-services

# Format code
go fmt ./...

# Lint code
golangci-lint run ./...
```

---

## Project Architecture

### Directory Structure

```
shieldx/
├── services/                    # Microservices
│   ├── shieldx-gateway/        # Orchestrator (port 8080)
│   │   └── orchestrator/
│   ├── shieldx-ingress/        # Ingress gateway (port 8081)
│   │   └── ingress/
│   ├── shieldx-forensics/      # Threat analysis
│   │   ├── guardian/           # Sandbox (port 9090)
│   │   └── anchor/             # Audit logging
│   ├── shieldx-auth/           # Authentication
│   │   └── contauth-service/   # Continuous auth (port 5002)
│   ├── shieldx-billing/        # Resource management
│   │   └── credits/            # Credits service (port 5004)
│   ├── shieldx-deception/      # Deception technology
│   │   ├── decoy-manager/
│   │   ├── shapeshifter/
│   │   └── sinkhole/
│   ├── shieldx-ml/             # Machine learning
│   │   └── ml-orchestrator/
│   ├── shieldx-policy/         # Policy management
│   │   └── policy-rollout/     # Policy deployment (port 5006)
│   └── shieldx-discovery/      # Service discovery
│       └── locator/            # Service locator (port 5008)
│
├── shared/shieldx-common/      # Shared libraries
│   ├── pkg/                    # Common packages
│   │   ├── auth/              # Authentication utilities
│   │   ├── metrics/           # Prometheus metrics
│   │   ├── observability/     # Logging, tracing
│   │   ├── sandbox/           # Firecracker integration
│   │   ├── ml/                # ML utilities
│   │   └── guardian/          # Threat scoring
│   └── core/                   # Core engines
│       ├── autoheal/          # Self-healing
│       └── maze_engine/       # Deception engine
│
├── pkg/                        # Additional shared code
│   ├── policy/                # OPA policy engine
│   ├── ebpf/                  # eBPF monitoring
│   ├── wch/                   # Webhook handling
│   └── database/              # Database utilities
│
├── infrastructure/             # Infrastructure as Code
│   ├── docker/                # Dockerfiles
│   ├── kubernetes/            # K8s manifests
│   └── infra/cloudflare/      # Edge workers
│
├── pilot/observability/        # Observability stack
│   ├── prometheus-scrape.yml
│   ├── grafana-dashboard-http-slo.json
│   └── docker-compose.yml
│
├── policies/                   # OPA policies
│   └── demo/                  # Demo policies
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   └── THREAT_MODEL.md
│
├── tools/cli/cmd/             # CLI tools
│   ├── policyctl/            # Policy management
│   ├── locator/              # Service discovery
│   └── migrate-db/           # Database migrations
│
├── scripts/                    # Build and deployment scripts
├── .github/workflows/          # CI/CD pipelines
├── Makefile                    # Build automation
├── go.mod                      # Go dependencies
└── docker-compose.full.yml     # Full stack compose
```

### Key Components

| Component | Port | Technology | Purpose |
|-----------|------|------------|---------|
| **Orchestrator** | 8080 | Go, OPA | Central routing & policy evaluation |
| **Ingress** | 8081 | Go, QUIC | Traffic gateway & rate limiting |
| **Guardian** | 9090 | Go, Firecracker, eBPF | Sandbox execution & threat analysis |
| **Credits** | 5004 | Go, PostgreSQL | Resource management & billing |
| **ContAuth** | 5002 | Go, Python ML | Continuous authentication |
| **Policy Rollout** | 5006 | Go | Controlled policy deployment |
| **Locator** | 5008 | Go, Consul | Service discovery |

---

## Contribution Workflow

### 1. Choose a Task

**For Beginners:**
- Look for issues labeled `good-first-issue`
- Documentation improvements
- Adding tests
- Fixing typos

**For Intermediate:**
- Bug fixes
- Feature enhancements
- Performance improvements
- Integration tests

**For Advanced:**
- New security features
- eBPF monitoring
- ML algorithms
- Firecracker integration

### 2. Create a Branch

Branch naming convention:

```bash
# Feature
git checkout -b feat/add-rate-limiting-metrics

# Bug fix
git checkout -b fix/guardian-race-condition

# Documentation
git checkout -b docs/update-architecture-diagram

# Refactoring
git checkout -b refactor/simplify-policy-engine

# Tests
git checkout -b test/add-integration-tests-ingress
```

### 3. Development Cycle

```bash
# Make changes
# Edit files in your IDE

# Format code
go fmt ./...

# Run tests
go test ./...

# Run specific tests
go test ./services/orchestrator/...
go test -run TestCircuitBreaker ./...

# Lint code
golangci-lint run ./...

# Build
make build-services

# Test locally
docker compose up -d
curl http://localhost:8080/health
```

### 4. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi-colons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `build`: Build system changes

**Examples:**

```bash
git commit -m "feat(orchestrator): add circuit breaker with exponential backoff

Implements circuit breaker pattern for backend services with:
- Configurable failure threshold
- Exponential backoff with jitter
- Prometheus metrics for state transitions

Closes #123"

git commit -m "fix(guardian): prevent race condition in job status updates

Use atomic operations for job status transitions to avoid
data races detected by -race flag.

Fixes #456"

git commit -m "docs(architecture): add mermaid diagram for request flow

Add visual representation of request processing through
Ingress -> Orchestrator -> Guardian pipeline."
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feat/add-rate-limiting-metrics

# Create Pull Request on GitHub
# Fill in the PR template
```

---

## Coding Standards

### Go Style Guide

Follow [Effective Go](https://go.dev/doc/effective_go) and [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md).

#### Package Organization

```go
// Good: Clear package structure
package metrics

import (
    "context"
    "time"
    
    "github.com/prometheus/client_golang/prometheus"
    "github.com/shieldx-bot/shieldx/shared/shieldx-common/pkg/observability"
)

// Bad: Mixed concerns
package utils
```

#### Naming Conventions

```go
// Packages: short, lowercase, single word
package auth
package metrics

// Files: lowercase with underscores
// circuit_breaker.go
// threat_scorer.go

// Exported functions: CamelCase
func CalculateRiskScore(events []Event) float64

// Unexported functions: camelCase
func parseConfig(path string) (*Config, error)

// Constants: CamelCase or UPPER_CASE
const MaxRetries = 3
const DEFAULT_TIMEOUT = 30 * time.Second

// Interfaces: -er suffix
type Runner interface {
    Run(ctx context.Context) error
}

type ThreatScorer interface {
    Score(event Event) float64
}
```

#### Error Handling

```go
// Good: Wrap errors with context
if err != nil {
    return fmt.Errorf("failed to parse policy bundle: %w", err)
}

// Good: Sentinel errors
var (
    ErrTimeout = errors.New("operation timed out")
    ErrNotFound = errors.New("resource not found")
)

// Good: Custom error types
type ValidationError struct {
    Field string
    Value interface{}
    Msg   string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for %s: %s", e.Field, e.Msg)
}

// Bad: Ignoring errors
_ = someFunction() // Only if truly safe

// Bad: Generic errors
return errors.New("error") // No context
```

#### Logging

```go
// Good: Structured logging
log.Printf("[orchestrator] policy evaluation: tenant=%s decision=%s latency=%dms",
    tenantID, decision, latency)

// Better: Use proper logger
logger.Info("policy evaluation completed",
    "tenant", tenantID,
    "decision", decision,
    "latency_ms", latency,
)

// Bad: Unstructured logging
log.Println("Policy evaluated")
```

#### Concurrency

```go
// Good: Proper mutex usage
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

// Good: Context handling
func ProcessRequest(ctx context.Context, req *Request) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    case result := <-processChan:
        return handleResult(result)
    }
}

// Good: Channel usage
func worker(ctx context.Context, jobs <-chan Job, results chan<- Result) {
    for {
        select {
        case <-ctx.Done():
            return
        case job := <-jobs:
            results <- process(job)
        }
    }
}
```

#### Comments

```go
// Good: Package documentation
// Package metrics provides Prometheus metrics collection
// and export functionality for ShieldX services.
//
// It includes pre-configured metrics for:
//   - HTTP request duration and count
//   - Circuit breaker state transitions
//   - Threat detection events
//   - Resource consumption
package metrics

// Good: Function documentation
// CalculateRiskScore computes a risk score (0.0-1.0) based on
// behavioral analysis of user events. Higher scores indicate
// higher risk of malicious activity.
//
// The score is calculated using:
//   - Event frequency analysis
//   - Anomaly detection
//   - ML model predictions
//
// Returns 0.0 for empty event lists.
func CalculateRiskScore(events []Event) float64 {
    // Implementation...
}

// Good: Inline comments for complex logic
// Use exponential backoff with jitter to avoid thundering herd
backoff := baseDelay * math.Pow(2, float64(attempt))
jitter := rand.Float64() * backoff * 0.1
time.Sleep(time.Duration(backoff + jitter))
```

---

## Testing Requirements

### Test Coverage

- **Minimum**: 70% coverage for new code
- **Target**: 80%+ coverage for critical paths
- **Required**: All public APIs must have tests

### Test Types

#### 1. Unit Tests

```go
// Good: Table-driven tests
func TestCircuitBreaker_StateTransitions(t *testing.T) {
    tests := []struct {
        name           string
        failures       int
        threshold      int
        expectedState  State
    }{
        {
            name:          "below threshold stays closed",
            failures:      2,
            threshold:     5,
            expectedState: StateClosed,
        },
        {
            name:          "at threshold opens",
            failures:      5,
            threshold:     5,
            expectedState: StateOpen,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            cb := NewCircuitBreaker(tt.threshold)
            for i := 0; i < tt.failures; i++ {
                cb.RecordFailure()
            }
            
            if cb.State() != tt.expectedState {
                t.Errorf("expected state %v, got %v",
                    tt.expectedState, cb.State())
            }
        })
    }
}
```

#### 2. Integration Tests

```go
func TestOrchestrator_PolicyEvaluation_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test")
    }
    
    // Setup
    ctx := context.Background()
    orch := setupTestOrchestrator(t)
    defer orch.Shutdown()
    
    // Test
    req := &Request{TenantID: "test-tenant"}
    resp, err := orch.EvaluatePolicy(ctx, req)
    
    // Assert
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if resp.Decision != "allow" {
        t.Errorf("expected allow, got %s", resp.Decision)
    }
}
```

#### 3. Benchmark Tests

```go
func BenchmarkThreatScorer_Score(b *testing.B) {
    scorer := NewThreatScorer()
    events := generateTestEvents(100)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        scorer.Score(events)
    }
}
```

### Running Tests

```bash
# All tests
go test ./...

# Specific package
go test ./services/orchestrator/...

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# With race detection
go test -race ./...

# Verbose output
go test -v ./...

# Short mode (skip integration tests)
go test -short ./...

# Benchmarks
go test -bench=. -benchmem ./...

# Specific test
go test -run TestCircuitBreaker ./...
```

---

## Security Guidelines

### Security-First Development

ShieldX is a security platform. All contributions must follow security best practices.

#### 1. Input Validation

```go
// Good: Validate all inputs
func ProcessRequest(req *Request) error {
    if req.TenantID == "" {
        return &ValidationError{Field: "tenant_id", Msg: "required"}
    }
    if len(req.Payload) > MaxPayloadSize {
        return &ValidationError{Field: "payload", Msg: "too large"}
    }
    // Sanitize inputs
    req.TenantID = sanitize(req.TenantID)
    return nil
}
```

#### 2. Secrets Management

```go
// Good: Never hardcode secrets
apiKey := os.Getenv("API_KEY")
if apiKey == "" {
    return errors.New("API_KEY not set")
}

// Bad: Hardcoded secrets
const apiKey = "sk-1234567890" // NEVER DO THIS
```

#### 3. SQL Injection Prevention

```go
// Good: Use parameterized queries
query := "SELECT * FROM users WHERE id = $1"
row := db.QueryRow(query, userID)

// Bad: String concatenation
query := fmt.Sprintf("SELECT * FROM users WHERE id = %s", userID)
```

#### 4. Authentication & Authorization

```go
// Good: Check permissions
func HandleRequest(w http.ResponseWriter, r *http.Request) {
    user, err := authenticate(r)
    if err != nil {
        http.Error(w, "Unauthorized", http.StatusUnauthorized)
        return
    }
    
    if !authorize(user, "read:policies") {
        http.Error(w, "Forbidden", http.StatusForbidden)
        return
    }
    
    // Process request
}
```

#### 5. Cryptography

```go
// Good: Use standard crypto libraries
import "crypto/rand"

func generateToken() (string, error) {
    b := make([]byte, 32)
    if _, err := rand.Read(b); err != nil {
        return "", err
    }
    return base64.URLEncoding.EncodeToString(b), nil
}

// Bad: Custom crypto
func generateToken() string {
    return fmt.Sprintf("%d", time.Now().Unix()) // Predictable
}
```

### Security Checklist

Before submitting PR:

- [ ] All inputs validated and sanitized
- [ ] No hardcoded secrets or credentials
- [ ] SQL queries use parameterized statements
- [ ] Authentication and authorization implemented
- [ ] Sensitive data encrypted at rest and in transit
- [ ] Error messages don't leak sensitive information
- [ ] Rate limiting implemented where appropriate
- [ ] CSRF protection for state-changing operations
- [ ] Security headers set correctly

---

## Documentation

### Code Documentation

```go
// Package-level documentation
// Package orchestrator implements the central routing and policy
// evaluation engine for ShieldX. It coordinates between ingress,
// guardian, and other services to make security decisions.
//
// Architecture:
//
//	Request → Ingress → Orchestrator → Policy Engine → Decision
//	                         ↓
//	                    Guardian (if suspicious)
//
// The orchestrator uses Open Policy Agent (OPA) for policy evaluation
// and maintains circuit breakers for backend service resilience.
package orchestrator

// Type documentation
// CircuitBreaker implements the circuit breaker pattern to prevent
// cascading failures when backend services are unhealthy.
//
// States:
//   - Closed: Normal operation, requests pass through
//   - Open: Failures exceeded threshold, requests fail fast
//   - HalfOpen: Testing if service recovered
//
// Example:
//
//	cb := NewCircuitBreaker(5) // Open after 5 failures
//	if cb.Allow() {
//	    err := callBackend()
//	    if err != nil {
//	        cb.RecordFailure()
//	    } else {
//	        cb.RecordSuccess()
//	    }
//	}
type CircuitBreaker struct {
    // ...
}
```

### README Updates

When adding features, update relevant README files:

- `README.md` - Main project README
- `services/*/README.md` - Service-specific documentation
- `docs/ARCHITECTURE.md` - Architecture changes
- `docs/API.md` - API changes

### Changelog

Update `CHANGELOG.md` for all user-facing changes:

```markdown
## [Unreleased]

### Added
- Circuit breaker pattern for backend resilience (#123)
- Prometheus metrics for policy evaluation latency (#124)

### Changed
- Improved error messages in Guardian service (#125)

### Fixed
- Race condition in job status updates (#126)

### Security
- Updated dependencies to patch CVE-2024-1234 (#127)
```

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run full test suite**
```bash
make test
make lint
go test -race ./...
```

3. **Update documentation**
- Add/update code comments
- Update README if needed
- Update CHANGELOG.md

4. **Self-review**
- Review your own changes
- Check for debug code or TODOs
- Verify commit messages

### PR Template

```markdown
## Description
Brief summary of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement
- [ ] Security fix

## Related Issues
Closes #123
Related to #456

## Changes Made
- Added circuit breaker to orchestrator
- Implemented exponential backoff
- Added Prometheus metrics

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Tested with `-race` flag
- [ ] Manual testing performed

## Security Considerations
- Input validation added for all endpoints
- No new dependencies with known vulnerabilities
- Secrets properly managed

## Performance Impact
- Benchmark results: 15% improvement in p99 latency
- Memory usage: No significant change
- CPU usage: Reduced by 10% under load

## Breaking Changes
None

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
- [ ] No new warnings
- [ ] Commit messages follow convention

## Screenshots/Logs
(If applicable)

## Additional Notes
(Any other context)
```

### Review Process

1. **Automated Checks** (must pass):
   - Build successful
   - All tests pass
   - Linting passes
   - Security scan passes
   - Coverage meets threshold

2. **Code Review**:
   - At least 1 maintainer approval required
   - Address all review comments
   - Re-request review after changes

3. **Merge**:
   - Squash and merge (default)
   - Rebase and merge (for clean history)
   - Merge commit (for feature branches)

### Review Timeline

- **Initial response**: Within 2 business days
- **Full review**: Within 5 business days
- **Follow-up**: Within 1 business day

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: support@shieldx.dev
- **Security**: security@shieldx-project.org (private)

### Getting Help

**For Contributors:**
- Tag `@mentor-needed` in issues
- Ask in GitHub Discussions
- Email support@shieldx.dev

**For Maintainers:**
- Review PRs promptly
- Provide constructive feedback
- Help onboard new contributors

### Recognition

Contributors are recognized through:
- Listed in release notes
- Credited in CHANGELOG.md
- Added to CONTRIBUTORS.md
- GitHub profile badges
- Contributor spotlight (significant contributions)

### Contributor Levels

- 🌱 **First Timer**: First merged PR
- 🌿 **Regular**: 5+ merged PRs
- 🌳 **Core**: 20+ PRs + code review rights
- 🏅 **Maintainer**: Trusted with releases and governance

---

## Additional Resources

### Documentation
- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Model](docs/THREAT_MODEL.md)
- [Roadmap](docs/ROADMAP.md)

### External Resources
- [Go Documentation](https://go.dev/doc/)
- [Effective Go](https://go.dev/doc/effective_go)
- [Uber Go Style Guide](https://github.com/uber-go/guide)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

### Tools
- [golangci-lint](https://golangci-lint.run/)
- [gosec](https://github.com/securego/gosec)
- [govulncheck](https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck)
- [syft](https://github.com/anchore/syft)
- [cosign](https://docs.sigstore.dev/cosign/overview/)

---

## License

By contributing to ShieldX, you agree that your contributions will be licensed under the Apache License 2.0.

---

## Questions?

If you have questions not covered in this guide:
1. Check existing [GitHub Discussions](https://github.com/shieldx-bot/shieldx/discussions)
2. Search [GitHub Issues](https://github.com/shieldx-bot/shieldx/issues)
3. Open a new discussion
4. Email support@shieldx.dev

---

**Thank you for contributing to ShieldX! Together, we're building the future of cloud security.** 🛡️
