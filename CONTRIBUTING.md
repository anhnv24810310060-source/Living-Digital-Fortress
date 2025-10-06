# Contributing to ShieldX

Thank you for your interest in contributing to ShieldX! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Go**: 1.22 or later
- **Docker**: 24.0 or later
- **Docker Compose**: 2.20 or later
- **Make**: For build automation
- **Git**: For version control

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/shieldx.git
   cd shieldx
   ```

2. **Install Dependencies**
   ```bash
   go mod download
   ```

3. **Verify Setup**
   ```bash
   make test
   make lint
   ```

4. **Run Development Stack**
   ```bash
   docker compose -f docker-compose.full.yml up -d
   ```

## Making Changes

### Branch Naming Convention

- `feat/<short-description>` - New features
- `fix/<short-description>` - Bug fixes
- `refactor/<short-description>` - Code refactoring
- `docs/<short-description>` - Documentation updates
- `test/<short-description>` - Test improvements
- `chore/<short-description>` - Maintenance tasks

Example: `feat/add-rate-limiting-metrics`

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make Changes**
   - Write code following [Coding Standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   ```bash
   # Run all tests
   make test
   
   # Run specific package tests
   go test ./services/orchestrator/...
   
   # Run with race detection
   go test -race ./...
   
   # Run integration tests
   ./scripts/test-deploy-full-stack.sh
   ```

4. **Lint and Format**
   ```bash
   make fmt
   make lint
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add rate limiting metrics"
   ```

## Testing

### Test Requirements

- **Unit Tests**: Required for all new logic
- **Coverage**: Aim for ≥70% coverage on modified files
- **Integration Tests**: Required for new services or major features
- **Race Detection**: All tests must pass with `-race` flag

### Running Tests

```bash
# All tests
go test ./...

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Specific package
go test -v ./services/guardian/...

# Integration tests
./scripts/test-deploy-full-stack.sh

# Benchmarks
go test -bench=. -benchmem ./services/orchestrator/
```

### Writing Tests

- Use table-driven tests for multiple scenarios
- Test both success and failure paths
- Mock external dependencies
- Use meaningful test names: `TestComponentName_Scenario`

Example:
```go
func TestCircuitBreaker_OpenThreshold(t *testing.T) {
    // Arrange
    backend := &Backend{URL: "http://test"}
    
    // Act
    for i := 0; i < 5; i++ {
        recordBackendFailure(backend, fmt.Errorf("test"))
    }
    
    // Assert
    if backend.cbState.Load() != 1 {
        t.Errorf("expected OPEN state, got %d", backend.cbState.Load())
    }
}
```

## Pull Request Process

### Before Submitting

1. **Ensure Tests Pass**
   ```bash
   make test
   make lint
   ```

2. **Update Documentation**
   - Update README if adding features
   - Add/update comments for public APIs
   - Update CHANGELOG.md

3. **Rebase on Latest Main**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### PR Description Template

```markdown
## Description
Brief summary of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Tested with `-race` flag

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Dependent changes merged

## Performance Impact
(If applicable) Describe performance implications.

## Breaking Changes
(If applicable) Describe what breaks and migration path.
```

### Review Process

1. **Automated Checks**: CI must pass (build, test, lint, security scans)
2. **Code Review**: At least 1 maintainer approval required
3. **Review Time**: Expect response within 3-5 business days
4. **Feedback**: Address reviewer comments and update PR
5. **Merge**: Maintainer will merge once approved

## Coding Standards

### Go Style

- Follow [Effective Go](https://go.dev/doc/effective_go)
- Use `gofmt` and `goimports`
- Run `golangci-lint` before committing

### Code Organization

```
services/          # Microservices (main packages)
pkg/              # Shared libraries
core/             # Core engines
scripts/          # Automation scripts
docs/             # Documentation
```

### Naming Conventions

- **Packages**: Short, lowercase, single word
- **Files**: Lowercase with underscores (e.g., `circuit_breaker.go`)
- **Functions**: CamelCase for exported, camelCase for unexported
- **Constants**: CamelCase or UPPER_CASE for exported
- **Interfaces**: Name after primary method + "er" suffix (e.g., `Runner`)

### Error Handling

```go
// Good: Wrap errors with context
if err != nil {
    return fmt.Errorf("failed to parse config: %w", err)
}

// Good: Sentinel errors
var ErrTimeout = errors.New("operation timed out")

// Bad: Ignore errors
_ = someFunction()  // Only if truly safe to ignore
```

### Logging

```go
// Use structured logging
log.Printf("[component] event: field1=%v field2=%v", val1, val2)

// Production: Use proper logger
logger.Info("event occurred", 
    "field1", val1,
    "field2", val2)
```

### Concurrency

- Use `sync.Mutex` for protecting shared state
- Prefer channels for communication between goroutines
- Always handle context cancellation
- Document goroutine lifecycles

### Testing

- Test files: `*_test.go`
- Benchmark functions: `Benchmark*`
- Example functions: `Example*`
- Use `testing.T` for tests, `testing.B` for benchmarks

## Commit Messages

### Conventional Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test addition/modification
- `docs`: Documentation update
- `chore`: Maintenance (deps, build, etc.)
- `style`: Code style changes (formatting)
- `ci`: CI/CD changes

### Examples

```
feat(orchestrator): add circuit breaker metrics

Add Prometheus metrics for circuit breaker state transitions.
Includes counters for open, close, and half-open events.

Closes #123
```

```
fix(guardian): prevent race in job status update

Use atomic operations for job status transitions to avoid
data races detected by -race flag.
```

```
docs: add architecture diagram to README

Add Mermaid diagram showing service interactions and data flow.
```

### Breaking Changes

If introducing breaking changes:

```
feat(api)!: change policy evaluation response format

BREAKING CHANGE: Policy evaluation now returns structured JSON
instead of plain text. Clients must update to parse new format.

Migration guide: docs/migrations/v2-policy-format.md
```

## Dependencies

### Adding Dependencies

1. **Evaluate Need**: Ensure dependency is necessary
2. **Check License**: Must be permissive (Apache-2.0, MIT, BSD)
3. **Assess Maintenance**: Active maintenance and security record
4. **Small Size**: Prefer smaller, focused libraries
5. **Update go.mod**:
   ```bash
   go get github.com/example/package@latest
   go mod tidy
   ```

### Security

- Run `go mod tidy` to clean unused dependencies
- Use `govulncheck` to scan for vulnerabilities
- Keep dependencies up to date via Dependabot

## Documentation

### Code Comments

- **Exported symbols**: Must have doc comments
- **Complex logic**: Add inline comments explaining why
- **TODO/FIXME**: Include ticket reference

```go
// CircuitBreaker implements automatic failure detection and recovery
// for backend services. It transitions between CLOSED, OPEN, and HALF_OPEN
// states based on configurable failure thresholds.
type CircuitBreaker struct {
    // ...
}
```

### Documentation Files

- Update `README.md` for user-facing changes
- Add to `docs/` for detailed documentation
- Update `CHANGELOG.md` for all notable changes

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

### Communication

- Be respectful and professional
- Assume good intentions
- Provide constructive feedback
- Be patient with reviewers and contributors

## Recognition

Contributors will be:
- Listed in release notes
- Credited in CHANGELOG.md
- Added to CONTRIBUTORS.md (if maintaining)

## Questions?

If you have questions not covered here:
1. Check existing issues and discussions
2. Open a new discussion
3. Reach out to maintainers

---

Thank you for contributing to ShieldX! 🛡️
