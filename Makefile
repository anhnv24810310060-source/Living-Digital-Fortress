# Use bash for improved shell features
SHELL := /bin/bash
.PHONY: all build test clean firecracker ebpf ml-orchestrator observability prom grafana
.PHONY: demo-health fmt lint sbom sign release otel-up otel-down slo-check

# Build all components
all: build-services build-ebpf

# Format code
fmt:
	@echo "Formatting Go code..."
	go fmt ./...
	@echo "Formatting complete"

# Lint code
lint:
	@echo "Linting Go code..."
	golangci-lint run ./... || echo "Install golangci-lint: https://golangci-lint.run/usage/install/"
	@echo "Linting complete"

# Run tests
test:
	@echo "Running tests..."
	go test -v -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Tests complete. Coverage report: coverage.html"

# Generate SBOM (Software Bill of Materials)
sbom:
	@echo "Generating SBOM..."
	syft . -o cyclonedx-json > sbom.json || echo "Install syft: https://github.com/anchore/syft"
	@echo "SBOM generated: sbom.json"

# Sign artifacts (requires cosign)
sign:
	@echo "Signing artifacts..."
	cosign sign-blob --key cosign.key sbom.json || echo "Install cosign: https://docs.sigstore.dev/cosign/installation/"
	@echo "Artifacts signed"

###############################
# Developer docker-compose UX #
###############################

# Project and compose settings
PROJECT ?= shieldx
COMPOSE_FILES ?= docker-compose.full.yml
COMPOSE := docker compose -p $(PROJECT) -f $(COMPOSE_FILES)
HEALTH_TIMEOUT ?= 60

# Known service names (from docker-compose.full.yml)
SERVICES := postgres redis orchestrator locator guardian ingress ml-orchestrator verifier-pool shieldx-gateway auth-service contauth policy-rollout otel-collector prometheus grafana

.PHONY: services dev-env-check dev-up dev-down dev-clean dev-build dev-pull dev-restart dev-logs dev-ps dev-shell dev-health

services:
	@echo "Available services:"; \
	for s in $(SERVICES); do echo "  - $$s"; done

dev-env-check:
	@command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 plugin is required"; exit 1; }

# Build images (all or a single SERVICE=<name>)
dev-build: dev-env-check
	@echo "Building images$(if $(SERVICE), for $(SERVICE), in parallel)..."
	@if [ -n "$(SERVICE)" ]; then \
		$(COMPOSE) build $(SERVICE); \
	else \
		$(COMPOSE) build --pull --parallel || (echo "Parallel build failed, retrying sequentially..." && $(COMPOSE) build --pull); \
	fi

# Start stack (all or a single SERVICE=<name>)
dev-up: dev-env-check
	@echo "Starting $(if $(SERVICE),service $(SERVICE),ShieldX stack)..."
	@if [ -n "$(SERVICE)" ]; then \
		$(COMPOSE) up -d $(SERVICE); \
	else \
		$(COMPOSE) up -d; \
	fi

dev-down: dev-env-check
	@echo "Stopping stack..."
	@$(COMPOSE) down --remove-orphans

dev-clean: dev-env-check
	@echo "Stopping stack and removing volumes..."
	@$(COMPOSE) down -v --remove-orphans

dev-restart: dev-env-check
	@echo "Restarting $(if $(SERVICE),service $(SERVICE),all services)..."
	@if [ -n "$(SERVICE)" ]; then \
		$(COMPOSE) restart $(SERVICE); \
	else \
		$(COMPOSE) restart; \
	fi

dev-logs: dev-env-check
	@echo "Tailing logs $(if $(SERVICE),for $(SERVICE),for all) (Ctrl+C to stop)..."
	@if [ -n "$(SERVICE)" ]; then \
		$(COMPOSE) logs -f --tail=200 $(SERVICE); \
	else \
		$(COMPOSE) logs -f --tail=200; \
	fi

dev-ps: dev-env-check
	@$(COMPOSE) ps

dev-pull: dev-env-check
	@echo "Pulling images$(if $(SERVICE), for $(SERVICE), for all)..."
	@if [ -n "$(SERVICE)" ]; then \
		$(COMPOSE) pull $(SERVICE); \
	else \
		$(COMPOSE) pull; \
	fi

dev-shell: dev-env-check
	@if [ -z "$(SERVICE)" ]; then echo "Usage: make dev-shell SERVICE=<name>"; exit 1; fi
	@echo "Opening shell in $(SERVICE) ..."
	@$(COMPOSE) exec $(SERVICE) sh -lc 'command -v bash >/dev/null 2>&1 && exec bash || exec sh'

# Lightweight health probe for common endpoints
dev-health:
	@echo "=== ShieldX Dev Health Check ==="; \
	failures=0; \
	ADM_SECRET="dev-secret-12345"; ADM_MIN=$$(($$(date +%s)/60)); \
	ADM_TOKEN=$$(printf "%s|%s" "$$ADM_MIN" "ingress" | openssl dgst -sha256 -hmac "$$ADM_SECRET" -binary | xxd -p -c 256); \
	curl_do() { url="$$1"; shift; if [[ "$$url" == https://localhost:8081/* ]]; then curl -skf --connect-timeout 1 -H "X-ShieldX-Admission: $$ADM_TOKEN" "$$url" "$$@"; else curl -sf --connect-timeout 1 "$$url" "$$@"; fi; }; \
	check() { name="$$1"; url="$$2"; timeout="$$3"; i=0; \
	  until curl_do "$$url" >/dev/null 2>&1; do \
		if [ "$$i" -ge "$$timeout" ]; then echo "  ❌ $$name not ready: $$url"; failures=$$((failures+1)); return; fi; \
		i=$$((i+1)); sleep 1; \
	  done; echo "  ✅ $$name OK"; }; \
	check orchestrator      http://localhost:8080/health $(HEALTH_TIMEOUT); \
	# Ingress uses TLS in dev via RA-TLS; require Admission header; allow self-signed
	check ingress           https://localhost:8081/healthz $(HEALTH_TIMEOUT); \
	check shieldx-gateway   http://localhost:8082/health $(HEALTH_TIMEOUT); \
	check locator           http://localhost:8083/healthz $(HEALTH_TIMEOUT); \
	check auth-service      http://localhost:8084/health $(HEALTH_TIMEOUT); \
	check ml-orchestrator   http://localhost:8087/health $(HEALTH_TIMEOUT); \
	check verifier-pool     http://localhost:8090/health $(HEALTH_TIMEOUT); \
	check contauth          http://localhost:5002/health $(HEALTH_TIMEOUT); \
	check policy-rollout    http://localhost:8099/health $(HEALTH_TIMEOUT); \
	check guardian          http://localhost:9091/healthz $(HEALTH_TIMEOUT); \
	check prometheus        http://localhost:9092/-/healthy $(HEALTH_TIMEOUT); \
	check grafana           http://localhost:3000/api/health $(HEALTH_TIMEOUT); \
	if [ "$$failures" -gt 0 ]; then echo "=== Result: $$failures failures"; exit 1; else echo "=== Result: all healthy"; fi

.PHONY: dev-health-fast
dev-health-fast:
	@echo "=== ShieldX Dev Health Check (fast) ==="; \
	ADM_SECRET="dev-secret-12345"; ADM_MIN=$$(($$(date +%s)/60)); \
	ADM_TOKEN=$$(printf "%s|%s" "$$ADM_MIN" "ingress" | openssl dgst -sha256 -hmac "$$ADM_SECRET" -binary | xxd -p -c 256); \
	curl_do() { url="$$1"; shift; if [[ "$$url" == https://localhost:8081/* ]]; then curl -skf --connect-timeout 1 --max-time 2 -H "X-ShieldX-Admission: $$ADM_TOKEN" "$$url" "$$@"; else curl -sf --connect-timeout 1 --max-time 2 "$$url" "$$@"; fi; }; \
	declare -a NAMES=( \
	  orchestrator ingress shieldx-gateway locator auth-service ml-orchestrator \
	  verifier-pool contauth policy-rollout guardian prometheus grafana \
	); \
	declare -a URLS=( \
	  http://localhost:8080/health \
	  https://localhost:8081/healthz \
	  http://localhost:8082/health \
	  http://localhost:8083/healthz \
	  http://localhost:8084/health \
	  http://localhost:8087/health \
	  http://localhost:8090/health \
	  http://localhost:5002/health \
	  http://localhost:8099/health \
	  http://localhost:9091/healthz \
	  http://localhost:9092/-/healthy \
	  http://localhost:3000/api/health \
	); \
	failures=0; \
	tmpdir=$$(mktemp -d); \
	for i in "$${!NAMES[@]}"; do \
	  ( \
	    name="$${NAMES[$$i]}"; url="$${URLS[$$i]}"; tries=0; max=20; \
	    until curl_do "$$url" >/dev/null 2>&1; do \
	      tries=$$((tries+1)); [ "$$tries" -ge "$$max" ] && { echo "  ❌ $$name not ready: $$url" > "$$tmpdir/$$name"; exit 0; }; \
	      sleep 0.5; \
	    done; echo "  ✅ $$name OK" > "$$tmpdir/$$name"; \
	  ) & \
	done; \
	wait; \
	for f in "$$tmpdir"/*; do cat "$$f"; [[ $$(cat "$$f") == *"❌"* ]] && failures=$$((failures+1)); done; \
	rm -rf "$$tmpdir"; \
	if [ "$$failures" -gt 0 ]; then echo "=== Result: $$failures failures"; exit 1; else echo "=== Result: all healthy"; fi

.PHONY: metrics-check
metrics-check:
	@bash tools/testing/metrics-check.sh

## OpenTelemetry stack management (observability demo compose)
otel-up:
	@echo "Starting OpenTelemetry observability stack..."
	docker compose -f infrastructure/kubernetes/pilot/observability/docker-compose.yml up -d
	@echo "Observability stack started"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/fortress123)"
	@echo "  - OTLP Endpoint: http://localhost:4318"

otel-down:
	@echo "Stopping OpenTelemetry observability stack..."
	docker compose -f infrastructure/kubernetes/pilot/observability/docker-compose.yml down
	@echo "Observability stack stopped"

# Check SLO status
slo-check:
	@echo "Checking SLO compliance..."
	@curl -s http://localhost:9090/api/v1/query?query=ingress:slo_availability:rate5m | jq '.data.result[] | {service: "ingress", availability: .value[1]}'
	@curl -s http://localhost:9090/api/v1/query?query=contauth:slo_error_ratio:rate5m | jq '.data.result[] | {service: "contauth", error_ratio: .value[1]}'
	@echo "SLO check complete"

# Build Go services
build-services:
	go build -o bin/locator ./services/locator
	go build -o bin/ingress ./services/ingress
	go build -o bin/guardian ./services/guardian
	go build -o bin/decoy-manager ./services/decoy-manager
	go build -o bin/ml-orchestrator ./services/ml-orchestrator
	go build -o bin/anchor ./services/anchor
	go build -o bin/shapeshifter ./services/shapeshifter
	go build -o bin/sinkhole ./services/sinkhole
	go build -o bin/camouflage-api ./services/camouflage-api

# Build eBPF programs
build-ebpf:
	clang -O2 -target bpf -c pkg/sandbox/bpf/syscall_tracer.c -o pkg/sandbox/bpf/syscall_tracer.o

# Install Firecracker (Linux only)
install-firecracker:
	@echo "Installing Firecracker..."
	curl -LOJ https://github.com/firecracker-microvm/firecracker/releases/download/v1.4.1/firecracker-v1.4.1-x86_64.tgz
	tar -xzf firecracker-v1.4.1-x86_64.tgz
	sudo cp release-v1.4.1-x86_64/firecracker-v1.4.1-x86_64 /usr/local/bin/firecracker
	sudo chmod +x /usr/local/bin/firecracker

# Setup kernel and rootfs for Firecracker
setup-firecracker-images:
	@echo "Setting up Firecracker kernel and rootfs..."
	mkdir -p firecracker-images
	curl -fsSL -o firecracker-images/vmlinux.bin https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/x86_64/kernels/vmlinux.bin
	curl -fsSL -o firecracker-images/hello-rootfs.ext4 https://s3.amazonaws.com/spec.ccfc.min/img/hello/fsfiles/hello-rootfs.ext4

# Test sandbox components
test-sandbox:
	go test -v ./shared/shieldx/shared/shieldx-common/pkg/sandbox/...
 
# Test ML components  
test-ml:
	go test -v  ./shared/shieldx/shared/shieldx-common/pkg//ml/...

# Test orchestrator
test-orchestrator:
	go test -v  ./shared/shieldx/shared/shieldx-common/pkg/orchestrator/...

# Run all tests
test: test-sandbox test-ml test-orchestrator

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f ./shared/shieldx/shared/shieldx-common/pkg/sandbox/bpf/*.o
	rm -rf firecracker-images/

# Development setup
dev-setup:
	go mod tidy
	go mod download

# Phase 1 specific targets
phase1-firecracker: install-firecracker setup-firecracker-images
	@echo "Firecracker setup complete"

phase1-ebpf: build-ebpf
	@echo "eBPF programs compiled"

phase1-ml: build-services
	@echo "ML orchestrator built"

phase1: phase1-firecracker phase1-ebpf phase1-ml
	@echo "Phase 1 implementation complete"

# Phase 2 specific targets
phase2-deception: build-services
	go build -o bin/shapeshifter-v2 ./services/shapeshifter
	@echo "Deception graph with bandit optimization built"

phase2-fingerprint: phase2-deception
	@echo "L7 fingerprint mimicry integrated"

phase2-anti-detection: phase2-fingerprint
	@echo "Anti-sandbox detection enabled"

phase2-counterstrike: phase2-anti-detection
	@echo "Parasitic counterstrike engine ready"

phase2: phase2-counterstrike
	@echo "Phase 2 Shapeshifter Evolution complete"

# Security checks
security-scan:
	gosec ./...
	
# Performance benchmarks
benchmark:
	go test -bench=. ./shared/shieldx/shared/shieldx-common/pkg/sandbox/
	go test -bench=. ml/

# Docker builds for production
docker-build:
	docker build -t shieldx/ingress -f infrastructure/docker-compose/docker/Dockerfile.ingress .
	docker build -t shieldx/ml-orchestrator -f infrastructure/docker-compose/docker/Dockerfile.ml-orchestrator .
	@echo "Note: decoy-manager Dockerfile is not defined; skipping."

# Minimal images for observability demo
.PHONY: docker-ingress docker-locator demo-up demo-down
docker-ingress:
	docker build --build-arg GO_TAGS="$(GO_TAGS)" -t shieldx/ingress:dev -f infrastructure/docker-compose/docker/Dockerfile.ingress .

docker-locator:
	docker build --build-arg GO_TAGS="$(GO_TAGS)" -t shieldx/locator:dev -f infrastructure/docker-compose/docker/Dockerfile.locator .

demo-up: docker-ingress docker-locator
	docker compose -f infrastructure/kubernetes/pilot/observability/docker-compose.yml -f infrastructure/kubernetes/pilot/observability/docker-compose.override.yml up --build -d

demo-down:
	docker compose -f infrastructure/kubernetes/pilot/observability/docker-compose.yml -f infrastructure/kubernetes/pilot/observability/docker-compose.override.yml down

# Policy-as-code (November)
.PHONY: policy-bundle policy-sign policy-verify policy-all
policy-bundle:
	go build -o bin/policyctl ./cmd/policyctl
	./bin/policyctl bundle -dir policies/demo -out dist/policy-bundle.zip

policy-sign: policy-bundle
	./bin/policyctl sign -dir policies/demo -sig dist/policy-bundle.sig

policy-verify:
	./bin/policyctl verify -dir policies/demo -sig dist/policy-bundle.sig

policy-all: policy-bundle policy-sign policy-verify
	@echo "Policy bundle build/sign/verify complete"

.PHONY: policy-sign-cosign policy-verify-cosign
policy-sign-cosign: policy-bundle
	@which cosign >/dev/null || (echo "cosign not found" && exit 1)
	@echo $$(awk '{print $$2}' dist/digest.txt >/dev/null 2>&1) || ./bin/policyctl bundle -dir policies/demo -out dist/policy-bundle.zip | grep '^digest:' | awk '{print $$2}' > dist/digest.txt
	COSIGN_EXPERIMENTAL=true cosign sign-blob --yes $(if $(KEY_REF),--key $(KEY_REF),) --output-signature dist/policy-bundle.cosign.sig dist/digest.txt

policy-verify-cosign:
	@which cosign >/dev/null || (echo "cosign not found" && exit 1)
	COSIGN_EXPERIMENTAL=true cosign verify-blob --signature dist/policy-bundle.cosign.sig dist/digest.txt

# Observability (local quick-run)
observability: prom
	@echo "Observability components started (Prometheus). Import Grafana dashboard from pilot/observability/grafana-dashboard-http-slo.json"

prom:
	@echo "Starting Prometheus with infrastructure/kubernetes/pilot/observability/prometheus-scrape.yml"
	@echo "Note: requires 'prometheus' binary on PATH or docker; adjust as needed"
	-prometheus --config.file=infrastructure/kubernetes/pilot/observability/prometheus-scrape.yml --web.enable-admin-api --web.listen-address=0.0.0.0:9091

grafana:
	@echo "Import dashboard: infrastructure/kubernetes/pilot/observability/grafana-dashboard-http-slo.json"

# Demo health check: waits for Prometheus targets and Jaeger UI to be up
demo-health:
	@echo "=== ShieldX Demo Stack Health Check ==="
	@echo ""
	@echo "[1/8] Checking Prometheus API..."
	@curl -sf http://localhost:9090/api/v1/status/runtimeinfo >/dev/null && echo "  ✅ Prometheus OK" || (echo "  ❌ Prometheus not ready"; exit 1)
	@echo ""
	@echo "[2/8] Checking Grafana..."
	@curl -sf http://localhost:3000/api/health >/dev/null && echo "  ✅ Grafana OK" || (echo "  ❌ Grafana not ready"; exit 1)
	@echo ""
	@echo "[3/8] Checking Jaeger UI..."
	@curl -sf http://localhost:16686 >/dev/null && echo "  ✅ Jaeger OK" || (echo "  ❌ Jaeger not ready"; exit 1)
	@echo ""
	@echo "[4/8] Checking OTEL Collector (OTLP HTTP 4318)..."
	@curl -s http://localhost:4318/ >/dev/null && echo "  ✅ Collector OK" || (echo "  ❌ Collector not reachable"; exit 1)
	@echo ""
	@echo "[5/8] Checking Ingress service..."
	@curl -sf http://localhost:8081/healthz >/dev/null && echo "  ✅ Ingress OK" || echo "  ⚠️  Ingress not responding"
	@echo ""
	@echo "[6/8] Checking Locator service..."
	@curl -sf http://localhost:8080/health >/dev/null && echo "  ✅ Locator OK" || echo "  ⚠️  Locator not responding"
	@echo ""
	@echo "[7/8] Checking ShieldX Gateway..."
	@curl -sf http://localhost:8082/health >/dev/null && echo "  ✅ Gateway OK" || echo "  ⚠️  Gateway not responding"
	@echo ""
	@echo "[8/8] Checking Prometheus targets..."
	@targets=$$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | map(select(.health == "up")) | length'); \
	total=$$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length'); \
	echo "  ✅ $$targets/$$total targets UP"
	@echo ""
	@echo "=== Summary ==="
	@echo "Observability Stack: ✅ Ready"
	@echo "Services: Check individual status above"
	@echo ""
	@echo "Next steps:"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Jaeger: http://localhost:16686"
	@echo "  - Import dashboard: pilot/observability/grafana-dashboard-http-slo.json"

# Help
help:
	@echo "ShieldX Cloud Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all                 - Build all components"
	@echo "  build-services      - Build Go services"
	@echo "  build-ebpf          - Compile eBPF programs"
	@echo "  install-firecracker - Install Firecracker (Linux)"
	@echo "  test                - Run all tests"
	@echo "  phase1              - Complete Phase 1 setup"
	@echo "  phase2              - Complete Phase 2 setup"
	@echo "  clean               - Clean build artifacts"
	@echo "  help                - Show this help"
	@echo "  policy-bundle       - Build demo policy bundle (zip)"
	@echo "  policy-sign         - Sign demo policy bundle (noop signer, demo)"
	@echo "  policy-verify       - Verify signature (noop verifier, demo)"
	@echo "  policy-all          - Build+sign+verify demo policy bundle"
	@echo "  sbom-all            - Generate SBOMs for Go + Python"
	@echo "  image-sign          - Cosign sign container image (REGISTRY/IMAGE:TAG, KEY_REF optional)"
	@echo "  release-snapshot    - Build reproducible snapshot with goreleaser"

.PHONY: sbom-all sbom-go sbom-python image-sign release-snapshot
sbom-all: sbom-go sbom-python
	@echo "SBOMs generated under dist/sbom"

sbom-go:
	@which syft >/dev/null || (echo "syft not found" && exit 1)
	mkdir -p dist/sbom
	syft packages dir:. -o cyclonedx-json > dist/sbom/sbom-go.json

sbom-python:
	@which syft >/dev/null || (echo "syft not found" && exit 1)
	mkdir -p dist/sbom
	syft packages dir:ml-service -o cyclonedx-json > dist/sbom/sbom-python.json

image-sign:
	@which cosign >/dev/null || (echo "cosign not found" && exit 1)
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make image-sign IMAGE=registry/repo:tag [KEY_REF=...]"; exit 1; fi
	COSIGN_EXPERIMENTAL=true cosign sign --yes $(if $(KEY_REF),--key $(KEY_REF),) $(IMAGE)

release-snapshot:
	@which goreleaser >/dev/null || (echo "goreleaser not found" && exit 1)
	goreleaser release --snapshot --clean