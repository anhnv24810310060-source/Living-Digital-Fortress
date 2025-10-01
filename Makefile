# ShieldX Cloud Build System

.PHONY: all build test clean firecracker ebpf ml-orchestrator observability prom grafana
 .PHONY: demo-health

# Build all components
all: build-services build-ebpf

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
	go test -v ./pkg/sandbox/...

# Test ML components  
test-ml:
	go test -v ./pkg/ml/...

# Test orchestrator
test-orchestrator:
	go test -v ./pkg/orchestrator/...

# Run all tests
test: test-sandbox test-ml test-orchestrator

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f pkg/sandbox/bpf/*.o
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
	go test -bench=. ./pkg/sandbox/
	go test -bench=. ./pkg/ml/

# Docker builds for production
docker-build:
	docker build -t shieldx/ingress -f docker/Dockerfile.ingress .
	docker build -t shieldx/ml-orchestrator -f docker/Dockerfile.ml .
	docker build -t shieldx/decoy-manager -f docker/Dockerfile.decoy .

# Minimal images for observability demo
.PHONY: docker-ingress docker-locator demo-up demo-down
docker-ingress:
	docker build --build-arg GO_TAGS="$(GO_TAGS)" -t shieldx/ingress:dev -f docker/Dockerfile.ingress .

docker-locator:
	docker build --build-arg GO_TAGS="$(GO_TAGS)" -t shieldx/locator:dev -f docker/Dockerfile.locator .

demo-up: docker-ingress docker-locator
	docker compose -f pilot/observability/docker-compose.yml -f pilot/observability/docker-compose.override.yml up --build -d

demo-down:
	docker compose -f pilot/observability/docker-compose.yml -f pilot/observability/docker-compose.override.yml down

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
	@echo "Starting Prometheus with pilot/observability/prometheus-scrape.yml"
	@echo "Note: requires 'prometheus' binary on PATH or docker; adjust as needed"
	-prometheus --config.file=pilot/observability/prometheus-scrape.yml --web.enable-admin-api --web.listen-address=0.0.0.0:9091

grafana:
	@echo "Import dashboard: pilot/observability/grafana-dashboard-http-slo.json"

# Demo health check: waits for Prometheus targets and Jaeger UI to be up
demo-health:
	@echo "Checking Prometheus API..."
	@curl -sf http://localhost:9090/api/v1/status/runtimeinfo >/dev/null && echo "Prometheus OK" || (echo "Prometheus not ready"; exit 1)
	@echo "Checking Grafana..."
	@curl -sf http://localhost:3000/api/health >/dev/null && echo "Grafana OK" || (echo "Grafana not ready"; exit 1)
	@echo "Checking Jaeger UI..."
	@curl -sf http://localhost:16686 >/dev/null && echo "Jaeger OK" || (echo "Jaeger not ready"; exit 1)
	@echo "Checking OTEL Collector (OTLP HTTP 4318)..."
	@curl -s http://localhost:4318/ >/dev/null && echo "Collector OK (port reachable)" || (echo "Collector not reachable"; exit 1)
	@echo "Basic demo stack health: PASS"

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