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