 ## 🧑‍💻 Developer Environment (for Contributors)

This section helps you quickly launch the entire system in a development environment using Docker Compose and control each service easily via Makefile. This is recommended for most contributors.

### Requirements

- Docker 24+ and Docker Compose v2 (can run `docker compose` command)
- Make, Git; Go only needed if you build outside of a container
- Linux with 4GB+ RAM recommended; Guardian requires Linux + KVM

Optional: load default environment variables

```bash
cp -n .env.dev .env || true
```

### Quick launch (full stack)

# (optional) build full image
```bash
make dev-build
```

# start all services
```bash
make dev-up
```
# wait for endpoints to be ready
```bash
make dev-health
```
Default endpoints after launch:

- Orchestrator: http://localhost:8080/health
- Ingress: http://localhost:8081/health
- Gateway: http://localhost:8082/health
- Locator: http://localhost:8083/healthz
- Auth Service: http://localhost:8084/health
- ML Orchestrator: http://localhost:8087/health
- Verifier Pool: http://localhost:8090/health
- ContAuth: http://localhost:5002/health
- Policy Rollout: http://localhost:8099/health
- Guardian: http://localhost:9090/healthz
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/fortress123)

### Work with each service

You can build/run/log/restart a specific service using variables `SERVICE`:

# build a service
```bash
make dev-build SERVICE=ingress
```
# start a service (dependencies should already be running)
```bash
make dev-up SERVICE=ingress
```
# view realtime logs
```bash
make dev-logs SERVICE=ingress
```
# quick restart
```bash
make dev-restart SERVICE=ingress
```
# open a shell in the container (bash if available, fallback sh)
```bash
make dev-shell SERVICE=ingress
```

View a list of valid services:

```bash
make services
```

Some commonly used default ports:

| Service | Port |
| --- | --- |
| orchestrator | 8080 |
| ingress | 8081 |
| shieldx-gateway | 8082 |
| locator | 8083 |
| auth-service | 8084 |
| ml-orchestrator | 8087 |
| verifier-pool | 8090 |
| contauth | 5002 |
| policy-rollout | 8099 |
| guardian | 9090 |
| prometheus | 9090 |
| grafana | 3000 |
| otel-collector (OTLP) | 4318 |
 ### Observability (optional)

- Launch basic observability stack (Prometheus, Grafana, OTEL Collector):

```bash
make otel-up
```

- Quick demo with compose override:

```bash
make demo-up
```

Shutdown/Revoke:

```bash
make otel-down
make demo-down
```

### Stop and cleanup

```bash
# stop stack, keep volumes
make dev-down
```
# stop and delete volumes (clean up data)
```bash
make dev-clean
```

### Troubleshooting

- "Docker Compose v2 plugin is required": need `docker compose` (not `docker-compose`). Install new Docker/Compose v2.
- Insufficient Docker privileges: add user to group `docker` or run with `sudo`.

- Busy port: change port in `docker-compose.full.yml` or stop the process occupying the port.

- Build error: try `make dev-build SERVICE=<name>` to localize; check Dockerfile at `infrastructure/docker-compose/docker/`.

- Guardian requires Linux + KVM: if `/dev/kvm` is not available, skip Guardian and develop other parts first.

----- 
## 🛠️ Service Setup Guides

To avoid duplication/inconsistent instructions between services, please see the separate READMEs in each folder under `services/` and `shared/`. General instructions for developer environments are above and are the recommended way to run the entire system.
 
 2. Ingress Service (`:8081`) 

**Purpose:** Traffic gateway with rate limiting and filtering.

**Setup:**

---
# Create .env file for configuration
 ```bash
cat > .env << EOF
INGRESS_PORT=8081
REDIS_HOST=localhost
RATE_LIMIT_PER_MINUTE=1000
ENABLE_QUIC=true
EOF
```
# Run the service
 ```bash
go run cmd/server/main.go
 ```

> See more: [`services/ingress/README.md`](https://www.google.com/search?q=services/ingress/README.md)

 
## 3. Guardian Service (`:9090`)
 

**Purpose:** Sandbox execution with Firecracker MicroVMs.

**Requirements:** Linux kernel 5.10+, KVM enabled, and root privileges (`sudo`).

**Setup:**

 ```bash
cd services/guardian
```
# Verify KVM support

```bash
ls -l /dev/kvm && lsmod | grep kvm
```

# Create .env file with paths to your kernel and rootfs

```bash
cat > .env << EOF
GUARDIAN_PORT=9090
FIRECRACKER_KERNEL=/path/to/vmlinux
FIRECRACKER_ROOTFS=/path/to/rootfs.ext4
SANDBOX_TIMEOUT=30
MAX_MEMORY_MB=512
EOF
```

# Run with elevated privileges

```bash
sudo go run cmd/server/main.go
```

> **Note:** Guardian requires Linux. On Windows/macOS, it will run in a limited stub mode.
> See more: [`services/guardian/README.md`](https://www.google.com/search?q=services/guardian/README.md)


 ## 4. Credits Service (`:5004`) 
 
**Purpose:** Resource consumption tracking and billing.

**Setup:**

```bash
cd services/shieldx-credits
```
# Ensure PostgreSQL is running via docker-compose

# Create .env file
 ```bash
cat > .env << EOF
CREDITS_PORT=5004
CREDITS_DB_HOST=localhost
CREDITS_DB_PORT=5432
CREDITS_DB_USER=credits_user
CREDITS_DB_PASSWORD=credits_pass
CREDITS_DB_NAME=credits
CREDITS_DB_SSL_MODE=disable
EOF
```
# Run database migrations before starting
# (The 'make migrate-up' command handles this)

# Run the service
 ```bash
go run cmd/server/main.go
 ```

> See more: [`services/shieldx-credits/CREDITS-SERVICE.md`](https://www.google.com/search?q=services/shieldx-credits/CREDITS-SERVICE.md)

 
