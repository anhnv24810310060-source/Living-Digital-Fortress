### Prerequisites

**Required:**

* Go `1.22` or higher
* Docker `24.0+` & Docker Compose `2.20+`
* `make`
* `git`

**Recommended:**

* 4GB+ RAM
* Linux Kernel `5.10+` with KVM support (`/dev/kvm`) for the Guardian service.

### Quick Start Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/shieldx-bot/shieldx.git
    cd shieldx
    ```

2.  **Install dependencies:**

    ```bash
    go mod download && go mod verify
    ```

3.  **Build all service binaries:**

    ```bash
    make build
    ```

4.  **Start infrastructure services (PostgreSQL, Redis):**

    ```bash
    docker compose up -d postgres redis
    ```

5.  **Run database migrations:**

    ```bash
    make migrate-up
    ```

6.  **Start all ShieldX services:**

    ```bash
    make run-all
    ```

7.  **Verify that services are running:**

    ```bash
    make health-check
    ```

    **Expected Response from each healthy service:**

    ```json
    {
      "status": "healthy",
      "version": "0.1.0",
      "timestamp": "2025-10-08T10:00:00Z"
    }
    ```

-----

## 🛠️ Service Setup Guides

For detailed setup, configuration, and API documentation for each microservice, please refer to the `README.md` within its respective directory. The following are quick-start summaries.

##1. Orchestrator Service (`:8080`) 

**Purpose:** Central routing and policy evaluation engine.

**Setup:**

```bash
cd services/orchestrator
``
# Create .env file for configuration
```bash
cat > .env << EOF
ORCHESTRATOR_PORT=8080
REDIS_HOST=localhost
REDIS_PORT=6379
OPA_BUNDLE_URL=http://localhost:8181/bundles/latest
LOG_LEVEL=info
EOF
```
# Run the service
```bash
go run cmd/server/main.go
```

> See more: [`services/orchestrator/README.md`](https://www.google.com/search?q=services/orchestrator/README.md)

 ##2. Ingress Service (`:8081`) 

**Purpose:** Traffic gateway with rate limiting and filtering.

**Setup:**

```bash
cd services/ingress
```
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

 3. Guardian Service (`:9090`) 

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

 4. Credits Service (`:5004`) 
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

 

-----
