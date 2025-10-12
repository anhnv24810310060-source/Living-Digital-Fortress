Thiết lập dưới đây là con đường khuyến nghị cho contributors: dùng Docker Compose + Makefile để khởi chạy toàn bộ hệ thống hoặc từng service, tránh lỗi và cấu hình thừa.

### Yêu cầu

- Docker 24+ và Docker Compose v2 (có thể chạy lệnh `docker compose`)
- make, git; Go không bắt buộc nếu build trong container
- Khuyến nghị Linux với 4GB+ RAM; Guardian yêu cầu Linux + KVM (`/dev/kvm`)

### Bắt đầu nhanh

1) Clone mã nguồn

```bash
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx
```

2) (Tùy chọn) nạp biến môi trường mặc định

```bash
cp -n .env.dev .env || true
```

3) Khởi chạy toàn bộ stack

```bash
# (tùy chọn) build tất cả image
make dev-build

# khởi chạy toàn bộ services
make dev-up

# chờ các endpoint sẵn sàng
make dev-health
```

Sau khi khởi chạy thành công, truy cập nhanh:

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

### Làm việc với từng service

```bash
# build 1 service
make dev-build SERVICE=ingress

# khởi chạy 1 service
make dev-up SERVICE=ingress

# xem log realtime
make dev-logs SERVICE=ingress

# restart nhanh
make dev-restart SERVICE=ingress

# vào shell trong container (bash nếu có)
make dev-shell SERVICE=ingress

# liệt kê trạng thái containers
make dev-ps
```

Liệt kê tên service hợp lệ:

```bash
make services
```

### Observability (tùy chọn)

```bash
# khởi chạy stack observability cơ bản
make otel-up

# demo nhanh với compose override
make demo-up

# tắt/thu hồi
make otel-down
make demo-down
```

### Dừng và dọn dẹp

```bash
# dừng stack, giữ dữ liệu volumes
make dev-down

# dừng và xóa volumes (dọn sạch dữ liệu)
make dev-clean
```

### Khắc phục sự cố thường gặp

- "Docker Compose v2 plugin is required": cần dùng `docker compose` (không phải `docker-compose`).
- Quyền Docker: thêm user vào group `docker` hoặc dùng `sudo`.
- Cổng bận: đổi cổng trong `docker-compose.full.yml` hoặc dừng tiến trình đang chiếm cổng.
- Build lỗi: thử `make dev-build SERVICE=<tên>` để khoanh vùng; kiểm tra Dockerfile tại `infrastructure/docker-compose/docker/`.
- Guardian yêu cầu Linux + KVM: nếu không có `/dev/kvm`, bỏ qua Guardian và phát triển các phần khác trước.

-----

## 🧑‍💻 Môi trường Developer (dành cho Contributors)

Phần này giúp bạn khởi chạy nhanh toàn bộ hệ thống trong môi trường phát triển bằng Docker Compose và điều khiển từng service dễ dàng qua Makefile. Cách này là khuyến nghị cho hầu hết contributors.

### Yêu cầu

- Docker 24+ và Docker Compose v2 (có thể chạy lệnh `docker compose`)
- Make, Git; Go chỉ cần nếu bạn build bên ngoài container
- Khuyến nghị Linux với 4GB+ RAM; Guardian yêu cầu Linux + KVM

Tùy chọn: nạp biến môi trường mặc định

```bash
cp -n .env.dev .env || true
```

### Khởi chạy nhanh (toàn bộ stack)

```bash
# (tùy chọn) build toàn bộ image
make dev-build

# khởi chạy toàn bộ services
make dev-up

# đợi các endpoint sẵn sàng
make dev-health
```

Các endpoint mặc định sau khi khởi chạy:

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

### Làm việc với từng service

Bạn có thể build/chạy/log/restart cho một service cụ thể bằng biến `SERVICE`:

```bash
# build một service
make dev-build SERVICE=ingress

# khởi chạy một service (các phụ thuộc nên đã chạy sẵn)
make dev-up SERVICE=ingress

# xem log realtime
make dev-logs SERVICE=ingress

# restart nhanh
make dev-restart SERVICE=ingress

# mở shell trong container (bash nếu có, fallback sh)
make dev-shell SERVICE=ingress
```

Xem danh sách service hợp lệ:

```bash
make services
```

Một số cổng mặc định thường dùng:

| Service | Cổng |
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

### Observability (tùy chọn)

- Khởi chạy stack observability cơ bản (Prometheus, Grafana, OTEL Collector):

```bash
make otel-up
```

- Demo nhanh với compose override:

```bash
make demo-up
```

Tắt/thu hồi:

```bash
make otel-down
make demo-down
```

### Dừng và dọn dẹp

```bash
# dừng stack, giữ volumes
make dev-down

# dừng và xóa volumes (dọn sạch dữ liệu)
make dev-clean
```

### Khắc phục sự cố thường gặp

- "Docker Compose v2 plugin is required": cần dùng `docker compose` (không phải `docker-compose`). Cài đặt Docker/Compose v2 mới.
- Không đủ quyền Docker: thêm user vào group `docker` hoặc chạy với `sudo`.
- Cổng bận: đổi cổng trong `docker-compose.full.yml` hoặc dừng tiến trình đang chiếm cổng.
- Build lỗi: thử `make dev-build SERVICE=<tên>` để khoanh vùng; kiểm tra Dockerfile tại `infrastructure/docker-compose/docker/`.
- Guardian yêu cầu Linux + KVM: nếu không có `/dev/kvm`, hãy bỏ qua Guardian và phát triển các phần khác trước.

-----

## 🛠️ Service Setup Guides

Để tránh trùng lặp/hướng dẫn sai khác giữa các service, vui lòng xem README riêng trong từng thư mục dưới `services/` và `shared/`. Hướng dẫn chung cho môi trường developer đã có ở mục trên và là cách khuyến nghị để chạy toàn bộ hệ thống.

\<details\>
\<summary\>\<b\>2. Ingress Service (`:8081`)\</b\>\</summary\>

**Purpose:** Traffic gateway with rate limiting and filtering.

**Setup:**

---
# Create .env file for configuration
cat > .env << EOF
INGRESS_PORT=8081
REDIS_HOST=localhost
RATE_LIMIT_PER_MINUTE=1000
ENABLE_QUIC=true
EOF

# Run the service
go run cmd/server/main.go
```

> See more: [`services/ingress/README.md`](https://www.google.com/search?q=services/ingress/README.md)

\</details\>

\<details\>
\<summary\>\<b\>3. Guardian Service (`:9090`)\</b\>\</summary\>

**Purpose:** Sandbox execution with Firecracker MicroVMs.

**Requirements:** Linux kernel 5.10+, KVM enabled, and root privileges (`sudo`).

**Setup:**

```bash
cd services/guardian

# Verify KVM support
ls -l /dev/kvm && lsmod | grep kvm

# Create .env file with paths to your kernel and rootfs
cat > .env << EOF
GUARDIAN_PORT=9090
FIRECRACKER_KERNEL=/path/to/vmlinux
FIRECRACKER_ROOTFS=/path/to/rootfs.ext4
SANDBOX_TIMEOUT=30
MAX_MEMORY_MB=512
EOF

# Run with elevated privileges
sudo go run cmd/server/main.go
```

> **Note:** Guardian requires Linux. On Windows/macOS, it will run in a limited stub mode.
> See more: [`services/guardian/README.md`](https://www.google.com/search?q=services/guardian/README.md)

\</details\>

\<details\>
\<summary\>\<b\>4. Credits Service (`:5004`)\</b\>\</summary\>

**Purpose:** Resource consumption tracking and billing.

**Setup:**

```bash
cd services/shieldx-credits

# Ensure PostgreSQL is running via docker-compose

# Create .env file
cat > .env << EOF
CREDITS_PORT=5004
CREDITS_DB_HOST=localhost
CREDITS_DB_PORT=5432
CREDITS_DB_USER=credits_user
CREDITS_DB_PASSWORD=credits_pass
CREDITS_DB_NAME=credits
CREDITS_DB_SSL_MODE=disable
EOF

# Run database migrations before starting
# (The 'make migrate-up' command handles this)

# Run the service
go run cmd/server/main.go
```

> See more: [`services/shieldx-credits/CREDITS-SERVICE.md`](https://www.google.com/search?q=services/shieldx-credits/CREDITS-SERVICE.md)

\</details\>
