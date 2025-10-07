ShieldX-Cloud Service Map (Design vs Repo)
=========================================

Tài liệu này đối chiếu nhanh giữa “Bản Thiết Kế Hệ Thống” và mã nguồn hiện tại, nêu rõ cổng dịch vụ, health/metrics và vị trí mã.

Lớp biên/điều phối
- Edge Workers (Cloudflare)
  - Repo: infra/cloudflare/
  - Ghi chú: Camouflage/JA3/TLS fingerprint xoay theo policy. Hướng dẫn triển khai trong infra/cloudflare/README.md.
- Load Balancer (HAProxy)
  - Repo: infra/haproxy/haproxy.cfg
  - Port: 80/443, Stats: 8404
- Web Console (React)
  - Repo: web/console/

Core services (theo thiết kế)
- Orchestrator (8080)
  - Thiết kế: định tuyến theo policy, /health, /route, /metrics, /policy.
  - Hiện trạng: vai trò được thực thi thông qua locator + policy engine.
    - Service thực tế: services/locator/ (port 8080)
    - Policy engine: pkg/policy/*, policyctl
  - Hướng xử lý: giữ locator là Orchestrator tạm; xem services/orchestrator/README.md.

- Ingress (8081)
  - Repo: services/ingress/
  - Health: GET /healthz
  - Metrics: GET /metrics

- Guardian (9090)
  - Repo: services/guardian/
  - Health: GET /healthz
  - Metrics: GET /metrics

- Credits (5004)
  - Repo: services/credits/
  - Health: GET /health
  - DB: PostgreSQL (khởi tạo qua infra/db/init-scripts/)

- Continuous Authentication – ContAuth (5002)
  - Repo: services/contauth/
  - Health: GET /health
  - DB: PostgreSQL

- Shadow Evaluation (5005)
  - Repo: services/shadow/
  - Health: GET /health
  - DB: PostgreSQL

Dữ liệu & Hạ tầng
- PostgreSQL/Redis/HAProxy/Backup
  - Compose: infra/docker-compose.data.yml
  - Scripts: infra/db/init-scripts/, infra/db/backup-scripts/

Quan sát & SLO
- Prometheus/Tempo/Grafana/OTel Collector
  - Repo: pilot/observability/*
  - Prometheus scrape: pilot/observability/prometheus-scrape.yml
  - Rules SLO: pilot/observability/rules/

Kiểm tra nhanh
- Script: scripts/healthcheck_core.sh để kiểm tra 6 dịch vụ cốt lõi theo cổng thiết kế.

Ghi chú: Khi cần “Orchestrator” tách riêng, ưu tiên dựng service mới tái sử dụng pkg/policy/* và API tương đương; hiện tại “locator@8080” đóng vai trò điều phối theo policy.