## 2025-09-30 — Khởi tạo lộ trình 12 tháng và chuẩn bị Tháng 10/2025 (Observability & SLO)

- Đã bổ sung vào `Lộ Trình Cải Tiến.md` mục "Lộ trình 12 tháng (10/2025 → 09/2026)" với kế hoạch chi tiết từng tháng.
- Tập trung triển khai ngay Tháng 10/2025 — Nền tảng quan sát và SLO cơ bản:
	- Thiết lập OpenTelemetry cho các dịch vụ Go và Python.
	- Tạo dashboard SLO (p95/p99 latency, error rate, RPS) và cảnh báo theo error budget.
	- Phạm vi tác động: `pkg/metrics/`, `cmd/*`, `services/ingress/`, `services/contauth/`, `services/verifier-pool/`, `services/ml-orchestrator/`, `services/shieldx-gateway/`, `ml-service/feature_store.py`.
	- Chỉ số chấp nhận: 95% endpoints có trace; 100% dịch vụ mục tiêu có metrics; theo dõi error budget liên tục 1 tuần.

- Rủi ro & giảm thiểu ban đầu:
	- Tăng overhead do instrumentation: bật sampling và batch exporter hợp lý, chỉ instrument đường nóng.
	- Không đồng nhất nhãn/metric: chuẩn hóa tên service và labels ngay từ `pkg/metrics/`.

- Việc tiếp theo (chuẩn bị PR):
	- Thêm skeleton OTel vào `pkg/metrics/` và wiring mẫu cho 2–3 dịch vụ đại diện.
	- Khởi tạo dashboard SLO tối thiểu và tài liệu hướng dẫn.

### Cập nhật mã nguồn đã thực hiện (Observability foundation)
- `pkg/metrics/metrics.go`:
	- Thêm Histogram và HTTPMetrics middleware (đo requests_total, errors_total, request_duration_seconds).
	- Mở rộng Registry để xuất counter/gauge/histogram theo chuẩn Prometheus text.
- `services/ingress/main.go`:
	- Bọc server bằng HTTP metrics middleware; tiếp tục phục vụ `/metrics` qua Registry hiện có.
- `services/guardian/main.go`:
	- Thêm HTTP metrics middleware; giữ nguyên `/metrics` qua Registry.
- `services/ml-orchestrator/main.go`:
	- Chuyển sang `http.ServeMux`, thêm Registry và `/metrics`; bọc middleware để thu thập HTTP metrics.
- `pilot/docs/kpi-dashboard.md`:
	- Ghi chú vận hành endpoints `/metrics` mới để dashboard kéo số liệu.
- `services/locator/main.go`:
	- Thêm HTTP metrics middleware; giữ nguyên `/metrics` qua Registry.
	- Cập nhật tài liệu KPI để thêm endpoint Locator.

## 2025-09-30 — Bổ sung instrumentation cho ContAuth và Verifier Pool

- `services/contauth/main.go`:
	- Chuyển sang `http.ServeMux`, thêm `pkg/metrics` Registry và `/metrics`.
	- Bọc middleware để thu thập *_http_* metrics mặc định.
- `services/verifier-pool/main.go`:
	- Chuyển sang `http.ServeMux`, thêm Registry và `/metrics`; bọc middleware.
- `pilot/docs/kpi-dashboard.md`:
	- Cập nhật danh sách endpoints để bao phủ ContAuth và Verifier Pool.

Lưu ý build: Build toàn repo vẫn yêu cầu đồng bộ go.sum của một số module không liên quan phạm vi (docker, ebpf, quic, jwt…). Các thay đổi lần này không thêm phụ thuộc mới ngoài `pkg/metrics`, nên an toàn để merge theo từng dịch vụ.

## 2025-09-30 — Metrics cho ML Service (Python)

- `ml-service/feature_store.py`:
	- Thêm `/metrics` sử dụng `prometheus_client`; đếm requests_total và đo duration theo endpoint/method.
	- Trang bị decorator `track_metrics` để bọc các route `/process`, `/training-data`, `/health`.
- `ml-service/requirements.txt`:
	- Bổ sung `prometheus-client==0.20.0`.
- `pilot/docs/kpi-dashboard.md`:
	- Cập nhật thêm endpoint metrics cho ML Service.

Ghi chú: Cần cài dependencies Python để kích hoạt metrics ML service.

## 2025-09-30 — Artefacts cho SLO Dashboard (Prometheus + Grafana)

- Thêm `pilot/observability/prometheus-scrape.yml` — cấu hình scrape mẫu cho các services được instrument.
- Thêm `pilot/observability/grafana-dashboard-http-slo.json` — dashboard mẫu theo dõi error rate (%) và p95 latency cho Ingress, ContAuth, Verifier Pool, ML Orchestrator; kèm biểu đồ throughput requests theo service.
- Cập nhật KPI docs trước đó đã liệt kê endpoints `/metrics`; dashboard này sử dụng các metric name mặc định vừa bổ sung.

### Bổ sung
- `pilot/observability/alert-rules.yml` — rule cảnh báo mẫu: error rate Ingress >1% (critical), p95 latency ContAuth >500ms (warning).
- `Makefile` — thêm targets `observability`, `prom`, `grafana` để chạy nhanh Prometheus và hướng dẫn import dashboard Grafana.

## 2025-09-30 — Tracing skeleton (OpenTelemetry) + Compose stack

- `pkg/observability/otel/otel.go`:
	- Hàm `InitTracer(serviceName)` cấu hình OTLP/HTTP exporter (endpoint từ `OTEL_EXPORTER_OTLP_ENDPOINT`), no-op nếu không đặt env.
- `services/ingress/main.go`, `services/locator/main.go`:
	- Gọi `InitTracer()` sớm trong `main()` và `defer` shutdown; không phá vỡ nếu collector vắng mặt.
- `pilot/observability/otel/collector-config.yml`:
	- Collector nhận OTLP/HTTP và export `debug` (in ra log) cho mục đích demo.
- `pilot/observability/docker-compose.yml`:
	- Stack tối thiểu: Prometheus, Grafana, OTEL Collector (4318). Import dashboard JSON để xem SLO, set env `OTEL_EXPORTER_OTLP_ENDPOINT` trong service để bật tracing.

### Bổ sung (per-path metrics + tracing demo override)
- `pkg/metrics/metrics.go`:
	- Thêm LabeledCounter/Histogram và emit metrics theo method/path: *_http_requests_by_path_total, *_http_request_duration_by_path_seconds (cảnh báo cardinality khi dùng rộng rãi).
- `pilot/observability/docker-compose.override.yml`:
	- Ví dụ chạy `ingress` và `locator` với `OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4318` để demo tracing end-to-end.

Lưu ý: chưa chạy `go mod download` toàn repo để tránh thay đổi ngoài phạm vi; build tổng thể sẽ yêu cầu đồng bộ `go.sum`. Các file thay đổi biên dịch sạch theo kiểm tra tĩnh nội bộ.

## 2025-10-01 — Dockerfiles demo + OTEL build tag

- Thêm `docker/Dockerfile.ingress` và `docker/Dockerfile.locator` (multi-stage, distroless, nonroot). Hỗ trợ `--build-arg GO_TAGS="otelotlp"` để bật exporter thật.
- `Makefile`: thêm các target `docker-ingress`, `docker-locator`, `demo-up`, `demo-down` để build images và chạy nhanh stack demo (`pilot/observability/docker-compose*.yml`).
- `pkg/observability/otel/`: giữ `InitTracer` mặc định no-op; thêm biến thể thực sự trong `otel_otlp.go` (build tag `otelotlp`) dùng OTLP/HTTP (`otlptracehttp`).
- Kết quả: có thể chạy Prometheus/Grafana/Collector + ingress/locator demo. Muốn bật tracing: build image với `GO_TAGS=otelotlp` và set `OTEL_EXPORTER_OTLP_ENDPOINT`.

## 2025-10-01 — Bật tracing cho demo, thêm ShieldX Gateway vào compose, cố định scrape và build tags

- Tracing và build tags (Go):
	- Thêm build constraint cho biến thể no-op để tránh xung đột khi bật `-tags otelotlp`:
		- `pkg/observability/otel/otel.go`: `//go:build !otelotlp` (no-op InitTracer)
		- `pkg/observability/otel/httpwrap.go`: `//go:build !otelotlp` (no-op HTTP wrapper)
		- Giữ `otel_otlp.go` và `httpwrap_otlp.go` cho biến thể thật khi build với `otelotlp`.
- ShieldX Gateway:
	- `services/shieldx-gateway/main.go`: 
		- Gọi `InitTracer("shieldx_gateway")` và bọc `http.Handler` bằng `otelobs.WrapHTTPHandler` (server spans).
		- Bọc HTTP metrics middleware + phục vụ `/metrics`; đọc cổng từ env `GATEWAY_PORT`.
	- `services/shieldx-gateway/go.mod`: thêm `replace shieldx => ../..` để import `shieldx/pkg/metrics` trong module con.
	- `docker/Dockerfile.shieldx-gateway`: 
		- Nâng builder lên Go 1.24; build ngay trong `services/shieldx-gateway` (module riêng); runtime distroless; `ENV GATEWAY_PORT=8082`.
- Compose + Prometheus:
	- `pilot/observability/docker-compose.override.yml`:
		- Thêm service `shieldx-gateway` (8082) và truyền `OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4318`.
		- Bật tracing cho `ingress`, `locator`, `shieldx-gateway` qua `build.args: { GO_TAGS: otelotlp }`.
	- `pilot/observability/prometheus-scrape.yml`:
		- Sửa job `ingress` sang `ingress:8081` (đúng port runtime).
		- Thêm job `shieldx_gateway` trỏ `shieldx-gateway:8082`.

- Kết quả chạy demo:
	- `make demo-up` khởi chạy thành công: Prometheus (9090), Grafana (3000), OTEL Collector (4318), Ingress (8081), Locator (8080), ShieldX Gateway (8082).
	- Các service xuất `/metrics`; Collector nhận spans (exporter `debug`).

Xác nhận nhanh (sanity):
- Prometheus targets OK (ingress:8081, locator:8080, shieldx-gateway:8082).
- Health endpoints: `/healthz` (ingress), `/health` (shieldx-gateway) phản hồi 200.

Tiến độ Tháng 10/2025 — Nền tảng quan sát và SLO cơ bản
- Metrics: Đạt (100%) cho phạm vi mục tiêu: ingress, contauth, verifier-pool, ml-orchestrator, locator, shieldx-gateway, và ML service (Python) đều có `/metrics`.
- Tracing: Đang triển khai. Đã bật cho ingress, locator, shieldx-gateway (qua `otelotlp`). Cần nối tiếp cho contauth, verifier-pool, ml-orchestrator để đạt ≥95% endpoints có trace. Collector đã hoạt động (debug exporter).
- Dashboard & Alerts: Đã có dashboard SLO và alert rules mẫu (Prometheus + Grafana). Cần thời gian chạy để lấp dữ liệu SLO.
- Error budget tracking: Bắt đầu thu thập; cần 1 tuần runtime liên tục để đánh giá.

Việc tiếp theo (nhỏ, rủi ro thấp):
- Bổ sung Tempo/Jaeger vào compose để quan sát trace trực quan trong Grafana.
- Bọc tracing cho contauth, verifier-pool, ml-orchestrator bằng `otelobs.WrapHTTPHandler` và `InitTracer()`.
- (Tùy chọn) Thêm whitelist cho path-label để kiểm soát cardinality metrics HTTP.


## 2025-10-01 — Hoàn thiện demo Observability: sửa metrics histogram, mở rộng compose, xác thực traces

- Sửa lỗi xuất metrics Prometheus cho histogram có nhãn:
	- File: `pkg/metrics/metrics.go` — gom nhãn `le` vào cùng một cặp `{}` với `method`/`path` thay vì in hai cặp, loại bỏ lỗi Prometheus: "expected value after metric, got '{l' ('BOPEN')".
- Build & restart các dịch vụ demo với `otelotlp` để bật tracing: `ingress`, `locator`, `shieldx-gateway`, `verifier-pool`, `ml-orchestrator`, `contauth`.
	- Dockerfiles cập nhật: `docker/Dockerfile.contauth`, `docker/Dockerfile.verifier-pool`, `docker/Dockerfile.ml-orchestrator` — build trong thư mục module con; runtime distroless, nonroot.
- ContAuth chế độ demo không DB:
	- Thêm `services/contauth/dummy_collector.go` và chuyển động qua biến môi trường `DISABLE_DB=true` (đã thiết lập trong compose) để chạy không cần Postgres.
- Compose & Prometheus:
	- `pilot/observability/docker-compose.override.yml`: thêm `DISABLE_DB=true` cho contauth; đổi ánh xạ cổng `ml-orchestrator` thành `8086:8087` (trong container vẫn 8087); giữ `GO_TAGS=otelotlp` và `OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4318` cho các service.
	- `pilot/observability/prometheus-scrape.yml`: bỏ job `ml_service` (không chạy trong demo); bổ sung chú thích scrape trong mạng compose.
- Kết quả xác nhận:
	- Tất cả targets trong Prometheus ở trạng thái up: `ingress:8081`, `locator:8080`, `shieldx-gateway:8082`, `verifier-pool:8087`, `ml-orchestrator:8087` (xuất cổng host `8086`), `contauth:5002`.
	- `/metrics` của từng service phản hồi OK từ host và trong mạng compose; lỗi BOPEN biến mất.
	- OTEL Collector (debug exporter) ghi nhận spans liên tục, xác nhận tracing end-to-end hoạt động khi build với `otelotlp`.
- Ghi chú:
	- Metrics theo path có rủi ro cardinality; sẽ thêm whitelist/chuẩn hoá sau khi có dữ liệu thực tế.
	- Build toàn repo có thể còn lỗi ở module/kiểm thử ngoài phạm vi demo; không ảnh hưởng mục tiêu Tháng 10 (demo stack chạy tốt).

Tiêu chí chấp nhận Tháng 10 (cập nhật):
- Metrics: đạt 100% cho 5 dịch vụ mục tiêu.
- Traces: đã bật trên các dịch vụ trong demo; Collector nhận span đều đặn. Dashboard SLO đang thu thập dữ liệu, sẵn sàng theo dõi error budget 1 tuần.


### 2025-10-01 — Bổ sung Jaeger + Blackbox và propagation traces
- Compose: thêm Jaeger all-in-one và Blackbox Exporter vào `pilot/observability/docker-compose.yml`; mount provisioning Grafana.
- Prometheus: thêm job `blackbox` trong `prometheus-scrape.yml` để probe các endpoint `/health(z)` và `/metrics`.
- ShieldX Gateway: bọc outbound HTTP client bằng `otelobs.WrapHTTPTransport` để propagate trace context; tránh bọc handler trùng lặp.
- Grafana: thêm datasource Jaeger và dashboard tối thiểu `ShieldX HTTP Overview` với link sang Explore để xem traces theo service.

