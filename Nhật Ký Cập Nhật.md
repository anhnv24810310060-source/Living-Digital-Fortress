## 2025-10-01 — RA‑TLS (SPIFFE) + wiring shieldx-gateway/ingress + cảnh báo hết hạn cert (prepend)

- Thư viện RA‑TLS (pkg/ratls):
	- Thêm `AutoIssuer` (CA in‑memory) phát hành cert ngắn hạn có SPIFFE SAN, tự xoay vòng (rotate) theo cấu hình (`RATLS_ROTATE_EVERY` < `RATLS_VALIDITY`).
	- API TLS: `ServerTLSConfig(requireClientCert, trustDomain)` và `ClientTLSConfig()` để bật mTLS nội bộ theo trust domain.
	- Metric helper: `LeafNotAfter()` để đọc thời gian hết hạn chứng chỉ hiện tại (phục vụ metric cảnh báo).
	- Kiểm thử: mTLS thành công, reject sai trust domain, và rotation hoạt động — tất cả PASS.

- Tích hợp dịch vụ:
	- `services/shieldx-gateway/main.go`
		- Đọc env RA‑TLS (`RATLS_ENABLE`, `RATLS_TRUST_DOMAIN`, `RATLS_NAMESPACE`, `RATLS_SERVICE`, `RATLS_ROTATE_EVERY`, `RATLS_VALIDITY`).
		- Bật mTLS inbound bằng `issuer.ServerTLSConfig(true, trustDomain)` khi `RATLS_ENABLE=true`.
		- HTTP client outbound dùng `issuer.ClientTLSConfig()` (giữ OTEL transport).
		- Metric `ratls_cert_expiry_seconds` (giây còn lại tới hạn cert) và cập nhật định kỳ để quan sát.
	- `services/ingress/main.go`
		- Bật mTLS inbound tương tự gateway khi bật RA‑TLS qua env.
		- Chuẩn hóa toàn bộ outbound (Locator/Guardian/Decoy) qua shared HTTP client bọc OTEL + mTLS client cert.
		- Thêm metric `ratls_cert_expiry_seconds` và cập nhật theo `LeafNotAfter()`.

- Quan sát & Cảnh báo:
	- Prometheus rule mới `RATLSCertExpiringSoon`: bắn cảnh báo khi `ratls_cert_expiry_seconds < 600` trong 5 phút (khả năng rotation bị kẹt).
	- Tài liệu rollout ngắn gọn: `pilot/docs/ratls-rollout.md` (envs, mẫu wiring server/client, metric, rule cảnh báo, ghi chú sản xuất).

- Ảnh hưởng build/chạy:
	- Không thêm phụ thuộc ngoài chuẩn thư viện Go. Các dịch vụ gateway/ingress build sạch; test `pkg/ratls` PASS.
	- Khi bật RA‑TLS, yêu cầu tất cả gọi nội bộ sang service khác dùng HTTPS + mTLS.

## 2025-12-01 — Báo cáo Tháng 12: Done (SBOM + ký image + build tái lập)

- CI `supply-chain.yml` hiện build + push ma trận tất cả images trong `docker/`, ký bằng Cosign keyless (OIDC) theo digest, và xuất SBOM CycloneDX cho từng image (đính kèm artifact). Nguồn (Go + Python) cũng có SBOM.
- GoReleaser snapshot cấu hình tái lập (trimpath, buildid rỗng) cho `cmd/policyctl`; có thể mở rộng binaries sau.
- Tài liệu đã bổ sung hướng dẫn enforce trong cluster với `pilot/hardening/image-signing.yml` (kèm `kubectl apply -f ...` và lưu ý issuer/subject).
- KPI: 100% images phát hành từ CI có chữ ký + SBOM; release có thể tái lập. Việc enforce verify trong runtime phụ thuộc bước apply manifest vào cluster (đã có hướng dẫn).

## 2025-12-01 — Tiến độ Tháng 12: SBOM + Ký image + Build tái lập

- Đã thêm workflow CI `supply-chain.yml`: sinh SBOM (Syft CycloneDX), build snapshot (GoReleaser) và tùy chọn ký image (Cosign keyless qua OIDC) khi cung cấp input `image`.
- Đã bổ sung tài liệu `pilot/docs/supply-chain.md` hướng dẫn chạy local và CI.
- Makefile đã có: `sbom-all`, `image-sign`, `release-snapshot`.
- Ghi chú: GoReleaser hiện build `cmd/policyctl`; có thể mở rộng thêm binary khác sau.

## 2024-12-01 — Khởi động Tháng 12: SBOM + Ký image + Build tái lập (reproducible)

- Makefile: thêm targets `sbom-all` (Syft CycloneDX), `image-sign` (Cosign keyless hoặc KEY_REF), `release-snapshot` (Goreleaser snapshot).
- CI: thêm workflow `.github/workflows/supply-chain.yml` tạo SBOM, build snapshot, và ký image theo input.
- Tài liệu: `pilot/docs/supply-chain.md` hướng dẫn chạy local/CI.
- Ghi chú: dùng OIDC cho Cosign trong CI; SBOM xuất ra `dist/sbom/**`.

## 2025-11-01 — Báo cáo Tháng 11: Done; chuẩn bị Tháng 12 (SBOM + ký image + reproducible builds) — prepend

- Tháng 11: Trạng thái = Done
	- Policy bundle ký số + CI verify (Cosign keyless): Hoàn tất
	- Conftest + Rego unit tests: Hoàn tất
	- Canary rollout + drift detection + metrics: Hoàn tất
	- Promote workflow (upload approved-bundle + webhook /apply tùy chọn): Hoàn tất
	- Tracing rollout (otelotlp build tag): Sẵn sàng
	- Spec bundle v0: Có
	- KPI: PR policy phải pass verify + tests; canary mô phỏng/metrics có sẵn

- Chuẩn bị Tháng 12 (đặt nền tảng, rủi ro thấp):
	- Makefile: targets `sbom-all`, `image-sign`, `release-snapshot` (goreleaser) — thêm ngay
	- CI `supply-chain.yml`: sinh SBOM (Syft/CycloneDX), build snapshot reproducible (goreleaser --snapshot), tải artifact SBOM
	- Docs: `pilot/docs/supply-chain.md` mô tả luồng SBOM → ký image → verify; yêu cầu secrets
	- Lưu ý: ký image (cosign) sẽ bật khi có registry + secrets; hiện chỉ chuẩn bị targets và workflow

## 2025-11-01 — Promote workflow, tracing rollout, registry URL callback (prepend)

- Promote CI: `.github/workflows/policy-promote.yml` chạy sau khi "Policy Bundle CI" thành công:
	- Tải (hoặc build lại) bundle, ký/verify bằng Cosign keyless, upload artifact `approved-bundle` (zip+sig+digest).
	- Tùy chọn gọi webhook `/apply` của `policy-rollout` nếu cấu hình `ROLLOUT_ENDPOINT_URL` và `ARTIFACT_BASE_URL` (presign/serve artefacts).
- Tracing rollout: `services/policy-rollout` bọc handler bằng `otelobs.WrapHTTPHandler` (build tag `otelotlp` để bật); thêm header phản hồi x-verify-* như span attributes thô (demo).
- Registry thực: khuyến nghị dùng artefact store/GitHub Releases/S3; workflow đã để ngỏ biến `ARTIFACT_BASE_URL` cho URL public hoặc presigned.

## 2025-11-01 — Rollout kết nối bundle thật (URL+cosign), compose wiring, Dockerfile (prepend)

- Policy Rollout service mở rộng:
	- `/apply` nhận `{url, sig}`: tải bundle zip từ URL, tính digest, verify bằng Cosign (nếu có chữ ký) rồi bắt đầu canary.
	- `/metrics` bổ sung thông tin nguồn và thời gian xác minh (qua log); giữ các metric verify/drift/rollout hiện hữu.
- Loader: `pkg/policy/zipload.go` đọc bundle từ zip và tính digest theo manifest/files.
- Compose: thêm service `policy-rollout` vào `pilot/observability/docker-compose.override.yml` (port 8099).
- Dockerfile: `docker/Dockerfile.policy-rollout` (multi-stage, distroless, nonroot).

## 2025-11-01 — Cosign keyless (CI), Make targets, rollout/drift skeleton, Rego tests (prepend)

- CI (GitHub Actions): cập nhật `.github/workflows/policy.yml` để dùng Cosign keyless:
	- Bật permissions `id-token: write`.
	- Cài `cosign` và chạy `cosign sign-blob`/`verify-blob` với OIDC.
	- Giai đoạn bundle tạo `dist/digest.txt` để ký/verify theo digest.
- Makefile: thêm targets `policy-sign-cosign` và `policy-verify-cosign` (KEY_REF tùy chọn; mặc định keyless).
- Rollout & Drift detection: tạo skeleton service `services/policy-rollout/`:
	- Endpoints: `/health`, `/metrics`, `/apply` (nhận digest), canary 10% và mô phỏng promote/rollback.
	- Metrics: `policy_verify_success_total`, `policy_verify_failure_total`, `policy_drift_events_total`, `policy_rollout_percentage`.
- Tests:
	- Go: `pkg/policy/bundle_test.go` (build/hash/zip, cosign adapter skip nếu thiếu cosign).
	- OPA: thêm `policies/demo/policy_test.rego` cho allow/deny; mẫu Conftest/OPA trước đó giữ nguyên.

## 2025-11-01 — Khởi động Tháng 11/2025: skeleton Policy Bundle + CLI + Makefile (ghi chú mới ở đầu file)

- Quy ước ghi nhật ký: Từ thời điểm này, mọi cập nhật mới sẽ được thêm ở ĐẦU file để dễ theo dõi tiến độ gần nhất.
- Đã tạo skeleton Policy-as-code:
	- `pkg/policy/bundle.go`: Manifest/Bundle, `LoadFromDir`, `Hash()` (SHA-256 canonical), `WriteZip()`, `Signer/Verifier` interface, `NoopSigner/NoopVerifier` demo, `BuildAndWrite`, `SignDigest`, `VerifyDigest`.
	- CLI `cmd/policyctl`: lệnh `bundle`, `sign`, `verify` để thao tác nhanh với bundle.
	- Demo policy: `policies/demo/manifest.json`, `rules/allow.rego`, `rules/deny.rego` (đường đi E2E).
	- Makefile: targets `policy-bundle`, `policy-sign`, `policy-verify`, `policy-all`.
- Xác nhận chạy E2E:
	- Build CLI, tạo bundle zip, ký (noop), và verify thành công; in ra digest.
- Việc tiếp theo (ngắn hạn):
	- Thêm Spec tài liệu `pilot/docs/policy-bundle-spec.md`.
	- Thay `NoopSigner/Verifier` bằng adapter Cosign CLI (tối thiểu) và thêm test.
	- Thiết lập Conftest + unit test Rego; workflow CI `policy.yml` verify chữ ký trên PR.

## 2025-11-01 — Kế hoạch Tháng 11/2025 — Policy-as-code ký số và kiểm thử (Checklist)

Mục tiêu: Policy bundle có ký số, kiểm thử và canary 10% an toàn; drift detection. PR policy phải có chữ ký và test đi kèm.

Phạm vi tác động: `pkg/policy/`, `services/policy/` (hoặc `services/plugin_registry/`), `Makefile`, `.github/workflows/`, `pilot/docs/`.

Các hạng mục cần làm (checklist):

- Đặc tả & tài liệu
	- [ ] Soạn "Policy Bundle Spec v0" (pilot/docs/policy-bundle-spec.md):
		- Manifest: name, version, created_at, opa_version, policies[], annotations.
		- Canonicalization: sort keys, normalize LF, exclude signature fields khi băm.
		- Hash: SHA-256 digest cho toàn bundle (manifest + policy files theo canonical order).
		- Ký số: Sigstore/cosign (keypair hoặc keyless OIDC); tùy chọn DSSE envelope.
		- Metadata chữ ký: subject, issuer, expiry, annotations (env, tenant, purpose).
	- [ ] Hướng dẫn Dev: quy trình build/sign/verify bundle + lưu trữ khóa an toàn.

- Thư viện & công cụ
	- [ ] `pkg/policy/bundle.go`: types (Manifest, Bundle), builder, `Hash()`, `Sign()`, `Verify()`; load/save `.tar.gz` hoặc `.zip`.
	- [ ] Tính năng verify cosign (ban đầu mock/exec cosign CLI; module hóa để có thể thay thế lib sau):
		- [ ] Interface `Signer`/`Verifier`, implementation `CosignCLI`.
	- [ ] Makefile targets: `policy-bundle`, `policy-sign`, `policy-verify` (kèm docs/usage).
	- [ ] Mẫu bundle demo với 1–2 file Rego (ví dụ allow/deny rule đơn giản) để kiểm thử đường đi.

- Kiểm thử & CI
	- [ ] Thiết lập Conftest trong repo (policies mẫu + tests).
	- [ ] Thêm unit test Rego (ví dụ deny on missing field, allow on valid schema).
	- [ ] `.github/workflows/policy.yml` (hoặc Makefile + CI sẵn có):
		- [ ] Chạy `policy-bundle` trên PR.
		- [ ] Xác minh chữ ký bundle (`policy-verify`).
		- [ ] Chạy Conftest và unit tests.
		- [ ] Đính kèm artifact bundle đã ký vào job (nếu cần).

- Rollout & Drift detection
	- [ ] Dịch vụ/Job canary rollout (services/policy/): áp dụng bundle mới cho 10% workload; nếu error rate vượt ngưỡng SLO -> rollback tự động.
	- [ ] Drift detection worker: so sánh hash bundle đang chạy với registry; cảnh báo Prometheus + event log khi lệch.
	- [ ] Endpoint quan sát: `/metrics` cho verify_success_total, verify_failure_total, drift_events_total, rollout_status.

- Quan sát & bảo mật
	- [ ] Metrics/traces cho đường verify/sign và rollout; log có cấu trúc, audit trail tối thiểu.
	- [ ] Chiến lược quản lý khóa cosign: file-based (demo) -> keyless (OIDC) sau; rotate và revoke notes.

- Acceptance & Demo
	- [ ] Kịch bản demo E2E: build -> sign -> verify -> canary -> promote/rollback.
	- [ ] Tiêu chí chấp nhận: 100% policy PR có chữ ký hợp lệ + test pass; rollback tự động < 5 phút trong canary lỗi.

Gợi ý thực thi theo tuần (tham khảo, không bắt buộc):
- Tuần 1: Spec + `pkg/policy` skeleton + Makefile targets + bundle demo.
- Tuần 2: Conftest + unit tests Rego + workflow CI base.
- Tuần 3: Canary rollout + drift detection + metrics/observability.
- Tuần 4: Hardening key mgmt, tài liệu, demo E2E và chốt chấp nhận.




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

## 2025-10-01 — Kiểm soát cardinality cho metrics HTTP theo path (allowlist/regex/mode)

- `pkg/metrics/metrics.go`:
	- Thêm cơ chế kiểm soát cardinality cho nhãn `path` của metrics HTTP:
		- Allowlist theo prefix (`pathAllowlist`).
		- Allowlist theo biểu thức regex (`pathRegexps`).
		- Chế độ chuẩn hóa `pathMode`: `heuristic` (mặc định, thay thế các segment giống ID thành `:id`) hoặc `strict` (không thuộc allowlist/regex sẽ gộp về `:other`).
	- Cấu hình qua biến môi trường (ưu tiên theo service, fallback global):
		- `<SERVICE>_HTTP_PATH_ALLOWLIST` hoặc `HTTP_PATH_ALLOWLIST` (CSV, ví dụ: `/health,/metrics,/api/v1/login`).
		- `<SERVICE>_HTTP_PATH_REGEX` hoặc `HTTP_PATH_REGEX` (CSV regex, ví dụ: `^/api/v1/users/[a-z0-9-]+/profile$`).
		- `<SERVICE>_HTTP_PATH_MODE` hoặc `HTTP_PATH_MODE` (`heuristic` | `strict`).
	- Thay đổi mặc định an toàn: bỏ `"/"` khỏi allowlist mặc định để tránh vô tình giữ nguyên toàn bộ đường dẫn (giảm rủi ro bùng nổ cardinality).
	- Giữ tương thích ngược: nếu không đặt biến môi trường, hành vi vẫn theo heuristic như trước, nhưng an toàn hơn về cardinality.

- Ảnh hưởng dashboard/Prometheus:
	- Nhãn `path` ổn định hơn; giảm rủi ro high-cardinality time series. Có thể tinh chỉnh thêm allowlist/regex theo service khi quan sát thực tế.

- Hướng dẫn nhanh:
	- Ví dụ giới hạn cardinality nghiêm ngặt cho Ingress:
		- `INGRESS_HTTP_PATH_ALLOWLIST="/healthz,/metrics,/route"`
		- `INGRESS_HTTP_PATH_MODE=strict`
	- Ví dụ cho phép một số pattern động qua regex cho ContAuth:
		- `CONTAUTH_HTTP_PATH_REGEX="^/sessions/[a-f0-9-]{36}$,^/users/[0-9]+/risk$"`

Ghi chú: tiếp tục theo dõi cardinality sau 24–48 giờ; nếu số series vẫn cao, chuyển `HTTP_PATH_MODE` sang `strict` cho dịch vụ có lưu lượng lớn hoặc mở rộng allowlist hợp lý.

### Bổ sung cấu hình demo
- `pilot/observability/docker-compose.override.yml`: thêm biến môi trường mặc định cho các dịch vụ (ingress, locator, shieldx-gateway, contauth, verifier-pool, ml-orchestrator):
	- `<SERVICE>_HTTP_PATH_ALLOWLIST` tập trung vào `/health(z)` và `/metrics`.
	- `<SERVICE>_HTTP_PATH_MODE=strict` để ổn định series trong demo.





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
