# Security Findings and Improvement Proposals

Ngày: 2025-10-13

Tài liệu này tổng hợp các lỗ hổng bảo mật và điểm cần cải tiến theo từng service/thành phần. Mỗi mục gồm: (1) số thứ tự, (2) mô tả lỗi, (3) hướng giải quyết đề xuất và vị trí, (4) tiêu chí hoàn thành.

---

## A. Cấu hình chung / Hạ tầng (docker-compose, images, secrets)

1) Lộ/weak secrets trong docker-compose (dev defaults)
- Mô tả: Secrets/creds để ở docker-compose và có giá trị yếu/mặc định: 
  - POSTGRES_PASSWORD=shieldx123; JWT_SECRET=dev-jwt-secret; ADMISSION_SECRET=dev-secret-12345; Grafana admin password fortress123; ML_API_ADMIN_TOKEN="" (trống) → vô hiệu hóa bảo vệ admin.
- Vị trí: `docker-compose.full.yml` (postgres, auth-service, ingress, grafana, ml-orchestrator).
- Đề xuất:
  - Di chuyển secrets sang Docker/K8s secrets; bỏ giá trị mặc định trong compose production.
  - Bắt buộc cấu hình mạnh qua biến môi trường/tệp secret khi deploy; bật rotate định kỳ.
  - Tách profile dev và prod; trong prod cấm commit secrets và cấm giá trị mặc định yếu.
- Hoàn thành khi: Secrets không còn xuất hiện trong repo; pipelines/helm chart yêu cầu secret inputs; kiểm tra runtime xác nhận biến mặc định không dùng ở prod.

2) Redis/Postgres không có hardening
- Mô tả: Redis không auth; Postgres dùng mật khẩu yếu và port/volume mặc định; nguy cơ lateral movement nếu lộ network nội bộ.
- Vị trí: `docker-compose.full.yml` (redis, postgres).
- Đề xuất: Bật auth/password cho Redis; sử dụng network segment tách biệt; Postgres dùng user/pass mạnh, TLS giữa app↔DB, bật pgaudit ở prod.
- Hoàn thành khi: Cấu hình auth Redis/PG áp dụng; kiểm tra kết nối yêu cầu cred; security baseline doc cập nhật.

3) Pin phiên bản container Images
- Mô tả: Prometheus/Grafana dùng `latest`; khó kiểm soát SBOM/CVE.
- Vị trí: `docker-compose.full.yml` (prometheus, grafana).
- Đề xuất: Pin version cụ thể hoặc digest; thêm image scanning (Trivy) trong CI.
- Hoàn thành khi: Tất cả images pin phiên bản/digest; CI báo cáo scan pass.

---

## B. Ingress Gateway

4) mTLS chưa bắt buộc trong prod cấu hình hiện tại
- Mô tả: `RATLS_REQUIRE_CLIENT_CERT=false` trong compose dev; Admission header là cơ chế bảo vệ chính → yếu nếu secret lộ.
- Vị trí: `services/shieldx-gateway/ingress/main.go`, `docker-compose.full.yml` (ingress env).
- Đề xuất: Prod bật RA-TLS + bắt buộc client cert; cấu hình allowlist SAN qua `INGRESS_ALLOWED_CLIENT_SAN_PREFIXES`; lưu Admission secret trong secret store và rotate.
- Hoàn thành khi: Ingress yêu cầu mTLS ở prod; test handshake client cert pass/fail; secret không nằm trong repo.

5) Admission HMAC có cửa sổ 1 phút (replay window)
- Mô tả: Token hợp lệ trong bucket phút hiện tại/trước đó → có thể replay trong ~60s nếu bị sniff.
- Vị trí: `pkg/guard/guard.go` VerifyHeader.
- Đề xuất: Thu hẹp cửa sổ (ví dụ 30s), gắn thêm nonce một-lần (LRU cache), và ràng buộc IP/source. Bắt buộc HTTPS/mTLS.
- Hoàn thành khi: Unit test VerifyHeader cập nhật; metrics cho reject replay; pen test không replay được.

6) Expose /metrics và /health public
- Mô tả: /metrics,/health thường mở; có thể rò rỉ thông tin nội bộ.
- Vị trí: Ingress và các services nói chung.
- Đề xuất: Chỉ expose nội bộ (network policy) hoặc bảo vệ bằng auth/gateway; ẩn thông tin nhạy cảm, thêm rate-limit.
- Hoàn thành khi: Network policy áp dụng; scan từ ngoài không truy cập /metrics.

---

## C. Policy Rollout Service

7) Unauthenticated policy apply + SSRF khả dĩ
- Mô tả: `/apply` nhận `url` và tải bundle qua `http.Get`, tùy chọn `sigURL`. Không kiểm soát schema/host/redirect/size → SSRF/DoS.
- Vị trí: `services/shieldx-policy/policy-rollout/main.go` (hàm `fetchAndVerify`).
- Đề xuất:
  - Bắt buộc auth (JWT/mTLS/Admission) + RBAC cho `/apply`.
  - Validator URL: chỉ cho phép `https`, chặn private IP (127.0.0.0/8, 10/8, 172.16/12, 192.168/16, link-local, 169.254/16), chặn file://, gopher://, ftp://, và chặn redirect.
  - Thiết lập timeout ngắn, giới hạn dung lượng (Content-Length và io.LimitReader), kiểm tra Content-Type/ZIP, verify cert.
  - Bắt buộc chữ ký (loại bỏ `NoopVerifier` ở prod), quy trình key management cho cosign.
- Hoàn thành khi: Thêm middleware auth; unit/integration test SSRF pass; benchmark tải lớn bị chặn; logs/metrics phản ánh reject case.

8) No size/time limits khi tải bundle
- Mô tả: Đọc toàn bộ vào RAM (bytes → zip reader) → dễ DoS.
- Vị trí: `policy-rollout/main.go` (`fetchAndVerify`).
- Đề xuất: Dùng http.Client với timeout; io.LimitReader (ví dụ 20MB); từ chối nếu vượt ngưỡng.
- Hoàn thành khi: Thử tải file > ngưỡng bị 413/400; memory stable trong test.

---

## D. ThreatGraph

9) ThreatGraph Writer: Arbitrary Cypher + No Auth
- Mô tả: `/graph/query` nhận trực tiếp chuỗi Cypher từ query param và thực thi; `/graph/node`/`/graph/edge` ghi dữ liệu không auth.
- Vị trí: `services/shieldx-forensics/threatgraph/writer.go`.
- Đề xuất:
  - Bắt buộc auth (JWT/mTLS) và RBAC; bỏ endpoint query tùy ý.
  - Chỉ cho phép các truy vấn được parameter hóa; thêm rate limit + audit log.
- Hoàn thành khi: Test không thể chạy Cypher tùy ý; kiểm thử xác thực bắt buộc.

10) ThreatGraph API thiếu auth/rate-limit
- Mô tả: `/ingest`, `/query`, `/stats` không có auth/rate-limit.
- Vị trí: `services/shieldx-forensics/threatgraph/main.go`.
- Đề xuất: Thêm middleware auth (JWT/mTLS) + rate limit và input validation; schema cho `ThreatQueryRequest`.
- Hoàn thành khi: E2E test yêu cầu token; rate-limit hoạt động.

---

## E. ML Orchestrator

11) Admin endpoints không được bảo vệ khi token rỗng
- Mô tả: `makeAdminMiddleware()` trả về passthrough nếu `ML_API_ADMIN_TOKEN==""`; trong compose dev để trống → mọi người có thể gọi /train, /model/*.
- Vị trí: `services/shieldx-ml/ml-orchestrator/main.go` và `docker-compose.full.yml`.
- Đề xuất: Bắt buộc token khác rỗng ở mọi môi trường; hỗ trợ mTLS hoặc Admission guard cho admin APIs; giới hạn IP.
- Hoàn thành khi: Gọi không token trả 401; token sai 401; token đúng 2xx; test CI cover.

12) Upload/model ops thiếu hạn mức
- Mô tả: Tiềm ẩn DoS nếu tải tệp lớn hoặc nhiều multipart; (code có xử lý nhưng cần xác nhận quotas/limits).
- Vị trí: `ml-orchestrator/main.go` (điểm xử lý upload/model save/load).
- Đề xuất: Thiết lập `MaxBytesReader`, hạn mức trên file count/kích cỡ, timeout; kiểm tra định dạng.
- Hoàn thành khi: Upload vượt ngưỡng trả 413; log/metrics phản ánh.

---

## F. Camouflage API (Deception)

13) CORS "*" và API key mặc định yếu
- Mô tả: `Access-Control-Allow-Origin: *`. Hàm `validateToken` dùng `CAMOUFLAGE_API_KEY` với default "default_key"; nếu không set đúng → token yếu/biết trước.
- Vị trí: `services/shieldx-deception/camouflage-api/main.go`.
- Đề xuất: Hạn chế CORS origin (allowlist); bắt buộc API key mạnh từ secrets; cân nhắc JWT/mTLS; log và rate-limit.
- Hoàn thành khi: CORS chỉ cho phép origin hợp lệ; yêu cầu Authorization với key hợp lệ.

---

## G. Admin WebAPI

14) Proxy endpoints không có auth
- Mô tả: `/api/shadow/*` proxy đến Shadow mà không xác thực.
- Vị trí: `services/shieldx-admin/webapi/main.go`.
- Đề xuất: Bắt buộc JWT/mTLS; áp dụng role-based access; rate-limit và audit log.
- Hoàn thành khi: Không có token → 401; token sai → 401; token đúng → 2xx; test OK.

---

## H. Verifier Pool

15) `/nodes` công khai, cần access control
- Mô tả: Mục đích liệt kê node; comment ghi chú "requires access control" nhưng hiện trả thông tin không auth.
- Vị trí: `services/shieldx-auth/verifier-pool/main.go` (handleListNodes).
- Đề xuất: Bảo vệ bằng JWT/mTLS; ẩn thông tin nhạy cảm; limit fields; rate-limit.
- Hoàn thành khi: `/nodes` yêu cầu auth; contract test cập nhật.

---

## I. Deception DSL Loader

16) LoadFromURL thiếu kiểm soát nguồn
- Mô tả: `LoadFromURL` gọi `http.Get(url)` trực tiếp; nếu URL do user cung cấp → SSRF/DoS tương tự (redirect/size/timeouts).
- Vị trí: `shared/shieldx-common/pkg/deception/dsl.go`.
- Đề xuất: Cho phép chỉ `https`, timeout và giới hạn kích thước, cấm địa chỉ nội bộ, tắt redirect; cân nhắc tải qua proxy allowlist.
- Hoàn thành khi: Unit test SSRF pass; limit áp dụng.

---

## J. TLS/Crypto Hygiene

17) Dịch vụ chạy HTTP (dev) chưa đảm bảo TLS ở prod
- Mô tả: Nhiều service lắng nghe HTTP plain trong dev.
- Vị trí: nhiều `main.go`; compose env.
- Đề xuất: Bật RA-TLS/mTLS trong prod; enforce TLS 1.3; cấm HTTP trừ health nội bộ; security header chuẩn (HSTS khi có TLS).
- Hoàn thành khi: Prod chỉ chấp nhận TLS; kiểm tra sslyze/nmap pass.

18) Key management/rotation
- Mô tả: Chưa thấy quy trình rotate keys (JWT, Admission, Cosign keyref).
- Vị trí: chung; `go.mod`/docs không ghi nhận.
- Đề xuất: Thiết kế quy trình rotate; dùng secret manager; TTL/rotation policy; logging rotation events.
- Hoàn thành khi: SOP/Docs và script rotate có; audit log xác nhận.

---

## K. Observability và Data Exposure

19) /metrics có thể lộ thông tin (paths, durations)
- Mô tả: Dữ liệu metrics có thể rò đường dẫn nội bộ, tên dịch vụ.
- Vị trí: tất cả services.
- Đề xuất: Chỉ scrape từ Prometheus nội bộ; filter nhãn nhạy cảm; không expose công khai.
- Hoàn thành khi: Prometheus target chỉ nghe private; thử từ ngoài không truy cập được.

20) Logs chứa dữ liệu có thể nhạy cảm
- Mô tả: Một số handler phản ánh lại headers/
- Vị trí: ingress, gateway, deception…
- Đề xuất: Redact PII/secrets trong logs; thêm structured logging với policy.
- Hoàn thành khi: Kiểm tra log không chứa secrets; CI lint rule cho log redaction.

---

## L. Quy trình & Phòng thủ chiều sâu

21) Bổ sung rate limiting/circuit breaker cho endpoints nhạy cảm
- Mô tả: Một số endpoints quản trị chưa có limiter.
- Vị trí: policy-rollout /apply, threatgraph writer, admin webapi.
- Đề xuất: Middleware limiter theo IP/user; CAPTCHA/2FA cho admin GUI; circuit breaker với upstream.
- Hoàn thành khi: Stress test không làm sập dịch vụ; metrics limiter xuất hiện.

22) Bổ sung SAST/DAST trong CI
- Mô tả: Chưa thấy pipelines bảo mật.
- Vị trí: CI/CD.
- Đề xuất: Gosec, Semgrep, Trivy image scan, dependency audit; chính sách PR block khi fail.
- Hoàn thành khi: CI pipelines thêm job security và cổng chất lượng bật.

---

## Phụ lục: Mức độ ưu tiên (Cao → Thấp)
- Cao: (7), (9), (11), (1), (4), (8), (16)
- Trung bình: (5), (12), (14), (15), (6), (3), (2)
- Thấp/Cải tiến: (17), (18), (19), (20), (21), (22)

---

Ghi chú: Một số cấu hình có thể chấp nhận trong môi trường dev (ví dụ secrets yếu, HTTP thay TLS), nhưng cần bắt buộc biện pháp mạnh ở prod. Các mục trên tập trung ưu tiên xử lý trước các lỗ hổng thực sự có thể khai thác.

---

## Bổ sung phát hiện mới

23) ORCH_ALLOW_INSECURE cho phép chạy Orchestrator trên HTTP
- Mô tả: Biến `ORCH_ALLOW_INSECURE=1` bật chế độ HTTP thuần (không TLS) cho orchestrator, phù hợp dev nhưng nếu bật nhầm ở prod → rủi ro MITM.
- Vị trí: `services/shieldx-gateway/orchestrator/main.go` (block cảnh báo ORCH_ALLOW_INSECURE).
- Đề xuất: Tách profile dev/prod; trong prod bỏ hoàn toàn đường đi này hoặc fail-fast khi không có TLS; thêm guard CI/CD không cho set biến này ở prod.
- Hoàn thành khi: Deploy prod không còn cờ này; kiểm tra endpoint buộc TLS.

24) MASQUE QUIC server dùng self-signed insecure config
- Mô tả: Hàm `generateInsecureTLSConfig()` tạo cert tự ký cho QUIC MASQUE; có thể chấp nhận cho lab nhưng không an toàn prod.
- Vị trí: `services/shieldx-gateway/masque/main.go`.
- Đề xuất: Thay bằng RA-TLS hoặc TLS hợp lệ từ PKI; buộc xác thực client.
- Hoàn thành khi: QUIC server yêu cầu mTLS; test xác thực client thành công/thất bại.

25) Deception DSL LoadFromURL thiếu hạn chế redirect/size
- Mô tả: `shared/shieldx-common/pkg/deception/dsl.go` và `pkg/deception/dsl.go` dùng `http.Get` và đọc toàn bộ body; thiếu timeout/limit.
- Vị trí: các hàm `LoadFromURL` tương ứng.
- Đề xuất: Dùng http.Client với timeout; chặn redirect; io.LimitReader (ví dụ 5–10MB); chỉ cho phép https + allowlist host.
- Hoàn thành khi: Unit tests SSRF/DoS pass; limit size áp dụng.

26) ML Orchestrator upload/model operations cần ràng buộc kích cỡ
- Mô tả: Tiềm ẩn tải tệp lớn gây memory spike nếu thiếu `MaxBytesReader` trên request.
- Vị trí: `services/shieldx-ml/ml-orchestrator/main.go` (các handler /model/*, /train).
- Đề xuất: Áp dụng `r.Body = http.MaxBytesReader(w, r.Body, MAX)`; kiểm tra header Content-Length; từ chối vượt ngưỡng; validate MIME.
- Hoàn thành khi: Test tải > MAX trả 413.

27) Admin WebAPI proxy không auth
- Mô tả: `/api/shadow/*` không kiểm soát quyền truy cập; có thể lạm dụng gọi sang Shadow service.
- Vị trí: `services/shieldx-admin/webapi/main.go`.
- Đề xuất: JWT/mTLS + RBAC; audit log; giới hạn IP nguồn quản trị.
- Hoàn thành khi: Không token/role sai → 401/403; test pass.

28) Verifier-pool `/nodes` lộ thông tin vận hành
- Mô tả: Trả về thông tin node mà không xác thực; có ghi chú “requires access control”.
- Vị trí: `services/shieldx-auth/verifier-pool/main.go`.
- Đề xuất: Bảo vệ JWT/mTLS; ẩn chi tiết nhạy cảm; thêm rate-limit.
- Hoàn thành khi: `/nodes` yêu cầu auth; test hợp lệ.

29) CORS “*” ở Camouflage API
- Mô tả: `Access-Control-Allow-Origin: *` và API key mặc định `default_key` khiến lộ diện cross-origin.
- Vị trí: `services/shieldx-deception/camouflage-api/main.go`.
- Đề xuất: Allowlist domain; buộc key mạnh/JWT; loại default ở prod.
- Hoàn thành khi: CORS chỉ chấp nhận origin hợp lệ; key bắt buộc.

30) /metrics exposure cần network policy
- Mô tả: Dù đã chuẩn hóa HELP/TYPE, /metrics vẫn không nên public.
- Vị trí: tất cả services.
- Đề xuất: Chỉ scrape qua mạng nội bộ; cấm cổng public; thêm auth nếu buộc phải mở.
- Hoàn thành khi: Scan bên ngoài không truy cập /metrics.

31) Gateway JWT Secret có default dev-only-secret
- Mô tả: Khi không đặt `GATEWAY_JWT_SECRET`, service dùng mặc định `dev-only-secret` → dễ đoán nếu chạy prod.
- Vị trí: `services/shieldx-gateway/main.go`.
- Đề xuất: Bắt buộc secret từ secret manager; fail-fast nếu biến không tồn tại ở prod; hỗ trợ rotation.
- Hoàn thành khi: Prod không còn secret mặc định; test env missing gây fail khởi động.

32) Verifier-pool /validate thiếu auth
- Mô tả: Endpoint `/validate` xử lý yêu cầu xác minh mà không ràng buộc danh tính caller.
- Vị trí: `services/shieldx-auth/verifier-pool/main.go`.
- Đề xuất: Bảo vệ JWT/mTLS; rate-limit; audit log.
- Hoàn thành khi: Gọi không token trả 401; token không quyền trả 403.

33) Orchestrator admin endpoints không có guard riêng
- Mô tả: `/admin/pools` (và biến thể) cần auth mạnh; hiện chỉ có Admission guard tổng.
- Vị trí: `services/shieldx-gateway/orchestrator/main.go`.
- Đề xuất: Thêm middleware yêu cầu role admin (JWT/mTLS) cho `/admin/*`; rate-limit chặt hơn; audit log.
- Hoàn thành khi: Contract test admin yêu cầu quyền riêng.

34) Missing body size caps ở một số POST endpoints
- Mô tả: Một số service đã có MaxBytesReader (contauth), nhưng endpoints khác (policy-rollout /apply, threatgraph writer) chưa giới hạn kích cỡ.
- Vị trí: `services/shieldx-policy/policy-rollout/main.go`, `services/shieldx-forensics/threatgraph/writer.go`.
- Đề xuất: Áp dụng `http.MaxBytesReader` (ví dụ 10–20MB cho bundle, <1–5MB cho ingests) + Content-Length check + MIME.
- Hoàn thành khi: Tải > limit trả 413 và được log.

35) Path traversal/zip slip khi giải nén zip
- Mô tả: Policy bundle zip đọc qua `zip.NewReader` và load nội dung; cần xác thực đường dẫn nội bộ khi giải và từ chối `..`/absolute paths.
- Vị trí: `services/shieldx-policy/policy-rollout/main.go` (load zip), `shared/shieldx-common/pkg/policy` (nếu có giải nén file).
- Đề xuất: Khi giải file, chuẩn hóa path (filepath.Clean) và từ chối path thoát thư mục; scan entries trước khi xử lý.
- Hoàn thành khi: Unit test zip slip fail-case; linter rule thêm.

36) Cấu hình OPA cho phép /metrics & /health công khai
- Mô tả: Các rego rules demo cho phép /metrics và /health; nếu áp cho prod mà không chặn ingress từ ngoài sẽ lộ thông tin.
- Vị trí: `services/shieldx-policy/policies/advanced.rego`, `.../demo/rules/allow.rego`.
- Đề xuất: Trong prod profile, chỉ allow nội bộ hoặc sau auth; tách bộ rules dev/prod.
- Hoàn thành khi: Prod rules không allow public /metrics; kiểm tra access denied từ ngoài.

37) CORS & Headers hardening chưa đồng nhất
- Mô tả: Chỉ Camouflage API có CORS rõ; services khác không set security headers (HSTS, X-Content-Type-Options, CSP).
- Vị trí: nhiều services `main.go`.
- Đề xuất: Thêm middleware chung cho security headers; bật HSTS ở TLS; tùy chỉnh CSP theo UI/API.
- Hoàn thành khi: Kiểm tra headers qua scan; chuẩn hóa toàn bộ services.

38) QUIC/XDP đường đi mở rộng cần kiểm soát quyền bật
- Mô tả: Ingress có XDP attach/QUIC server optional; cần đảm bảo chỉ bật khi cấu hình phù hợp và quyền root; tránh surface không cần thiết.
- Vị trí: `services/shieldx-gateway/ingress/main.go` (XDP, QUIC), `.../masque/main.go`.
- Đề xuất: Flag gating chặt chẽ; yêu cầu mTLS; audit log khi bật; default off prod nếu không cần.
- Hoàn thành khi: Prod không expose QUIC/XDP trừ khi explicit; kiểm tra port scan.
