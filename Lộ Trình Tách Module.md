Dưới đây là file tài liệu đề xuất (bạn có thể lưu thành `docs/MODULAR_ARCHITECTURE.md`).

````markdown
# Kiến Trúc Module & Quy Ước Đóng Góp (ShieldX)

Mục tiêu:
- Giảm xung đột khi nhiều contributor làm song song.
- Tách rõ ranh giới domain (AI / Honeypot / Monitoring / Policy / Auth / Shared).
- Tạo “bề mặt đóng góp” (contribution surface) rõ ràng, dễ mở rộng.

## 1. Sơ đồ module (hiện tại + định hướng)

```
services/
  ai-service/               (Inference: anomaly, threat scoring)
  honeypot-service/         (Decoys, guardian runtime detection)
  monitoring-service/       (Metrics, SLO evaluation)
  policy-service/ (tương lai)
  auth-service/   (tương lai)
  edge-gateway/   (tương lai: ingress, rate limit)

shared/
  eventbus/                 (In-memory broker abstraction)
  metrics/                  (Common counters / histograms)
  observability/{otel,logcorr}/
  honeypotdeception/        (DSL + deception primitives)
  auth/                     (JWT, RBAC)          ← moved từ pkg/auth
  policy/                   (Policy load/eval)   ← moved từ pkg/policy
  ledger/                   (Append audit log)
  types/                    (DTO / typed events)
  config/                   (Env & file loader)
  logging/                  (Structured logger wrapper)
```

Nguyên tắc phụ thuộc (allowed):
```
shared/*  ← (không phụ thuộc vào services)
services/X → shared/*
services/X → (HTTP / event bus) → services/Y (không import code trực tiếp)
```
Không được: `services/a` import code Go trong `services/b`.

## 2. Ranh giới chức năng

| Module | Trách nhiệm | Không làm |
| ------ | ----------- | --------- |
| ai-service | Nhận sự kiện chuẩn hóa, phân tích & trả kết quả anomaly/threat | Lưu trữ dài hạn, xuất metrics trực tiếp (chỉ publish) |
| honeypot-service | Sinh/thu thập tương tác decoy, guardian logic ban đầu | Phân tích AI |
| monitoring-service | Thu thập metrics, SLO, counters, kết quả phân tích | Phân tích sự kiện thô |
| shared/eventbus | Abstraction publish/subscribe | Logic domain |
| shared/policy | Load & evaluate chính sách | Xử lý network |
| shared/auth | JWT, RBAC | Policy evaluation |
| shared/ledger | Ghi forensic / audit append-only | BI / analytics |

## 3. Loại giao tiếp

| Loại | Khi dùng | Ví dụ |
| ---- | -------- | ----- |
| Event Bus (async) | Chuỗi honeypot → AI → monitoring | honeypot.request, analysis.result |
| HTTP (sync) | Query trạng thái, health, admin | GET /healthz, GET /model/status |
| (Tương lai) Kafka/NATS | Scale cao | Streaming sản xuất |

## 4. Quy ước thư mục trong mỗi service

```
services/<name>/
  cmd/<binary-name>/main.go
  internal/            # Chỉ dùng trong service
  pkg/ (optional)      # API export (hook) cho integration/harness
  api/ (optional)      # OpenAPI / proto / schema
  README.md
  Makefile (tối giản hoặc dùng root)
  testdata/
```

## 5. Quy tắc “Ràng buộc” (Binding Rules)

1. Không import qua lại giữa các services (chỉ qua shared hoặc giao tiếp runtime).
2. Không dùng `internal/` của service khác.
3. Mỗi module phải có README mô tả scope + out-of-scope.
4. Event mới: bắt buộc khai báo ở `shared/types/events.go`.
5. Tên topic event: `domain.action` (vd: `honeypot.request`, `analysis.result`)
6. Mọi HTTP handler phải:
   - Log: request id / path
   - Timeout context nếu > 2s (trừ stream)
7. Metrics:
   - Chỉ monitoring-service expose `/metrics`
   - Service khác chỉ push/publish qua event hoặc shared/metrics collector.
8. Không hardcode config → dùng shared/config.
9. Không dùng global mutable state ngoài: logger, config cache, metrics registry.
10. Commit không được chứa “WIP” khi mở PR.
11. Unit test: tối thiểu 1 test file cho module mới.
12. Tên branch: `feat/<area>-<short>`, `fix/<area>-<short>`, `refactor/<area>-<short>`.
13. Không thêm dependency lớn mà không mở issue “dep-proposal”.
14. Không vendor dependencies (dùng `go mod tidy`).

## 6. Checklist khi tạo module mới

```
[ ] Tạo thư mục /services/<tên> hoặc /shared/<tên>
[ ] README.md (mục tiêu, boundary, ví dụ dùng)
[ ] go.mod (nếu service mới)
[ ] cmd/<service>/main.go (có /healthz)
[ ] Thêm vào go.work (nếu module mới)
[ ] Thêm target vào Makefile (build + test)
[ ] Thêm test tối thiểu
[ ] Cập nhật docs/MODULAR_ARCHITECTURE.md (mục “Thay đổi gần đây”)
[ ] Tạo issue “Module: <tên> tracking”
```

## 7. Chuẩn event (shared/types/events.go ví dụ)

````go
// Event chuẩn hóa gợi ý
type HoneypotRequest struct {
  Timestamp time.Time
  Protocol  string
  SourceIP  string
  Path      string
  PayloadSz int
}

type AnalysisResult struct {
  CorrelationID string
  IsAnomaly     bool
  Score         float64
  ModelVersion  string
  SourceEvent   HoneypotRequest
}
````

## 8. Quy trình đóng góp (Contributor Flow)

1. Fork & clone.
2. Chạy: `make bootstrap` (chuẩn bị tool).
3. Chọn issue (label: good-first-issue / help-wanted).
4. Tạo branch mới.
5. Implement + test: `go test ./...`
6. Run static check: `make lint`
7. Cập nhật docs nếu thay đổi public behavior.
8. Mở PR: mô tả ngắn + ảnh/JSON (nếu có).
9. Pass CI (build, unit test, smoke integration).
10. Chờ review tối thiểu 1 maintainer.

## 9. Phụ thuộc & sắp xếp (Dependency Discipline)

| Layer | Có thể phụ thuộc | Không được phụ thuộc |
| ----- | ---------------- | -------------------- |
| shared/* | (chỉ std lib + 3rd nhỏ) | services/* |
| services/* | shared/* | services khác trực tiếp |
| cmd/* harness | shared/* + các service packages export | internal/ của service khác |

Kiểm tra nhanh:
```
grep -R "services/.*/internal" -n | grep -v "$(go list)"  # không được có ngoại lệ
```

## 10. Migrate lịch sử (History Log)

| Wave | Nội dung | Trạng thái |
| ---- | -------- | ---------- |
| 1 | Tách ai-service, honeypot-service, monitoring-service | DONE |
| 2 | Move pkg/ml → ai-service/internal/ml | DONE |
| 3 | Move metrics → shared/metrics | DONE |
| 4 | Honeypot decoys + deception tách | DONE |
| 5 | Guardian + Observability (otel/logcorr/slo) | DONE |
| 6 | Event bus in-memory + harness | DONE |
| 7 | (Kế hoạch) Move auth, policy, ledger → shared | PENDING |

## 11. Plan tiếp theo (Next Planned Modules)

| Module | Loại | Ghi chú |
| ------ | ---- | ------- |
| shared/auth | cross-cutting | Trước khi tách auth-service |
| shared/policy | logic | Chuẩn hóa interface evaluate() |
| shared/ledger | util | Ghi forensic append-only |
| policy-service | service | Khi cần hot reload / push bundles |
| edge-gateway | service | Khi có nhu cầu central ingress |

## 12. Chất lượng & Testing

| Loại test | Mục tiêu | Ví dụ |
| --------- | -------- | ----- |
| Unit | Hàm / package | anomaly detector |
| Integration | Chuỗi event | harness smoke |
| Contract | Event schema | JSON schema validate |
| Load (tùy chọn) | Throughput | 1k req/s decoy |

Smoke test chạy:
```
./scripts/smoke_integration.sh
```

## 13. Logging & Tracing

- Format: JSON line.
- Trường tối thiểu: `ts, level, service, msg, correlation_id`.
- Tracing (Optional Wave): dùng shared/observability/otel; bật qua env: `OTEL_ENABLED=1`.

## 14. Bảo mật (Security Notes)

| Chủ đề | Quy tắc |
| ------ | ------- |
| Secrets | Không hardcode; dùng env + vault (tương lai) |
| JWT | Exp ≤ 15m refresh token (tương lai) |
| Input | Validate trước khi chuyển sang AI |
| File I/O | Ledger append-only, không mutate file cũ |

## 15. Quy tắc commit & PR

- Commit msg: `feat(ai): add model status endpoint`
- Types: feat, fix, refactor, chore, docs, test, perf.
- PR tiêu đề = commit chính.
- > 400 dòng thay đổi: tách nhỏ (bắt buộc nếu có logic + test + docs lẫn lộn).

## 16. Tự động kiểm (Automation Gợi ý)

Thêm vào CI:
```
go vet ./...
golangci-lint run
go test -race ./...
./scripts/smoke_integration.sh
```

## 17. FAQ

Q: Muốn gọi trực tiếp code của AI trong honeypot?  
A: Không. Gửi event hoặc HTTP.

Q: Thêm topic sự kiện mới ở đâu?  
A: Cập nhật shared/types + docs / Event Registry.

Q: Có cần mở issue trước khi thêm dependency lớn?  
A: Có (label: dep-proposal).

---

Maintainers có thể cập nhật phần “History Log” sau mỗi merge lớn.
````

Bạn muốn mình tạo thêm các README khung cho `shared/auth`, `shared/policy`, `shared/ledger` không? Chỉ cần yêu cầu là được.
````

## 18. Event Registry (Danh mục topic chuẩn)

| Topic | Producer | Consumer(s) | Payload (tối thiểu) |
|-------|----------|-------------|----------------------|
| honeypot.request | honeypot-service (decoy-http) | ai-service | path, ua, remote, ts |
| analysis.result | ai-service | monitoring-service | is_anomaly, score, confidence |
| policy.reload (tương lai) | policy-service | gateway, ai-service | version, changed_keys |
| auth.token.revoked (tương lai) | auth-service | gateway, policy-service | jti, user_id |

Quy tắc đặt tên: `<domain>.<action>`; tránh thêm hơn 2 segment trừ khi phân tầng bắt buộc.

## 19. Ownership (Module Owners)

| Area | Owners (GitHub handles) | Escalation |
|------|-------------------------|------------|
| ai-service | @ml-core, @sec-ml | @arch-core |
| honeypot-service | @decoy-team | @arch-core |
| monitoring-service | @obs-team | @arch-core |
| shared/auth | @security-platform | @arch-core |
| shared/policy | @policy-engine | @arch-core |
| shared/eventbus | @platform-runtime | @arch-core |

Maintainer thêm / thay đổi bảng này qua PR có label `governance`.

## 20. Versioning nội bộ

- shared packages: sử dụng pseudo-version qua commit; khi stable đánh tag `shared/vX.Y.Z`.
- services: semantic version riêng nếu được phát hành độc lập (ví dụ container tag `ai-service:v0.2.0`).
- Breaking change: phải tăng MINOR nếu vẫn pre-1.0; post-1.0 tuân thủ SemVer.

## 21. Chính sách Breaking Change

| Mức | Ví dụ | Yêu cầu |
|-----|-------|---------|
| Minor | Thêm field optional vào event | Cập nhật Event Registry + docs |
| Major | Đổi tên topic, xóa field bắt buộc | Proposal issue + 2 maintainer approve |
| Critical | Thay đổi payload format gây panic consumer | Deprecation notice ≥2 release trước |

Phải có mục "Migration" trong PR description cho Major & Critical.

## 22. Externalization Roadmap (Broker / Storage / Auth)

| Hiện tại | Mục tiêu ngắn hạn | Mục tiêu dài hạn |
|----------|-------------------|-------------------|
| In-memory event bus | Abstraction interface Broker | Kafka / NATS pluggable |
| File-based ledger | Optional async writer + rotate | Central audit pipeline (e.g. Loki) |
| JWT in-memory revocation | Pluggable store (Redis) | Multi-region revocation sync |
| Policy load file | Hot reload + ETag | Policy distribution service |

## 23. Testing Matrix mở rộng

| Dimension | Scope | Tool/Gợi ý |
|-----------|-------|------------|
| Unit | shared/*, internal logic | go test ./... |
| Integration | harness event flow | scripts/smoke_integration.sh |
| Contract | Event JSON schema | (tương lai) schema tests |
| Performance | AI predict throughput | benchmark test + pprof |
| Security | JWT tampering, policy bypass | fuzz + table tests |

## 24. PR Template (Gợi ý)

```markdown
### Mục tiêu

### Loại thay đổi
- [ ] feat  - [ ] fix - [ ] refactor - [ ] docs - [ ] test - [ ] perf - [ ] chore

### Liên quan
Issues: Closes #ID

### Chi tiết
<mô tả ngắn gọn>

### Validation
- [ ] go build ./...
- [ ] go test ./...
- [ ] scripts/smoke_integration.sh PASS

### Event / API Impact
- Topic mới? (Y/N) Nếu Y: cập nhật Event Registry
- Breaking? (Y/N) Nếu Y: kèm Migration

### Screenshots / Logs (optional)

### Checklist bổ sung
- [ ] Cập nhật docs
- [ ] Không thêm dependency lớn
```

## 25. Code Review Guidelines

1. Giữ PR < 500 LOC net changes (tách phases nếu lớn).
2. Reviewer xác minh: boundary giữ nguyên? event mới có docs? tests tối thiểu?
3. Không approve khi có TODO bỏ ngỏ trong code (trừ khi gắn issue rõ ràng).
4. Log phải có ngữ cảnh (service, action, id).
5. Panics bị cấm trong path runtime bình thường (dùng error + fallback).

## 26. Performance Guardrails (Baseline)

| Path | Ngưỡng mục tiêu | Ghi chú |
|------|------------------|--------|
| ai-service Predict | p95 < 50ms | Batch inference cân nhắc sau |
| honeypot HTTP decoy | p95 < 10ms | Không tính intentional delay (tarpit) |
| monitoring /metrics | p95 < 30ms | Scrape interval 15s |

## 27. Observability Default Fields

| Field | Mục đích |
|-------|----------|
| trace_id | Correlate multi-service |
| span_id | Child operations |
| correlation_id | Business correlation (chuỗi sự kiện) |
| model_version | AI model tracking |
| anomaly | Bool flag |

## 28. Deprecation Process

1. Đánh dấu @deprecated trong comment (GoDoc) + lý do.
2. Thêm vào bảng Deprecations (mục 29).
3. Sau ≥2 minor: loại bỏ (nếu không phản đối trong issue).

## 29. Deprecations (Theo dõi)

| Item | Deprecated Since | Target Removal | Ghi chú |
|------|------------------|----------------|---------|
| (trống) | - | - | - |

## 30. Security Review Checklist (nhỏ)

| Câu hỏi | Đáp ứng? |
|---------|----------|
| Input validation đủ chưa? |  |
| JWT expire hợp lý? |  |
| Secrets qua env, không commit? |  |
| Logging không lộ PII? |  |
| Policy evaluation có default deny? |  |

## 31. Glossary

| Thuật ngữ | Nghĩa |
|-----------|-------|
| Correlation ID | Chuỗi gắn kết nhiều event liên hệ |
| Decoy | Dịch vụ mồi để thu hút attacker |
| Anomaly Score | Điểm định lượng bất thường |
| SLO | Service Level Objective |

---
Tài liệu này là "living document" — mở PR để cải tiến khi thực tế thay đổi.