# Orchestrator Service (8080)

Orchestrator chịu trách nhiệm policy-based routing, health checking và metrics. Triển khai đã sẵn sàng production với:

- TLS tối thiểu 1.3 (bắt buộc). Hỗ trợ RA-TLS tự xoay vòng chứng chỉ hoặc cung cấp cặp cert/key.
- Thuật toán LB hiệu suất cao: ewma, least_conn, round_robin, rendezvous hashing, p2c (power-of-two với EWMA).
  - Hỗ trợ trọng số backend (weights) cho EWMA/P2C và rendezvous hashing.
- Health probe + EWMA + circuit-breaker per backend.
- Rate limiting per-IP (Redis optional cho phân tán).
- Tích hợp OPA (shadow/enforce) + JSON policy cơ bản.
- Prometheus metrics tại `/metrics`.

API
- GET /health — tình trạng pool/backends, EWMA, lỗi gần nhất
- GET /policy — cấu hình policy hiện hành, OPA state
- POST /route — chọn backend theo policy + LB
- GET /metrics — Prometheus metrics
  
Admin (bảo vệ bởi Admission header nếu cấu hình)
- GET /admin/pools — liệt kê pools và backends
- PUT/POST /admin/pools/{name} — upsert pool
  body: {"urls":["http://10.0.0.2:8081"], "algo":"p2c", "weights": {"http://10.0.0.2:8081":2.0}}
- DELETE /admin/pools/{name} — xóa pool

POST /route request body
{
  "service": "ingress",        // tên pool, hoặc dùng candidates
  "tenant": "t1",
  "scope": "api",
  "path": "/",
  "hashKey": "user:123",       // cho rendezvous
  "algo": "ewma",              // override per-request
  "candidates": ["http://10.0.0.2:8081"]
}

Biến môi trường chính
- ORCH_PORT=8080 (không đổi)
- ORCH_LB_ALGO=ewma|least_conn|round_robin|rendezvous|p2c
- ORCH_POOL_ALGO_<NAME>=ewma|least_conn|round_robin|rendezvous|p2c (mặc định theo pool)
- ORCH_POOL_WEIGHTS_<NAME>={"http://10.0.0.2:8081":2.0,"http://10.0.0.3:8081":1.0} hoặc CSV: "http://10.0.0.2:8081=2.0,http://10.0.0.3:8081=1.0"
- ORCH_BACKENDS_JSON='{"ingress":["http://127.0.0.1:8081"],"guardian":["http://127.0.0.1:9090"]}'
- ORCH_POOL_<NAME>=url1,url2 (chấp nhận dạng rút gọn "host:port" -> auto http://)
- ORCH_POLICY_PATH=policy.json, ORCH_OPA_POLICY_PATH=policy.rego, ORCH_OPA_ENFORCE=1
- ORCH_HEALTH_EVERY=5s, ORCH_HEALTH_TIMEOUT=1500ms, ORCH_EWMA_DECAY=0.3
- ORCH_CB_FAILS=3, ORCH_CB_OPEN_FOR=15s  (circuit breaker)
- ADMISSION_SECRET=... (header HMAC guard), ADMISSION_HEADER=X-Admission
- REDIS_ADDR=host:6379 (rate limit phân tán)
- RATLS_ENABLE=true | ORCH_TLS_CERT_FILE, ORCH_TLS_KEY_FILE

Security ràng buộc (theo Phân chia công việc)
- Không hard-code credentials; luôn validate input; bắt buộc TLS 1.3.
- Không vô hiệu rate limiting/filtering.
- Log đầy đủ security events vào ledger.

Ghi chú
- Service lắng nghe đúng port 8080. Ingress lắng nghe 8081.
- Khi dùng rendezvous hashing, nên truyền hashKey ổn định (ví dụ userID) để dính backend tốt hơn.
 - Trọng số mặc định của backend là 1.0. Trọng số cao hơn ưu tiên backend đó hơn (rendezvous) và giảm cost (EWMA/P2C).
