# Orchestrator Service (Design placeholder)

Theo thiết kế, Orchestrator (cổng 8080) chịu trách nhiệm định tuyến theo policy và thu thập metrics.

Hiện tại trong repo, vai trò này được đảm nhiệm bởi `services/locator/` sử dụng các thư viện ở `pkg/policy/*`.

Trạng thái:
- Endpoint health: `GET /health` trên locator (8080)
- Policy engine: `pkg/policy`, công cụ `policyctl`

Kế hoạch chuyển đổi (không phá vỡ):
1) Duy trì `locator@8080` như Orchestrator tạm thời
2) Tạo service `services/orchestrator/` tái sử dụng `pkg/policy/*`, expose API: `/health`, `/route`, `/metrics`, `/policy`
3) Cập nhật HAProxy/Ingress trỏ về `orchestrator` khi sẵn sàng

Cho đến khi có service riêng, tài liệu và scripts sẽ coi `locator` như Orchestrator.
