Cloudflare Edge Workers (Camouflage)
====================================

Thành phần này triển khai các Worker dùng để ngụy trang (camouflage) và điều phối nhẹ ở biên.

- Code/mẫu: tham khảo các tệp trong thư mục này (scripts, worker templates nếu có)
- Tích hợp với policy: tham khảo pkg/policy và policies/ để đồng bộ cấu hình
- Triển khai:
  1. Cài Wrangler và đăng nhập
  2. Cấu hình KV/Buckets nếu cần cho profile
  3. Deploy worker và kiểm tra trên routes mong muốn

Ghi chú: Worker nên có khả năng xoay JA3/TLS fingerprints theo policy nếu được bật.
