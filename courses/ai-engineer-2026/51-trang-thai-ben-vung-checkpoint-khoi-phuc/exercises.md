# Bài tập 51 — Trạng thái bền vững, checkpoint và khôi phục

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Các đoạn thực hành chạy từ thư mục bài 51, process sạch, dữ liệu tạm synthetic. Không cần API key hoặc GPU.

## 1. Lưu ticket có phải đã lưu agent?

Giải thích vì sao mở lại world.sqlite chưa resume được run ở bài 50. Liệt kê ít nhất sáu thành phần execution state cần giữ. Context, checkpoint, memory và log khác nhau thế nào?

## 2. Vẽ state machine

Vẽ các phase ready/planning/pending/paused/final, điều kiện chuyển và invariant. Vì sao pending phải chứa action/op_id? Vì sao answered cần final result đã commit?

## 3. Thiết kế ID

Gán run_id, step_id, op_id, attempt, observation_id, checkpoint revision và owner epoch cho search S2 retry sau crash. ID nào giữ, số nào tăng? Có được lấy MCP request ID làm idempotency key mà không biết contract server không?

## 4. Commit trước/sau I/O

Viết thứ tự dành planning turn, commit intent, reserve attempt, gọi tool, commit observation và commit final. Vì sao không giữ database transaction mở trong lúc đợi LLM/tool? Transaction nào phải chứa cả append checkpoint và update head?

## 5. Crash gap

Chạy after_response. Trước crash có bao nhiêu acknowledgement, observation và attempt reservation? Sau resume có bao nhiêu? Tại sao thiếu observation không chứng minh tool chưa chạy? Nếu tool tạo ticket thì phải thay recovery thế nào?

## 6. Reserve mà chưa gửi

So attempt_reserved với after_response. Hai case cùng 5 reservations nhưng acknowledgement khác nhau ra sao? Có nên hoàn budget của attempt cũ chỉ vì không thấy observation không?

## 7. Commit observation rồi chết

Chạy observation_committed. Kiểm S2 không bị gọi lại, nhưng agent vẫn có thể sửa query bằng S4. Vì sao replay observation và retry operation khác nhau?

## 8. Atomic transaction

Chạy terminal_transaction. Worker chết sau INSERT snapshot nhưng trước UPDATE head/COMMIT. Mở lại journal thấy checkpoint finished chưa? Vì sao không có nửa snapshot/head? Phép thử này đã chứng minh an toàn khi mất điện thật chưa?

## 9. Mất kết quả planner

Chạy planning_reserved. Vì sao tổng lượt đề xuất tăng từ 5 lên 6? Nếu LLM thật đã trả response nhưng chưa commit thì điều gì còn unknown về output/cost? Vì sao không ép model sinh lại y hệt để gọi là replay?

## 10. Pause và cached final

Chạy pause_resume và terminal_committed. So ownership, counters và số tool calls sau resume. Final có thể được gửi lại không? Làm thế nào tránh UI tạo hai thông báo cho một run? Deadline execution có tự là TTL của cached answer không?

## 11. Ngân sách và deadline

Chạy attempt_budget, per_op_budget, turn_budget và expired. Vì sao resume không được reset counters hoặc tạo TTL mới? Clock trong suite là thật hay giả? Thiếu gì để có hard deadline giữa mọi bước?

## 12. Hai worker cùng ghi

Tạo hai handle cùng run. Kiểm run_busy, stale_revision và stale_owner sau takeover. Revision và epoch chặn loại lỗi nào? Vì sao fence trong checkpoint DB chưa ngăn remote API call của worker cũ còn sống?

## 13. Quyền hiện tại và binding

Thử task/limit/health config khác, actor inactive và runbook revoke. Có được đọc lại final cũ khi source bị revoke không? Chặn export có xóa raw snapshots/probe/backup không?

## 14. Corruption và migration

Sửa một snapshot trong DB để hash không khớp. Journal phải làm gì? Hash chain có chống người sửa DB rồi tính lại toàn bộ chain không? Schema/release mới cần migration kiểu gì để giữ pending identity và ngân sách?

## 15. Đọc counters và giới hạn thí nghiệm

Giải thích 54 planning reservations, 44 attempt reservations, 43 acknowledgements và 25 worker launches. Mẫu số nào là physical execution, mẫu số nào chưa đủ xác định? Vì sao 13 expected outcomes không là 100% LLM recovery quality hoặc RTO đã đo?

## 16. Mở rộng sang hành động ghi

Thiết kế state machine cập nhật ticket có approval, payload/version, op key bền vững, attempt/unknown/status/reconciliation và audit. Nêu ranh giới transaction local/remote, điều kiện retry, quyền khi resume và failure cases phải kiểm. Chỉ rõ phần nào chưa được code bài 51 thực hiện.
