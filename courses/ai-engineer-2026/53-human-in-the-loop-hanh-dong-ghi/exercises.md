# Bài tập 53 — Human-in-the-loop và hành động ghi có kiểm soát

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Dùng process sạch, thư mục tạm synthetic. Không nhập credential hay ticket thật. Reviewer trong helper là fixture/controller, chưa có con người đánh giá.

## 1. Những mốc không được gộp

Phân biệt investigation answered, proposed, approved, in_flight, unknown và confirmed. Khi nào được nói “note đã ghi”? Có thể nói “sự cố đã sửa” chỉ từ receipt không?

## 2. Review view đủ thông tin

Thiết kế view cho internal note: action/target/version/body/source/uncertainty/expiry và ID nào cần có? Vì sao “agent muốn cập nhật ticket, đồng ý không?” chưa đủ? Hash có chứng minh reviewer đã đọc kỹ không?

## 3. Reviewer và actor scope

Thử agent tự approve và reviewer tenant B đọc draft tenant A. Trường hợp nào phải bị chặn, trước khi có effect nào? Principal/role được lấy từ đâu, có được nhận từ một field model tự viết không?

## 4. Sửa draft và stale view

Approve draft version 1, edit comment, rồi thử approve bằng view cũ hoặc execute ngay. Vì sao cần view/review mới? Draft version, local workflow revision và target version khác nhau ra sao?

## 5. Expiry và source freshness

Approval ở clock 100 hết hạn 160: execute 159/160/161 có policy gì? Memory dùng trong source investigation hết hạn 150 thì sao dù approval còn hạn? Clock suite có đo thời gian thật không?

## 6. Commit trước/sau POST

Vẽ local reserve → target transaction → local result commit. Transaction nào chứa effect+receipt? Local rollback có hoàn tác target không? Vì sao không giữ một write transaction mở trong lúc chờ mạng/human review?

## 7. Mất acknowledgement

Chạy lost_ack. Trước reconcile, local phase/receipt và target notes/receipts là gì? Sau status hợp lệ, có cần POST mới không? Vì sao timeout/5xx chưa tự là no effect?

## 8. Status not_found và retry cùng key

Chạy status_missing_retry. Vì sao status không thấy receipt nhưng target vẫn có note? POST thứ hai có bao nhiêu effect? Thử cùng key với payload khác và giải thích vì sao phải conflict.

## 9. Target version đổi

Sau approve, external actor tăng ticket version lên 2 rồi agent gửi expected_version=1. Ai phải chặn? Có được tự tăng expected version và gửi lại theo approval cũ không? Mirror nguồn bài 52 có phải target authoritative không?

## 10. Kiểm receipt identity

Chạy bad_receipt. Một response state=applied nhưng ticket_id sai có được confirmed không? Ngoài ticket, cần bind những trường nào? Receipt version 2 có chứng minh ticket hiện vẫn ở version 2 không?

## 11. Cancel và compensation

So cancel trước send với cancel_unknown. Vì sao không đổi local flag rồi nói đã hủy mọi effect? Đề xuất compensation cho một note sai và chỉ ra quyền/review/audit cần thiết; phần nào chưa có trong lab?

## 12. Crash và ngân sách bền vững

Chạy crash_before_send/crash_after_response/attempt_budget. So reservations, POST handlers và note effects. Resume có được reset attempts hoặc đổi key không? Status polling có giới hạn gì?

## 13. Quyền hiện tại và effect lịch sử

Revoke actor/reviewer/source trước fresh write, rồi so với source revoke/approval expiry sau một lost acknowledgement. Khi nào được đọc status để xác nhận effect cũ? Trả metadata có đồng nghĩa được export lại source/note body không?

## 14. Demo permit và review artifact

Thử sửa payload mà giữ permit, signature giả và thiếu permit. HMAC demo chứng minh điều gì, chưa chứng minh gì khi key công khai? Vì sao review.html phải escape text và chưa là UI approval production?

## 15. Audit và evaluation

Thiết kế audit kiểm event chain, approval đúng draft, target effect+receipt, idempotency và cross-tenant scope. Vì sao 26 cases/546 checks chưa là LLM/human quality hoặc exactly-once guarantee? Metric nào cần thêm cho người duyệt?

## 16. Tích hợp thật vào agent

Đề xuất tool proposal/review/execute/status cho LLM thật, permission boundary, durable keys/attempts, human UI, target API contract và unknown recovery. Nêu cách không cho retrieved text/memory tự cấp approval; chỉ rõ điều cần kiểm trước khi đưa vào production.
