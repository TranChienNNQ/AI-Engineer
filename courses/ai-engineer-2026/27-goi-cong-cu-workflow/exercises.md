# Bài 27 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán state/effect trước khi chạy. Mọi tình huống dùng dữ liệu tổng hợp; không cần API key hoặc ticket thật.

## 1. Vòng gọi công cụ

Mô tả các bước từ user hỏi ticket đến model nhận result. Ai tạo proposal, ai kiểm quyền, ai thực thi, ai quyết định dừng? Tại sao một object đúng schema chưa chứng minh một hàm đã chạy? Protocol nội bộ của bài có phải API của mọi provider không?

## 2. Contract tool

Thiết kế mô tả `append_note` có mục đích, input/output, side effect, khi không dùng và lỗi. Vì sao không đưa field tùy ý `tenant`, `grant`, `shell_command` cho model tự điền? Thử unknown tool, extra field, bool version, note trắng và duplicate JSON key.

## 3. Ghép call và result

Tạo hai read call, trả kết quả theo thứ tự ngược. Ghép bằng ID. Thử missing ID, duplicate ID và sai tên tool. Vì sao call ID mới không đủ để xác định một write là ý định mới?

## 4. Phụ thuộc và pattern

Vẽ đồ thị: ticket → {runbook, health} → draft → review → append. Đâu là chaining, routing, fan-out/fan-in, gate? Vì sao không chạy lookup_runbook theo service đoán trước hoặc song song hai write cùng ticket/version?

## 5. Đường găng

Từ 40/80/120/50/60 ms, tính sequential, parallel critical path và tổng công việc tool. Có nên cộng thời gian con người review vào 350 ms trong ví dụ không? So với wall_ms của `parallel.py`, có thể khẳng định latency mạng được cải thiện đúng tỷ lệ không?

## 6. Xác suất và retry

Tính `.98×.97×.99` và `1−(1−.8)^3`. Nêu giả định để dùng từng phép tính. Dependency outage chung làm thay đổi cách hiểu thế nào? Nếu 4 tầng mỗi tầng tối đa 3 attempt thì một request có thể gây tối đa bao nhiêu attempt ở tầng cuối?

## 7. Ngân sách và backoff

Budget 500 ms, hai attempt mỗi attempt 200 ms và backoff 50 ms: còn bao nhiêu? Lần thứ ba có thể bắt đầu với full timeout 200 ms không? Chạy fixture read_retry, call_limit và time_budget; phân biệt call count với node/state count.

## 8. Partial result

Chạy optional_health_down, no_runbook và bad_read_output. Tại sao health có thể unknown nhưng runbook rỗng cần chuyển người xử lý? Có nên bắt mọi Exception rồi thay bằng `{}` không? “Không biết health” khác “dịch vụ healthy” ra sao?

## 9. Review là một sự kiện có thẩm quyền

Chạy prepare, thử execute trước approve, rồi reject. Sau approve, sửa note hoặc source version: điều gì xảy ra? Lab có thực sự xây review UI/login không? Được phép tự động theo workflow đã xác lập có bắt buộc hỏi lại mọi action không?

## 10. Idempotency và key conflict

Thực thi một write, gửi lại cùng args/key nhưng đổi call ID, rồi đổi note mà giữ key. Đếm effect. Phân biệt grant, request hash và idempotency key. Vì sao dùng key mới sau timeout có thể gây duplicate dù schema/quyền đều đúng?

## 11. Kết quả chưa xác định

Chạy lost_write_response, unknown_no_receipt và invalid_write_output. Nêu state/effect trước/sau reconcile. Vì sao output invalid ở write không được coi chắc chắn “chưa có effect”? Tại sao không tìm thấy receipt vẫn có thể phải giữ UNKNOWN?

## 12. Thu hồi quyền và phiên bản

Review rồi đổi quyền write hoặc ticket version trước execute. Chứng minh không có effect. Nếu quyền bị thu hồi sau một write đã commit nhưng trước reconcile, nên kết luận operation thất bại hay chỉ không còn đủ quyền đối soát? Nêu giới hạn của lock trong RAM.

## 13. Song song và hủy task

Chạy read_fanout, rồi thử health unavailable với optional_health=True. Chuyện gì khác khi lỗi thuộc nguồn bắt buộc hoặc có timeout ở group? Có thể suy task local bị cancel nghĩa là server đã rollback không? Demo có scheduler nhiều worker, concurrency cap tổng hoặc remote API không?

## 14. Metric đúng mẫu số

Từ report, đếm các state trước/sau đối soát. Giải thích tại sao 22/22 case đạt kỳ vọng không phải 22 task ghi thành công. Chọn metric riêng cho quality note, permission, duplicate write, partial evidence, UNKNOWN, latency và cost; phần nào lab chưa đo?

## 15. Compensation và giao dịch nhiều hệ thống

Mở rộng ý tưởng: thêm note xong thì gửi email thông báo. Email thất bại. Xóa note có phải rollback hoàn hảo không? Đề xuất state, idempotency cho từng effect và cách báo partial success. Khi nào cần transaction/outbox thay vì một chuỗi try/except?

## 16. Dự án nhỏ

Tạo workflow hỗ trợ một nghiệp vụ backend khác, có ít nhất một route, hai read độc lập và một write. Tách proposal/contract/runtime/backend, giữ danh tính và review ngoài model args. Có case retry read, nguồn bắt buộc thiếu, từ chối review, stale version, lost response và unknown không giải được. Báo expected state/effect, trace và giới hạn. Nếu thêm LLM, đo tool choice/args/task quality riêng; không thay số fixture bằng tuyên bố model đã đạt 100%.
