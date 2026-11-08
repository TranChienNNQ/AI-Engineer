# Bài 31 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán admission, state, HTTP/error layer và physical work trước khi chạy. Chỉ dùng dữ liệu tổng hợp trong bài.

## 1. Từ model đến API

Vẽ đường từ request đến model/result. Những gì không tồn tại trong hàm predict nhưng cần ở dịch vụ? Tính score khi hai feature bằng 0,5 và giải thích vì sao score ấy chưa là xác suất khẩn cấp đã kiểm chứng.

## 2. Load và feature contract

Kiểm hai request chỉ load model một lần trong lifespan. Warm-up có chứng minh accuracy/SLO không? Nếu đổi thứ tự feature hoặc preprocessing nhưng giữ weights, API có thể vẫn trả 200 không? Một artifact/version nên gắn với những thành phần nào?

## 3. Ba nghĩa sync/async

API chờ response, async def và job 202 khác nhau ở đâu? Gửi job rồi poll GET đến terminal. GET 200 có nghĩa job thành công không? 202 trong lab có bền qua restart không?

## 4. FIFO burst tính tay

Với một worker service 100 ms, Q=2, arrival A/B/C/D/E ở 0/10/20/30/40, tính start/finish/wait và quyết định admission. Tăng Q lên 4 có tăng μ không? Nếu C chỉ có budget 150 ms từ arrival 20 thì nó có bắt đầu đúng hạn không?

## 5. Tải và Little’s Law

Tính μ với 100 ms/job. Trong M/M/1 μ=10, tính W tại λ=5/8/9,5. Với admitted rate 8 và W=0,5, L là bao nhiêu? Nêu giả định và vì sao không dùng attempted λ khi có reject hoặc áp thẳng M/M/1 vào lab hữu hạn.

## 6. Hàng đợi có giới hạn

Chạy overload demo, đọc statuses/queue/busy/model calls. Vì sao queue_size=0 bị từ chối? Semaphore quanh inference có đủ giới hạn số request đang chờ không? Await put vô hạn có chuyển backlog sang chỗ khác không?

## 7. Một deadline xuyên suốt

Client budget 500 ms, đã dùng 120 ms và cần chừa 30 ms, còn bao nhiêu? Nêu mốc bắt đầu budget_ms của API lab. Tại sao không cấp lại 500 ms khi worker dequeue? Monotonic deadline có được gửi nguyên sang máy khác không?

## 8. Hết hạn khi còn chờ

Chạy expired_in_queue và queue_wait_timeout. Hai code khác nhau thế nào? Job expired có đến model không? Vì sao request mới vẫn có thể queue_full dù một entry đã expired? Queue.join có nghĩa mọi job thành công không?

## 9. Hết hạn khi đang chạy

Chạy running timeout, kiểm state, backend cancelled, busy và worker cleanup. Cooperative backend của bài dừng ở đâu? Nếu forward là CPU/native blocking, timer có buộc nó dừng đúng deadline không?

## 10. Hủy caller và thread

Chạy thread demo và caller_cancel. Caller hết chờ có chứng minh thread kết thúc không? Shield làm gì, không làm gì? Khi nào được trả semaphore/worker credit nếu đã đưa inference vào thread?

## 11. Timeout HTTP qua TCP

Chạy socket mode nếu môi trường cho bind loopback. So ReadTimeout client với job_after_client_timeout. Vì sao không mâu thuẫn với ASGI task cancellation check? Phân biệt connect/read/write/pool, timeout tổng và keep-alive timeout.

## 12. HTTP status và retry

Phân loại 200/202/422/500/503, GET record FAILED và network exception. Khi nào 504 phù hợp? Vì sao không gọi mọi local deadline là 408/504? Ba lớp mỗi lớp 3 attempts có thể thành bao nhiêu downstream attempts? Retry-After có phải cam kết capacity không?

## 13. Record retention và bảo toàn state

Đặt record_limit=2, hoàn tất ba job rồi GET job đầu. Thử cancel terminal SUCCEEDED. Vì sao không evict running/expired-but-physical-active record? Snapshot copy có ngăn client sửa state nội bộ không; có phải security sandbox không?

## 14. Lifecycle và workers

Phân biệt livez/readyz/queue full. Cho active và queued job, shutdown với grace ngắn, kiểm task/queue/model cleanup. Hai asyncio worker khác hai Uvicorn worker process thế nào? Với RAM job API, GET đi process khác có vấn đề gì?

## 15. Đánh giá đúng phạm vi

45 check, 11 scenario và score logistic nói điều gì, chưa nói gì? ASGI khác TCP test và load test ra sao? Thiết kế 5 negative case bổ sung về body size, fair quota, model load, native cancellation và restart; ghi rõ case nào hiện chưa được triển khai.

## 16. Dự án nhỏ

Phục vụ một model nhỏ của bài trước trong môi trường synthetic/local: version hóa preprocessing+weights, bounded queue/concurrency, deadline xuyên queue+run, job API nếu cần. Có overload, queued expiry, cancellation semantics, model error, startup/shutdown và kiểm end-to-end qua HTTP. Nếu model không cooperative, nêu/kiểm physical completion; không giả đã dừng chỉ vì client timeout. Tách kết quả đúng số học, lifecycle checks và performance benchmark.
