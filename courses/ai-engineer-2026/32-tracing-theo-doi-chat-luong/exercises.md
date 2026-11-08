# Bài 32 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán trace graph, mẫu số và chất lượng trước khi chạy. Không cần tài khoản, collector hoặc dữ liệu thật.

## 1. Chọn tín hiệu

Với “API chậm”, “nhiều request503”, “mobile bị ưu tiên sai” và “đã hết trace trong backend”, cần log/metric/trace/quality record nào? Vì sao HTTP200 hoặc span UNSET chưa chứng minh prediction đúng?

## 2. Đọc một trace

Chạy trace mode; vẽ parent graph 7 span cho một prediction, ghi kind và service. Vì sao job.process có parent queue.publish dù producer đã kết thúc? Có được cộng mọi span duration thành latency không?

## 3. W3C Context và ID

Giải nghĩa bốn phần traceparent mẫu. Trace ID/span ID/prediction ID khác nhau thế nào? Header toàn zero phải được xử lý ra sao? Một trace ID hợp lệ có chứng minh user được quyền đọc ticket không?

## 4. Queue context và concurrency

So trace bình thường với --drop-context: statuses, span_count, trace_count và parent nào đổi? Vì sao worker tạo ở startup cần context trong message? Dùng global current_trace_id có thể sai thế nào khi hai request xen kẽ?

## 5. Critical path

Hai operation song song 40/60 ms rồi reducer10 ms: tính end-to-end lý tưởng và tổng work. Nêu khác biệt với hai operation tuần tự. Vì sao parent/child timestamps của asynchronous work không luôn containment?

## 6. Histogram tính tay

Với latency10/20/30/40/500 ms, tính mean, nearest-rank p95 và cumulative buckets20/50/100/+Inf. Nếu chỉ biết bucket≤100 có4/5, bạn có biết chính xác p95 không? Vì sao không lấy trung bình p95 của replica?

## 7. Mẫu số và cardinality

Một API request gọi model một lần tạo bao nhiêu HTTP completion trong lab? Vì sao queue_wait histogram không gồm rejected request? Tính 3 routes×2 versions×3 status và khi thêm10.000 predictionID. Thiết kế labels không chứa raw URL hoặc ID vô hạn.

## 8. Sampling và exporter

Chạy --no-sampling và --export-fail. So prediction count, HTTP counters, spans, logs và failure counter. Vì sao không tính tỷ lệ lỗi từ chỉ error traces đã giữ? Tail sampling có khôi phục span đã drop ở head không?

## 9. Data minimization

Giải thích test SECRET32 trong body/header/query/path/exception. Vì sao tắt auto exception recording và dùng error code? Attribute key allowlist có ngăn secret trong mọi value/name không? Hash user ID có tự ẩn danh và giảm cardinality không?

## 10. Join feedback đúng thời điểm

Vẽ Prediction và Feedback records, chỉ ra join key và ba mốc thời gian. Nếu label_event_at=105 nhưng available_at=200, snapshot as_of=120 có được dùng không? Duplicate, wrong model version và label conflict phải xử lý ra sao?

## 11. Nhãn về chưa đủ

Tính accuracy/coverage của current ở as_of120 và220. Vì sao 6/6 không đồng nghĩa10/10? Khi maturity lag đủ nhưng chỉ còn6 nhãn, state và quality alert phải thế nào? Maturity/80%coverage có chứng minh không selection bias không?

## 12. Confusion matrix và slices

Từ TP1/FP4/FN0/TN5, tính accuracy/precision/recall/F1. Đọc mobile/web n và coverage; vì sao recall100% và overall60% vẫn cần điều tra? Với zero positive labels, recall nên là0 hay undefined theo contract nào?

## 13. Drift và quality

Tính TV giữa [0.5,0.5] và [0.2,0.8]. So current và shift: proxy alert có tương đương quality alert không? Phân biệt feature drift, concept drift và training-serving skew; histogram đơn biến có bỏ sót thay đổi nào?

## 14. Alert và chẩn đoán

Đọc alert overall/mobile của mature cohort. Viết runbook cho “API200 nhưng mobile accuracy giảm”; trước rollback cần kiểm gì về join, n, coverage, versions và labels? Quy tắc thresholds lab có phải kiểm định thống kê không? Bài có tự gửi thông báo/retrain không?

## 15. Phạm vi bằng chứng và tái lập

Phân biệt48 checks,6 tracing scenarios,2 live predictions,30 replay predictions và14 span. ID/timestamp thay đổi thì phần nào cần ổn định? ASGI propagation chứng minh gì, chưa chứng minh gì về TCP/collector/OTLP? File runtime nào không nên đóng vào ZIP chia sẻ?

## 16. Dự án nhỏ

Instrument một workflow synthetic nhiều bước: có queue hoặc HTTP boundary, propagation đúng và lỗi có chủ đích, counters không phụ thuộc sampling, privacy canary. Tạo prediction/feedback ledger riêng với version và delayed labels; có thiếu/trùng/conflict, cohort maturity, slice và một proxy không đủ phát hiện quality regression. Viết report tách operation success, telemetry completeness và labeled quality; chưa làm phần nào thì ghi rõ.
