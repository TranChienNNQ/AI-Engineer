# Bài 34 — Bài tập

[Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán trước khi chạy. Các ví dụ tiền/token đều giả định, không phải giá nhà cung cấp.

## 1. “Nhanh hơn” nghĩa là gì?

TeamA báo mean20 ms trên20 request tuần tự; teamB báo p95200 ms ở100 arrival/giây, có2% lỗi. Có đủ cơ sở chọnA nhanh hơn không? Liệt kê ít nhất sáu thông tin cần thêm, gồm workload, population, window và quality.

## 2. Bộ đếm không được mất mẫu

Có100 planned arrival,10 generator_drop,90 sent. Trong90 sent có70 valid200,15 HTTP503 và5 timeout. Trong70 success có50 đạt scheduled latency SLO.

Tính success/sent, success/planned, attainment/planned. Với window2 giây, tính attempt rate, successful throughput và goodput. Vì sao không xếp10 generator_drop thành503?

## 3. Mean và nearest-rank percentile

Tính mean/p50/p95/p99 cho10/20/30/40/500 ms. Tại sao p99 ở đây không đủ để khẳng định production p99? Tạo code cho empty sample, invalid q và NaN; nêu ý nghĩa None.

## 4. Đặt lại đồng hồ

Một request planned0, sent100, done120 ms, success; SLO40 ms. Tính dispatch lag/client latency/scheduled latency và good hay không. Thêm một generator_drop ở planned10 rồi tính attainment. Đâu là dấu hiệu generator không tạo được tải?

## 5. Queue, execution và Little's Law

Giả định steady population có5 accepted request/giây, mean time in system .2 giây. Tính mean in-system count. Có thể suy số worker cần đúng bằng số đó không? Nếu offered rate10 nhưng chỉ5 accepted, dùng10×.2 có đúng cùng boundary không?

Hai nhánh40/60 ms chạy song song, reduce10 ms. Tính critical path và total work, nêu vì sao không cộng span lồng nhau.

## 6. Closed/open và omission

Một worker service100 ms. So10 arrival ở0/100/…/900 với0/20/…/180, queue đủ lớn. Tính dãy latency, mean và p95 của từng lịch. Run nào phù hợp10 user events vẫn tới độc lập mỗi20 ms? Closed model có luôn sai không?

## 7. Burst và goodput

Mô phỏng8 request cùng lúc,2 worker, queue2, service20 ms, SLO30 ms. Tính admitted/rejected, completion timeline, success/error latency, số good, success rate và goodput trên cả cửa sổ drain. Kiểm FIFO, giải thích rejected latency0 là giả định.

## 8. Sweep không chỉ chọn RPS lớn nhất

Đọc simulation sweep20/60/100/140/200. Vì sao từ100 lên140/200 không làm goodput tăng tương ứng? Nếu báo mean trên cả503 nhanh thì có thể gây hiểu nhầm gì? Capacity của mô phỏng có phải capacity của lab HTTP/GPU không?

## 9. Bounded generator

Dùng mock transport bị giữ bởi một event. Generator open4 arrival, rate500/s, max_inflight1, lag budget đủ lớn. Giữ request đầu đến khi đã phát xong lịch rồi mới mở event.

Dự đoán sent/drop và lỗi nếu request đầu trả503. Vì sao không đặt sleep sau mỗi response rồi gọi đó là constant-arrival-rate? Chạy code kiểm, không gửi tới URL bên ngoài.

## 10. Đọc một phép đo TCP thật

Chạy `lab.py --mode all --repeats 2 --out <thư mục mới>`. Với từng profile ghi planned/sent/drop/success/error, window, p95 success/scheduled và goodput. Kiểm conservation, queue/active bounds.

Nêu ít nhất bốn giới hạn của phép đo. Nếu open-250 bị nhiều generator drop, có thể kết luận server capacity250/s không? Vì sao code không assert cố định p95 hoặc số503 của máy tác giả?

## 11. Streaming boundaries

Gửi0 ms, token tại120/150/190/220, response end240. Tính TTFT, ITL, TPOT tới last token, TPOT dùng response end và E2E. Nếu chỉ có một output token cần xử lý sao? Một SSE chunk có mặc định là một token không?

## 12. Warmup, lặp và aggregation

Một run có3 warmup,24 measured request và drain. Giải thích window của lab; warmup có nhất thiết bị loại khỏi chi phí không? Có thể lấy mean của hai p95 để gọi global p95 không? Thiết kế cách so hai release tránh order/cache bias và nói phần lab chưa làm.

## 13. Tính chi phí token và retry

Với rate giả định input2/cached.5/output8 USD trên một triệu token, tính từng attempt A1/A2/B1 trong ledger. Tính total, cost/attempt, cost/successful task và cost/qualified task. Vì sao không cộng cached input thêm vào tổng input? Vì sao không bỏ failed attempt đã có usage?

## 14. Usage chưa biết và duplicate

Thêm taskC failed/usage=None vào ledger. Có nên coi costC=0 không? Nêu giá trị total/per-unitcost nên báo. Nếu cùng(task,attempt) xuất hiện hai lần có nội dung khác nhau, bạn cần reconcile gì thay vì cộng cả hai hoặc drop tùy ý?

## 15. Self-host cost và quality tradeoff

Cấp một replica trọn giờ giá3 USD,60.000 request/54.000 success/45.000 qualified. Tính ba unit cost. Nếu compute bận30% giờ, có tự nhân tổng cost với .3 không? Đề xuất những khoản/scope còn thiếu khi so self-host với API; không cần chọn vendor.

## 16. Protocol cho sản phẩm AI

Thiết kế protocol so hai release RAG hoặc LLM: workload/mix/cache, clocks/window, open/closed, warm/cold, caps, repeats, quality/SLO và cost attribution. Nêu ba tiêu chí dừng/không được kết luận capacity, ba kiểm tra sau tối ưu và cách gắn release/rollback.

Kiểm `stable_facts()` phân biệt phần nào phải tái tạo nhất quán và phần nào là số đo có thể thay đổi. Không yêu cầu TCP raw timings giống hệt từng run.
