# Bài 34 — Đào sâu phép đo và unit economics

[Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Một protocol trước khi có bảng kết quả

Viết trước câu hỏi, release/source hash, environment, dataset/mix và các giới hạn. Chọn đơn vị công việc: một HTTP attempt, một logical inference, một turn hay một task nhiều tool call. Ghi tiêu chí response hợp lệ và quality đủ dùng.

Định nghĩa latency endpoints, SLO/window/denominator, workload open/closed, connection reuse, timeout/retry và warmup/drain. Sau đó mới chạy và giải thích số đo. Nếu thay threshold/SLO sau khi nhìn kết quả, cần ghi đó là phân tích khám phá, không là điều kiện đã đăng ký trước.

Metadata lab có Python/platform/package versions và source hash của mọi .py + requirements.txt. Source hash giúp nhận diện bộ code, chưa bao toàn OS/kernel/hardware/load/background state hoặc cryptographic provenance. Không suy report trùng code hash thì mọi timing phải giống nhau.

## 2. Đồng hồ và resolution

`perf_counter()` đo elapsed time, tính cả thời gian await/sleep/scheduler. CPU time loại một số thời gian chờ nên không thay thế end-to-end duration. Clock resolution khác accuracy của thí nghiệm và overhead của instrumentation. Đọc/ghi log mỗi request có thể ảnh hưởng phép đo.

Latency client của bài gồm HTTPX pool/connect/write/read, parsing và numerical validation. Server queue/run bắt đầu sau validation FastAPI và không bao toàn frontend. Residual client−queue−run có nhiều loại overhead; chưa là packet RTT thuần.

Không trừ một timestamp monotonic từ máyA với timestamp từ máyB. Dùng local duration/correlation hoặc phương pháp clock synchronization có sai số được tính đến. Tracing có thể giúp phân tích path nhưng span sampling/collector lag không nên thay counters của toàn phép đo.

Simulation dùng Fraction cho thời gian nội bộ và so SLO tại boundary trước khi chuyển sang số JSON để hiển thị. Đây là exact rational model, không dùng real clock. Các arrival sweep cũng được tạo bằng rational i×1000/rate; tránh để lỗi nhị phân float quyết định một completion có đúng bằng boundary hay không.

## 3. Một vòng closed có thể đúng hoặc sai câu hỏi

Nếu sản phẩm có số user cố định và mỗi user luôn chờ phản hồi trước hành động tiếp theo, closed user + think time có thể hợp lý. Nếu arrivals từ queue/sự kiện bên ngoài tiếp tục tới khi service chậm, open workload phù hợp hơn để bộc lộ backlog và admission.

Một closed test không quan sát nhu cầu độc lập “đáng lẽ” tới giữa hai request chậm. Coordinated omission không có nghĩa sửa lại bất kỳ closed result nào bằng nhân một hệ số; cần hiểu arrival process. Lab so hai timeline tự tạo để chỉ ra khác biệt, chưa triển khai statistical correction hoặc replay production traces.

Generator open của bài chờ `origin+i/r` thay vì sleep sau completion. Late arrival không được tự động dời toàn bộ lịch. Nếu quá max_lag_ms hoặc cap in-flight thì drop có record; các arrival còn lại tiếp tục theo lịch. Điều này có nghĩa test không tạo đủ offered workload khi generator yếu, cần xem run không đạt yêu cầu tải thay vì công bố capacity của server.

Cap áp cho in-flight tasks của open mode. Closed concurrency là số user, giới hạn1..8; max_inflight không điều chỉnh closed mode. Count tối đa96/rate1..500/repeats1..3 nhằm giữ bài thực hành nhỏ. Đây chưa là generator thiết kế cho stress hàng triệu request.

## 4. Window, cohort và drain

Finite cohort hỏi: “N arrival đã định có kết quả gì và mất bao lâu để kết thúc?”. Fixed wall window hỏi: “Trong khoảngT có bao nhiêu completion?”. Hai câu hỏi cho rate khác nhau ở đầu/cuối, nhất là queue dài. Không lấy numerator finite cohort đã drain chia cho arrival-only duration rồi so với rate completion trong wall window của cấu hình khác.

Lab TCP bắt đầu measurement sau3 warmup đã drain/reset. Open horizon=count/rate, last planned arrival=(count−1)/rate. Window kéo dài ít nhất horizon, thêm phần client terminal sau horizon nếu có; closed window chạy đến client terminal cuối. Server có thể còn chạy sau timeout; harness đợi idle riêng, nhưng không âm thầm đổi window client để tính lại rate.

Warmup/request reset chỉ loại chúng khỏi population và summary của từng run; không chứng minh startup/cache/thermal đã ổn định. Nếu tính cost của toàn phiên cấp tài nguyên, warmup/idle/teardown vẫn có thể thuộc chi phí tùy scope. Không lấy measured successful calls làm toàn bộ công việc đã trả tiền.

Lab chưa dùng external arrivals/fixed duration/soak test hoặc đo cold model load. Các profile được chạy theo thứ tự cố định, nên hai repeat không loại trừ order/cache/thermal bias. Hai con số khác nhau chưa đủ là bằng chứng cải thiện đáng tin.

## 5. Concurrency và hàng đợi nào đang được giới hạn?

Service admission kiểm `outstanding < workers+queue_cap` trước khi await, rồi semaphore giới hạn execution. Một event loop đảm bảo đoạn check/increment không bị coroutine khác chen giữa qua await. Đây không là mutex thread-safe/multi-process admission; chạy nhiều worker process sẽ có nhiều bộ đếm/queue riêng.

`peak_waiting=max(outstanding−workers,0)` là proxy số slot chờ trong admission model của bài. Chưa đo OS listen backlog, số HTTP connections, request trước validation, connection-pool wait, memory toàn process hoặc queue của proxy. Bounded service queue không tự bảo đảm toàn hệ thống bounded memory.

Fake run time dùng cooperative sleep và forward rất nhỏ. Tăng concurrency có thể chồng các sleep; không suy tương tự với CPU-bound Python, GPU memory bandwidth hoặc kernel batching. Chưa đo CPU/GPU utilization hoặc memory peak. Utilization trong simulation là occupancy của execution slot có mô hình thời gian cố định.

## 6. Error, timeout và right censoring

Một503 nhanh là terminal error; latency error nên được báo riêng. Timeout cho biết client đã chờ tới một mốc nhưng không thấy kết quả; completion service có thể xảy ra sau đó hoặc không xảy ra. Đó là quan sát bị giới hạn, không phải một sample thành công có duration bằng timeout.

Lab lưu elapsed client tới terminal timeout trong latency_errors_ms và giữ lỗi trong attainment denominator. Không ước lượng phần đuôi latency của những request timeout bằng survival analysis. Nếu nhiều timeout, p95 của các success còn lại chỉ là conditional success latency; không đại diện trải nghiệm mọi request.

Fixture timeout dùng MockTransport với async sleep để kiểm outer asyncio deadline. TCP benchmark mặc định có500 ms outer timeout và HTTPX timeouts, nhưng không ép server fault để tuyên bố đã kiểm TCP disconnect/unknown effect. Bài31 có tình huống timeout TCP riêng; không nhập kết quả cũ thành bằng chứng của bài này.

## 7. Percentile, aggregation và độ tin cậy

Nearest rank chọn vị trí ceil(qn). Với n5, p99=max; với n24, p95 thường là một trong hai giá trị lớn nhất. Sai số sampling, dependency giữa request và nonstationarity có thể làm đuôi dao động mạnh. Không dùng n nhỏ để khẳng định p99 ổn định hoặc extrapolate một năm traffic.

Pooling raw samples tạo một percentile có trọng số theo số request; mean của per-run p95 là summary các run, không cùng đại lượng. Cần giữ count, workload, estimator và histogram boundary/units. Nếu một run có1.000 request còn run khác10 request, average hai percentile cho hai run trọng số ngang nhau dù traffic khác100 lần.

Nếu thiết kế CI performance gate, nên đo baseline/candidate cùng điều kiện, xen kẽ thứ tự khi cần, đủ repeat và kế hoạch uncertainty. Threshold quá sít có thể tạo flaky CI; threshold quá rộng bỏ lọt regression. Chưa có statistical test/CI speed gate trong lab, chỉ có structural regression predicates.

## 8. SLO attainment, quality và goodput

Một good request phải đạt tất cả điều kiện đã đặt, không lấy trung bình các tỷ lệ đạt riêng. Nếu80/100 đạt latency và80/100 đạt quality nhưng là hai nhóm khác nhau, số đạt cả hai có thể thấp hơn80.

Trong lab, numerical response validation + scheduled latency≤40 ms tạo goodput. Cùng response chính xác về công thức chưa chứng minh label urgency đúng ngoài đời. Khi có feedback đến muộn, quality-qualified goodput cần thêm attribution/window/maturity như bài32; không tự đánh missing label là failed task.

Attainment của bài dùng planned arrivals để generator drops không biến mất. Một công cụ có thể loại incomplete hoặc không có measurement cho SLO nào đó; so kết quả phải đọc định nghĩa population. Đừng gộp rate theo một window với attainment theo window khác rồi suy số request good.

## 9. Streaming và token accounting

TTFT phụ thuộc định nghĩa first event: first byte, first content chunk, first token hoặc first visible output. Nếu chỉ nhìn first-byte chứa metadata, người dùng có thể chưa thấy nội dung. Chunking/buffering ở proxy/client có thể làm số đo khác token generation timestamps.

ITL được đo giữa event liên tiếp theo định nghĩa công cụ. TPOT per-request có thể tính từ first output tới final response; average các TPOT theo request khác aggregate gap mean theo token. Cần nêu weighting và output token count. n1 thì mẫu số n−1 không hợp lệ.

Streaming fixture chỉ có bốn token timestamp giả định và finish overhead20 ms. Không chạy SSE, tokenizer, usage parser, speculative decoding hoặc token-throughput benchmark. Không dùng những timestamp này làm thông số của model thật.

## 10. Chi phí task nhiều bước và usage chưa biết

Ledger dùng key(task,attempt), cộng usage của mọi attempt đã có dữ liệu, gồm lần failed. Task là một logical unit trong fixture; taskA có hai attempt nhưng chỉ một successful task. Nếu task có nhiều stage/tool call, cần thêm operation ID, usage category và quy tắc attribution; tránh gộp retry HTTP với một user retry thành cùng identity không rõ.

Cached input nằm trong total input. Tính uncached=total−cached; không tính total×full_input_price rồi cộng cached×cache_price. Rates và Decimal giúp phép tính thập phân nhất quán, chưa mô phỏng rounding/tax/credit/minimum của invoice. USD chỉ là đơn vị của ví dụ giả định.

Missing usage không được coi miễn phí. Trong quy tắc không có negative charge của bài, known subtotal là phần chi phí đã biết; total/per-unitcost được để None tới khi đối soát đủ. Nếu hệ thống có credit/refund thì subtotal không tự là invoice lower bound. Duplicate record cần reconcile theo provider receipt/idempotency, không chỉ drop trùng key nếu nội dung conflict.

Qualified là annotation có sẵn trong cost fixture, không phải model judge. Một task có nhiều outcomes cần định nghĩa terminal/revision/as_of; account() đơn giản dùng tập task đã có ít nhất một succeeded/qualified attempt. Không dùng hàm này làm ledger kế toán cho task bị revoke/compensated mà chưa bổ sung state machine.

## 11. Phân bổ fixed cost và quyết định tối ưu

Allocated hour3 USD là một fixed window. Cost trên qualified output tăng khi traffic ít hoặc failures/retries cao dù giá/giờ không đổi. Compute-only cost chưa là total cost of ownership; cần scope phù hợp quyết định và tránh cộng trùng hạ tầng chung cho nhiều workload.

Có thể tính incremental cost cho một thay đổi, hoặc fully allocated cost cho sản phẩm; hai mục đích khác nhau. Chi phí cơ hội, nhân lực vận hành và reserved capacity còn có giả định riêng. Bài không tính hóa đơn thực, exchange rate hay recommendation mua tài nguyên.

So tối ưu cần giữ quality, error, latency, freshness/security và workload. Một cache làm giảm cost nhưng trả dữ liệu cũ không đạt qualified output; một model nhanh nhưng sai nhiều có thể giảm goodput. Liên kết kết quả thí nghiệm với version manifest, CI và rollback ở bài33.

## 12. Từ bài thực hành sang phép đo sản phẩm

Giữ lại protocol, source/release/environment, workload snapshot và raw records cần thiết theo retention. Với production-like experiment, tách generator khỏi server, kiểm generator headroom, đo proxy/network/replicas/cold/warm và representative mix. Có thể cần soak/burst/failure test cùng kế hoạch dừng.

Đánh giá chất lượng bằng dataset/outcome riêng, không chọn cấu hình chỉ theo throughput. Xác định vùng có thể vận hành ở error/latency/quality mục tiêu, rồi kiểm failure envelope và chi phí theo cùng đơn vị công việc. Tài liệu này cung cấp cơ chế và lab nhỏ, chưa là capacity plan cho một hệ thống đang chạy.
