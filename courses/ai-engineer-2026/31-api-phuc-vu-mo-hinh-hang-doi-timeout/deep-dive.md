# Bài 31 — Đào sâu queue, deadline và vận hành

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Ranh giới và đơn vị phải có trước công thức

Một request HTTP có thể bị từ chối trước inference; một job API có thể có nhiều GET poll; một inference batch có thể chứa nhiều request. “RPS” cần chỉ rõ là attempted HTTP requests, admitted predictions, model calls hay completed jobs. Throughput phục vụ và goodput — kết quả còn hữu ích, đúng hạn — cũng có thể khác nhau.

Stats lab đếm attempts đi vào `Service.submit()`, sau validation; GET/DELETE và input 422 không cộng vào số này. accepted là đã admission, không là thành công. Counter SUCCEEDED/FAILED/EXPIRED/CANCELLED tăng đúng một lần ở terminal transition. Records chỉ là số snapshot còn giữ, không là tổng accepted từ startup vì có eviction.

Đường cong M/M/1 dùng λ<μ, queue vô hạn, arrival Poisson, service exponential và stationarity. FIFO burst 100 ms là ví dụ deterministic hữu hạn riêng. Little dùng cùng boundary/population và admitted rate khi có rejection; không lấy attempted λ áp vào W chỉ của accepted jobs. Bài chưa làm performance fit hoặc load test.

## 2. Bounded queue và những nơi vẫn có thể xếp hàng

`asyncio.Queue(maxsize=Q)` giới hạn entry chưa dequeue, còn c worker giữ job đang chạy. Queue full dùng put_nowait để không tạo thêm vô hạn producer task chờ put. Gate admission không có await giữa check và mutation nên tuần tự trong một event loop, chưa thread-safe hay distributed atomicity.

Một semaphore c quanh inference chỉ giới hạn phần đang chạy; nếu hàng nghìn endpoint chờ semaphore thì memory/latency vẫn có thể tăng. Các tầng socket accept, proxy pending, body parse, connection pool, tokenizer và scheduler GPU cũng có thể có queue riêng.

Service giới hạn registry 16, trace ring 128 và backend diagnostic history/signal queue 128. Đây chưa là memory budget toàn process: raw HTTP body, JSON decoder, ASGI task, network buffer và model memory vẫn thuộc các tầng khác. Registry eviction chỉ chọn terminal job đã physical_done; expiry không được làm mất record của công việc còn thực sự chiếm slot.

Queue có thể giữ tombstone của expired/cancelled job. Worker bỏ qua và task_done khi đến entry ấy; không chạm private `_queue` để xóa. Cách này đơn giản, bảo toàn FIFO bookkeeping nhưng giảm khả năng nhận việc trong lúc head đang chậm. Nếu cần compact/prioritize mạnh hơn, phải thiết kế scheduler khác và kiểm race/cancellation/fairness.

## 3. Deadline propagation qua nhiều lớp

Trong process, deadline dùng loop.time() monotonic. Không gửi con số loop.time() thô sang máy khác vì clock origin không chung. Qua network có thể truyền remaining duration được clamp bởi server policy, hoặc absolute wall-clock timestamp với thỏa thuận clock/skew phù hợp. Client tự khai timeout không được tự tăng cap của server.

Lab nhận integer budget_ms 10..2000, bắt đầu sau validation. Queue deadline=min(request deadline, now+queue_wait cap). Worker kiểm thời gian trước dequeue execution và khi actual task dispatch; lúc RUNNING chỉ còn request deadline ban đầu.

Khi chia budget, dành headroom cho serialization/response/cleanup và các lớp ngoài application. Nếu downstream chỉ biết một timeout duration mới, phải truyền remaining, không cấp lại toàn budget ban đầu cho mỗi retry. Lab chưa propagating budget sang model server từ xa, proxy hoặc DB.

Timers/cancellation cần event loop được scheduling. Blocking native/Python code giữ event loop có thể làm thông báo expiry đến muộn. Post-inference check không công bố SUCCEEDED khi deadline đã qua, nhưng không quay ngược thời gian để giải phóng tài nguyên sớm. Muốn hard isolation cho tác vụ không cooperative có thể cần process/backend-level control; bài chưa triển khai kill/restart worker process.

## 4. Một terminal state không mô tả mọi trạng thái vật lý

Job state EXPIRED nghĩa ứng dụng không chấp nhận kết quả đúng hạn nữa. CANCELLED nghĩa host đã chốt trạng thái hủy logic; inference task được yêu cầu cancel nếu có. Worker credit chỉ trả khi actual await đã unwind, không dựa duy nhất vào future terminal.

Cooperative backend của bài chỉ await gate/sleep rồi forward số học rất nhỏ, nên CancelledError dừng được phần await. Nó không tương đương thread, remote server, GPU kernel hoặc tác vụ tự bắt và nuốt cancellation. close() có contract cooperative, chưa là shutdown watchdog chống mọi backend treo.

Thread demo dùng shield và giữ task để quan sát physical completion. Nếu await trực tiếp to_thread rồi cancel wrapper, thread có thể vẫn chạy trong khi wrapper Future đã cancelled; bạn mất một tín hiệu completion dễ quản lý. Không cắm bridge thread ấy vào Service rồi giải phóng busy ngay ở CancelledError. Một integration đúng cần giữ job-owned execution handle/capacity cho đến khi computation thật kết thúc.

Socket client ReadTimeout và cancel coroutine test là hai input khác nhau. Code endpoint hiện không monitor `http.disconnect`, vì vậy thực nghiệm TCP thấy job còn hoàn tất. Nếu bổ sung disconnect watcher, cần xác định request nào nên hủy (interactive predict) và request nào đã nhận thành job độc lập (202); disconnect không nên vô tình hủy durable job đã cam kết.

## 5. HTTP contract và error taxonomy

Client network exception không là HTTP response từ ứng dụng. Local service EXPIRED/503 khác ReadTimeout phía client; cả hai có thể xảy ra cho cùng một ý định nhưng ở các lớp khác nhau.

503 trong lab gắn overload/not-ready/local deadline; 504 là gateway/proxy upstream timeout theo RFC, không là status bắt buộc cho local inference. Nếu tương lai API proxy tới một model server, 504 có thể phù hợp cho upstream timeout tùy contract. Không dùng 408 để chỉ model compute chậm.

GET /jobs trả 200 khi đọc được record, rồi client kiểm state. 202 submit không là completion; Location là đường theo dõi, chưa receipt bền. Cancel terminal job giữ nguyên trạng thái, tránh biến SUCCEEDED thành CANCELLED chỉ vì client gửi DELETE muộn. Unknown/evicted trả 404, chưa phân biệt bằng 410 hoặc tombstone retention dài.

Input validation không admission. NaN trong JSON input có thể được decoder tiếp nhận rồi validator từ chối; handler không trả nguyên input trong error để tránh JSONResponse lại lỗi khi serialize NaN. Parser JSON toàn hệ thống, duplicate-key policy, content-length/body-stream cap chưa được bài xây thành một security layer tổng quát.

## 6. Job bền, delivery và effect

RAM queue, future và registry đều mất qua restart. Dù trả 202, Service chưa có durability. Nếu cần công việc dài, thiết kế persist admission, trạng thái và result; queue/broker delivery, ack, lease/visibility timeout, retry cap và poison-message handling; quan hệ giữa DB update và publish có thể cần outbox hoặc transaction phù hợp.

Broker “at least once” có thể redeliver; inference thuần đọc thường dễ tính lại nhưng vẫn tốn tài nguyên. Nếu job tạo email, thanh toán hoặc ghi business state, cần stable business intent/idempotency key, receipt và expected version như bài 27. Queue message ID, request ID và job ID có thể phục vụ mục đích khác nhau.

Đừng reset quota/retry budget khi job chuyển queue hoặc resume. Nếu retry ở client, gateway và worker độc lập thì tổng attempts có thể nhân lên. Jitter giảm đồng loạt, không tạo thêm service capacity. Bài giải thích cơ chế, chưa cài broker, distributed retry hoặc idempotency store.

## 7. Model lifecycle và scaling

Model artifact cần gắn với preprocessing/feature order và output contract. Trong lab, Model giữ snapshot và hash từ weights+bias+feature order+threshold/version; đây không là chữ ký hay checksum source/dependency toàn bộ. Không có endpoint upload model hoặc hot reload.

Warm-up chạy một input tổng hợp trước ready. Với model thật có thể cần nhiều shape/dtype/device path; load xong chưa chắc hết lazy initialization, CUDA synchronization hoặc memory allocation. Serving model cần chọn thread/process/native inference backend theo đặc tính workload, thay vì chỉ thêm async def.

c asyncio worker chia một model trong process. N Uvicorn worker process có N model và N queue/registry, không tự chia sẻ job ID hoặc GPU memory. Bài có test hai coroutine worker bằng barrier; chưa chạy nhiều Uvicorn process hoặc benchmark scaling CPU/GPU.

Batching là một scheduler khác: có max batch size, max wait và padding/token budget, đồng thời phải giữ deadline từng request. Continuous batching của LLM còn phải quản lý sequence đang decode/KV cache; không suy delay sleep của lab mô tả cơ chế ấy.

## 8. Liveness, readiness và shutdown

Liveness trả lời ứng dụng còn đáp ứng; readiness nói đã sẵn sàng nhận traffic theo policy. Full queue ngắn hạn không nhất thiết là lý do làm mọi replica unready. Ngược lại model load thất bại không được báo ready. Routing/probe interval và recovery cần kiểm cùng orchestrator thực tế, chưa được lab triển khai.

Shutdown: accepting=False, chờ queue.join với grace, hủy những job còn lại khi grace hết, drain tombstones, dừng worker, hủy timer và release model. Timer phải thuộc lifecycle, không tạo task nền vô chủ. `queue.join()` không chứng minh job nào thành công, chỉ kiểm bookkeeping đã được xử lý.

Nếu backend không cooperative hoặc thread/native operation tiếp tục chạy, không được coi các bước close trên đã chứng minh bounded shutdown. Trong production cần thiết kế grace/hard-kill boundary, replay/durability và cleanup resource khác phù hợp. Bài chỉ kiểm fake async backend cooperative và thread riêng được gate-release/join rõ ràng.

## 9. Những gì các kiểm đã chứng minh

ASGI tests đi qua HTTP route/validation/lifespan và actual asyncio scheduling, không qua TCP. Gate/Event ép thứ tự running→queued→reject; timer dùng monotonic time thật, không fake clock hoặc modeled ms như bài 30. Không assert request phải xong đúng một số millisecond để suy performance.

TCP test tạo Uvicorn loopback ephemeral socket, HTTPX client với trust_env=False, rồi đóng. Đã kiểm ready/predict/202→GET và ReadTimeout client khi server vẫn hoàn tất; chưa DNS, TLS, proxy, container, network outage hoặc production load.

45 predicate regression khác 11 scenario và khác 3 model calls trong TCP. Score model được đối chiếu phép tính tay, không có label dataset/accuracy/calibration hoặc LLM generation. Một vài diagnostic history là ring hữu hạn, không phải trace đầy đủ cho auditing lâu dài.

Report giữ trạng thái/count/code xác định của từng scenario; thời gian scheduling đo được không dùng làm benchmark. Kết quả có thể không giữ nguyên nếu máy quá tải đến mức vượt các test guard; cần chẩn đoán môi trường và lỗi trước kết luận model/API regression.

## 10. Bước tiếp theo có ranh giới rõ

Bài 32 tiếp tracing/quality: nối admission, queue wait, model call, HTTP response và client observation bằng correlation IDs; tách queue age, deadline remaining, useful completions và rejection/error layer. Bài 34 sẽ đo tải, percentiles, workload và cost.

Những phần chưa làm ở bài này: model trained phục vụ thật, GPU/LLM scheduler, streaming endpoint, shared durable broker/result DB, auth/ownership/rate quota, multi-process routing, distributed clock/budget, production body/proxy limits, model hot reload và load test/SLO. Bài tập nhỏ dùng synthetic data để học từng boundary, không cần triển khai dịch vụ Internet.
