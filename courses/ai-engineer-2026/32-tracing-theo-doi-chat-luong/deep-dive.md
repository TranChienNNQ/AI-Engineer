# Bài 32 — Đào sâu tín hiệu, nhãn và chẩn đoán

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Một metric cần tên population, boundary và clock

“Error rate=1%” chưa đủ nếu chưa biết numerator/denominator: số HTTP error/số completed requests, số failed jobs/số admitted jobs hay số wrong prediction/số labeled predictions? Retry, polling và timeout có thể làm các số khác nhau.

Lab HTTP counters ghi mọi completion tại middleware theo service/route/status_class. Một API request gọi model còn có một model HTTP completion. Failed/rejected request không tạo prediction, nên prediction ledger không có cùng mẫu số với API attempts. Runtime instrumentation không đại diện toàn user journey ngoài ứng dụng.

OTel span timestamps đo trên wall-clock ns của SDK; duration histogram dùng monotonic clock của process. Waterfall thật từ một run không là lý do công bố SLO; muốn so latency nhiều replica cần window/workload, bucket/rate và đồng bộ ngữ nghĩa đo. Trace ID và timestamp được giữ trong report runtime, còn stable_facts bỏ variability khi kiểm cấu trúc.

## 2. Parent, link và vòng đời asynchronous

Consumer có thể là child của producer span đã kết thúc. Producer mô tả gửi message, consumer mô tả xử lý về sau; causality vẫn hợp lệ dù không có temporal containment. queue.wait là sibling riêng để đo chờ, không phải producer đang mở lâu.

Khi một batch nhận message từ nhiều trace, một span chỉ có một parent; có thể dùng links cho nhiều nguồn. Job nhiều ngày có thể dùng trace riêng và link về request nhận việc. Bài chỉ triển khai một message–một parent, chưa fan-in, span links, broker instrumentation hoặc long-running trace storage.

Python ContextVar/task context giúp trong một event loop, không tự đưa context vào serialized message hoặc HTTP request. Worker tạo từ startup không được tự đọc context của request tương lai. Lab serialize W3C carrier và extract riêng cho mỗi message; không dùng global “current_trace_id” vì concurrent requests có thể ghi đè nhau.

## 3. Trace propagation và trust boundary

External traceparent có thể bị giả mạo; trace ID là correlation metadata, chưa authenticate caller. Cùng trace_id không tự chứng minh mọi span đều từ cùng user hoặc được cấp quyền. Tại boundary public, sản phẩm cần policy context/sampling phù hợp; không để sampled flag trở thành cách buộc export quá tải hoặc lối tắt quyền.

Lab chỉ dùng TraceContextTextMapPropagator, không cài global Baggage propagator. Context carrier vẫn có thể có tracestate vendor data; việc propagate và việc log/export raw headers là hai quyết định khác nhau. Không nhét credential, review grant hoặc dữ liệu nhạy cảm vào baggage/context để truyền tiện.

Manual instrumentation ở đây dùng custom app.* keys và tên span cố định. Không có auto-instrumentation FastAPI/HTTPX để tránh hai bộ span trùng nhau; nếu thêm sau này cần kiểm duplicate span, route mapping, exception recording và attributes mới được thu. Package semantic-conventions được SDK kéo vào nhưng bài chưa áp trọn bộ naming/version contract của nó.

## 4. Histogram: tính đúng và không trộn population

Cumulative bucket b đếm mọi observation≤b. Bucket +Inf bằng count. Sum/count cho mean; percentile từ buckets là xấp xỉ phụ thuộc boundary và thuật toán. Nearest-rank trên năm giá trị đã biết trong phần tính tay là phép tính khác với histogram_quantile của Prometheus.

Không lấy trung bình p95 của replica để có p95 toàn hệ thống; phải tổng hợp dữ liệu phân bố cùng boundary rồi tính percentile. Cũng không cộng duration histogram API và model vào một population nếu câu hỏi là latency client API; có request góp vào cả hai.

Cardinality không chỉ đến từ user ID: raw exception message, raw URL, dynamic tool name, hash của từng prompt hoặc model version tăng vô hạn đều có thể tạo series/index quá nhiều. Bounded categories nên có chính sách version/retention. Traces/logs cho phép ID riêng nhưng vẫn cần cost/privacy controls; sampling không chữa một metric label sai.

## 5. Sampling và quan sát bị thiếu

Giữ toàn error trace và một phần success trace hữu ích cho debug, nhưng error fraction trong tập được giữ không là error rate thật của traffic. Biết inclusion probability có thể cho phép estimator phù hợp; nhiều sampling/drop path thực tế không đơn giản như một tỷ lệ cố định. Lab không triển khai estimator từ sampled trace.

Default ParentBased(ALWAYS_ON) và demo ALWAYS_OFF là hai cấu hình khác: parent-based có thể giữ/drop theo parent flag, còn ALWAYS_OFF demo bỏ toàn bộ. Logs vẫn có correlation IDs dù trace ấy không export. Đường link tới trace không tồn tại trong backend có thể đơn giản là không được sample hoặc bị evict/export failure.

LocalExporter trả SpanExportResult.FAILURE để kiểm business path không hỏng; SimpleSpanProcessor chưa retry/buffer bền hoặc bảo đảm latency nếu exporter thật chặn lâu. Ring cap256 span/log cap128 kiểm memory fixture; evicted/failure counter phải được theo dõi độc lập, không suy trace backend luôn đầy đủ. Telemetry có thể mất do crash trước flush, SDK limits hoặc downstream collector nữa; chưa được lab mô phỏng hết.

## 6. Privacy bắt đầu trước exporter

Auto record_exception có thể đưa exception text/stack vào span. Tắt automatic recording rồi set error_code cố định giúp tách chẩn đoán khỏi raw message. Lab model cố ý ném exception chứa SECRET32 để kiểm code này, không serialize body/header/query vào span hoặc log.

Attribute allowlist chỉ kiểm key; nếu developer đặt một bí mật vào value của app.route hoặc span.name thì filter key không tự hiểu meaning. Nên chọn value từ category host giữ và review mọi instrumentation path. Canary test là regression có phạm vi, không là privacy proof hoặc hệ thống phát hiện PII.

Hash của ID dễ đoán có thể bị dò lại; nó còn là một dimension high-cardinality dù không hiện plaintext. Quality ledger có thể cần dữ liệu giàu hơn debug traces, nhưng cần quyền/access/retention riêng. Bài dùng 30 synthetic record và không gửi telemetry/feedback ra ngoài, chưa triển khai production consent hoặc deletion pipeline.

## 7. Event time, available time và cohort

Một label có label_event_at sớm nhưng available_at muộn. Báo cáo as_of chỉ dùng label có available_at≤as_of; dùng label_event_at đơn lẻ có thể làm backtest biết trước dữ liệu pipeline chưa có. Prediction cohort được chọn theo predicted_at trong window half-open [start,end), không theo ngày label đến.

Nếu lấy mọi label mới đến hôm nay rồi so accuracy với mọi label đến hôm qua, có thể đang so hai tập prediction khác tuổi/model/version. Cần giữ model/feature/policy lineage và những window hợp nghĩa. Lab filter model_version; mọi replay cùng feature_version. Khi production đổi preprocessing/threshold trong cùng model version, cần tách/group version đúng hơn, chưa được fixture này bao phủ.

Maturity lag100 chỉ là policy nói cohort đủ tuổi theo giả định. Nó không là watermark bảo đảm label đã hết trễ, không chứng minh missingness random và không sửa selection bias. Lab còn yêu cầu n≥8/coverage≥.8 overall, slice n≥4/coverage≥.8. Threshold nhỏ để thấy mechanics, không là khuyến nghị thống kê sản phẩm.

## 8. Duplicate, conflict và correction

Một prediction/version key phải unique; duplicate prediction record gây ValueError để không double count. Feedback cùng feedback_id và cùng nội dung được idempotent; khác feedback_id nhưng cùng prediction/label cũng chỉ đóng góp một label. Nếu có hai label khác nhau cho cùng prediction, key bị quarantine khỏi metrics thay vì lấy event cuối một cách ngầm định.

Đây là policy append-only cho một binary outcome, chưa workflow label correction/supersession, reviewer agreement hoặc adjudication. Nếu nhãn được phép sửa, cần explicit label revision/provenance và định nghĩa as-of cho cả bản sửa. Không reset coverage hoặc drop ca conflict chỉ để metric đẹp; report phải cho biết bị quarantine bao nhiêu.

Join diagnostics trong snapshot hiện tính cho toàn input visible-as-of, còn quality metrics theo window/model. Alert scope=join là pipeline-level; chưa attribution mọi dữ liệu lỗi vào từng cohort riêng. Unknown prediction hoặc version mismatch không được nối bằng cách đoán ID gần giống. Job/trace/prediction ID không thay credential hoặc primary key policy.

## 9. Drift, slices và chất lượng

Fixture current giữ feature_value và score distribution nhưng đổi labels, tạo quality regression dù TV proxy bằng0. Đây là constructed counterexample cho việc “không drift thì quality không giảm”; không phải một detector concept drift hoàn chỉnh.

Fixture shift đổi histogram một feature nhưng nhãn tương ứng cho accuracy1. TV=.3 chỉ nói độ khác biệt của hai-bin marginal histogram. Hai phân bố có thể đổi bên trong bin hoặc ở joint relation mà TV này vẫn0; .3 không là p-value. Feature drift, label shift, concept drift và training-serving skew có nghĩa khác nhau; bằng chứng từ một dashboard thường chưa chỉ ra nguyên nhân duy nhất.

Overall accuracy=.6 che mobile=.2/web=1. Cần predefined slices quan trọng và coverage/label age trong từng slice. Tìm hàng nghìn slice sau khi nhìn data có thể tạo false alarm do multiple comparisons. Các alert threshold trong bài chỉ là rule minh họa, chưa confidence interval, sequential test, smoothing/hysteresis hoặc significance claim.

## 10. Theo dõi LLM/agent và business outcome

Đối với RAG/agent, tracing nên tách retrieval/rerank/model/tool/action boundary; quality còn cần task correctness, nguồn hỗ trợ claim, false refusal, quyền/effect, human labels hoặc calibrated judge. Token/cost/TTFT đều hữu ích nhưng không thay quality. Không log chain-of-thought/raw prompt tùy tiện để có một trace “đầy đủ”.

User thumbs-up không tự ground truth; người bấm feedback có thể không đại diện người im lặng. Agent chọn hành động cũng thay đổi dữ liệu/labels về sau (feedback loop). Muốn kết luận cải tiến làm business outcome tốt hơn cần thiết kế đo/experiment phù hợp, không chỉ correlation giữa model version và metric.

Lab không gọi LLM, judge, retriever, tool ngoài hoặc business service. API prediction ledger hai record minh họa independent sampling; quality replay ba cohort riêng kiểm monitoring math. Chưa nối labeler thật vào live prediction API, chưa DB/stream/collector/alert notification hoặc production quality SLO.

## 11. Từ alert đến hành động có bằng chứng

Latency tăng: kiểm traffic mix/admission/queue age, trace critical path, model/downstream và saturation. Quality giảm: kiểm join, label freshness/coverage, schema/preprocessing/version và slices; reproduce bằng eval trước quyết định. Trace correlation giúp định vị, không tự chứng minh root cause.

Monitoring bắt regression sản phẩm nhưng cũng cần tự kiểm monitoring: mất span, timestamp lệch, duplicate label, wrong version, unknown prediction, nil metric, stale baseline và stale dashboard. Không có alert khi nhãn thiếu có thể là policy gate hoạt động đúng, đồng thời là tín hiệu cần theo dõi feedback pipeline bằng rule riêng.

Bài 33 sẽ nối version dữ liệu/mô hình, CI và rollback. Bài 32 mới trả alert records để đọc, chưa gửi tin, tự retrain, rollback hoặc tạo background monitor.
