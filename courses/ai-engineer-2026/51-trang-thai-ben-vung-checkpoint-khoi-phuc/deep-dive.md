# Đào sâu bài 51 — Checkpoint không bao trùm thế giới bên ngoài

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Bền vững là một cam kết có phạm vi

Một object Python còn trong RAM chỉ sống cùng process. Một file đã ghi nhưng chưa flush/commit có thể có mức bảo đảm khác một transaction đã commit. Một bản sao trên cùng ổ đĩa cũng chưa là backup khi ổ đĩa hỏng. Khi nói “durable”, phải nói theo failure model nào: exception, worker exit, OS crash, power failure, mất máy hoặc mất vùng dữ liệu.

Suite làm worker process chết thật bằng os._exit và đọc lại DB bằng process mới. Case terminal_transaction chết khi transaction đang mở cho thấy dữ liệu chưa commit bị rollback khi đọc lại. Các phép thử đó không cắt điện thiết bị, không làm hỏng filesystem và không chứng minh RPO=0 trong mọi lỗi phần cứng. synchronous FULL là cấu hình sử dụng, không kết quả benchmark hoặc bảo hiểm chống mọi tầng lưu trữ sai.

State DB cần được backup/restore nhất quán theo cơ chế của database. Nếu đổi sang WAL sau này, việc sao chép riêng main DB trong khi có writer không thể mặc nhiên được coi là snapshot đầy đủ; cần hiểu backup/checkpoint/WAL của engine. Bài hiện dùng rollback journal DELETE, không triển khai backup/replication hoặc so sánh WAL performance.

## 2. Execution checkpoint, snapshot và event sourcing

Journal của bài lưu toàn bộ state sau mỗi transition, kèm kind/epoch/previous_hash. Có thể kiểm chuỗi và xem tiến độ; state mới nhất đọc từ snapshot cuối. Đây không phải một implementation event sourcing hoàn chỉnh chỉ lưu domain events rồi reducer dựng lại mọi state.

Full snapshots dễ học và dễ đối chiếu, nhưng lặp lại observations/plan history nhiều lần. Nếu state tăng gần tuyến tính theo số bước và snapshot mỗi bước, tổng bytes có thể tăng gần bậc hai trong ví dụ đơn giản. Không có phép đo storage ở đây; payload thực, compression và tần suất checkpoint thay đổi kết quả. Bài giới hạn tối đa 128 revisions và không thực hiện pruning.

Một thiết kế lớn có thể lưu append-only events + periodic snapshots + content-addressed blobs, với schema/reducer version và garbage collection rõ. Phải bảo đảm head, event và referenced blob được công bố đúng thứ tự. Nếu blob ở object store khác transaction, lại có một ranh giới lỗi cần thiết kế, không mất đi nhờ đổi định dạng.

## 3. Atomicity cục bộ và khoảng trống hai hệ thống

Checkpoint transaction có thể atomically lưu observation và xóa pending. Nó không atomically buộc một API bên ngoài thực thi cùng transaction. Hai trình tự “gọi API rồi commit” và “commit rồi gọi API” đều có cửa sổ crash với các ý nghĩa khác nhau.

Intent trước I/O giúp biết operation nào có thể đã bắt đầu. Reservation trước I/O giữ budget không vượt qua restart. Observation commit giúp bỏ qua việc gọi lại khi kết quả đã được lưu. Không bước nào tự cho biết remote effect của một request mất acknowledgement.

Với API ghi, mô hình hữu ích là proposed → approved → intent committed → attempted → confirmed/unknown. Unknown phải dẫn tới status query/reconciliation/review phù hợp, không tự biến thành failed rồi retry bằng key mới. Bài 51 chỉ thực thi reads; hàm recovery_rule trả reconcile_or_review với tên tool ngoài allowlist, chưa triển khai một write state machine hoặc production reconciliation.

## 4. Idempotency cần cả identity lẫn hệ đích thực thi contract

Một op_id ổn định giúp theo dõi cùng ý định. Muốn deduplicate effect còn cần server nhận key, ràng buộc payload/tenant, lưu kết quả và định nghĩa cửa sổ retention/concurrency. Hai request cùng key nhưng args khác phải được xử lý theo contract, không tùy tiện dùng lại kết quả khác nghiệp vụ.

Trong lab, support-504:S2 chỉ là logical identity trong checkpoint/probe. MCP server không được truyền một idempotency key như vậy. Vì thế after_response có hai acknowledged executions cho cùng op là kết quả có chủ ý. “Một observation được lưu” và “một external execution” là hai tính chất khác nhau.

Ngay cả hệ đích hỗ trợ dedup cũng cần xử lý timeout khi query status, key hết hạn, thay release và read-your-writes. Exactly-once thường cần nói rõ effect nào, trong boundary nào, dưới giả định nào. Không suy từ UNIQUE(run_id, revision) hoặc một receipt local ra một bảo đảm end-to-end.

## 5. Retry reads và tính nhất quán của observations

Read thường dễ retry hơn write, nhưng không tự là miễn phí hoặc ổn định. Hai lần đọc health có thể thấy hai status, một search mới có thể dùng corpus mới, hoặc quyền nguồn đổi giữa các lần. Nếu agent giữ observation cũ, cần ghi timestamp/version và đánh giá còn dùng được cho quyết định hiện tại không.

Lab chọn mô hình nhỏ: current snapshot/release và seed phải khớp, source còn được cấp quyền. Nếu health config đổi, binding khác và run bị chặn thay vì tự ghép hai thời điểm. Nếu body ticket đổi, seed identity khác. Đây là chính sách bảo thủ để dễ giải thích; không là giải pháp tối ưu cho ticket đang được người dùng cập nhật liên tục.

Hệ thật có thể invalidate một phần observations, tạo revision điều tra mới hoặc tiếp tục trên snapshot đã pin có TTL. Cần xác định semantics: câu trả lời về thời điểm nào, dữ liệu nào có thể stale, lúc nào phải refresh, và refresh có tạo op mới không. Giữ một checksum chưa trả lời các câu hỏi đó.

## 6. Ownership, CAS và fencing không thay thế nhau

Expected checkpoint revision chặn lost update nếu hai writer cùng dựa trên một head. Epoch chặn writer của thế hệ trước ngay cả khi nó giữ một snapshot cũ. Cả hai được kiểm trong transaction cùng việc append/head update; kiểm ngoài transaction rồi ghi sau sẽ có race.

Owner trong bài là nhãn controller cấp, không credential. Actor/principal là lớp khác, được bind từ world authorization. Không cho model đề xuất owner hoặc takeover_epoch như một cách tăng quyền. SQL file và Journal API là thành phần tin cậy của runtime, không API public nhận arbitrary state từ người dùng.

Takeover chỉ được controller thử sau subprocess.run đã đợi worker cũ chết. Không có lease/heartbeat hoặc cơ chế kiểm PID từ xa. Nếu cho takeover một worker còn sống, epoch chặn checkpoint commit cũ nhưng chưa ngăn nó gọi remote tool sau lần kiểm gần nhất. Muốn fence effect cần hệ đích kiểm fencing token hoặc một protocol tương đương. Demo không chứng minh loại guarantee đó.

## 7. Checkpoint replay không cần tái sinh token

Replay trong lab dựng PlanGuard từ dữ liệu đã commit và đúng observation prefix tại lúc revision được chấp nhận. Nó không gọi model để kể lại cùng lý do. Pending action đã được chấp nhận sẽ được dùng nguyên, trong scope/budget hiện tại.

Nếu proposal chưa được commit, runtime có thể phải gọi planner lại và tính thêm lượt. Với LLM live, nên giữ request/sample/attempt IDs và usage known/unknown, không coi crash trước checkpoint là inference miễn phí. Stored proposal cần strict parsing, version/source lineage và policy; một transcript chưa là executable authority.

Bài chưa có live provider history, opaque reasoning items hoặc tool protocol replay của một provider. Không serialize tùy tiện object model/client bằng pickle. Nếu sau này tích hợp Responses adapter, phải kiểm lại request/response correlation và resume contract riêng thay vì coi một JSON state là đủ cho mọi API.

## 8. Deadline, budget và cached final là các policy khác nhau

Attempt reservation được tính trước I/O nên có thể lớn hơn số acknowledgement. Khi process chết sau reserve nhưng trước dispatch, budget vẫn bị tiêu. Hoàn lại vì “không có observation” sẽ tạo lỗ hổng reset budget qua crash. Một hệ có status oracle đủ mạnh có thể có chính sách accounting khác, nhưng phải chứng minh căn cứ.

Lab có absolute deadline cố định và business clock giả kiểm tại đầu run. Nó chứng minh resume không tạo lại TTL; chưa chứng minh hard timeout cho toàn run đang chạy. Link có timeout cục bộ cho requests; timeout đó cũng chưa là remote cancellation. Product cần clock source, elapsed/cumulative/absolute budgets và cancellation semantics rõ.

Final commit trước response cho phép trả lại cùng final nếu ack ở client bị mất. Đây là redelivery, chưa exactly-once notification. Client/UI có thể dedup theo run/result version. Đọc cached final được tái kiểm nguồn/quyền; bài không áp execution expiry để cấm đọc kết quả, nên cached-answer TTL là một policy riêng.

## 9. Retention, privacy và migration

Checkpoint chứa ticket body, symptom search query và runbook quotes. Chặn export sau revoke không xóa bản cũ trong snapshots, backup hoặc probe. Probe chỉ synthetic trong bài; triển khai thật cần tránh nhân bản nội dung nhạy cảm vào test/log không kiểm soát. Hash/ID chưa tự là anonymization.

Hash chain phát hiện corruption theo cách triển khai, không xác thực writer như chữ ký/HMAC. Người có quyền sửa DB có thể tính lại chain và head. Bộ code không nhận checkpoint file lạ như dữ liệu an toàn, không triển khai encryption/rotation hoặc tamper-resistant remote audit.

Khi schema/release đổi, reject là lựa chọn mặc định. Một migration đúng cần copy/validate, giữ logical IDs và counters, ghi source/target version và rollback plan; không chỉ sửa version string rồi tiếp tục. Nếu tool semantics đổi, còn cần quyết định pending cũ có thể được thực thi hay phải reconcile/handoff. Bài chưa có migration runner.

## 10. Đo recovery và giới hạn bằng chứng

RTO là thời gian khôi phục đến một mức phục vụ được định nghĩa; RPO nói mức mất dữ liệu chấp nhận theo chính sách. Không thể suy RTO từ số Python turns. Suite chỉ báo process exits, checkpoint/attempt/ack/observation counters và outcomes synthetic, không benchmark disk hoặc service availability.

Một chương trình đánh giá rộng hơn cần crash ở nhiều step, lặp theo workload, timeout trong I/O, disk full/locked/corruption, actor/source mutation, model nondeterminism và write unknown. Phải kiểm cả trường hợp dừng đúng khi không đủ budget, không chỉ giữ những case cuối cùng answered.

60 checks mới + 349 hồi quy giúp chặn lỗi trong implementation và fixture đã công bố; không phải formal verification hoặc bằng chứng LLM hiểu recovery. Nhánh hiện vẫn single-agent, tuần tự, local SQLite, MCP memory trong worker. Context/memory xuyên run là bài 52; human review và write effects là bài 53, hiện mới đề cương.
