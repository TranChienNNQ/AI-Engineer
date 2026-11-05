# Bài 27 — Đào sâu giao thức, recovery và phạm vi bảo đảm

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Tool definition là một API contract hướng tới hai đối tượng

Code cần schema/typing/versioning để dispatch an toàn; model cần ý nghĩa và điều kiện sử dụng để chọn đúng. Một tool quá tổng quát dồn công việc kiểm vào chuỗi ngôn ngữ; một bộ quá nhiều tool gần giống nhau có thể làm lựa chọn khó hơn. Không có số tool tối ưu chung; phải đo trên nhiệm vụ thật.

Bài dùng bốn tool với tham số nhỏ và output hữu hạn. Tool description được trình bày ở bài chính, nhưng chưa được gửi tới LLM; không có bằng chứng thực nghiệm model chọn đúng chỉ nhờ mô tả ấy. Khi tích hợp, giữ validation ở host kể cả provider có strict output mode.

Tên tool/schema không đủ để suy semantics. Một hàm tên get_ticket vẫn có thể ghi telemetry; safe/idempotent xét intended business effect, không buộc không có log hay biến động dữ liệu theo thời gian. Tool read có thể tốn tiền/rate limit nên vẫn cần budget.

## 2. Validate toàn batch không làm execution thành transaction

`read_batch` parse tất cả trước khi chạy; invalid schema ở call thứ hai không tạo dispatch call thứ nhất. Nhưng nếu batch hợp lệ rồi read thứ hai lỗi mạng/quyền, read thứ nhất có thể đã chạy. Đây không phải all-or-nothing distributed transaction.

Runtime kiểm call ID duy nhất theo run; internal retry giữ cùng ID, cùng logical call, tăng attempt. Result correlation chỉ chứng minh ghép đúng cặp và output shape, chưa chứng minh service trả dữ kiện thật hoặc model suy luận đúng từ kết quả.

Provider có thể yêu cầu vị trí message/result cụ thể sau tool call; adapter phải đọc tài liệu API đang dùng. Protocol trung lập giúp tách orchestration khỏi provider nhưng không xóa nghĩa vụ ánh xạ đúng giao thức ngoài.

## 3. State ownership và quyền kiểm soát

Model có thể đề xuất note, không được tự set `state=APPROVED`, sửa `subject`, cấp grant hoặc tăng budget. Workflow object thuộc coordinator tin cậy, proposal là dữ liệu. Snapshot được trả để quan sát, không được nhận lại từ client rồi tin như executable state.

Threat model không bao gồm arbitrary Python execution cùng process. Code được toàn quyền đọc RAM/reference có thể sửa Backend hoặc grant store; thư viện này không là sandbox. Khi thêm browser/code/network tool cần thiết kế lại effect/egress boundary.

Policy partial health=unknown chỉ phù hợp note triage cần người kiểm tiếp. Nếu nghiệp vụ là tự động refund/deploy/delete, mất một dependency có thể bắt buộc dừng. Workflow phải thể hiện ý nghĩa nghiệp vụ thay vì một rule “error thì bỏ qua” áp dụng mọi nơi.

## 4. Review phải gắn với đúng proposal và evidence

Controller giả định reviewer đã nhìn ticket/action/note và nguồn liên quan. Helper `approve()` chưa kiểm sự kiện UI hay chữ ký người dùng. Source refs giữ ID/version/hash để review không bị dùng với runbook đã đổi; source lineage chỉ giữ binding, chưa chứng minh note trung thành với nguồn.

Ticket version kiểm ngay khi write. Nếu conflict, không tự đọc version mới rồi thay expected_version và dùng review cũ. Thay đổi có thể làm intent/note không còn hợp lý; cần lấy lại context và xác lập ý định cho proposal mới.

Grant TTL dùng đồng hồ logic. Trong service thật cần clock/expiry semantics phù hợp và không cho model/client tự chọn now. Không lấy TTL làm deadline end-to-end; một cái giới hạn quyền còn hiệu lực, một cái giới hạn thời gian xử lý.

## 5. Retry có ownership, không chỉ một vòng for

Một SDK retry 3 attempt, một service retry 3 và workflow retry 3 có thể gây khuếch đại 27 attempt ở dependency. Cần biết tầng nào sở hữu retry, thời lượng còn lại, loại lỗi và effect semantics. Backoff giảm nhịp; jitter giảm đồng bộ giữa nhiều caller, không làm operation tự an toàn.

Lab read retry tối đa hai attempt và không retry write tự động. Runtime biết duration của fault trong simulator để tính deadline; production không biết trước điều đó. Timer I/O, connection/read timeout, total deadline và cancellation cần được định nghĩa riêng.

Bản code không chờ thật ở workflow; không dùng modeled_ms để công bố p95. Demo TaskGroup có wall_ms thật nhưng chỉ đo asyncio + sleep + xử lý local, không đo throughput/latency tool service.

## 6. Ba lớp bảo đảm của idempotency

1. **Nhận diện ý định:** coordinator giữ key ổn định cho cùng business action, không dùng mỗi response model một key mới.
2. **Thực thi tại nơi có effect:** backend so request hash và lưu receipt cùng mutation, không chỉ dedup trong prompt.
3. **Durability/retention/recovery:** receipt và mutation còn đồng bộ qua crash/retry/late arrival trong khoảng thời gian đã cam kết.

Lab minh họa hai lớp đầu trong cùng lock/RAM. Không có crash recovery hay datastore transaction nên chưa đạt lớp ba. Dedup ở client không đủ nếu client mất state hoặc service đã commit nhưng response thất lạc.

Cùng key/cùng request trả receipt cũ không cần grant mới vì không phát sinh effect mới; caller vẫn phải có quyền hiện tại để đọc receipt qua đường này. Receipt có thể nhạy cảm. Nếu policy cần status read riêng với quyền hẹp hơn write, phải thiết kế endpoint/scope tương ứng; lab dùng cùng quyền write cho đơn giản.

Key conflict là tín hiệu request mới hay state sai, không phải lỗi tạm thời để retry với key khác. Thay key có thể tạo duplicate; cần reconciliation hoặc xác lập một intent mới có thẩm quyền.

## 7. UNKNOWN là thông tin về tri thức của caller

Simulator biết timeout_before chưa tạo effect, nhưng workflow chỉ được thấy lỗi ambiguous như qua mạng. Giữ UNKNOWN giúp tránh “biết lén” ground truth từ fixture. timeout_after và invalid output có effect thật trong RAM nhưng caller chưa có receipt hợp lệ.

Không có receipt không tự là NOT_APPLIED. Endpoint trạng thái production phải định nghĩa consistent read, pending, not found, retention và late request; nếu không, caller chưa đủ cơ sở tự động gửi lại non-idempotent operation.

Trường hợp tác vụ bị cancel sau gửi write cũng tương tự: cancellation không hồi tố effect đã xảy ra. Nên báo “kết quả chưa xác định, operation ID/key đã được lưu để đối soát” thay vì báo “đã hủy nên không thay đổi gì”. Lab không có remote cancellation hoặc storage bền để giữ ID qua restart.

## 8. Concurrency và TaskGroup không tự là workflow engine

TaskGroup quản lý lifetime của một nhóm coroutine trong process. Nó không lưu event durable, không tự resume sau crash, không giải quyết distributed transaction hoặc rate limit đa worker. Khi bắt lỗi optional thành result error, bạn đang thay đổi failure policy của group một cách có chủ đích.

Kết quả có thể hoàn tất khác thứ tự; correlation bằng ID độc lập với scheduling. Deadline của group có thể cancel coroutine, nhưng handler blocking/thread/remote server có thể không bị hủy đồng bộ. Demo chỉ read, tránh dùng hành vi cancel để suy rollback write.

Nếu tăng fan-out, cần concurrency cap, per-tenant limit, queue/backpressure, timeout và budget tổng. Hai read nhanh trong lab chưa là bằng chứng có thể fan-out hàng nghìn tool hiệu quả.

## 9. Compensation, outbox và workflow lâu dài

Một note và một email ở hai hệ thống không tự nằm trong cùng transaction. Saga/compensation là thiết kế nghiệp vụ cho chuỗi effect đã commit; rollback local exception không thể undo email đã được nhận. Outbox giúp lưu ý định phát event cùng transaction dữ liệu khi datastore hỗ trợ, nhưng worker vẫn cần xử lý duplicate delivery.

Không triển khai những pattern này trong bài. Chúng là giới hạn quan trọng khi mở rộng từ one-process demo sang workflow production. Bài 28 sẽ tiếp tục state/memory của vòng agent; các phần vận hành sau học sâu persistence/tracing/CI theo đề cương.

## 10. Kiểm đúng phần nào thì kết luận phần đó

22 fixture kiểm routing, policy, contract, budget và effect/state trong code cụ thể. Template không phải model; poisoned body không được diễn giải để tạo action nên kết quả không là tần suất model chống prompt injection. Note template giữ “cần kiểm tra” cũng chưa là đánh giá chuyên môn note do LLM.

Khi thêm LLM, đo tool selection, args semantic, thiếu/thừa call, task correctness, grounding, false refusal và outcome từng effect. Ghi model/revision/sampling, input snapshots, tool versions, quyền và deadline; reset fixture để tránh state lẫn giữa các trial. Không gộp expected denial thành task success, hoặc chỉ tính quality trên những request không bị block.
