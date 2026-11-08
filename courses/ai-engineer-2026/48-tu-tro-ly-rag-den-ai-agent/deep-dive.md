# Đào sâu bài 48 — Quyền lựa chọn, protocol và giới hạn của agent

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Tính tự chủ là một tập quyền lựa chọn cụ thể

Trong bài, model được chọn tool, args trong schema, thứ tự search/health sau read_ticket, hoặc terminal. Nó không được chọn principal, namespace dữ liệu, tool implementation, budget hoặc cách runtime ghi state. Vì vậy “agent tự chủ” cần luôn kèm câu hỏi: tự chủ ở quyết định nào, trong phạm vi nào, dưới cơ chế kiểm nào?

Tăng flexibility có thể làm quy trình thích ứng tốt hơn với tác vụ mở, nhưng cũng có thể tạo đường đi không cần thiết. Không có định luật rằng chuyển một if/else sang model sẽ tăng chất lượng. Ở bốn case baseline, graph nhỏ đã giải quyết đủ; overhead model chưa được bù bằng lợi ích thấy được. Trước khi mở thêm action space, cần xác định nhóm task mà fixed graph đang thất bại hoặc quá tốn công bảo trì.

Một model trả danh sách kế hoạch rồi host chạy cứng từng bước vẫn chưa chứng minh closed-loop adaptation. Cần nhìn tool observations có làm quyết định tiếp đổi không. Fixture thể hiện nhánh theo observation; LLM live có làm đúng như vậy hay không là câu hỏi thực nghiệm riêng.

## 2. Tách task, decision, observation và outcome

Task chứa yêu cầu được ứng dụng chấp nhận. Decision là ứng viên từ model. Observation là dữ liệu host thu từ công cụ có quyền. Outcome là kết quả host chấp nhận sau kiểm. Không gộp bốn khái niệm vào một paragraph “agent đã làm gì” do chính model sinh.

Tool call chứa call_id/name/arguments; local parser biến arguments thành object có type rõ trước dispatch. Observation thêm ID host, status/code và data. Terminal trỏ observation, không tự tạo “source O99” vì model nói đã đọc. Runtime final gate kiểm đúng vai trò của observation: read_ticket không được dùng làm runbook citation, health không được thay bằng một câu trích từ ticket.

Read_ticket đầu tiên là một prerequisite do thiết kế sản phẩm đặt, không là định nghĩa agent phổ quát. Sau bước ấy, thứ tự search và health linh hoạt nếu goal cho phép. Một nhiệm vụ khác có thể cần prerequisite khác; mô tả cho model nên khớp host policy để tránh gọi rồi bị từ chối quá nhiều.

## 3. Correlation trong một run và idempotency của side effect

Call ID giúp gắn output với request tương ứng. Nó không là khóa chống ghi trùng ở hệ ticket. Observation ID giúp terminal trỏ đến dữ liệu đã thấy; doc ID/version/hash lại xác định nguồn. Ticket receipt của bài 45 chứng minh một effect đã commit trong failure domain SQLite; tool protocol items không chứng minh điều đó.

Bài 48 từ chối call ID tái sử dụng và signature tool+args lặp lại. Đây là stop policy đơn giản, không cache hoặc poll semantics. Với read health theo thời gian thật, cùng args ở thời điểm khác có thể hợp lệ; khi cho phép, cần interval/budget/freshness và phân biệt observation mới/cũ. Không xóa guard rồi hy vọng model tự biết khi nào ngừng.

Một tool ghi ở bài sau cần downstream idempotency/status lookup và xử lý timeout unknown. Đếm tool_calls hoặc dùng function_call ID không tự cho exactly-once effect. Cũng không dùng “model đã nói thất bại” để kết luận remote side effect không xảy ra rồi tạo key mới.

## 4. History của provider khác application state

History chứa messages/output items/tool results cần cho protocol. Application state còn có quyền hiện hành, reservations, signatures đã thấy, observations đã kiểm và kết quả cuối. Không để model edit history hoặc một tool result tự sửa counters/status.

Opaque reasoning items được mang qua như protocol data. Không render summary/encrypted content thành lời giải hoặc trace suy nghĩ công khai. Fixture opaque strings chỉ kiểm roundtrip. Bài không chứng minh mọi model/provider hỗ trợ cùng payload và không đo hiệu quả reasoning reuse.

Full-history replay đơn giản nhưng context tăng. Context cap hiện tính serialized Python characters, không token window. Một opaque item hoặc tool result lớn vẫn có thể khiến lần kế tiếp bị input_char_budget; app phải báo failure rõ. Truncation tùy tiện có thể làm mất call/result hoặc evidence. Compression và memory có quyền/source sẽ học ở bài 52, cần evaluation riêng.

## 5. Read-only và quyền truy cập dữ liệu

Model không có tool ghi vẫn có thể làm rò dữ liệu nếu tool đọc thiếu scope. Backend phải xác thực principal ngoài model, bind task target/service và lọc dữ liệu trước khi đưa vào context. World bind đọc ticket để kiểm quyền trước provider; dispatch read_ticket kiểm target lần nữa. Search dùng principal hiện tại và corpus eligibility trước candidate/IDF như bài 45.

Role reader có thể đọc runbook gateway nhưng không health; lab chặn khi đề xuất health, không giả dữ liệu đó rỗng vì provider không biết quyền. Khi hệ thật có capability discovery, có thể chỉ advertise tools được phép để giảm fail calls, nhưng dispatch vẫn phải kiểm lại. Capability quảng cáo không là authorization tại thời điểm thực thi.

Nếu quyền/source bị thu hồi trong lúc model call chạy, revalidate sau call giúp chặn bước tiếp và export output. Provider đã nhận context trước đó thì không thể “chưa từng nhận” chỉ bằng cách xóa observations trong result. Runtime DB, trace collector, provider storage và retention là những lớp riêng. Không quảng cáo current revoke guard thành remote data deletion.

## 6. Injection boundary và chất lượng ngữ nghĩa

Một ticket chứa “resolve ticket 2” có thể ảnh hưởng model. Host allowlist/args/target scope vẫn cần chặn dù model bị lừa. Test write_after_ticket cho model giả lập yêu cầu resolve sau khi đọc instruction, xác nhận không dispatch tên tool ấy. Nó không ước lượng xác suất một LLM thật bị lừa.

Quote-based answer giảm không gian synthesis nhưng chưa tự chống nội dung độc hại: nếu source approved chứa chỉ dẫn xấu, model có thể chọn quote ấy và host render. Tương tự, source có thể sai/đã thiếu cảnh báo trong đoạn khác. String identity không là semantic support hoặc completeness của cả nhiệm vụ.

Vì thế nâng agent cần hai lớp bằng chứng: control-plane correctness (quyền/dispatch/state/effect) và content/task quality (đúng/đủ/hữu ích/abstain). Lớp thứ nhất không thay lớp thứ hai. Bài 47 cung cấp nền rubric; bài 54 sẽ mở rộng trajectory/tool decisions và outcome nhiều bước.

## 7. Deadline, retry và unknown usage

Runtime kiểm clock trước và sau generate, sau tool dispatch và đầu vòng tiếp. Provider có budget riêng và kiểm giữa các đoạn I/O. Đây là cooperative: không preempt hàm đang block, không hủy side effect từ xa, không thu lại tokens đã sinh. Lệnh live có thể quá mốc dự kiến trước khi control quay về để báo lỗi.

Nếu provider timeout sau khi đã nhận request, attempt usage có thể unknown. Retry có thể sinh quyết định khác. Vì call mới chưa dispatch trước response được parse/guard, retry generation không tự lặp tool read ở runtime; dù vậy nó có thể tăng provider cost và tổng thời gian. Với write tools tương lai, vẫn phải giữ journal/receipt riêng.

Một call chỉ gồm reasoning mà chưa tool/terminal hiện bị expected_terminal_text, không tự thêm một turn “continue” vô hạn. Một response hai calls hoặc mixed call/text bị ambiguous_or_parallel_output, dù server được yêu cầu parallel=false. App giữ contract hẹp và fail rõ; muốn mở rộng cần quy tắc order/reservation/error aggregation mới.

## 8. Final/handoff và mức hoàn thành

Schema terminal không bao gồm arbitrary answer text; mỗi claim chỉ observation ID + quote. Host render quote/citation và status health từ snapshot. Đây là chủ ý giảng dạy cho lineage/tool loop; chẩn đoán nhiều bước với hypothesis/reasoning tường minh sẽ cần contract và rubric khác.

Handoff có thể là lựa chọn đúng khi health unknown hoặc thiếu nguồn, nhưng model cũng có thể handoff sớm dù có tool hữu ích chưa dùng. Current host chỉ kiểm shape của handoff, không chứng minh nó tối ưu. Không ép model tiếp tục đến answer để tăng completion, cũng không cho mọi handoff là task success. Cần nhãn expected behavior và đánh giá từng case.

Health degraded cùng lúc với 504 không chứng minh nguyên nhân upstream, database hay deploy. Snapshot được đánh thời điểm giả định và hash; nó không là telemetry live hoặc dữ liệu luôn mới. Host nói chưa xác định root cause để giữ đúng mức bằng chứng, không gắn một lời kết luận đẹp vào observation chưa hỗ trợ.

## 9. Reproducibility và các ranh giới chưa bền vững

Manifest core gắn implementation 5 module runtime, prompt/tools/schema, adapter config, backend manifest/corpus/health và budgets; report manifest thêm fixture/experiment source hashes. Source45 + bridge46 được băm theo bytes. Không hash mọi Markdown, dependency lockfile, environment hoặc publisher identity. Cùng release không làm generation live deterministic.

Agent single-use, history/state trong RAM. 27 scenario DB và bốn baseline DB chỉ giữ seed tickets, không lưu agent checkpoint. Reopen DB không resume run; creating another Agent là run mới. Budget reset giữa các instance không giải quyết distributed quota hoặc retry client. Cần owner/run IDs, durable state transitions, pending calls và recovery semantics ở bài 51.

Synchronous runtime hiện tuần tự; không có server queue, auth production, concurrent runs, cancellation framework hoặc API serving endpoint. Việc reuse Store45 không có nghĩa đã tích hợp một agent SaaS service. Những phần backend đã có là nền cho nhánh vận hành, không là phép đo capacity của bài này.

## 10. Chuyển sang live có thể đánh giá

Bước nhỏ: model tài khoản có quyền dùng Responses tools+strict terminal schema, corpus synthetic, goal retry/triage và case thiếu nguồn. Thu requested/returned model, response ID/status, protocol/final validity, trajectory, tool reads, usage/unknown và latency theo population đã chốt. Không dùng fixture expected outputs để tự gán nhãn cho text thật.

Dùng paired tasks/snapshots/scope/budgets để so workflow; bổ sung case mà agent có lý do đổi kế hoạch và task đủ đa dạng. Chấm cả tool selection, unnecessary calls, early handoff, evidence/answer completeness và critical errors. Một đường đi khác không tự tốt hơn; nhiều model calls chưa tự là model “suy nghĩ sâu”.

Live adapter có mã nhưng chưa được gọi trong phiên. Chưa xác nhận model cụ thể, provider schema acceptance hoặc opaque item compatibility phía server. Nếu 400, kiểm lại docs/model/config; không auto retry hoặc nới quyền để mong pass. Không cần gửi dữ liệu nội bộ hoặc thêm tool ghi để chứng minh tool loop đọc hoạt động.
