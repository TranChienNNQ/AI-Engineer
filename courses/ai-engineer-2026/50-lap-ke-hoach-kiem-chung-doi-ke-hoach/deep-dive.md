# Đào sâu bài 50 — Kế hoạch là giả thuyết có điều kiện

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Structured plan không đồng nghĩa formal planning

Plan DAG của bài mô tả dependency và postconditions của vài tool đọc. Nó không tìm kiếm trên một state-transition model đầy đủ, không PDDL/STRIPS solver, không bảo đảm optimality/completeness. Tên gọi “verifier” phải đi cùng câu hỏi: kiểm được điều gì và theo giả định nào?

Schema kiểm shape/type; DAG guard kiểm cấu trúc; runtime/MCP kiểm scope/current source; fixture verifier kiểm những điều kiện như evidence không rỗng hoặc health status không unknown. Không lớp nào ở đây tự suy ra nguyên nhân vận hành từ văn bản. Một graph hợp lệ có thể là kế hoạch tốn công, không đầy đủ hoặc không hữu ích.

Khi cần formal guarantees cho một domain, cần định nghĩa trạng thái/hành động/preconditions/effects và validator tương ứng, cùng việc kiểm domain model có phản ánh môi trường thật không. Không lấy một plan JSON thỏa schema làm thay thế.

## 2. Đơn vị tiến triển: tool execution, postcondition và task outcome

Một tool invocation đã trả response là một sự kiện thực thi. Postcondition đòi hỏi dữ liệu response có cung cấp điều đang cần hay không. Task outcome lại phụ thuộc nhiều postconditions và điều kiện nghiệp vụ. Ba mức này không nên bị trộn trong một status “done”.

S2 search empty đã chạy, có observation O2 và có thể tiêu tốn MCP operations/context; nhưng runbook_available chưa supported. Nếu model đánh dấu S2 successful chỉ vì ok=true, bước S3 có thể chạy theo một assumption sai. Guard tính trạng thái từ observations, không tin trường status model tự thêm.

Sau S4 tìm thấy nguồn, nghĩa vụ runbook có thể thỏa dù S2 vẫn empty. Vì thế “mọi plan step từng được đề xuất đều phải thành công” không phải định nghĩa task success phù hợp ở đây. Một kế hoạch có nhánh thay thế cần giữ thất bại lịch sử và chứng minh nghĩa vụ cuối đã được đáp ứng.

## 3. Plan identity, parent và immutable past

Revision gồm nội dung mới, parent hash, trigger và observation refs. Parent ràng buộc đúng bản plan mà người sửa đang dựa trên; revision liên tiếp giúp phát hiện proposal cũ hoặc bỏ qua lịch sử. Nó không tự là concurrency protocol phân tán hoặc signature của người duyệt.

Executed steps không được xóa/đổi ID/args/dependency/postcondition. Nếu cần query khác, tạo S4 thay vì sửa S2 đã chạy. Điều này giữ trace trung thực và giúp chấm cost/recovery. Unexecuted steps có thể được đổi dependency khi revision có trigger hợp lệ.

Hash không bảo đảm semantic progress. Một model có thể thêm query khác chữ nhưng không thêm thông tin. Rule chống reused trigger/no-op và finite budgets chỉ chặn một số vòng lặp rõ; cần task quality evaluation, không tuyên bố mọi revision hợp lệ là thông minh hoặc tối ưu.

## 4. Kế hoạch trước hành động và replanning sau observation

Một initial plan hữu ích để thấy dependency và ngân sách dự kiến. Khi môi trường chưa biết đủ, không nên xem nó như script chắc chắn đúng. Receding-horizon thực dụng là chỉ thực thi bước ready tiếp theo, quan sát, rồi giữ hoặc sửa phần tương lai.

Trong bài, proposal có thể mang revision và một action cùng lượt. Host nhận plan trước, kiểm action trong plan/ready rồi giao runtime. Valid plan có thể được lưu dù action bị chặn, hoặc revision accepted nhưng runtime tool budget cạn trước dispatch. Vì thế accepted_plan_count không bằng executed_steps hay task_success.

Port này không có planning-only turn hoặc nhiều candidates/search tree. Muốn thêm chúng cần budget riêng, request/trace schema và cách chấm lợi ích của lượt lập kế hoạch không gọi tool. Không tự coi thêm một lớp planner là miễn phí hoặc luôn tăng quality.

## 5. Verification ledger và sự phụ thuộc vào nguồn

Ledger lưu status kèm observation refs. Ticket chứa symptom/reported_health/report_at là report của fixture, không oracle độc lập về trạng thái thật. Health là snapshot tổng hợp có thời điểm/hash. Runbook cung cấp hướng dẫn, không observation khẳng định remediation đã chạy.

Nếu report và snapshot cùng mốc mà khác status, lab coi là discrepancy cần clarification. Nếu khác mốc, status có thể đã thay đổi nên không trực tiếp so như contradiction. Đây là rule minh họa; hệ thật còn cần timezone, freshness, scope, nguồn đo và độ tin cậy của từng report.

root_cause_established luôn unknown vì không có causal diagnostic tool. Điều đó không có nghĩa mọi sự cố thật đều không xác định được root cause; chỉ là action/evidence space của bài chưa đủ. Khi mở rộng, phải định nghĩa điều gì hỗ trợ kết luận, dữ liệu nào bác bỏ nó và ai chịu trách nhiệm xác nhận.

## 6. Refuted, unknown, inconsistent và stale không giống nhau

Không tìm được evidence chưa phải evidence rằng giả thuyết sai. Tool unavailable chưa phải hệ thống đích bình thường. Hai status khác nhau ở hai thời điểm chưa phải contradiction. Một nguồn đúng lúc trước có thể stale ở thời điểm phục vụ hiện tại. Những phân biệt này quyết định bước tiếp.

Lab dùng supported/empty/unknown/tool_error/discrepancy/not_comparable theo check cụ thể, chưa có một logic đa trị hoàn chỉnh hoặc hệ cập nhật niềm tin Bayes. Không gắn confidence .9 chỉ vì model viết tự tin. Nếu có confidence sau này, cần định nghĩa xác suất gì, dữ liệu hiệu chuẩn và metric tương ứng.

Một verifier rule-based vẫn có thể sai nếu parser/report model/assumptions sai. Closed-world checks giúp kiểm behavior đã biết, không tự thừa kế danh hiệu “sound” cho mọi văn bản mới. Human review và semantic evaluation vẫn cần khi dùng dữ liệu ngoài generator fixture.

## 7. Thông tin có giá trị khi có thể đổi quyết định

Trước tool call, hãy hỏi kết quả có thể làm chọn action khác, dừng hoặc giảm bất định quan trọng không? Đọc lại cùng snapshot có thể không thêm gì. Query mới từ symptom đã thấy có thể hữu ích nếu query cũ chỉ chứa một mã không có trong corpus.

Expected value p×gain−cost trong bài chỉ là ví dụ giả định. p không được rút ra từ “LLM nghĩ có khả năng”; gain/cost còn phụ thuộc tác động, reviewer effort và thời gian chờ. Quyết định tool cũng phải thỏa scope, ngay cả khi expected value giả định cao.

Một guard semantic progress tổng quát khó hơn so chữ/hash. Query khác có thể là paraphrase vô ích; query giống có thể hữu ích nếu dữ liệu đã đổi. Bài chọn finite/no-poll policy đơn giản và ghi giới hạn. Khi cần polling hoặc dynamic data, bổ sung snapshot/freshness/time budgets và evaluation riêng.

## 8. Verification, authorization và export privacy

Plan có dependency đúng không làm tool được cấp quyền. Runtime 49 tiếp tục bind principal/target/service, revalidate source, pin catalog và kiểm result identity. Không cho plan chứa tenant/approval/budget override rồi coi đó là task requirement.

Một plan có thể chứa query từ ticket, còn planning trace có terminal quote/evidence refs. Nếu chỉ redact observations khi quyền bị thu hồi mà vẫn export plan/trace cũ, app vẫn có thể rò dữ liệu. PlannedAgent xóa các artifact planning khỏi result bị blocked, và có kiểm regression revoke sau lượt đầu.

Đó là boundary export của một run. Contexts vẫn có thể đã ở RAM trong script/controller; production logs/artifact store/provider data cần retention/quyền riêng. Không mô tả việc xóa result fields như remote deletion hoặc không từng gửi dữ liệu.

## 9. Baseline, ablation và mẫu số

Phép so trong bài đổi cả scripted policy, planning representation và discrepancy gate. Vì thế khác outcome không định lượng riêng đóng góp của plan DAG hoặc verifier. Đây là demonstration cơ chế, không một ablation nghiên cứu đã kiểm soát mọi biến.

Một evaluation live tốt có thể so reactive và planned với cùng model/task/tools/snapshots/budgets/final rubric, rồi ablate checker/repair strategy có chủ ý trong môi trường thử. Báo unsuccessful calls, early handoff, false confirmation, extra plans/calls/latency và human effort. Không chỉ giữ case recover thành công mà bỏ case thrash/timeout hoặc cost của proposals bị chặn.

Planning turns 77, accepted plans 31 và MCP calls 52 của suite là counter cho các việc khác nhau. Usage provider là not_applicable vì không có LLM/HTTP ở port 50. Không so cost token của script với một model thật rồi gọi là cải tiến hiệu suất.

## 10. Giới hạn runtime và bước tiếp

Guard state/plan chain ở RAM, single-use run, không checkpoint/replay/cross-process recovery. SQLite ghi report seed trước run và phục vụ tool reads, không persist plan transitions. Reopen DB chưa resume counters hay biết tool đang pending. Bài 51 sẽ cần run IDs, durable events/ownership và crash boundaries cụ thể.

PlanningPort sử dụng prompt/view 50 riêng, không cùng request body của ToolAdapter 48. Nếu nối LLM thật, cần adapter cho Proposal schema, strict parsing/time/input-output limits/usage/provenance và mới chấm outputs; không nói chỉ đổi profile là đã có live planner. Lịch proposal hiện do tác giả lập trình và biết fixtures.

Nguồn nghiên cứu của bài chỉ đọc abstract/metadata. Những kết quả cũ trên vài model/domain không chứng minh mọi LLM hiện tại không lập kế hoạch, cũng không chứng minh plan JSON/critique loop của một ứng dụng đã đáng tin. Kết luận của lab chỉ dựa trên các contract và dataset synthetic đã công bố.
