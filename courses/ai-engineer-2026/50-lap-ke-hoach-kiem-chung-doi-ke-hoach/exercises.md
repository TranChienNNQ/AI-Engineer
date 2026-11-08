# Bài tập 50 — Lập kế hoạch, kiểm chứng và đổi kế hoạch

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Chạy code từ thư mục bài 50 trong process sạch, giữ source 45–46–48–49 cạnh bên. Không cần API key hoặc dữ liệu công ty. Planner scripted, MCP thực thi thật.

## 1. Plan khác lời kể

Viết một kế hoạch điều tra 504 có bước, input, prerequisite và postcondition. Vì sao “tôi sẽ kiểm tra rồi sửa lỗi” chưa đủ để runtime biết hoàn thành? RAG/workflow/reactive agent/planned agent có thể dùng chung tools không?

## 2. Thực thi và tiến triển

Search trả ok=true/evidence=[] có trạng thái gì ở tool, step và nghĩa vụ runbook? So với health unknown, tool unavailable và successful runbook search. Vì sao không lấy HTTP/MCP success làm task success?

## 3. DAG và schema

Tạo một dependency cycle, dependency không tồn tại hoặc step ID trùng. Guard nên chặn gì? Model thêm status=done có hợp lệ không? Một DAG không cycle có bảo đảm plan hữu ích hoặc tối ưu không?

## 4. Bước ready và authority

Ở plan ban đầu, step nào ready? Nếu action chọn S3 ngay, hoặc args ticket 2 khác plan/task 1 thì sao? Plan_guard và MCP/runtime scopes có vai trò khác nhau thế nào?

## 5. Sửa query nhưng không sửa quá khứ

Chạy recover_memory. Đọc hai revisions, S2 và S4 cùng observations. Tại sao S2 vẫn empty khi task đã answered? S3 đổi dependency từ S2 sang S4 có phải viết lại lịch sử không?

## 6. Parent và trigger của revision

Thử stale parent_hash, based_on=O999 hoặc replan missing_evidence nhưng chỉ dựa O1 ticket thành công. Vì sao các trường hợp phải bị chặn? Nếu chỉ đổi cách diễn đạt plan mà không đổi steps/intent hoặc thêm evidence, có progress chưa?

## 7. Discrepancy và thời gian

So discrepancy với different_time. Cùng status report degraded/snapshot ok, vì sao check khác nhau? Có thể kết luận bên nào sai hoặc database là root cause không? Khi nào agent nên hỏi thêm/handoff?

## 8. Giả thuyết và kết luận chưa được hỗ trợ

Chạy claim_root. Tại sao có ticket/runbook/health vẫn không được root_cause_confirmed? Verifier bài này kiểm những gì và không kiểm những gì? Phân biệt unknown với refuted hoặc một false statement đã được chứng minh.

## 9. Dữ liệu không có quyền sửa plan state

Model tự thêm status=done/budget=999, đổi args của step đã chạy hoặc đề xuất write tool. Host phải làm gì? Đề xuất trong plan có thay được principal/target/MCP policy hoặc review requirement không?

## 10. Đọc phép so baseline

So straight/recover/discrepancy trong comparison. Recovery thêm bao nhiêu turns/tool calls? Discrepancy outcome khác vì những thay đổi nào? Vì sao chưa thể quy khác biệt đó cho “LLM planning tốt hơn”?

## 11. Giá trị thông tin giả định

Với p=.6, gain=10, cost=2, tính expected net value. Nêu một tool output có thể đổi quyết định và một tool call thêm không có giá trị rõ. Những giả định/dữ liệu nào còn thiếu để dùng công thức trong sản phẩm thật?

## 12. Counter và contract success

Đọc 25 cases/31 accepted plans/77 planning turns/52 MCP calls. Các mẫu số có cùng nghĩa không? Vì sao 25/25 không là 100%LLM success hoặc model token savings? Liệt kê metric semantic/trajectory còn cần.

## 13. Budget và chống thrash

Chạy revision_budget, tool_budget và thrash. Revision mới có reset tool/model counters không? Vì sao same trigger O2 không được dùng mãi dù plan text khác? Rule này đã phát hiện mọi semantic no-op chưa?

## 14. Thu hồi quyền và export

Nếu revoke actor sau read_ticket, chỉ xóa observations nhưng giữ planning_trace có đủ không? Artifact nào có thể chứa dữ liệu nhạy cảm? Lab có xóa mọi RAM/context/provider bytes khi redact result không?

## 15. Reproduce và MCP stdio

Chạy recover_stdio, kiểm link closed, plan parent/hash và seed/effects. Vì sao setup UPDATE body không phải action do model thực hiện? Mở lại SQLite có tự resume plan không?

## 16. Thiết kế live planning experiment

Lập kế hoạch thu output Proposal từ LLM thật: prompt/schema/view, tools/scope/budgets, baseline/gates, annotation/verification, stop/quality/cost/latency và artifact lineage. Chỉ ra phần nào port 50 chưa có, và vì sao không chỉ dùng prompt 48 cũ hoặc author fixture labels cho model mới.
