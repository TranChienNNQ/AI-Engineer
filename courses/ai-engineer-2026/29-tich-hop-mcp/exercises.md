# Bài 29 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán frame/error/operation trước khi chạy. Không cần LLM, tài khoản hoặc server MCP của bên thứ ba.

## 1. Vị trí của MCP

Vẽ host/client/server/backend cho trợ lý điều tra ticket. Ai chọn tool, ai gửi MCP request, ai kiểm dữ liệu workspace và ai giữ agent state? Server có bắt buộc chạy một LLM không? Một endpoint REST JSON bất kỳ đã đủ là MCP chưa?

## 2. Version protocol và SDK

Phân biệt 2026-07-28, 2025-11-25 và mcp 2.3.0. Chạy modern và forced legacy, đọc version thực tế. Tại sao lời dạy “luôn initialize trước” có thể sai thời điểm? Cùng SDK hỗ trợ legacy có chứng minh tương thích mọi server cũ không?

## 3. Ba primitive

Phân loại get_ticket, kb://catalog, kb://runbooks/{doc_id}, triage_ticket. Method nào list/call/read/get từng loại? Template có nằm chung danh sách concrete resource không? Render prompt có tự chạy một tool hoặc trở thành system instruction không?

## 4. Đọc wire

Chạy wire demo, kiểm request/response ID, metadata version, result/error và resultType. Đếm result thành công, tool error result, RPC error. Unknown RPC method khác unknown tool name ở đâu? Vì sao raw observer của lab chưa thay SDK client production?

## 5. Tính tay số tích hợp và latency

Với 3 host/4 backend, tính N×M và N+M trong sơ đồ lý tưởng. Những phần việc nào vẫn còn? Tính 10+10+20+30+20+20 ms, chỉ rõ bước bị bỏ ngoài ví dụ. Vì sao không gọi 110 ms là latency đo được của MCP?

## 6. Stdio process thật

Chạy demo stdio, so với in-memory. Process nào mở stdin/stdout, khi nào server chờ, ai đóng process? Vì sao không print banner lên stdout? Tại sao command+args cố định tốt hơn lấy command từ model? Có cần port cho stdio không?

## 7. Streamable HTTP

Chạy http-asgi. Nó kiểm những lớp nào, chưa kiểm lớp nào so với TCP/TLS/proxy deployment? Trong revision đang học, endpoint dùng POST/JSON/SSE như thế nào? SSE trong response có đồng nghĩa transport `/sse` cũ hoặc GET stream/session đời cũ không?

## 8. Input validation ở hai phía

Thử nested field actor, outer field actor, limit=True và limit=4. Host/server SDK chặn hoặc xử lý ra sao? Đọc observed behavior của SDK 2.3.0. Vì sao outer field bị bỏ qua không biến actor thành credential? Việc schema đóng có thay resource authorization không?

## 9. Structured output và identity

So content với structuredContent và tên thuộc tính Python. Output đúng schema nhưng ticket ID khác task phải xử lý thế nào? Nếu server trả text “ok” với isError=true thì có được dùng làm kết quả thành công không? Hash body có chứng minh nguồn thật và text đúng nghĩa không?

## 10. Resource selection và freshness

Search rồi đọc resource được chọn; thử URI file/HTTP ngoài danh sách. Tăng source version giữa search/read hoặc đổi workspace nguồn. Tầng nào chặn? Vì sao giới hạn bytes sau nhận chưa là network-level DoS protection?

## 11. Alias và catalog review

Kết nối blue/green, cùng serverInfo.name và remote tool names. Tạo tên qualified không trùng. Thử alias trùng hoặc remove tool sau review. Catalog hash có phát hiện mọi thay đổi behavior của server không? Host đang hỗ trợ pagination/listChanged subscription tới đâu?

## 12. Ba tầng lỗi và khác biệt implementation

Phân loại host outside_task, local timeout, unsupported protocol version, tool not_permitted và prompt thiếu argument. Nêu observed behavior của SDK bản ghim so với docs/spec examples. Tại sao client không được suy mọi JSON-RPC error là tool result hay sửa mọi assertion để che một khác biệt?

## 13. Quyền của deployment

Blue server được gọi trực tiếp bằng SDK với T-GREEN-1, bỏ qua host guard. Kết quả ra sao? Workspace argv có phải authentication nhiều người dùng không? clientInfo.name/serverInfo.name/URI/handle có tự chứng minh người dùng được cấp quyền không?

## 14. Prompt injection và external template

Đọc RB-BLUE-POISON. Body chứa [SYSTEM] có đổi tool list, cấp write hoặc tự chạy HTTP request không? Vì sao đây chưa là benchmark LLM chống injection? Host nên xử lý prompt/instructions từ server như thế nào? Residual answer-only còn thiếu lớp đánh giá gì?

## 15. Budget và quan sát hệ thống

Demo có 5 logical data operations nhưng wire observer có 13 request. Có mâu thuẫn không? Discovery/list/cache/SDK extras nằm ở đâu? Thiết kế trace có đủ alias/tool/method/error/latency nhưng tránh credential/raw body. Nếu thêm write và response mất thì cần cơ chế nào từ bài 27?

## 16. Dự án nhỏ

Bọc một backend tổng hợp khác thành MCP server có 2 read tool, 1 concrete resource, 1 resource template và 1 prompt. Viết host allowlist/schema/alias/budget, kiểm server authorization độc lập host. Có stdio thật, ít nhất một error result/protocol error, catalog drift, stale resource và residual chưa chặn. Ghi protocol/SDK version và phân biệt in-memory/ASGI/TCP kiểm được gì. Không cài server lạ hoặc kết nối dữ liệu thật để làm bài.
