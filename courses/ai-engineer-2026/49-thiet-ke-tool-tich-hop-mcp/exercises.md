# Bài tập 49 — Thiết kế tool và MCP cho agent

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Code chạy từ thư mục bài 49 trong process sạch, giữ source 45–46–48 cạnh bên. Không yêu cầu API key hoặc dữ liệu công ty. MCP stdio là subprocess local thật; model vẫn fixture.

## 1. Thiết kế một tool rõ nghĩa

So do_action(data), execute_sql(sql) với ticket_read(request). Viết description nêu mục đích, input, scope, output/freshness và side effect. Những field nào nên bind ở host/backend thay vì để model chọn?

## 2. Ánh xạ hai giao diện

Vẽ đường read_ticket model→host→MCP ticket_read→observation→model lượt tiếp. Tên/args ở hai giao diện có giống nhau không? Vì sao MCP không tự thay planner? Đọc mcp_provenance trong observation.

## 3. Catalog và pagination

Chạy list tools, đọc schema. Chia catalog thành hai pages giả rồi merge; thử duplicate tool ở page sau. Cursor lặp/page budget có rủi ro gì? Lab đã test pagination server thật hay typed client giả?

## 4. Schema an toàn và provider compatibility

Thử external $ref, $dynamicRef hoặc dialect lạ. Vì sao host không resolve tùy ý URI schema? MCP inputSchema có $defs/wrapper có nên forward nguyên sang provider tool schema không? Bài này chọn cách nào?

## 5. Lifecycle và phiên bản

Giải thích mode auto/protocol_version đã quan sát. Có phải mọi MCP connection đều initialize handshake không? So Client(server) với Client(StdioServerParameters). Vì sao giữ một async lifecycle qua run và dùng portal thay vì asyncio.run từng call?

## 6. Output schema chưa là resource identity

Tạo result có envelope hợp lệ nhưng ticket_id sai, quote/hash không khớp hoặc health snapshot khác. Lớp nào chặn mỗi lỗi? Thử text content chứa instruction nhưng structuredContent hợp lệ; phần nào được đưa vào model context?

## 7. Quyền hai phía

Gọi server trực tiếp bằng client, bỏ host, yêu cầu ticket 2 hoặc health dưới goal retry. Server phải làm gì? Điều này có phải OAuth/per-user authorization production không? Những thông tin nào model không được gửi để tăng scope?

## 8. Phân loại lỗi

So HostError local_input_contract, mcp_protocol_error, tool_execution_error, business unavailable, source_result_identity và mcp_timeout. Trường hợp nào có tools/call, có backend read, có observation để model handoff? Vì sao không retry tất cả?

## 9. Request budget và counters

Với max_requests=2, review catalog dùng một operation, refresh dùng operation thứ hai. Tool call có được gửi không? Tính8 operations của triage và giải thích tại sao chưa là toàn wire round trips. Calls sent33 nhưng backend reads31 nghĩa gì?

## 10. Catalog drift

Đổi description của service_health ở real in-process server giữa run rồi gọi ticket_read. Host có phát hiện dù tool đang gọi không phải health không? Vì sao hash không là signature, và refresh/call vẫn có TOCTOU?

## 11. IDs và namespace

Phân biệt server alias, serverInfo.name, remote tool name, model call_id, MCP JSON-RPC ID, observation ID và document hash. Hai server có cùng tên tool xử lý thế nào? Có thể lấy call_id làm ticket idempotency receipt không?

## 12. Đọc14 scenario đúng phạm vi

Đếm answered/handoff/failed/blocked. Vì sao14/14 đúng contract chưa là 100% LLM success? Ca nào chạy stdio thật, ca nào in-process, ca nào dùng test double trong checks? Cần bằng chứng gì trước claim model chọn tool tốt?

## 13. Integration parity

Chạy direct Agent48 và stdio MCP Agent49 cùng goal. So answer/citation/path/model calls và tools. Vì sao parity có ích nhưng chưa chứng minh MCP cải thiện chất lượng hoặc giảm latency? Phép so này khác workflow baseline bài 48 thế nào?

## 14. Stdio và đóng lifecycle

Server có được print debug vào stdout không? Hãy chạy một Link stdio rồi đóng; Client có được dùng lại không? closed=True chứng minh điều gì và chưa chứng minh loại crash recovery nào?

## 15. Release và provenance

Manifest49 thêm gì so bài 48? Vì sao cần catalog/deployment/budgets/source hashes, nhưng hash chưa xác thực publisher hoặc facts? Nếu runtime/output đổi mà vẫn dùng rating hoặc source snapshot cũ thì cần xử lý gì?

## 16. Thiết kế tool mới và live evaluation

Đề xuất tool đọc trace của dịch vụ: input/scope/output/source/time/error và điều cấm. Lập kế hoạch kiểm server, host mapping, model decisions và paired evaluation trước dùng thật. Chỉ ra phần cần xây thêm để gọi LLM thật qua MCP49; không yêu cầu thực hiện call có phí hoặc tool ghi.
