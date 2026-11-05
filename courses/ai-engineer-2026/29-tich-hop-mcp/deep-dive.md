# Bài 29 — Đào sâu version, boundary và cách tích hợp

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Giao thức chuẩn hóa phần giao tiếp, không chuẩn hóa mọi semantics nghiệp vụ

Hai server cùng có tool search vẫn có thể khác nguồn, quyền, tokenizer, freshness và giới hạn. Host cần hiểu contract/effect cụ thể, không giả rằng cùng JSON shape là cùng semantics. Việc backend dùng REST/SQL/file ở phía sau không đổi nghĩa vụ kiểm quyền/freshness.

Discovery giảm code glue tự chế, nhưng mapping model tools, task policy, output validation, retry và tracing vẫn là ứng dụng. Lab không gọi model: agent.py là fixed host workflow để tách kiểm protocol khỏi quality của agent loop. Khi nối bài 28, reducer có thể lưu data/ref/alias, nhưng không lưu credential vào model context.

## 2. MCP 2026 per-request state và agent state là hai thứ khác nhau

Revision 2026-07-28 yêu cầu request tự khai protocol metadata. Không dựa vào connection trước đó để suy negotiated capability hoặc người dùng. Software clientInfo là thông tin interoperability/debug; authentication cần kênh xác thực phù hợp.

State ứng dụng vẫn có thể tồn tại: ticket, cart, job, checkpoint. Nó cần explicit handle/identifier và authorization theo caller ở từng request; không được suy cùng stdio process là cùng cuộc hội thoại. Một process có thể phục vụ nhiều task.

Legacy initialize/session có semantics khác. SDK auto/legacy là compatibility mechanism, không là quyền mới. Lab modern transcript có server/discover, còn forced legacy kiểm SDK với chính server ấy; chưa là compatibility matrix của mọi peer v1/v2.

## 3. Metadata cũng là đầu vào có thể sai

Tool description, annotations, server name, instructions và prompt đều có thể chứa nội dung gây hiểu sai. Catalog review cần biết server nào, chạy code gì, có quyền nào và tool có effect gì; không thể suy tin cậy từ readOnlyHint hoặc icon/tên đẹp.

Host alias đến từ cấu hình, không lấy server name tự khai làm global route. Fingerprint giữ snapshot của descriptor, phát hiện một số thay đổi cần review; nó không chứng thực code execution, source correctness hoặc ngăn race hoàn toàn giữa list và call.

review_catalog trong lab là trusted helper cho source local đã biết. Nó không dựng UI review/consent hoặc phân tích semantics của metadata. Khi gắn server thật, phải thiết kế approval/policy theo quyền/ý định đã có, không tự approve server lạ chỉ vì kết nối được.

## 4. Schema là contract chứ không là auth hay sandbox

Pydantic nested request đóng field, còn outer argument schema SDK sinh có thể không đóng. Host overlay outer policy nên từ chối field không dùng; direct SDK behavior được kiểm và ghi riêng. Không mặc định mỗi framework tự bảo đảm mass-assignment safety ở mọi lớp.

Host dùng JSON Schema 2020-12 và chỉ local refs. External ref có thể tạo fetch không mong muốn hoặc SSRF nếu validator được cấu hình tự tải; lab từ chối thay vì tải. Kích thước schema được giới hạn nhưng code vẫn là teaching subset, chưa sandbox cho mọi JSON Schema adversarial/recursive.

State/output đúng schema vẫn có thể sai resource/task/meaning. Host kiểm returned ticket ID, selected URI, source version và body consistency. Nó chưa kiểm factual truth, entailment hoặc summary faithful; residual answer-only minh họa khoảng trống ấy.

## 5. Discovery/list/caching và protocol extras

tools/list, resources/list và prompts/list có thể phân trang; code hiện báo pagination_not_implemented nếu tool list có nextCursor. Không có traversal/gộp mọi trang hoặc subscriptions/listen; server SDK có thể advertise capability rộng hơn tính năng host đang dùng.

SDK có thể tự discovery/re-list khi thiếu/stale header/schema metadata; caching cũng có thể làm không phải mọi method invocation tương ứng một network round trip. Counter host.ops chỉ đếm data operations do workflow thực hiện, chưa đo request count/cache hit hoặc provider bill.

Nếu triển khai cache, key phải xét server deployment/version, auth scope, metadata và freshness; không cache credential hay dùng một listing của người có quyền cho người khác. Lab trả ttlMs=0 ở các listing/resource SDK quan sát được nên không benchmark cache behavior. Token generation, auth/user caches và notifications không được kiểm ở đây.

## 6. Transport lifecycle khác task lifetime

Stdio child process tồn tại trong async with; đóng stdin là shutdown signal bình thường. Nó không cần restart cho mỗi model turn hoặc tương ứng một user session. Lab tạo child mới cho từng demo vì muốn isolation và cleanup rõ, không là policy tối ưu production.

Raw observer dùng một request rồi chờ một response, không có dispatcher cho notifications xen kẽ, concurrent IDs, multi-round input hoặc subscription. Nó chỉ để nhìn wire. Không bọc observer thành production SDK dựa vào vài frame chạy được.

HTTP-ASGI dùng real HTTP client/endpoint logic nhưng không socket/TLS/proxy. Transport URL 127.0.0.1:8000 chỉ là tên trong ASGI test; server không listen ở đó. Optional server.py --transport http là đường để tự thử local, chưa được tự động kiểm qua TCP trong bài.

## 7. Lỗi implementation cần được ghi như một quan sát có version

Spec minh họa unknown tool là protocol error; SDK 2.3.0 được chạy trả isError result. SDK docs main minh họa missing prompt argument -32602; bản ghim trả -32603. Các observation được giữ trong fixture, kèm khác biệt, không bị xóa hoặc đổi để báo “tuân thủ hoàn toàn”.

Đừng suy mọi server sẽ giống bản này. Có thể dependency patch thay behavior trong tương lai; cần integration tests, pin compatible version và đọc error envelopes/types. Public docs cũng có thể theo main vượt bản release: lab đã phải dùng import `mcp.client.streamable_http.streamable_http_client` thay vì giả mọi re-export cấp root đều có trong 2.3.0.

Một error string có thể chứa input nhạy cảm hoặc schema chi tiết. Host normalize layer/code và trace metadata; report nghiên cứu tổng hợp giữ payload để học, không là mẫu logging production. Fixtures chỉ che diagnostics lỗi dự kiến trong process test; server logging vẫn giữ.

## 8. Authentication, authorization và confused deputy

Server lab cố định workspace từ deployment config như một service account. Đây không phải cách xác thực người dùng từ model args, clientInfo hoặc connection. Nếu triển khai cho nhiều người, credential và scope phải đi theo request đúng người và backend vẫn kiểm resource, không chỉ check đã có token.

OAuth flow, token audience, downstream delegation, SSRF trong discovery URLs và Origin/DNS-rebinding cần thiết kế riêng theo transport. Tài liệu security/authorization được dùng để đặt boundary; lab không triển khai token issuer/verifier hoặc kiểm những attack ấy. Không mang token của downstream API đi làm token của MCP server tùy tiện.

Local server process có quyền OS của nó; command injection qua config, mã package lạ hoặc môi trường credential quá rộng đều nằm ngoài việc JSON-RPC có đúng format hay không. Stdio không là sandbox. Lab dùng file đã biên soạn, executable tuyệt đối và arguments riêng; không install vào host app hoặc đọc thư mục riêng của người dùng.

## 9. Cancellation và write semantics

Client timeout/cancel chỉ cho biết caller không còn đợi. Remote backend có thể đã có effect; việc pipe/SSE đóng không rollback tác vụ đã commit. Lab chỉ có read nên không tái hiện write ambiguity qua MCP.

Nếu thêm write, giữ permission/intent grant, expected version, idempotency key và receipt/reconciliation bài 27. MCP request ID khác business intent key; retry transport với ID mới có thể vẫn là cùng ý định write. Protocol `resultType=complete` cũng không chứng minh write/task đúng nghĩa.

Không cho model tự gọi tool review/grant hoặc thay credential. Elicitation/input_required có thể thu thập thêm input nhưng vẫn cần host policy xác định thẩm quyền; không tự coi mọi form response là quyền cho chuỗi effect không giới hạn.

## 10. Các nhánh nâng cao được gọi tên nhưng chưa triển khai

MRTR/input_required, subscriptions/listen, elicitation, extensions/Tasks và multimodal content là phần cần đọc theo đúng revision khi thực sự dùng. Sampling thuộc nhóm deprecated ở revision hiện hành theo architecture page; không dạy nó như khả năng bắt buộc phải có để chạy một MCP server.

Bài không chạy Inspector, đăng registry, cài server vào desktop host, hoặc OAuth/remote TCP. Muốn thêm những phần đó cần quy trình cài/deploy/test riêng, không thể suy đã hoàn thành từ in-memory/stdio/ASGI demo.

## 11. Đánh giá từ protocol tới task

Tách protocol exchange, schema/policy, task correctness, source support, privilege violations và latency/cost. 33 predicate có integration, unit, expected denial và observed SDK behavior; không cùng mẫu số với 13 wire requests hoặc 5 data operations.

Poison body đi qua đúng protocol vẫn là dữ liệu có instruction độc hại. Scripted host không chọn tool từ ngôn ngữ nên chưa đo LLM attack success. Khi thêm model, cần benign/adversarial trials, model/revision/sampling, reset data/state và semantic task labels; giữ residual/false positive, không chỉ báo số deny.
