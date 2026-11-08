# Đào sâu bài 49 — Tool contract và ranh giới MCP

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Tool design là thiết kế quyết định, không chỉ bọc API

Một backend API có thể cần hàng chục field theo schema nghiệp vụ, nhưng model không nhất thiết phải tự chọn tất cả. Nếu tenant/service/target đã có trong task trusted, đưa chúng thành input tùy ý chỉ tạo thêm cơ hội nhầm scope. Host nên bind phần authority, để model chọn các tham số phục vụ tác vụ như query tìm kiếm.

Tool quá rộng làm model phải đoán hành vi, tool quá nhỏ có thể làm một tác vụ tốn nhiều vòng gọi. Cần đo trên task cụ thể: selection đúng không, args sai ở đâu, tool có thiếu thông tin để quyết định không, có tool nào bị gọi thừa không? Không có số tool tối ưu chung. Lab giữ3 tool vì đây là mức đủ kiểm vòng triage/retry từ bài 48.

Tên, description và examples cần nhất quán với policy thật. Nếu description nói “có thể đọc mọi ticket” nhưng host chỉ cho một ticket, model có thể liên tục đề xuất bị chặn. Sửa description giúp giảm ambiguity nhưng không thay scope guard. Không báo cải thiện tool usability nếu chưa thu model decisions hoặc nghiên cứu người dùng tương ứng.

## 2. Ba contract cùng tồn tại

Contract model48 là logical tool names/flat args và terminal schema. Contract MCP49 là remote names/nested wrapper/inputSchema/outputSchema/results. Contract nghiệp vụ là ticket/source/health identity, role/goal/task scope và invariants hiện hành. Không coi chuyển JSON hợp lệ giữa hai giao thức là đã giữ toàn bộ nghiệp vụ.

Adapter có chủ ý là mapping hữu hạn read_ticket→ticket_read, search_runbooks→runbook_search, read_health→service_health. Nó không convert mọi JSON Schema remote thành provider function schema. Khi thêm tool, cần cập nhật allowlist/mapping/input-output models, mô tả model và tests; không tự expose toàn bộ catalog vì server mới trả thêm một tool tên hấp dẫn.

Remote result có thể thỏa outputSchema nhưng không thỏa local data contract. Envelope data là dict chung; local Ticket/Search/Health kiểm cụ thể theo tool. Sau đó identity/source validation còn kiểm dữ liệu ấy đúng resource của task. Tách các lớp giúp nhìn lỗi nằm ở schema, integration hay domain.

## 3. Discovery, trust và pinning

Discovery trả catalog/capabilities/metadata; nó chưa chứng minh server là đối tác tin cậy. Lab chỉ kết nối module local đã đóng gói hoặc subprocess từ đường dẫn cố định của giáo trình. Việc pin catalog ngay sau discovery là chấp nhận snapshot của deployment đó theo cấu hình đã biết, không một quy trình duyệt server bên ngoài.

Hash catalog bao tên/descriptions/schemas/annotations và metadata có trong snapshot. Thay đổi bất kỳ thành phần ấy làm refresh không khớp. Đây là bảo vệ khỏi thay đổi giữa run hoặc join nhầm snapshot; không là chữ ký, semantic approval hoặc authorization. Một description độc hại đã nằm trong snapshot ban đầu vẫn có thể được băm bình thường.

Trong lab, raw descriptions không được forward model. Nếu sản phẩm muốn cho model thấy tool descriptions từ nhiều server, phải xem chúng như một bề mặt input không đáng tin, có quy trình review/curation phù hợp và evaluation injection. Annotation read-only cũng có thể sai; backend implementation và policy mới quyết định có side effect hay không.

## 4. Catalog mutation và TOCTOU

Host refresh trước call để không dùng schema vừa thay đổi mà không biết. Tuy nhiên list rồi call là hai request riêng. Server có thể thay contract hoặc dữ liệu giữa chúng. Mục đích là giảm drift dễ thấy, không đạt atomic binding giữa descriptor và execution.

Envelope contract_version, outputSchema/local validation và resource/source checks tiếp tục chặn nhiều mismatch. Với remote service thật, có thể cần API version/ETag/capability handle được backend kiểm và deployment rollout có kiểm soát. Chọn cơ chế nào phụ thuộc loại effect và server; client tự tính hash rồi gửi không tự buộc server tôn trọng contract ấy.

Pagination cũng là một phần snapshot. Duplicate tên ở page sau, cursor lặp hoặc page budget cạn không được âm thầm lấy page đầu rồi cho model dùng catalog thiếu. Lab kiểm nhánh pagination bằng typed fake client; server thật hiện có một page. Không suy từ những tests đó rằng pagination/caching trên mọi remote server đã tương thích.

## 5. Async client và sync agent

Agent48 có run/dispatch đồng bộ. MCP SDK là async. Link dùng một BlockingPortal chạy event loop riêng, giữ async Client context qua toàn bộ run và gọi coroutine từ thread host. Nhờ đó không asyncio.run từng tool call với một client bị đóng/mở liên tục.

Bridge này có overhead và lifecycle cần quản lý. Không gọi portal wrapper sai thread hoặc dùng client sau close. Không coi việc mỗi run có một portal là mô hình tối ưu cho hàng nghìn concurrent agents; khi xây service thật nên thiết kế async orchestration/ownership/connection pooling và admission rõ hơn.

Client(server) thực hiện SDK interaction trong process, không socket/subprocess. Stdio khởi chạy server process thật, SDK quản lý transport và request metadata/correlation. Report Link.ops đếm operation do host gọi, không đếm negotiation/internal frames. Không có TCP/TLS/proxy/auth HTTP test trong bài 49.

## 6. Error semantics và unknown outcome

Một tool result báo isError khác business envelope unavailable, và cả hai khác protocol exception. Người gọi có thể cần hành vi khác: sửa input theo rubric, thử phương án khác, handoff hoặc dừng cấu hình. Một retry policy “gặp exception nào cũng retry” có thể làm tăng lỗi/tải và che input contract sai.

Lab chuyển isError thành observation lỗi đã sanitize; AdaptiveScript handoff. Nó không cho model tự đọc raw private exception stack để sửa server hoặc cấp thêm quyền. Protocol error mapping có test riêng với MCPError có kiểm soát; classification của một server cụ thể cần được quan sát, không suy mọi tool lỗi đều là JSON-RPC error.

Timeout là local deadline chờ. Trong slow_health fixture, coroutine bị hủy trước World.read, và audit cho thấy admitted reads ít hơn calls sent. Đó là kết quả của kịch bản async cụ thể; một hàm sync hoặc remote API có thể tiếp tục sau khi client timeout. Bài không có automatic MCP retry hoặc exactly-once claim. Tool ghi tương lai cần receipt/status/idempotency riêng.

## 7. Structured data, provenance và nội dung model thấy

Host lấy structuredContent làm canonical, kiểm kiểu/identity, rồi tạo observation. Text content có thể là serialized copy hoặc nội dung khác; lab không nối nó vào context. Resource links hoặc image/audio items cũng không tự tạo một fetch/action mới. Đây là subset integration hẹp, không bộ renderer MCP đa phương thức tổng quát.

Mcp_provenance được host gắn sau validation: server_alias, remote_name, catalog_hash và sequence của tools/call đã gửi. Model call_id vẫn từ protocol provider; MCP JSON-RPC ID do SDK dùng; observation ID do runtime48 cấp. Không dùng một ID để thay tất cả các loại correlation hoặc authorization.

Provenance không là truth label. Server có thể trả đúng schema nhưng sai facts; local oracle của giáo trình chỉ kiểm với corpus/ticket/snapshot đã biết. Với dữ liệu thật, cần nguồn authority, freshness và quy trình content evaluation. Citation hash bảo vệ identity của bytes, không tự biết runbook có đúng vận hành không.

## 8. Identity binding của server trong lab

Mỗi server nhận actor/goal/task từ trusted constructor hoặc CLI deployment. Model không cung cấp bearer token, tenant hoặc service override. Host giữ các kiểm scope bài 48 và server chạy World scope lại, nên bypass host bằng client trực tiếp vẫn không đọc được ticket 2 trong deployment A.

Đây không là login/session auth production. Server local không có OAuth access token validation, per-request external user scopes hay tenant routing ngoài fixture. Cấu hình một actor trên server process chỉ phù hợp môi trường học và scope synthetic đã chọn; không dùng chung process đó như mọi khách hàng có cùng quyền.

Host/remote World cùng chia SQLite fixture, để host có authority oracle kiểm source/ticket identity. Nếu triển khai SaaS thật, host thường không có bản sao local luôn đồng bộ. Cần thiết kế lại resolver/reference/freshness semantics; không copy phép so dict với DB local rồi tuyên bố đã giải quyết remote identity.

## 9. Parity, metrics và chi phí

Direct và MCP branches cùng model script, task và final gate. Hai goal giữ cùng answer/citation/path; qua MCP không làm model có thêm năng lực. Metric này kiểm integration không phá behavior, chưa kiểm retrieval/task quality trên population. Các invalid result cases còn cho thấy transport có thể thành công mà app vẫn đúng khi từ chối output.

Model calls43, MCP calls33 và backend admitted reads31 của suite là ba mẫu số khác nhau. Link ops thêm discovery-related host work như list/refresh và controller audit, chưa bao toàn SDK negotiation. Đừng nhân một giá/token/latency giả định lên mọi counter rồi gọi là bill thực. Muốn đo performance cần các request/time populations và tool/backend costs thật.

Các fault cases được cố ý lấy nhiều để kiểm boundary, không là phân bố lỗi tự nhiên. 14/14 đúng kỳ vọng có nhiều expected fail/block, không model accuracy. Khi có model live, cần dataset/annotation/trajectory eval tương ứng, giữ failures và so baseline theo cùng snapshot/scope/budget.

## 10. Mở rộng sau bài này

Để thêm tool mới, bắt đầu từ task cần quyết định gì, dữ liệu/quyền nào cố định và lỗi nào model có thể phục hồi. Thiết kế schema/envelope/identity trước, kiểm server độc lập, rồi thêm mapping/model descriptor và cả positive/negative cases. Nếu endpoint/descriptor ngoài môi trường trusted local, thêm trust/auth/review theo scope thực tế.

Một live collector có thể dùng runtime49 nhưng thay model fixture bằng adapter thật được kiểm theo provider/model; cần thu actual decisions, usage/errors và chấm lại. Bài49 không có CLI live ấy; mode live của48 chưa đi qua MCP49. Không dùng author labels hoặc fixture parity làm bằng chứng cho live agent.

Bài50 dự kiến tập trung planning/verification/replanning; durable resume, memory, write review và production deployment thuộc các chặng sau. Không cần thêm tool ghi, framework hoặc multi-agent để hoàn thành contract/MCP integration hiện tại.
