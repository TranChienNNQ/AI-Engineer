# Thiết kế đào sâu bài 45 — Từ prototype tới một dự án có thể review

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Bản đặc tả ngắn có thể giao cho một backend engineer khác

**Problem statement:** nhân viên hỗ trợ cần tìm runbook đúng và chuẩn bị thay đổi ticket; hệ phải giữ quyền theo tenant và có bằng chứng cho answer/effect. **Success hypothesis:** giảm effort tìm/nhập liệu mà không tăng hành động sai. Lab chỉ kiểm behavior, chưa kiểm success hypothesis bằng người dùng.

**User stories:** reader hỏi câu có nguồn hoặc nhận abstain; agent chọn intent và xem proposal; reviewer đọc payload cụ thể rồi approve; agent execute/retry cùng key; operator đối chiếu run/proposal/receipt/audit khi lỗi. Một sự cố thật có thể cần nhiều người và nhiều hệ thống hơn, ngoài phạm vi local workflow này.

**Invariants:** source cấm không vào evidence; cùng run key khác input không tạo nhánh mới; proposal pending không tạo ticket; approval không được sửa target/args; local effect/receipt/audit commit cùng transaction; replay receipt không ghi lại; version stale không lost update. Đây là acceptance criteria của thiết kế, không phải một danh sách hứa hẹn model sẽ làm đúng.

**Nonfunctional scope:** request message tối đa 1.000 ký tự; context 600 ký tự/tối đa hai paragraph; max_steps từ 1 đến 10; synchronous bounded local work. Chưa có rate limiter, queue, timeout/cancellation provider, HA, encryption at rest hay percentile SLO. Cần nêu chúng như phần thiếu, không giấu sau chữ end-to-end.

## 2. Những lựa chọn kiến trúc và trade-off

| Quyết định | Lợi ích trong bài học | Đổi lại / khi cần đổi |
|---|---|---|
| Workflow xác định | Theo dấu được từng bước và side effect | Không tự lập kế hoạch từ câu hỏi mở |
| BM 25 + lexical overlap | Không tải model, dễ kiểm source | Paraphrase/synonym/đa nghĩa có thể thất bại |
| Extractive answer | Quote/offset kiểm được cơ học | Chưa tổng hợp hay giải đáp nuance |
| SQLite local tickets | Effect và receipt nằm chung transaction | Không mô phỏng atomicity với SaaS bên ngoài |
| Reviewer riêng cho writes | Minh họa payload binding rõ | Thêm effort/latency, chưa đánh giá reviewer UX |
| RLock trên Engine/Store | Tránh dùng chung connection đồng thời trong process | Serialize gây head-of-line; không đo throughput |
| Manifest trong source/runtime | Có identity behavior để pin/check | Chưa full registry/code/env/model artifact lineage |
| Fixture tokens/roles | Chạy offline, thử tenant/role dễ | Không phù hợp authentication/authorization production |

ADR có ích khi ghi rõ trigger đổi quyết định. Ví dụ chỉ thêm embedding khi development cho thấy lexical baseline bỏ synonym quan trọng; chỉ thêm planner nếu explicit workflow không bao phủ task cần thiết. Một framework được chọn trước chưa cho biết nó giải lỗi nào.

## 3. Data lineage: source, context, answer và action payload

Từ source text có ID/version, retriever lấy paragraph đủ quyền. Paragraph được pack nguyên đoạn nếu vừa budget. Evidence giữ offsets/hash; answer lấy đúng quote đầu. Validator không chấp nhận một citation ngoài evidence dù source đó có trong corpus. Citation còn phải khớp nguồn hiện hành/role/tenant/service.

Tại proposal, citation trở thành bằng chứng đã dùng và được ràng trong payload hash. Nhưng body/title ticket lấy từ message người gọi; chúng **không được model xác nhận là sự thật**. Trong hệ có LLM, cần phân biệt user allegation, observed telemetry và model inference trong ticket draft. Không viết mọi câu tóm tắt thành fact đã kiểm chứng.

Source cũ v1 có thể được quote chính xác nhưng không được dùng cho run hiện tại. Source hiện hành vẫn có thể viết sai. Một metadata flag `approved` chỉ cho biết quy trình ingest đã cho phép; không thay thế đánh giá nội dung. Fixture poison chưa approved bị loại, còn test eligible malicious text cho thấy baseline không thông dịch nó thành operation. Chưa có model để kết luận injection robustness.

[OWASP RAG Security](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html) là nguồn đối chiếu cho ACL trước model và output/action boundaries; các policy/service cụ thể ở đây do bài tự thiết kế. Không dựa vào một classifier “prompt injection” để cấp quyền ghi.

## 4. API contract không nên mang quyền từ client

Client có thể chọn service để định hướng tìm kiếm, nhưng service không cấp quyền. Principal chứa tenant/role từ fixture server; request tự thêm tenant bị extra-forbid. Một API input validated vẫn có thể nhắm ticket của người khác, nên authorization nằm ở read/propose/execute gateway. Case không nhìn thấy trả 404 để không lộ record tồn tại theo target ID.

`/v1/runs` trả initial result; `/v1/runs/{id}` chỉ owner cùng tenant đọc được. Reviewer có endpoint proposal cùng tenant để xem payload, không tự đọc mọi run. `/v1/tickets/{id}` trong bài là tenant-level access, chưa có ACL chi tiết theo người được assign hoặc sensitivity của từng ticket. Nếu domain yêu cầu per-ticket scope, phải thêm nó vào mọi read/write/replay path.

POST run trả JSONResponse sau explicit `RunOut.model_validate`. Response model/JSON schema chỉ là shape contract. Không có filter PII hoặc semantic validator tổng quát cho text. UI tương lai cần render text an toàn; JSON quote chứa instruction không được render thành executable HTML hoặc client action.

Fixture auth state/revocation nằm trong RAM. Restart sẽ nạp lại fixture mặc định; database bền vững không làm authorization state cũng bền vững. Khi dùng thật, identity/policy service phải là nguồn quyền hiện tại, và cached result phải tái kiểm nó. Không ghi token thật trong query, ticket hoặc audit.

## 5. State machine và binding của review

Một proposal pending có thể được approved bởi reviewer cùng tenant khác owner. Approval lưu payload hash, reviewer và expiry. Execute dùng đúng proposal ID và owner; không nhận args mới từ body execute, vì nếu cho sửa args sau approval thì review đã duyệt việc khác. Request execute chỉ có request_key.

Approval expired không được gia hạn bằng approve lại; proposal state không còn pending. Hệ yêu cầu một proposal mới và review mới. Với target ticket, expected_version là snapshot người đề xuất đang dựa vào. Nếu hai note cùng expected 1, note đầu tạo version 2; note sau bị stale thay vì ghép lên trạng thái mới mà reviewer chưa thấy.

`resolve` chuyển open→resolved và version+1; không tự giải quyết production incident. Nếu ticket đã resolved, một request note/resolve mới bị từ chối theo policy nhỏ này. Một sản phẩm muốn reopen hoặc cho note sau close cần thêm transition, permission và eval, không chỉ bỏ guard.

Review là một control, không chứng minh reviewer luôn đọc kỹ. Với LLM, phải ngăn review UI bị nội dung model lừa che target/args. [OWASP Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/) nhấn mạnh scope quyền và kiểm downstream; lab không đo human approval error rate.

## 6. Hai lớp idempotency và một hiệu ứng đã commit

Run key scope `(tenant,actor,key)` gắn với hash request. Execute key scope tương tự gắn với proposal ID. Unique proposal receipt chặn execute cùng proposal bằng key khác. Receipt được đọc trước các kiểm còn cần để tạo effect mới như expiry/release, vì effect đã tồn tại cần trả lại sự thật cũ; identity/owner vẫn được kiểm trước đọc receipt.

Một effect nằm trong transaction:

`validate approval/current permission → validate target/version → mutate → insert receipt → mark executed → audit effect → commit`.

Ở SQLite, `BEGIN IMMEDIATE` lấy quyền writer sớm; writer khác có thể chờ hoặc bị busy. Check+update version thực hiện trong transaction một writer nên không lost update trong thiết lập này; không nên copy logic đó sang remote API không có CAS rồi cho rằng tương đương. [SQLite transaction docs](https://www.sqlite.org/lang_transaction.html) hỗ trợ cơ chế transaction, không chứng minh thiết kế distributed nào ngoài SQLite.

Fault trước commit rollback cả ticket/receipt/proposal transition/audit. Fault sau commit báo 503 cho caller nhưng DB đã lưu receipt. Reopen DB rồi retry cùng key trả cùng result; không cần approval mới để **đọc receipt**, không có ticket thứ hai. Các phương thức này không cung cấp exactly-once delivery của HTTP request, chỉ bảo vệ local committed effect theo scope/constraint đã nêu.

Pending run bị crash trước finish vẫn có result NULL; retry báo run_in_progress. Chưa có lease/recovery của run-level computation. Cần thiết kế owner/lease/status reconciliation nếu muốn tự động phục hồi thay vì để operator xử lý. Không lẫn receipt recovery sau effect với resume mọi stage workflow.

## 7. Khi ticket backend nằm ở hệ khác

Giả sử local DB gọi remote `create_ticket`, remote commit rồi network timeout. Local DB không biết ticket ID và chưa ghi receipt; rollback local không xóa ticket remote. Tự retry với key mới có thể tạo trùng. Một transaction hai bảng local không giải được điều đó.

Các option cần đối chiếu contract downstream: remote nhận idempotency key và cung cấp status/receipt; hoặc local outbox gửi command có operation ID và remote inbox/dedup; hoặc reconcile bằng stable external reference nếu domain cho phép. Những cơ chế đó có retry/lifetime/retention riêng. Nếu remote không có cách tìm trạng thái, kết quả có thể phải giữ UNKNOWN và chuyển review, thay vì giả succeeded hoặc failed.

Trong bài, Store là downstream thực thi local, không phải HTTP adapter Jira/Zendesk. Đổi adapter là một thay đổi kiến trúc cần test lost response, duplicate, timeout, revoke và conflict qua failure boundary thật. Không đưa credential thật vào demo hiện tại hoặc gọi ticket thật chỉ để hoàn thành bài học.

## 8. Chất lượng: vì sao 8/8 chưa đủ để chọn một sản phẩm

Dev 6/test 8 queries có cùng corpus nhỏ. Phần lớn service/tenant/role hiện có tối đa một paragraph đủ quyền, nên suite chưa thử ranking với nhiều hard negatives. Overlap gate quyết định nhiều kết quả; nó không đo được lợi ích embedding/reranker hoặc tổng hợp nhiều nguồn. Tên BM 25 không làm một evaluation nhỏ trở thành retrieval benchmark khó.

Exact-source/abstain metric kiểm source ID, chưa đo answer relevance/completeness/safety/giải quyết ticket. Một answer trích nguyên paragraph chứa nhiều bước có thể đúng nguồn nhưng chưa trả lời trường hợp cụ thể. Cần annotation claims/evidence, answer rubric, judge calibration và human review nếu dùng LLM; keep separate unauthorized/unsupported/failed cases.

Test có 4 answerablevà 4 unanswerable theo quyền/context của actor. Conditional source accuracy 4/4 cần đi cùng answer rate 4/8 và answerable recall 4/4. Luôn abstain đạt 4/8 exact nhưng recall 0. Trong domain khác, class balance khác làm overall exact đổi dù capability như cũ.

Đề xuất split tiếp theo: giữ cùng incident/ticket paraphrases trong một split; tách future runbook version, service/domain mới và tenant có policy khác theo câu hỏi cần kiểm. Dùng development để tune, reserve test mới cho release decision. Báo error slices và uncertainty phù hợp, không suy tỷ lệ thực từ 8 fixtures.

## 9. Cost/latency budget và observability có nghĩa gì?

Prototype chỉ có logical stages. Một LLM request có thể cần system prompt, retrieved context, output tokens, nhiều tool/model round trips và retries. Cần đo từ request/client timestamp phù hợp, không chỉ thời gian handler. Context chars không đổi được sang provider tokens bằng một hệ số đáng tin cho mọi ngôn ngữ.

Cost ledger nên giữ model/version, actual input/output usage nếu provider trả, retries, embedding/index build/query, storage và human review effort. Với cache, báo hit population và quality/invalidation policy. Không dùng giá cũ hoặc giả định của ví dụ .005 làm ngân sách production. Bài không có provider bill hoặc measured token usage.

Audit stage/status và release giúp truy vết nhưng chưa có trace_id xuyên service/span durations/exporter. Khi mở rộng, dùng bài 32 để nối instrumentation thật, kiểm data minimization và quyền xem trace. Run output/proposal/ticket có text, nên masking log không giải quyết data retention của toàn hệ.

## 10. Release và rollback: điều gì được pin?

Release ID gắn corpus hash và config behavior tối thiểu. Docs thay mà hash không đổi thì integrity check dừng run mới. Một config overlap mới tạo release ID khác, làm approval cũ không đủ cho effect mới. Đây là pinning, chưa có một release registry/atomic traffic switch như bài 33.

Receipt cũ chứa release đã thực hiện; trả lại receipt sau code/config rollback vẫn báo hiệu ứng cũ. Ticket DB không rollback theo manifest. Migration DB, external schema, idempotency retention và source deletion cần kế hoạch riêng. Không được “fix” một receipt khó hiểu bằng xóa nó rồi retry effect.

Hash là content identity theo canonical serialization, không chữ ký. Truncation 16 hex chỉ phục vụ ID hiển thị trong lab; chưa thiết kế collision/adversarial registry guarantee. Source/env/model artifacts cần được lưu hoặc truy xuất lại; hash không tự tạo được chúng.

## 11. Kế hoạch triển khai theo cổng bằng chứng

| Giai đoạn | Phạm vi | Bằng chứng cần trước bước sau |
|---|---|---|
| Local baseline | Fixture corpus + ticket SQLite | Contract, tenant, citation, transaction và replay checks |
| Read-only shadow | Dữ liệu được cấp quyền, không ghi ticket | Retrieval/answer eval và logging/quyền thật |
| Staging writes | Ticket sandbox/downstream adapter | Idempotency/status reconciliation, review/version faults |
| Limited pilot | Người dùng/tenant có phạm vi đã thống nhất | Quality/cost/latency, reviewer effort, rollback và support runbook |
| Broader rollout | Tăng traffic/scope có điều kiện | SLO, feedback/label lifecycle, security review và measured value |

Shadow không gửi writes rồi “bỏ kết quả”; effect vẫn đã xảy ra. Pilot không được dùng quyền admin chung chỉ vì model đang được theo dõi. Mỗi bước giữ action scope phù hợp và có đường quay lại read-only/manual. Đây là kế hoạch, chưa phải deployment đã thực hiện.

Ví dụ incident runbook: retrieval down→trả failed rõ và fallback manual; source/release mismatch→dừng run mới, đối chiếu artifact; effect timeout→query receipt/status cùng operation ID; version conflict→đọc snapshot mới/proposal mới; auth/source revoke→chặn cả đọc lại evidence theo current policy. Không tự xóa DB/receipt hoặc bypass review để làm demo chạy tiếp.

## 12. Những phần cần hoàn thiện trước dữ liệu thật

Cần hỏi đúng domain về retention, identity/tenant/per-ticket ACL, consent/scope đọc dữ liệu, review UX và mức tự động hóa. Cần nối secret management, provider/downstream adapters, service deadline/admission, tracing/error budgets, dependency lifecycle, migration/backup/restore và continuous evaluation. Không phải mọi project đều cần microservices hoặc agent framework; chọn theo bottleneck đo được.

Bộ prototype đã kiểm phần được nêu bằng fixture, SQLite và ASGI; socket bind bị sandbox từ chối nên nhánh Uvicorn/TCP chưa được xác minh trong phiên này. Dùng lab để review decision/data flow, rồi mở rộng bằng failure mode thật. Một hồ sơ dự án tốt mô tả rõ thiết kế, bằng chứng, kết quả và khoảng trống, thay vì chỉ có screenshot chatbot.
