# Đào sâu bài 46 — Hợp đồng giữa LLM và hệ thống hỗ trợ

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Model là bộ sinh ứng viên, không phải database transaction

Một LLM chọn output theo phân phối có điều kiện trên input. Nó có thể sinh text hợp lý nhưng sai, đổi cách diễn đạt giữa hai call, hoặc không trả output hoàn chỉnh. Backend transaction lại cần kiểm điều kiện và cập nhật state rõ. Không nên đặt cùng kỳ vọng cho hai thành phần này.

Interface hữu ích là `generate(messages) → structured text + attempt metadata`, rồi `parse/ground → accepted candidate hoặc typed error`. Sau đó workflow mới tạo proposal. Nếu đầu ra bị chặn, không cố biến mọi response thành thành công chỉ để UI có câu trả lời. Giữ abstain, refusal, integration failure và denied action riêng.

Bài có hai policy để học trade-off, không hai model. Một structured reply có JSON đúng và citation thật vẫn có thể sai ý. Exact-quote gate hẹp hơn semantic support: có false rejection của paraphrase; citation-only gate rộng hơn nhưng có false acceptance. Cả hai không xác thực source là chân lý.

## 2. Provider port, HTTP adapter và app workflow

HTTP adapter biết endpoint/request/envelope/retry/usage. Nó không biết tenant, reviewer hay ticket version. SupportEngine biết identity/evidence/policy/proposal; gateway 45 biết transaction/receipt. Tách những trách nhiệm này giúp test lỗi API mà không cần ghi ticket, và thay model mà không viết lại quyền DB.

Fixture dùng cùng HTTPX request builder và parser, chỉ thay transport. Scripted response không chứng minh remote provider chấp nhận request/schema, model có quyền dùng hoặc generated text có quality. Ngược lại, một live smoke 200 không kiểm hết transient faults, corrupt quote, cached-result revoke và retry receipt. Cần cả contract tests và evaluation phù hợp.

REST envelope có `output` gồm typed items; không được giả phần tử đầu tiên luôn là một answer text. Adapter bỏ qua reasoning item, chặn function/tool item vì workflow này không cấu hình tools, và chỉ chấp nhận một output_text JSON hoàn chỉnh. Nếu muốn hỗ trợ nhiều message/block hoặc streaming, cần protocol/validator mới thay vì nối strings tùy ý.

## 3. JSON Schema của provider và validator local không trùng hoàn toàn

Wire schema cố ý dùng subset cơ bản: root object, properties/required, enum, array, additionalProperties:false và nullable draft. Schema có thể được một số model/API hỗ trợ, nhưng phải xác minh với model cụ thể khi live. Không hứa mọi JSON Schema keyword dùng được cho mọi structured-output backend.

Local Pydantic thêm length/count/strict type. Parser chặn duplicate object keys: JSON parser mặc định có thể lấy giá trị sau, gây khác biệt giữa hệ kiểm và hệ thực thi. NaN/Infinity không là dữ liệu business hợp lệ ở đây. Field optional về nghiệp vụ được gửi `null` rõ thay vì bỏ key; extra field không được coi là quyền mới.

[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses) đối chiếu required/nullable và refusal/incomplete. Những ràng buộc nghiệp vụ như ask không draft hoặc expected ticket version không lấy từ model là policy tự thiết kế, không tự được provider schema bảo đảm.

## 4. Quote, offset và evidence capability

Model thấy E1/E2 trong context cục bộ, không thấy một namespace nguồn toàn công ty để tự chọn ID. Claim chỉ được nối vào evidence của request ấy. Backend dùng unique substring để tính offset từ parent start; kiểm source ID/version/hash/quyền hiện tại lần nữa. Nếu quote xuất hiện hai lần trong parent, helper từ chối thay vì đoán vị trí.

Một claim có text khác quote được chấp nhận trong citation_only không tự có entailment. Ví dụ quote có điều kiện “chỉ retry khi đọc hoặc có idempotency”, text lại “retry mọi request ghi vô hạn”. Citation mechanics đúng nhưng nội dung trái điều kiện. Source membership là phép kiểm cần thiết, chưa đủ cho grounded synthesis.

Ngay cả exact_quote có thể chọn câu không trả lời câu hỏi, thiếu cảnh báo ở câu khác hoặc quote source được approved nhưng sai. Ingest provenance, policy hiệu lực, context completeness và semantic evaluation vẫn cần. Đừng dùng một string check đơn giản làm nhãn “hallucination-free”.

## 5. Prompt boundaries và prompt injection

System instruction và user JSON là hai message do server tạo. User không được chèn thêm role ở JSON transport bằng một chuỗi có dấu ngoặc/quote; serialization giữ nó trong field. Nhưng model vẫn đọc các từ trong field, nên instruction trong query/evidence có thể ảnh hưởng hành vi. Delimiter/role tách bạch không là chứng minh chống injection.

Biện pháp quan trọng của lab là authority không do model nắm: no DB handle, no direct execution, không model-supplied tenant/role/operation/target. Unexpected function_call không được dispatch; generated draft chỉ là data cần review. Những kiểm này vẫn có ý nghĩa ngay cả khi model bị lừa tạo output sai.

Dữ liệu gửi provider gồm query và evidence hiện được phép. Thu hồi quyền sau khi call đã gửi không thu lại được bytes đã tới provider; hệ chỉ có thể chặn output/effect tiếp theo và áp chính sách dữ liệu thích hợp. Fixture callback kiểm revoke trong lúc generation không cho answer/draft ra client, nhưng chưa mô phỏng remote deletion hoặc confidentiality của mọi mô hình.

## 6. Deadline và retry không chỉ là một con số timeout

HTTPX timeout áp vào hoạt động I/O; adapter thêm elapsed clock trước attempt, trong đọc body và trước retry delay. Nó không preempt mọi blocking operation đúng mốc 10 giây. Muốn hard request deadline cần kiến trúc cancellation/worker ownership và đo end-to-end, không chỉ đổi constant.

Retries chỉ cho loại lỗi đã định, max 2 mặc định. Attempt 503/timeout có thể đã tạo usage remote; repeat generation còn có thể trả draft khác. App không được xem retry như hoàn toàn miễn phí hoặc deterministic. Vì chưa execute ticket trong generation, những retry này không tự ghi trùng ticket; chúng vẫn có thể làm tăng latency/bill provider.

Không tự repair JSON, tăng output limit hoặc bỏ validation vô hạn. Một repair strategy là một thuật toán khác: cần maxcalls, failure categories, budget, data redaction và eval về việc repair có làm sai semantics/authority không. Case malformed-model-JSON trong lab tiêu tốn một fixture usage rồi fail, không gọi lại đến khi pass.

## 7. Ledger có unknown khác ledger có zero

Known input/output counts là tổng usage hợp lệ đã quan sát trong response envelope. Nếu không nhận body, nhận body quá lớn/invalid, hoặc provider bỏ usage, một attempt có usage unknown. Đặt zero ở đó khiến chi phí báo cáo thấp giả.

Trong mỗi policy suite:12 attempts,11 response có 120 input/40 output fixture tokens,1 HTTP 503 unknown. Tổng known 1320/440 đúng theo fixture, chưa là total bill. Có cả refusal/incomplete/schema reject vẫn có known usage. Provider có thể tính giá/discount/reasoning/cache khác; bài không xác minh pricing hoặc usage accuracy live.

Result replay đọc lại metadata cũ. Nếu client nhận response hai lần rồi cộng cả hai ledger, nó double-count generation. Để tính usage, aggregate theo unique generation/run/attempt ID phù hợp, không theo số HTTP response được UI render. Provider response ID/returned model giúp truy vết, không tự là idempotency receipt của ticket.

## 8. Draft content và action authority

Model chỉ viết title/body. Backend lấy operation/ticket_id/expected_version từ request đã kiểm và giữ trong proposal. Không cho model tự sửa target sau khi xem tài liệu. Ask với draft bị chặn; write answer thiếu draft không tạo proposal. Draft title/body vẫn có thể sai factual incident, nên không auto-approve.

Reviewer thấy đúng payload sau validation, hash và release, rồi gateway 45 kiểm lại current actor/source/version/expiry trước effect. Nếu review UI chỉ cho xem một lời giải thích đẹp nhưng execute args khác, control đã thất bại. Hash binding không chứng minh người review hiểu nội dung; cần UX/rubric và đo effort/lỗi review nếu dùng thật.

Function calling của provider là protocol khác. Theo [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling), application nhận call, thực thi và trả output có call_id trước vòng model tiếp. Bài này chưa làm vòng đó; từ “draft ticket” không nên được quảng cáo thành autonomous agent hoặc đã có tool planning.

## 9. Dùng cùng database với bài 45

Bridge nạp các module 45 theo đường dẫn sibling cố định, kiểm collision và yêu cầu process sạch. Đây là cơ chế tái sử dụng cho giáo trình, không khuyến nghị tổ chức package production bằng sys.path. Khi làm dự án thật, đưa backend core vào package có version/import path rõ, giữ boundary tương tự.

Run 46 hash gắn request+release, nên key cũ 45 không được replay như generation mới. Existing ticket/receipt vẫn trong DB; test thật tạo ticket bằng Engine 45 rồi note bằng SupportEngine 46. Committed receipt cũ trả sự thật cũ sau current identity/owner check; approval chưa execute của release cũ không tự hợp lệ cho release mới.

Pending run crash trước finish vẫn chưa có recovery worker như bài 45. Generation remote có thể đã chạy nhưng output chưa lưu; retry không được giả chắc usage 0 hoặc effect 0. Ở flow hiện tại effect nằm sau một proposal/review riêng, nhưng nếu thêm tool-call loop thì cần journal/status cho từng tool và generation checkpoint.

## 10. Mở rộng evaluation trước khi chọn model

Contract cases lập trình sẵn trả lời câu hỏi “pipeline làm gì khi provider trả loại response này?”. Nó không trả lời “model trả loại response này bao nhiêu lần trên người dùng thật?”. Hai mẫu số khác nhau. Annotation supported/unsupported của ví dụ do tác giả gán; chưa có independent human labeling study.

Khi live, chọn dataset query thực đã được phép dùng, group theo incident/source/version, giữ dev/test riêng. Đo source relevance, answer completeness/support/abstention, invalid/refusal/incomplete rates, draft correctness, action proposal scope, reviewer effort và backend effect outcomes. Repeats giúp thấy variability, nhưng không được sample nhiều rồi chỉ giữ câu tốt nhất mà bỏ cost/failure của các mẫu khác.

LLM judge có thể hữu ích để mở rộng đánh giá, nhưng cần rubric, disagreement/error audit và một tập human-checked. Không dùng chính model tạo answer làm thẩm quyền duy nhất xác nhận answer đó. Một live smoke vài câu tốt chưa đủ thay contract suite hoặc human relevance evaluation.

## 11. Release identity và reproducibility

Manifest 46 thêm requested model/adapter config, prompt/schema/validator identity, support policy, implementation source hash và base 45 source hash. Thay các thành phần đó cần release mới; current integrity guard chặn mutation config trong process. Response/request hash giữ bằng chứng về input/output đã dùng, không khiến model deterministic.

Requested model alias và returned model snapshot có thể khác. Không ghim một tên “latest” rồi cam kết hành vi không đổi. Bài không chọn model cụ thể; người chạy kiểm quyền/schema compatibility, snapshot và trade-off theo domain. Một model đổi nội bộ vẫn có thể thay output dù string identity giống; cần evaluation khi release/provider behavior thay đổi.

Hashes không là signature hoặc nơi lưu source/model. Fixture case snapshot không thay evaluation provenance toàn bộ repo/environment; giữ source/package/report cùng các phiên bản đã chạy. Runtime DB chứa text/citation/draft, nên retention và quyền truy cập artifact cũng quan trọng như log hygiene.

## 12. Một kế hoạch live smoke có giới hạn

Trước hết dùng corpus synthetic hiện có, một query answerable và một query không có evidence. Chọn model có quyền và strict schema theo tài liệu hiện hành; cấu hình key bằng môi trường/secret manager, không truyền key vào prompt hoặc command argument. Mode live là thao tác chủ động có network/billing, không tự chạy khi mở tài liệu hay chạy all/check.

Kiểm schema acceptance, completed/refusal/incomplete, source/quote/draft và actual usage/returned model. Nếu 400 do model/schema không hỗ trợ, sửa cấu hình theo docs; không retry hàng loạt. Đo riêng call latency và full workflow nếu cần. Không gửi dữ liệu nội bộ thật hoặc execute ticket SaaS chỉ để chứng minh adapter chạy.

Bài chưa làm live smoke này; mô tả adapter/request shape đã đọc nguồn và được kiểm bằng transport fake. Sau smoke mới xây model evaluation/staging pilot, vẫn giữ scope/review/idempotency của backend. Giới hạn được công bố rõ giúp người học biết cần kiểm bước nào tiếp, thay vì hiểu nhầm mọi thứ đã production-ready.
