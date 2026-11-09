# Bài 22 — API, đầu ra có cấu trúc và đánh giá

[Mục lục](../README.md) · [Bài 21](../21-decoding-context-inference/README.md) · [Đào sâu và thiết kế hệ thống](deep-dive.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Đây là bài nối mô hình với ứng dụng backend. Ví dụ và dữ liệu tự tạo; lab không có tài khoản, API key, model tải ngoài hoặc cuộc gọi trả phí. API chạy cục bộ bằng FastAPI, có provider quy tắc để demo và provider giả lập lỗi để kiểm hệ thống. **Số liệu đánh giá trong bài là kết quả contract/fault fixtures, không phải chất lượng của một LLM thật.**

## 1. Mục tiêu và cầu nối

Bài 21 đã có logits → chọn token → EOS hoặc hết budget. Nhưng downstream không thể chỉ tin rằng một chuỗi sinh ra “có vẻ đúng”. API phải biết request hợp lệ chưa, upstream đã kết thúc chưa, output có parse/validate được không và hành vi có đáp ứng tác vụ hay không.

*(Giải thích tường minh cho kỹ sư backend:
- **Upstream (Phía nguồn / Nhà cung cấp mô hình):** Là nơi mô hình LLM hoạt động (ví dụ: OpenAI API, Anthropic, vLLM hoặc tiến trình chạy mô hình nội bộ). Upstream nhận prompt và trả về một chuỗi văn bản tự do (raw text).
- **Downstream (Phía hạ lưu / Ứng dụng nghiệp vụ):** Là các hệ thống phần mềm tiêu thụ kết quả của mô hình (ví dụ: Database PostgreSQL, hệ thống thanh toán Stripe, hàng đợi Kafka, hoặc giao diện Frontend). Downstream KHÔNG THỂ nhận một đoạn văn bản tự do nói "Tôi nghĩ là đội Network" mà bắt buộc phải nhận dữ liệu có cấu trúc chuẩn mực (như JSON hợp lệ với đúng kiểu dữ liệu).
- **Lớp trung gian (Application Orchestration Layer):** Là tầng API mà kỹ sư AI phải xây dựng: làm cầu nối trung gian để kiểm duyệt dữ liệu vào (Input Validation), gọi Upstream có kiểm soát (Retry/Timeout), phân tích cú pháp và kiểm duyệt dữ liệu ra (Output Validation & Schema Verification) trước khi chuyển cho Downstream).*

Sau bài này, bạn cần:

- Phân biệt API của ứng dụng với API model/provider; thiết kế request, response và mã lỗi rõ ràng. *(Hiểu ranh giới kiến trúc: client của bạn chỉ tương tác với API ứng dụng, không bao giờ được gọi trực tiếp LLM Provider để tránh lộ API key và mất kiểm soát chi phí).*
- Tạo schema cho output có nhánh route/abstain; kiểm JSON, type, field và quan hệ giữa các field. *(Làm chủ kỹ thuật ép buộc cấu trúc dữ liệu đầu ra bằng JSON Schema và Pydantic Strict Mode).*
- Xử lý finish reason, timeout, retry có giới hạn và thông tin usage không đầy đủ. *(Xây dựng cơ chế phòng vệ hệ thống: xử lý dứt điểm các trường hợp mô hình bị ngắt cụt, sập mạng hoặc quá tải).*
- Xây bộ đánh giá có denominator đúng, slice, confusion matrix và giới hạn được công bố. *(Nắm vững phương pháp đo lường chất lượng hệ thống AI thực tế, tránh các cạm bẫy toán học "thổi phồng" độ chính xác).*
- Phân biệt test hệ thống, eval chất lượng model và đánh giá tác vụ đầu cuối. *(Tách bạch giữa việc kiểm tra lỗi phần mềm (unit/integration test) với việc đánh giá độ thông minh ngữ nghĩa của mô hình (eval)).*

Cần bài 5 về leakage, bài 7 về metric, bài 19 về tokenizer, bài 20–21 về hành vi và inference. Bạn có nền backend nên có thể đọc nhanh phần HTTP quen thuộc; nên dành thời gian cho schema không đồng nghĩa đúng nội dung và cách đánh giá khi có abstention/error.

## 2. Ví dụ thực tế: từ ticket đến đề xuất đội xử lý

Một hệ thống nhận ticket:

```text
L1: API /orders gặp timeout khi đọc đơn hàng.
L2: Auth vẫn hoạt động bình thường.
```

Ứng dụng cần một đề xuất có cấu trúc:

```json
{
  "schema_version": "1",
  "decision": "route",
  "team": "NET",
  "evidence_ids": ["L1"]
}
```

*(Định nghĩa cấu trúc Đề xuất (Structured Proposal):
- `schema_version`: Đánh dấu phiên bản cấu trúc (Version pinning) để đảm bảo tính tương thích ngược khi nâng cấp hệ thống.
- `decision`: Quyết định hành động (`route` - định tuyến chuyển tiếp, hoặc `abstain` - từ chối kết luận).
- `team`: Tên định danh của đội phụ trách xử lý sự cố (`AUTH`, `DATA`, `NET`).
- `evidence_ids`: Danh sách các ID dòng log đóng vai trò làm bằng chứng (Grounding Evidence). Mô hình không được phán bừa mà bắt buộc phải chỉ ra căn cứ).*

Nhãn AUTH/DATA/NET là quy ước minh họa của khóa học, không phải chính sách vận hành phổ quát. Nếu ticket thiếu dữ kiện, contract cho phép:

```json
{
  "schema_version": "1",
  "decision": "abstain",
  "team": null,
  "evidence_ids": []
}
```

`abstain` là không đưa ra đề xuất đội với dữ kiện hiện có; không phải timeout hoặc HTTP error. *(Định nghĩa Abstention (Quyết định từ chối kết luận): Trong kỹ thuật AI, một trong những năng lực giá trị nhất không phải là luôn luôn cố đoán mò, mà là **biết khi nào mình không biết**. Khi dữ liệu đầu vào mơ hồ hoặc thiếu căn cứ, việc mô hình chủ động trả về `abstain` là một hành vi nghiệp vụ thành công và an toàn, giúp ngăn ngừa các quyết định sai lầm).* API của bài chỉ tạo **proposal**. Nó không đóng ticket, phân công nhân viên hoặc thực hiện công cụ. Việc thay đổi trạng thái hệ thống là một bước nghiệp vụ khác, cần quyền và điều kiện riêng. *(Nguyên tắc thiết kế an toàn: Tách rời bước Đề xuất (Proposal/Draft) khỏi bước Thực thi có tác dụng phụ (Execution/Action). Mô hình AI chỉ đóng vai trò tư vấn, quyết định cuối cùng cần có con người phê duyệt hoặc hệ thống kiểm soát quyền hạn (RBAC)).*

Tách rõ hai API:

| Lớp | Client biết gì? | Trách nhiệm |
|---|---|---|
| API ứng dụng `/v1/route-proposals` | Ticket, các dòng được cung cấp, contract nghiệp vụ | Input validation, orchestration, output validation, lỗi, observability |
| Adapter model/provider | Prompt/messages, timeout, cấu hình model, response envelope | Gọi upstream, chuẩn hóa finish/usage/lỗi về giao diện nội bộ |

Không để client ứng dụng tùy ý chọn URL upstream, system prompt hoặc model đặc quyền trong request công khai. Cấu hình đó thuộc phía server theo sản phẩm. Lab không kết nối nhà cung cấp nào; các field provider trong code là giao diện minh họa, không phải wire format của một vendor.

![Từ API ứng dụng tới adapter và các lớp kiểm](figures/01-architecture.png)

## 3. Hợp đồng request và response

Request của lab:

```json
{
  "schema_version": "1",
  "ticket_id": "ticket-demo-1",
  "lines": [
    {"id": "L1", "text": "event=timeout."}
  ]
}
```

`lines` có 1–8 dòng; mỗi ID dạng L 1, L 2, … và không trùng; mỗi text 1–500 code point theo giới hạn string. Server không nhận label đúng từ client để “giúp model trả lời”. Eval giữ reference ở bên scorer, không nhét và o prompt của ca đang đo.

Success envelope tách kết quả nghiệp vụ khỏi metadata. Các giá trị request_id/elapsed dưới đây là **minh họa**, không phải snapshot cố định:

```json
{
  "request_id": "server-generated-id",
  "provider": "local-keyword-rule-v1",
  "prompt_version": "route-v1",
  "result": {
    "schema_version": "1",
    "decision": "route",
    "team": "NET",
    "evidence_ids": ["L1"]
  },
  "metrics": {
    "attempts": 1,
    "elapsed_ms": 1.0,
    "known_input_tokens": 100,
    "known_output_tokens": 25,
    "usage_complete": true
  }
}
```

*(Giải thích mẫu thiết kế Phong bì siêu dữ liệu (Metadata Envelope Pattern):
- **Phần định danh & truy vết:** `request_id` (mã UUID phục vụ Distributed Tracing trên Datadog/Jaeger), `provider` (cho biết mô hình nào đã phục vụ), `prompt_version` (biết chính xác phiên bản prompt đã sinh ra kết quả để đối soát A/B Testing).
- **Phần kết quả nghiệp vụ (`result`):** Chứa dữ liệu đề xuất đã được kiểm duyệt nghiêm ngặt qua schema.
- **Phần chỉ số vận hành (`metrics`):** Số lần thử lại (`attempts`), thời gian chạy (`elapsed_ms`), số lượng token tiêu thụ (`known_input_tokens`, `known_output_tokens`) và cờ `usage_complete` cho biết dữ liệu đo lường có bị thiếu hụt do lỗi mạng hay không).*

Hai token counter 100/25 trong adapter lab là **giá trị giả lập**, không đo bằng tokenizer, không là hóa đơn. `elapsed_ms` ở API dùng monotonic clock từ lúc và o orchestration tới sau validation, không bao gồm toàn bộ mạng/queue của client. Eval dùng FakeClock riêng nên ghi `simulated_elapsed_ms` và không biến thành benchmark latency.

FastAPI dùng type/response_model để tạo tài liệu và kiểm/lọc response theo model khai báo. Lab còn kiểm candidate upstream trước khi tạo success; không dựa và o việc lọc response để che field lỗi. [FastAPI response model](https://fastapi.tiangolo.com/tutorial/response-model/).

## 4. JSON đọc được chưa đủ để downstream tin dùng

Có ít nhất năm câu hỏi riêng:

1. **Hoàn thành:** upstream báo kết thúc bình thường, hay length/refusal/timeout?
2. **Syntax:** raw text có đúng một giá trị JSON theo policy parser không?
3. **Schema:** root, field bắt buộc, type, enum và quan hệ field có đúng không?
4. **Tham chiếu/nghiệp vụ:** evidence ID có thuộc input được cung cấp cho request này không?
5. **Đúng tác vụ:** nhãn có thực sự được dữ kiện hỗ trợ và có đúng reference/rubric không?

*(Mổ xẻ phễu 5 tầng kiểm duyệt dữ liệu trong thực tế:
- **Tầng 1 — Trạng thái hoàn thành (Upstream Completion):** Mô hình có thực sự nói xong với `finish_reason='stop'` hay bị ngắt ngang xương vì cạn token (`finish_reason='length'`)? Nếu bị ngắt cụt, chuỗi JSON chắc chắn bị thiếu dấu ngoặc đóng `}` và không bao giờ được đưa tiếp vào parser.
- **Tầng 2 — Cú pháp JSON (Syntax Validity):** Chuỗi ký tự có parse được bằng `json.loads()` không? Có bị dính rào chắn markdown (như ` ```json...``` `) hay chứa các giá trị phi chuẩn như `NaN`, `Infinity` không?
- **Tầng 3 — Cấu trúc dữ liệu (Schema Conformance):** Có đủ các trường bắt buộc không? Kiểu dữ liệu có đúng là string/array/null không? Giá trị enum có nằm trong danh sách cho phép không?
- **Tầng 4 — Ràng buộc tham chiếu & Logic (Referential & Relationship Constraints):**
  - Nếu `decision == 'route'` thì `team` bắt buộc phải khác null và `evidence_ids` phải có ít nhất 1 phần tử.
  - Nếu `decision == 'abstain'` thì `team` bắt buộc phải là null và `evidence_ids` phải rỗng.
  - Các ID bằng chứng (`evidence_ids`) có thực sự tồn tại trong danh sách dòng log của ticket không? Hay mô hình tự "bịa" ra một ID ma như `L99`? (Ảo giác trích dẫn - Citation Hallucination).
- **Tầng 5 — Tính đúng đắn của nghiệp vụ (Task Correctness / Factuality):** Dù vượt qua cả 4 tầng trên, câu trả lời vẫn có thể sai về mặt thực tế! Ví dụ: Log ghi rõ ràng "timeout" (cần đội NET), nhưng mô hình lại phân cho đội AUTH.
- **Bài học cốt lõi:** *Schema Valid $\neq$ Factually True* (Đúng cấu trúc không đồng nghĩa với đúng sự thật)).*

Ví dụ `{"team":"AUTH"}` parse được nhưng thiếu decision/version/evidence. Ví dụ đầy đủ schema, team AUTH và evidence L 1 vẫn có thể sai khi L 1 mô tả timeout cần NET theo định nghĩa tác vụ. Một evidence ID tồn tại chỉ chứng minh nó có trong input, chưa chứng minh suy luận từ dòng đó là đúng.

Câu `timeout đã được loại trừ; event=denied` còn minh họa lỗi ngữ nghĩa: tìm thấy từ timeout không có nghĩa nó là lỗi hiện tại. Rule demo của API chỉ dựa từ khóa, nên không được coi là bộ hiểu ticket tiếng Việt hoàn chỉnh.

![Các tầng kiểm và giới hạn từng tầng](figures/02-validation.png)

## 5. Schema: required, null, enum và extra field

Xem [schema đầy đủ](output-schema.json), được sinh từ [contracts.py](contracts.py) theo JSON Schema Draft 2020-12. Phần lõi:

```json
{
  "type": "object",
  "required": ["schema_version", "decision", "team", "evidence_ids"],
  "additionalProperties": false,
  "properties": {
    "schema_version": {"const": "1"},
    "decision": {"enum": ["route", "abstain"]},
    "team": {"anyOf": [{"enum": ["AUTH", "DATA", "NET"]}, {"type": "null"}]},
    "evidence_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true, "maxItems": 8}
  }
}
```

Đây là **trích lược** để đọc; schema thật còn pattern cho ID và `oneOf` kiểm các nhánh:

- Route: team thuộc enum và evidence có ít nhất một ID.
- Abstain: team=null, evidence rỗng.

`required` kiểm key có mặt, không tự đồng nghĩa giá trị khác null. *(Giải thích: Trong JSON Schema, `required` chỉ kiểm tra sự tồn tại của tên thuộc tính (key). Một trường nằm trong danh sách `required` hoàn toàn có thể mang giá trị `null` nếu kiểu dữ liệu của nó cho phép. Đó là lý do trong nhánh abstain, trường `team` vẫn phải có mặt nhưng nhận giá trị `null`).* `team` vẫn bắt buộc ở nhánh abstain nhưng có giá trị null; thiếu key không được coi như null ngầm. `properties` tự nó không bắt field phải có; cần required. Extra field `"action":"close_ticket"` bị từ chối thay vì mặc nhiên truyền xuống hệ thống khác. [JSON Schema object](https://json-schema.org/understanding-json-schema/reference/object). *(Ý nghĩa của `additionalProperties: false` (hoặc `extra='forbid'` trong Pydantic): Cấm tiệt các trường lạ ngoài danh mục. Nếu mô hình tự ý thêm một trường bất thường, hệ thống lập tức từ chối để ngăn chặn các cuộc tấn công tiêm thuộc tính độc hại (Property Injection)).*

`oneOf` nghĩa là hợp lệ với đúng một nhánh, không phải chỉ một nhánh bất kỳ được kiểm rồi bỏ qua phần còn lại. Lab có decision const khác nhau nên hai nhánh không chồng lấn. Có thể thiết kế union theo discriminator khác, nhưng cần kiểm schema cuối thực sự được render. [Schema composition](https://json-schema.org/understanding-json-schema/reference/combining). *(Giải thích: `oneOf` thiết lập logic loại trừ lẫn nhau (Mutually Exclusive). Đảm bảo một đề xuất chỉ được rơi vào đúng 1 trong 2 trường hợp: Hoặc là Route hoàn chỉnh, Hoặc là Abstain hoàn chỉnh, không bao giờ chấp nhận tình trạng "nửa nạc nửa mỡ" như vừa chọn `abstain` lại vừa điền `team: "NET"`).*

### Strict type và parser policy

Pydantic mặc định có thể coerce một số kiểu; lab bật strict và extra=forbid cho request/response, không tự biến số thành text hoặc giữ field lạ. Strict mode vẫn có chi tiết khác nhau theo type/input, nên phải kiểm domain cần dùng. [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/). *(Giải thích Strict Mode: Pydantic thông thường sẽ tự động ép kiểu ngầm (type coercion) — ví dụ chuỗi `"123"` sẽ tự động bị ép thành số nguyên `123`. Trong hệ thống AI khắt khe, ta bật `strict=True` để cấm hoàn toàn hành vi này: chuỗi phải là chuỗi, số phải là số. Nếu client truyền sai kiểu, API sẽ trả về lỗi HTTP 422 ngay lập tức).*

Parser lab còn từ chối:

- Markdown fence trước/sau JSON; không tự cắt một object bằng regex. *(Vì sao cấm Markdown Fence? LLM thường có thói quen bọc JSON trong khối mã ` ```json ... ``` `. Một bộ parser nghiệp vụ chuẩn mực đòi hỏi chuỗi thô phải là JSON thuần túy, không dùng biểu thức chính quy (regex) để "đoán mò" và cắt tỉa vì regex rất dễ cắt nhầm vào các chuỗi bên trong dữ liệu).*
- Duplicate key trong cùng object, tránh hai thành phần hiểu khác nhau khi một bên giữ giá trị cuối. *(Bẫy Duplicate Key: Chuỗi `{"team": "AUTH", "team": "NET"}` là hợp lệ trong một số parser lỏng lẻo. Tuy nhiên, parser của Python có thể lấy giá trị sau (`NET`), trong khi parser của Go hoặc Java có thể lấy giá trị đầu (`AUTH`), dẫn đến sự bất đồng bộ cực kỳ nguy hiểm trong kiến trúc microservices).*
- NaN/Infinity ngoài JSON chuẩn, dù parser Python có thể cho phép theo mặc định.
- Candidate raw vượt 16,000 byte UTF-8.

Chuỗi root `"NET"` là JSON hợp lệ nhưng không là object theo schema. JSON trống hoặc nhiều object nối nhau cũng không hợp lệ theo contract một object. Các hook parser được đối chiếu với [Python json](https://docs.python.org/3.14/library/json.html). Giới hạn candidate không thay thế body-size limit ở reverse proxy/server; lab chưa xây lớp giới hạn transport đó.

```python
from contracts import RouteRequest, validate_candidate
from service import proposal_json
request = RouteRequest(schema_version='1',ticket_id='demo',lines=[{'id':'L1','text':'event=timeout.'}])
proposal = validate_candidate(proposal_json('AUTH'),request)
assert proposal.team == 'AUTH'  # Qua contract, nhưng sai nhãn tác vụ ở ví dụ này.
```

## 6. Prompt yêu cầu JSON, JSON mode và schema-constrained generation

Ba khái niệm không được gộp làm một:

| Cách | Ý nghĩa | Vẫn cần gì? |
|---|---|---|
| Viết “hãy trả JSON” trong prompt | Yêu cầu bằng ngôn ngữ, model có thể không tuân thủ | Kiểm finish, parse, schema, task |
| Chế độ bảo đảm JSON theo API/model cụ thể | Ràng buộc cú pháp JSON ở phạm vi được dịch vụ hỗ trợ | Không tự bảo đảm field/enum của ứng dụng |
| Sinh có ràng buộc schema/grammar | Giới hạn output theo tập cấu trúc được implementation hỗ trợ | Kiểm khả năng hỗ trợ, kết thúc, refusal, semantic/business rule |

*(Phân biệt rõ 3 cấp độ tạo JSON trong kỹ thuật AI hiện đại:
1. **Cấp độ 1 — Yêu cầu bằng Prompt tự nhiên:**
   - *Cách làm:* Viết câu lệnh: "Hãy trả lời dưới dạng JSON có cấu trúc sau: {...}".
   - *Hạn chế:* Kém ổn định nhất. Mô hình thường xuyên thêm lời chào mở đầu ("Dưới đây là kết quả của bạn:"), bọc trong markdown fence, hoặc quên đóng ngoặc nhọn. Tỷ lệ lỗi cú pháp có thể lên tới 5–15%.
2. **Cấp độ 2 — JSON Mode của API Provider:**
   - *Cách làm:* Bật cờ cấu hình `response_format={"type": "json_object"}` trong OpenAI/Anthropic API.
   - *Cơ chế:* Bộ giải mã của nhà cung cấp can thiệp vào quá trình decoding để đảm bảo chuỗi ký tự sinh ra chắc chắn là cú pháp JSON hợp lệ.
   - *Hạn chế:* Nhà cung cấp chỉ cam kết là *JSON hợp lệ*, nhưng KHÔNG cam kết JSON đó có chứa đúng các trường nghiệp vụ của bạn hay không! Mô hình vẫn có thể trả về `{"thong_bao": "khong_tim_thay"}` thay vì trường `team`.
3. **Cấp độ 3 — Sinh có ràng buộc Schema / Văn phạm (Structured Outputs / Grammar-constrained Decoding):**
   - *Cơ chế:* Bạn cung cấp trực tiếp file `output-schema.json`. Tại mỗi bước sinh token, thuật toán giải mã (như Outlines, Guidance hoặc Structured Outputs của OpenAI) sẽ xây dựng một máy trạng thái hữu hạn (FSM). Bất kỳ token nào vi phạm cấu trúc JSON Schema sẽ bị gán điểm logit bằng $-\infty$ (mặt nạ cấm tuyệt đối).
   - *Kết quả:* Đảm bảo 100% tuân thủ cả cú pháp JSON lẫn tên thuộc tính và kiểu dữ liệu! Tuy nhiên, tính đúng đắn về mặt ngữ nghĩa (Factuality) vẫn cần kiểm chứng riêng).*

Tên gọi và tập schema hỗ trợ thay đổi giữa provider; bài không tuyên bố vendor nào bảo đảm gì hoặc đưa config chưa chạy cho họ. Nếu dùng một API cụ thể, phải đọc đúng phiên bản tài liệu và kiểm raw envelope của nó. Draft 2020-12 của validator cục bộ không đồng nghĩa mọi structured-output API hỗ trợ toàn bộ draft ấy.

**Lab không triển khai constrained decoding.** Nó nhận raw candidate từ provider giả lập/rule rồi validate sau sinh. Passing schema không tạo bằng chứng sự thật, kể cả khi cấu trúc được bảo đảm trong quá trình generation.

## 7. Error không giống abstention

Lab map trạng thái theo contract riêng:

| Trường hợp | HTTP của API lab | Code/body |
|---|---:|---|
| Request không đúng type/field/ID | 422 | FastAPI validation response |
| Candidate hợp lệ và route hoặc abstain | 200 | Success với result.decision |
| JSON/schema/evidence sai, length, refusal, upstream bị từ chối | 502 | ErrorBody có code cụ thể |
| Lỗi transient/rate limit hết số lần thử | 503 | upstream_unavailable |
| Timeout hết lần thử hoặc tổng deadline | 504 | upstream_timeout/deadline_exceeded |

*(Giải thích chuẩn mực thiết kế mã trạng thái HTTP trong AI Backend:
- **Vì sao Abstain trả về HTTP 200 OK?**
  Bởi vì request hoàn toàn hợp lệ, hệ thống hoạt động trơn tru và mô hình đã đưa ra một kết luận nghiệp vụ chính xác rằng "không đủ bằng chứng để phân loại". Trả về HTTP 200 kèm `decision: "abstain"` giúp client hiển thị thông báo thân thiện cho người dùng mà không kích hoạt các cảnh báo lỗi hệ thống (PagerDuty/Sentry).
- **Vì sao lỗi JSON từ LLM lại trả về HTTP 502 Bad Gateway?**
  Theo chuẩn RFC 9110, mã 502 chỉ ra rằng server đóng vai trò Gateway/Proxy đã nhận được một phản hồi không hợp lệ từ máy chủ Upstream. Ở đây, LLM chính là Upstream. Khi LLM sinh ra chuỗi văn bản hỏng không parse được JSON, đó là lỗi của nhà cung cấp mô hình, không phải lỗi cú pháp của người dùng (400/422) và cũng không phải lỗi sập code của backend bạn (500). Việc trả về 502 giúp đội ngũ vận hành phân định rạch ròi trách nhiệm của đối tác cung cấp AI.
- **HTTP 503 Service Unavailable:** Xảy ra khi Upstream bị nghẽn (Rate Limit 429) hoặc sập tạm thời, hệ thống đã thử retry hết số lần cho phép nhưng vẫn không thành công.
- **HTTP 504 Gateway Timeout:** Xảy ra khi Upstream sinh quá chậm và vượt quá tổng thời gian chờ (Deadline) cho phép).*

Mapping refusal sang 502 là **quyết định minh họa của API này**, không là luật bắt buộc cho mọi sản phẩm. Cũng không có quy tắc rằng một HTTP 200 của provider đồng nghĩa API ứng dụng phải trả 200. Phải xem finish/content trước.

Các nghĩa HTTP 502/503/504 được đối chiếu với [RFC 9110, phần 15.6](https://www.rfc-editor.org/rfc/rfc 9110.html). ErrorBody không trả nguyên candidate hoặc stack trace upstream cho client. Request validation response mặc định vẫn có chi tiết input lỗi của FastAPI; ứng dụng có yêu cầu riêng về redaction cần thiết kế thêm.

Lỗi lập trình ngoài các lỗi provider/contract đã phân loại không được biến thành một answer giả. Lab không có middleware xử lý mọi exception hay tuyên bố ready cho production.

## 8. Timeout và retry phải có ngân sách

Policy mặc định: tối đa 2 attempt, tổng deadline 1,000 ms, mỗi attempt tối đa 400 ms, backoff đầu 50 ms. Lần thử lại dùng remaining budget; không khởi tạo lại deadline mỗi retry. Khi có retry-after ở adapter, chọn delay ít nhất bằng giá trị đó nếu còn budget.

*(Tư duy quản trị ngân sách thời gian (Deadline Budgeting):
- Sai lầm phổ biến của lập trình viên là chỉ cài đặt timeout cho từng lần gọi riêng lẻ (ví dụ `timeout=400ms`) và cho phép retry 2 lần. Nếu mỗi lần mất 400ms cộng thêm thời gian chờ, tổng thời gian có thể vượt quá 1 giây mà client đã hủy kết nối từ lâu!
- Giải pháp chuẩn: Cài đặt **Tổng ngân sách thời gian (Monotonic Deadline)**. Ví dụ tổng deadline là 1,000 ms. Lần thử 1 mất 400 ms và nghỉ backoff 50 ms, thời gian tối đa còn lại cho lần thử 2 chỉ là $1,000 - 400 - 50 = 550 \text{ ms}$. Lần thử 2 bắt buộc phải tuân theo giới hạn thời gian còn lại này).*

### Tính tay theo clock giả

Ca timeout 2 lần: attempt 1 đợi 400 ms → backoff 50 ms → attempt 2 đợi 400 ms =850 ms rồi trả 504. Không thử vô hạn dù tổng deadline còn 150 ms vì đã hết max_attempts.

Ca rate limit rồi hồi phục: attempt 1 mất 20 ms, yêu cầu chờ 100 ms; delay=max(50,100)=100 ms; attempt 2 mất 30 ms → tổng 150 ms,2 attempt và thành công. Fixture đánh dấu usage_complete=False vì lỗi attempt đầu có thể đã dùng tài nguyên upstream mà client không biết đầy đủ.

Không retry lỗi JSON/schema như retry network một cách mặc định. “Yêu cầu model sửa JSON” là một lượt model call khác, có thể đổi nội dung và tăng chi phí; muốn làm cần policy và eval riêng. Refusal/length cũng không được lặp mù quáng trong lab. *(Nguyên tắc Retry trong AI Engineering: Chỉ retry các lỗi quá cảnh của hạ tầng mạng (Network Drop, HTTP 429 Rate Limit, HTTP 503). TUYỆT ĐỐI KHÔNG tự động retry khi mô hình sinh sai cú pháp JSON hoặc vi phạm Schema! Vì mô hình có xu hướng lặp lại cùng một lỗi tư duy với cùng câu prompt; việc retry mù quáng chỉ làm tăng gấp đôi chi phí API mà không đem lại kết quả).*

Trong hệ thống nhiều client, retry thường cần jitter/backpressure để tránh cùng thức dậy; lab dùng delay cố định theo công thức để kiểm tái lập, **chưa có jitter hoặc circuit breaker**. Retry-After ở fixture là millisecond đã chuẩn hóa nội bộ, không là bộ parse HTTP date/seconds ngoài mạng.

![Retry có giới hạn và tổng deadline](figures/03-retry.png)

### Timeout parameter chưa phải phép hủy thần kỳ

`execute()` truyền per-attempt timeout và o adapter và kiểm monotonic deadline. Provider giả lập tuân thủ timeout. Nếu adapter thật chạy một thao tác blocking và phớt lờ timeout, caller đồng bộ này không tự ngắt được nó giữa chừng; kiểm deadline sau call không thay thế transport timeout/cancellation đúng. Đây là phần phải bổ sung khi nối upstream thật, không được nói lab đã kiểm hủy mọi cuộc gọi mạng.

### Request ID không là idempotency key

Request ID dùng trace; retry có thể tạo nhiều lượt tính và nhiều hóa đơn dù cùng nội dung. Nếu API sau này có side effect như tạo ticket/hoàn tiền, cần idempotency/dedup theo nghiệp vụ; không dựa temperature 0 hoặc cùng request ID để tránh hành động lặp. POST không tự có tính idempotent chỉ vì request body giống. [RFC 9110, mục 9.2.2](https://www.rfc-editor.org/rfc/rfc 9110.html). *(Phân biệt rõ:
- `request_id`: Chỉ là mã định danh phục vụ ghi log và truy vết hệ thống.
- `Idempotency Key (Khóa bất biến)`: Khóa nghiệp vụ do client gửi lên dành cho các thao tác có tác dụng phụ (như thanh toán tiền, tạo đơn hàng). Nếu mạng bị chập chờn khiến client gửi lại cùng một request, server nhận diện Idempotency Key để trả về kết quả cũ mà không thực hiện trừ tiền hai lần).*

## 9. Từ contract test đến eval chất lượng

| Loại kiểm | Ví dụ | Nó chứng minh gì? |
|---|---|---|
| Unit/contract test | Reject extra key, bắt null sai nhánh | Quy tắc parser/schema chạy theo thiết kế |
| Integration/fault test | HTTP 422, timeout→retry→503/504 | Các thành phần kết nối và xử lý lỗi đúng |
| Model eval | Chạy model thật trên ticket giữ riêng | Hành vi model trong tập và cấu hình được đo |
| End-to-end eval | Request thật qua context/provider/validator tới kết quả | Chất lượng và lỗi của toàn pipeline theo nhiệm vụ |

*(Phân định 4 tầng kiểm thử trong dự án AI:
1. **Unit / Contract Test:** Kiểm tra các hàm parser, schema Pydantic, cấm extra field. Chạy trong vài mili-giây, không cần mô hình AI.
2. **Integration / Fault Test:** Kiểm tra cơ chế xử lý lỗi HTTP, khả năng retry khi gặp sự cố mạng, ngắt kết nối khi hết deadline.
3. **Model Evaluation (Eval mô hình):** Đo lường năng lực trí tuệ của mô hình ngôn ngữ trên một tập dữ liệu chuẩn đã được gán nhãn độc lập.
4. **End-to-End Evaluation (Đánh giá toàn diện):** Đo lường tỷ lệ thành công của toàn bộ chuỗi mắt xích từ lúc nhận request của người dùng, qua prompt template, gọi mô hình, validate schema đến khi trả về kết quả cuối cùng).*

Bộ fake output của lab tốt để tạo lỗi chắc chắn và kiểm branch hiếm. Nó không cho một tỷ lệ lỗi production hoặc bảng xếp hạng model. Một model mới có thể giữ nguyên schema nhưng làm semantic accuracy giảm; unit test không thay thế eval ấy.

Đánh giá cần gắn nhiệm vụ cụ thể thay vì chỉ dùng benchmark chung. Bài [Emerging Patterns in Building GenAI Products, phần Evals](https://martinfowler.com/articles/gen-ai-patterns/) mô tả kinh nghiệm tổ chức scoring ở mức thành phần và toàn hệ thống. Đây là nguồn thực hành của tác giả, không phải bằng chứng một judge hay metric luôn đáng tin.

## 10. Thiết kế một bộ eval có thể dùng lại

Mỗi case nên có ID, group/nguồn, input, expected answer hoặc rubric, slice và metadata cần thiết. Reference chỉ scorer biết. Giữ dữ liệu nhạy cảm theo quyền và mục đích sử dụng; đừng mặc định đưa mọi production log và o bộ chia sẻ.

Các slice phù hợp route ticket:

- Rõ ràng theo lớp, nhiều cách diễn đạt. *(Ca dữ liệu sạch - Clean slice).*
- Thiếu dữ kiện, mâu thuẫn, nhiều sự cố cùng lúc. *(Ca mơ hồ - Ambiguous slice, đòi hỏi mô hình phải biết abstain).*
- Phủ định: “timeout đã được loại trừ”. *(Ca ngữ nghĩa phức tạp - Semantic slice, bẫy các mô hình chỉ tìm từ khóa ngây thơ).*
- Input có instruction nằm trong nội dung dữ liệu. *(Kiểm tra khả năng chống Prompt Injection).*
- Output/schema edge case và upstream fault (nên báo riêng với chất lượng semantic).

Giữ group incident/hội thoại qua split như bài 5. Dùng development/validation để sửa prompt/schema/policy; sau khi chốt mới đo test. Một fixture suite được mở để phát triển code, như lab này, **không phải holdout chất lượng chưa từng xem**. Không đổi tên nó thành “test độc lập” vì file tên test.

Khi gọi model thật nhiều lần, lưu version model, prompt, tokenizer/template khi có, decoding settings, dataset hash, scorer version và số lần lặp. So candidate với baseline trên cùng case; dùng paired analysis khi phù hợp thay vì so hai bộ khác nhau. Thay đổi provider/model phía server có thể làm kết quả cũ không tái lập hoàn toàn dù prompt không đổi.

## 11. Metric: phải công bố mẫu số

Giả sử cóNrequest:

- Completion pass: qua gate finish bình thường /N.
- JSON pass: qua completion rồi parse được /N.
- Schema pass: qua các gate trước và đúng schema /N.
- Acceptance: qua schema và kiểm reference ID /N; gồm route lẫn abstain.
- Task success: response accepted và đúng expected decision/team /N; lỗi hệ thống vẫn nằm trong mẫu số.
- Route coverage: số response accepted có decision route /N.
- Route accuracy: số route đúng /số route accepted.

*(Cạm bẫy toán học: Thao túng Mẫu số (The Denominator Trap):
- Nếu bạn báo cáo: *"Độ chính xác định tuyến của hệ thống là 57.14%"* (lấy 4 ca đúng chia cho 7 ca được route), con số này nghe có vẻ chấp nhận được.
- Nhưng sự thật trần trụi là: Trong tổng số 20 yêu cầu mà người dùng gửi vào ($N=20$), chỉ có đúng 4 yêu cầu được định tuyến chính xác!
- Toàn bộ 13 yêu cầu khác bị sập mạng, timeout hoặc vỡ cấu trúc JSON đã bị bạn "gạt ra ngoài mẫu số" để làm đẹp báo cáo.
- **Quy tắc đạo đức và kỹ thuật:** Luôn luôn công bố chỉ số **End-to-End Task Success** trên tổng mẫu số ban đầu $N$ ($5/20 = 25\%$), không được che giấu các thất bại kỹ thuật của hệ thống).*

Task success và conditional accuracy không thay nhau. Hệ thống chỉ nhận và i case dễ có thể conditional accuracy cao trong khi end-to-end success thấp. Abstain đúng trên case thiếu dữ kiện là một success; provider error không phải abstain hợp lệ.

Với câu mở như email/tóm tắt, exact match một câu tham chiếu thường quá chặt. Tách deterministic check (schema, con số, ID, điều cấm) khỏi rubric như đủ thông tin/đúng nguồn/giọng điệu. LLM-as-judge chỉ là một scorer có sai số: cần rubric, ví dụ chuẩn, đối chiếu người chấm và kiểm ổn định/thiên lệch; không để model tự chấm rồi gọi đó là ground truth.

Lab không chạy LLM judge hoặc suy calibration từ một field confidence do model tự viết. Nếu muốn dùng confidence để quyết định abstain, cần dữ liệu đánh giá và phép hiệu chỉnh/phân tích phù hợp như bài 4/7, không mặc định số 0.9 có nghĩa 90% đúng.

## 12. Kết quả lab và ví dụ tính metric

Lab có 20 case chủ đích:3 clean,2 ambiguous,2 semantic,3 format,4 schema,1 reference-ID,5 transport. Các case này nằm trong `fixtures()` ở [lab.py](lab.py), không lấy ngẫu nhiên từ một quần thể thực.

| Gate/metric | Kết quả đã chạy |
|---|---:|
| Completion pass | 16/20 =80% |
| JSON pass sau completion | 13/20 =65% |
| Schema pass sau các gate trước | 9/20 =45% |
| Accepted, gồm abstain | 8/20 =40% |
| Task success trên mọi case | 5/20 =25% |
| Đúng trong số accepted | 5/8 =62.5% |
| Route coverage | 7/20 =35% |
| Route accuracy có điều kiện | 4/7 ≈57.1429% |
| Số attempt quan sát | 23 cho 20 case |

*(Phân tích chi tiết từng tầng rớt đài trong 20 ca thử nghiệm:
1. **Tổng đầu vào:** 20 trường hợp.
2. **Gate 1 — Completion Pass (16/20 = 80%):** Có 4 ca rớt ngay tại cửa Upstream (1 ca bị cụt do hết token, 1 ca bị mô hình từ chối, 2 ca bị timeout và sập mạng).
3. **Gate 2 — JSON Syntax Pass (13/20 = 65%):** Trong 16 ca sống sót, có 3 ca sinh văn bản không parse được JSON (dính markdown, duplicate key).
4. **Gate 3 — Schema Pass (9/20 = 45%):** Trong 13 ca JSON hợp lệ, có 4 ca vi phạm cấu trúc Schema (JSON là chuỗi đơn thay vì object, thiếu trường bắt buộc, giá trị null sai nhánh).
5. **Gate 4 — Acceptance Pass (8/20 = 40%):** Trong 9 ca đúng Schema, có 1 ca bị loại vì trích dẫn ID bằng chứng ma không tồn tại trong input (`unknown_evidence`).
6. **Gate 5 — Task Success (5/20 = 25%):** Trong 8 ca được chấp nhận, chỉ có 5 ca đưa ra quyết định chính xác (4 ca route đúng đội và 1 ca abstain đúng khi thiếu dữ kiện). 3 ca còn lại bị sai ngữ nghĩa (ví dụ thấy chữ timeout bị loại trừ nhưng vẫn phán đội NET)).*

Một raw candidate JSON đầy đủ nhưng finish=length bị loại ở gate completion, nên không tăng “JSON pass sau completion”. Điều đó **không nói bản raw ấy parse lỗi**; metric ở bảng là pipeline pass, không là tỷ lệ syntax tách riêng khỏi status. Các timeout/refusal vẫn nằm trong mẫu số 20.

Ba accepted response vẫn sai tác vụ: sai team dù schema đúng; ép route khi có hai dấu hiệu mâu thuẫn; evidence ID tồn tại nhưng nội dung không hỗ trợ team được đề xuất. Không có validator schema nào trong lab “bảo đảm đúng semantic”.

Slice clean đúng 3/3, semantic đúng 0/2, transport có 1/5 case thành công sau retry. Nếu chỉ báo 3/3 clean, bạn che toàn bộ khó khăn còn lại. Nếu chỉ báo 5/8 accepted, bạn che 12 case không có output được chấp nhận. Bảng và confusion matrix giữ cả cột ERROR.

![Funnel trên fixture suite](figures/04-eval-funnel.png)
![Confusion matrix và slice](figures/05-confusion-slices.png)

### Chi phí và latency ở fixture

20 case dùng 23 attempt: overhead số lần gọi =23/20=1.15 attempt/case. Đây không là chi phí tiền thực. Timeout có thể phát sinh usage không được trả về; dùng counter biết được như một phần thông tin thay vì điền 0 rồi gọi “miễn phí”. Lab chỉ giữ usage counter trên success envelope, không có billing reconciliation cho mọi failure.

Clock giả cho 850 ms/150 ms/70 ms ở ba case có retry là tổng duration/backoff do fixture định nghĩa. Nhiều case khác có 0 ms là vì fake clock không tự đếm CPU parse/validation. Không tính p 95 latency production từ dãy số giả lập này.

## 13. API cục bộ và các file cần đọc

- [contracts.py](contracts.py): Pydantic model, JSON Schema và parser/validator.
- [service.py](service.py): adapter protocol, provider script/rule, clock, retry policy và execute.
- [api.py](api.py): FastAPI app với endpoint đề xuất và response model.
- [lab.py](lab.py): schema/API demo,20 fixture và scorer/report.
- [output-schema.json](output-schema.json): schema đầy đủ để đối chiếu.

`demo_provider` mặc định là **rule từ khóa**, không phải model học từ bài 20 hoặc API LLM. Nhờ tách adapter, có thể thay bằng provider thật sau này mà vẫn chạy contract/fault suite. Bài này không tự cấu hình account/key hoặc tiêu tiền để có output mẫu.

TestClient gửi request qua ASGI trong process, kiểm HTTP status/body/OpenAPI mà không cần mở cổng. Đó là integration check, không là benchmark mạng hoặc kiểm TLS/proxy. Lệnh Uvicorn localhost cũng đã được chạy để kiểm health, POST proposal và OpenAPI qua HTTP thật; server được dừng sau khi kiểm. [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).

```python
from fastapi.testclient import TestClient
from api import create_app
payload = {'schema_version':'1','ticket_id':'demo','lines':[{'id':'L1','text':'event=timeout.'}]}
with TestClient(create_app()) as client:
    response = client.post('/v1/route-proposals',json=payload)
    assert response.status_code == 200
    assert response.json()['result']['team'] == 'NET'
    assert client.post('/v1/route-proposals',json={**payload,'unexpected':True}).status_code == 422
```

![Ranh giới test hệ thống và eval model](figures/06-eval-plan.png)

## 14. Lệnh chạy

Đã kiểm Python 3.14.6, FastAPI 0.162.0, Pydantic 2.14.2, jsonschema 4.26.0, httpx 0.28.1, Starlette 1.10.0, Uvicorn 0.54.0 trên macOS arm 64. Phiên bản ghim ở [requirements.txt](requirements.txt); bài này **không cần PyTorch**. Từ thư mục bài 22:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode schema
.venv/bin/python lab.py --mode api
.venv/bin/python lab.py --mode all --out /tmp/lesson22-run-01
```

`--mode eval` chỉ chạy scorer; --out cần đường dẫn mới. Không có --out chỉ in kết quả. Run lưu report.json và output-schema.json; không có weight hoặc dữ liệu người dùng thật. JSON schema bản nguồn được sinh từ contracts.py và được kiểm không lệch so với file lưu.

Để mở server cục bộ ở terminal riêng:

```bash
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Sau đó gọi bằng terminal khác:

```bash
curl -sS http://127.0.0.1:8000/v1/route-proposals \
  -H 'Content-Type: application/json' \
  -d '{"schema_version":"1","ticket_id":"demo","lines":[{"id":"L1","text":"event=timeout."}]}'
```

Xem `/docs` hoặc `/openapi.json` trên cùng server; dừng bằng Ctrl+C. Windows thay đường dẫn Python tương ứng; các lệnh curl có thể cần cách quote phù hợp shell. Server chỉ là demo localhost, chưa có auth, tenant isolation, rate limit, idempotency store, cancellation thật hoặc deployment config.

Bộ phiên bản TestClient đã kiểm có thể in cảnh báo deprecation từ Starlette về adapter httpx; đây không là lỗi test. Bài ghim phiên bản để tái lập; nếu đổi HTTP client/Starlette cần chạy lại integration check, không giả định version mới giữ mọi hành vi.

Sáu hình PNG/SVG đã lưu; tái tạo với NumPy 2.5.4 và Matplotlib 3.11.2:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py --run /tmp/lesson22-run-01
```

## 15. Tự kiểm và bước tiếp

Làm [16 bài tập](exercises.md), rồi xem [lời giải](solutions.md). Bạn cần tự giải thích: schema valid chưa đủ đúng task; unknown evidence khác evidence không hỗ trợ; HTTP 200 khác semantic success; retry cần tổng budget; và vì sao metric phải giữ lỗi/abstain đúng mẫu số.

[Phụ lục](deep-dive.md) đào sâu schema contract, adapter và eval thực tế. Tiếp tục với [bài 23 — Tìm kiếm từ khóa và vector](../23-tim-kiem-tu-khoa-vector/README.md), đã có thực hành, bài tập và lời giải. [Nguồn và phạm vi đối chiếu](../SOURCES.md).
