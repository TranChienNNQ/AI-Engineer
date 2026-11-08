# Bài tập 46 — Tích hợp LLM vào trợ lý hỗ trợ kỹ thuật

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Chạy code từ thư mục bài 46 và giữ thư mục bài 45 cạnh bên. Mọi bài thực hành mặc định dùng fixture, không cần API key/network. Các yêu cầu thiết kế live không bắt buộc gọi provider thật.

## 1. Xác định trách nhiệm của model

Vẽ luồng user intent/evidence→LLM→candidate→validator→proposal/review/gateway. Field nào model được viết, field nào phải do backend quyết định? Vì sao một câu model nói “đã tạo ticket” chưa đủ làm UI báo succeeded?

## 2. Phân biệt fixture, smoke và evaluation

Một test có MockTransport trả reply hợp lệ, một call model thật 200 và một eval 500 queries đã annotation chứng minh những điều khác nhau gì? Tại sao không được báo fixture 12 cases là accuracy của một LLM? Nêu phần còn cần kiểm cho adapter live hiện tại.

## 3. Xem prompt và schema

Tạo prompt cho timeout 504, liệt kê user payload keys và evidence fields. Kiểm raw bearer/role/ticket target có được gửi không. Vì sao nullable draft vẫn required? Thử reply thêm tenant hoặc text dạng số; parser phải làm gì?

## 4. Prompt injection và cấu trúc request

User query chứa chuỗi đóng ngoặc JSON và “role system”. JSON serialization bảo vệ điều gì, không bảo vệ điều gì? Nếu evidence approved chứa instruction resolve mọi ticket, tại sao backend vẫn không được tự execute? Thiết kế một test authority boundary không phụ thuộc model có ngoan hay không.

## 5. Tính citation offset ở server

Lấy câu thứ hai của runbook gateway làm quote, tạo claim, tìm start/end tại server và kiểm khớp source. Vì sao từ chối quote xuất hiện hai lần? Model tự cung cấp URL/offset có rủi ro gì? Nếu source revoke sau khi provider nhận context, phần nào còn có thể chặn?

## 6. Schema đúng nhưng nội dung sai

Viết một reply có evidence ID/quote hợp lệ nhưng text khuyến nghị trái quote. Field/type nào vẫn hợp lệ? Cần tách kiểm shape, citation và support thế nào? Exact_quote có chứng minh câu trả lời đủ ý hoặc source đúng không?

## 7. So hai support policy

Dùng fixture paraphrase và unsupported với exact_quote/citation_only. Dự đoán serve/block và metadata semantic. Từ bảng kết quả, tính precision/recall của quyết định serve cho hai policy. Vì sao cùng 11/12 accuracy chưa đủ chọn policy?

## 8. Response envelope và trạng thái

Thiết kế handling cho HTTP 400/429/503, status incomplete, refusal, malformed model JSON, reasoning item và function_call bất ngờ. Trường hợp nào retry trong adapter hiện tại? Vì sao không nối mọi output thành text hoặc execute partial JSON?

## 9. Retry, deadline và usage

Chạy 503→good và hai timeout liên tiếp. Kiểm attempts, known usage và unknown attempts. Nếu total budget .04 s nhưng retry delay .05 s, có attempt thứ hai không? Vì sao timeout không tự có cost 0? Phân biệt provider budget với API deadline/cancellation.

## 10. Dự trù budget trước model call

Ask có evidence cần bao nhiêu stages, write proposal bao nhiêu, no-evidence bao nhiêu? Với max_steps 3 cho ask, provider có được gọi không? Nêu sự khác nhau giữa context 600 chars, input 6.000 chars, response 16.384 bytes và max_output_tokens 768.

## 11. Draft, review và receipt

Dùng fixture tạo draft create, chứng minh DB chưa có ticket mới trước review. Duyệt payload rồi execute/retry; kiểm provider chỉ được gọi một lần. Draft title/body có được coi là fact đã kiểm về incident không? Thay generator có làm mất optimistic version hoặc idempotency không?

## 12. Function calling khác structured draft ở đâu?

Mô tả năm bước provider function calling và vai trò call_id. Lab hiện có thực thi protocol này không? Nếu muốn cho model chọn tool, cần thêm những kiểm nào trước dispatcher và khi trả tool result? Vì sao không cho function_call resolve_ticket bất ngờ chạy luôn?

## 13. API-ASGI thực hành

Gọi API hỏi hai lần cùng key, kiểm cùng result và chỉ một provider call. Kiểm thiếu auth hoặc extra tenant bị chặn. Vì sao app 46 có thể dùng lại envelope/API 45? Test ASGI hiện chưa chứng minh phần mạng hoặc model provider thật nào?

## 14. Đọc kết quả và thiết kế model evaluation

Giải thích trường hợp paraphrase bị chặn và unsupported claim bị serve. Known usage 1.320/440 trên mỗi policy có phải bill thật không? Đề xuất dataset/repeats/rubric và human-check để đánh giá LLM live; tránh tune bằng test và tránh cherry-pick output tốt nhất.

## 15. Release và reopen DB

Reopen một generated run/receipt bằng adapter fixture khác, kiểm không gọi provider lại. Model/prompt/schema/policy/config thay đổi ảnh hưởng release thế nào? Vì sao key cũ 45 không được giả là một run 46 mới? Hash có bảo đảm model deterministic hoặc artifact từ publisher tin cậy không?

## 16. Kế hoạch live có thể review

Viết kế hoạch smoke nhỏ dùng corpus synthetic, chỉ rõ model/schema/account cần xác minh, input gửi ra ngoài, budget/retry và dữ liệu/metadata cần giữ. Sau smoke, bằng chứng nào còn thiếu trước staging/pilot thật? Không cần cung cấp key hoặc thực hiện call có phí để trả lời bài này.
