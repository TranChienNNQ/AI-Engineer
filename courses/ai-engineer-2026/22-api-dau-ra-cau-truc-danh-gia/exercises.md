# Bài 22 — Bài tập API, structured output và eval

[Bài học](README.md) · [Lời giải](solutions.md)

## 1. Hai hợp đồng API

Vẽ client → API ứng dụng → adapter provider → validator. Vì sao request nghiệp vụ không nên là bản sao mọi tham số của provider? API 200 của provider có bảo đảm ứng dụng thành công không?

## 2. Required và null

Phân biệt team bị thiếu với team=null. Viết hai object hợp lệ cho route/abstain và ba object mâu thuẫn phải bị loại. Vì sao chỉ nullable team chưa đủ?

## 3. Schema không thay syntax

Phân loại các candidate: `"NET"`, object thiếu team, JSON trong Markdown fence, object có action lạ và object có duplicate key. Case nào parse được nhưng schema sai? Policy lab xử lý ra sao?

## 4. Kiểm quan hệ field

Giải thích oneOf trong output schema. Nếu cả hai nhánh cùng cho decision bất kỳ, có thể xảy ra vấn đề gì? Dùng code kiểm Python validator và JSON Schema đồng ý trên cùng tập ca.

## 5. Đúng cấu trúc, sai task

L 1 có “timeout đã được loại trừ; event=denied”. Candidate NET, L 1 qua được những lớp nào và sai ở đâu? Vì sao evidence ID tồn tại không là bằng chứng đúng nhãn?

## 6. Finish reason

Raw candidate là JSON đầy đủ nhưng finish=length. Vì sao lab vẫn trả error? Refusal có phải abstain theo contract không? Vì sao không tự đổi mọi error thành abstain?

## 7. Retry tính tay

Policy 2 attempt, tổng 1000 ms, attempt 400 ms, backoff 50 ms. Hai timeout mất bao nhiêu thời gian giả lập? Rate-limit 20 ms, retry-after 100 ms, rồi reply 30 ms thì sao? Nếu total chỉ 100 ms có được reset budget để thử lại không?

## 8. Chính sách retry

Phân loại timeout/transient/rate-limit/fatal/schema-invalid/length/refusal: lab retry loại nào? Vì sao repair JSON không tự là cùng hành động với retry transport? SDK 3 attempt và app 3 attempt có thể thành bao nhiêu call?

## 9. Request ID và idempotency

Vì sao cùng request ID không bảo đảm một hành động chỉ xảy ra một lần? Nếu endpoint đổi từ proposal sang ghi assignment thì cần thêm gì? Timeout có chứng minh upstream chưa làm gì không?

## 10. Denominator

Từ 20 case,8 accepted,5 correct,7 route,4 route đúng: tính acceptance, task success, accuracy trên accepted, route coverage, route accuracy. Có thể thay metric này bằng metric kia không?

## 11. Slice và confusion

Clean 3/3 nhưng semantic 0/2 và task success tổng 5/20. Bản báo cáo chỉ ghi 100%clean thiếu gì? Vì sao cần giữ cột ERROR tách khỏi ABSTAIN?

## 12. Usage và latency

23 attempt cho 20 case nói gì về overhead? Vì sao usage không biết sau timeout không được ghi là 0 rồi coi đã biết chi phí? Có được báo p 95 production từ FakeClock của fixture không?

## 13. Test và eval

Phân biệt unit/schema test, integration fault test, model eval, end-to-end eval. Vì sao bộ fixture được xem trong quá trình viết code không là holdout model chất lượng? Nêu cách chia dữ liệu theo incident.

## 14. Judge và rubric

Thiết kế rubric ngắn cho email hỗ trợ hoặc tóm tắt incident. Những phần nào kiểm deterministic, phần nào cần người/judge? Vì sao LLM tự viết confidence 0.9 không tự là 90%đúng?

## 15. Thực hành endpoint

Dùng TestClient kiểm 200,422,502,503,504 và OpenAPI. Kiểm unknown evidence, extra field và timeout không trả success giả. Một schema-valid wrong-team có bị contract validator phát hiện không?

## 16. Thí nghiệm mở rộng

Thiết kế eval để thay ScriptedProvider bằng một model thật mà không làm lẫn kết quả fault suite với quality eval. Nêu dataset, reference, version, prompt, decode policy, cách chọn cấu hình, metric/chi phí và giới hạn. Không cần gọi dịch vụ trả phí để hoàn thành đề bài.
