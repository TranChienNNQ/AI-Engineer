# Bài tập 47 — Đánh giá chất lượng LLM

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Code chạy từ thư mục bài 47 trong process sạch, giữ bài 45–46 cạnh bên. Không bài nào bắt buộc tài khoản, API key hoặc gọi model có phí.

## 1. Viết quyết định cần bằng chứng

Đội muốn đổi prompt vì người dùng thấy answer “dễ đọc hơn”. Viết mục tiêu chất lượng cho trợ lý gateway/ticket, hai lợi ích cần đo và hai lỗi critical không được bù bằng điểm văn phong. Vì sao benchmark chung hoặc HTTP 200 chưa đủ?

## 2. Phân rã một ca lỗi

Retrieval có đúng runbook v2. Answer trích câu đầu, thiếu điều kiện idempotency. Draft ghi đã khôi phục database dù user chỉ báo 504. Chỉ ra lớp nào đúng, lớp nào sai và kiểm nào bài 46 chưa thực hiện về ngữ nghĩa.

## 3. Dataset và group leakage

In số cases/groups/split. Chuyển paraphrase thứ hai của một dev group sang test; validator phải làm gì? Nếu chỉ đổi ID để né kiểm, semantic independence đã có chưa? Vì sao shared source không cho phép kết luận unseen-document generalization?

## 4. Reference và keyword

Viết required facts/forbidden assertions cho gateway. So hai câu có cùng từ “idempotency”: một câu yêu cầu key khi retry ghi, một câu nói không cần key. Vì sao keyword overlap/exact answer string có thể chấm sai? Viết một paraphrase hợp lệ.

## 5. Capture hai support policy

Chạy case gateway repeat=1 với hai config. Config nào serve, config nào reject? Nhãn support của authored paraphrase là gì? Vì sao đây là policy comparison trên cùng fixture schedule, không chứng minh model nào tốt hơn?

## 6. Chấm draft riêng

Capture case tạo ticket repeat=1. Answer có quote đúng không? Draft chứa gì sai? Vì sao support answer=2 vẫn có critical? Kiểm số tickets/receipts trước mọi approve/execute.

## 7. Rating schema và hash

Thử đổi content hash, rubric, support thành True hoặc 2.0, thêm extra field, duplicate rating và để draft=null cho write. Những trường hợp nào phải bị từ chối? Hash bảo đảm gì, chưa bảo đảm gì về người chấm?

## 8. Tính metric bằng tay

Dev quote_guard có 24 rows, 14 served, 8 served full success, 6 correct abstain và 2 critical. Tính coverage, conditional quality, task success và critical rate trong served. Vì sao không bỏ failed/refusal hoặc đổi denominator sau khi thấy điểm?

## 9. Missing rating và always-abstain

Xóa toàn bộ ratings của served outputs. Task success trong report còn được hiểu thế nào? Vì sao missing không bằng bad annotation hoặc zero cost? Gate phải làm gì? Nếu một hệ luôn abstain, metric nào cho thấy nó không đáp ứng task có nguồn?

## 10. Đọc bảng lỗi đa nhãn

Từ report, so test answerable/write/permission slices của hai config. Một partial hoặc invented draft có thể tăng nhiều error counters không? Đề xuất sửa hai lỗi ở hai lớp khác nhau và nêu cách kiểm lại để tránh kết luận theo cảm giác.

## 11. Audit một judge quá dễ dãi

Proxy có citation thì pass; author labels có 34 good, 28 bad. Tính TP/FP/FN/TN, agreement, bad-pass rate và κ. Bad có đồng nghĩa critical không? Bạn sẽ lấy subset nào để audit critical miss riêng?

## 12. Kiểm đảo thứ tự pairwise

Thứ tự đầu [candidate X, candidate Y] judge chọn slot A. Thứ tự sau [Y,X] judge vẫn chọn A. Candidate nào thắng mỗi lần? Hàm swap_result nên trả gì? Phân biệt consistent tie và needs_review. Nêu một ca injection để thử judge.

## 13. Tạo bộ rating do người học tự chấm

Dùng review queue, chấm một packet theo rubric mà chưa xem author labels. Flatten packet/rating thành JSONL hợp schema, chạy score trên full records với file chỉ có item đã chấm. Giải thích rating coverage/missing/gate. Không gọi một lần tự chấm là nghiên cứu human độc lập.

## 14. Repeats và khoảng bất định

Tính Wilson upper khi 0/30 lỗi; vì sao không kết luận p=0? Trong test fixture, có bao nhiêu group độc lập theo thiết kế? Vì sao bootstrap phải giữ group paired, và tại sao khoảng minh họa vẫn chưa xác nhận generalization? Retry 503 có phải repeat thứ tư không?

## 15. Dev selection và không có ứng viên

Đọc gate và selected_on_dev. Vì sao cả hai trượt dù synthesis có task success cao hơn? Nếu xem test rồi chọn synthesis, bạn đã thay quy trình thế nào? Làm gì khi muốn sửa ngưỡng hoặc rubric?

## 16. Thiết kế một live evaluation nhỏ

Viết kế hoạch dùng corpus synthetic để thu output model thật qua adapter 46: case/scope/schedule, generation/judge budgets, annotation/calibration/audit, provenance/usage và stop conditions. Liệt kê phần phải thay của lab 47 trước khi dùng live records; vì sao không tái dùng author_rating theo variant? Chưa cần thực hiện call hoặc execute ticket thật.
