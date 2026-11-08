# Bài 30 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán state, counter và grade trước khi chạy. Không cần model hoặc dữ liệu ngoài bộ bài.

## 1. Đơn vị đánh giá

Phân biệt task, trial, assertion, grader và hai loại harness. Với 12 task, 4 profile và 41 regression check, có bao nhiêu task-trial? Có được nói “89 lần agent thành công” không?

## 2. Outcome thật

Tạo final NOTE có facts đúng nhưng không ghi note. Stop có thể là FINAL không? Outcome/accepted phải thế nào? Ngược lại nếu note đã ghi mà không có final, vì sao không tự kết luận chưa có effect?

## 3. Thứ tự hợp lệ

Đổi thứ tự health/runbook có làm task sai không? Vẽ các cạnh dependency cần giữ. Tạo REPORT đúng facts nhưng không đọc gì: grader nào cần phát hiện thiếu tiền đề? Vì sao exact transcript dễ từ chối đường hợp lệ?

## 4. Attempt và violation

Chạy D01 profile 2 ở strict và schema_only. So denial, outcome_ok, safe và accepted. “Đã chặn một proposal” có tự là task thất bại hoặc vi phạm đã thực thi không? Một violation trial có thể có nhiều unsafe read/write không?

## 5. Phát hiện semantic error

Chạy D03 profile 3. Vì sao JSON/schema/final đều hợp lệ nhưng task sai? Code grader đang kiểm semantics đến mức nào? Với giải thích tự do, đề xuất rubric và cách kiểm judge có bị transcript lừa không.

## 6. Reliability tính tay

Với p=0,75, k=3, tính pass@3 và pass^3 dưới giả định i.i.d. Tính 0,95^8 cho chuỗi 8 bước độc lập. Giải thích vì sao không lấy step accuracy trung bình rồi dùng lũy thừa nếu có phụ thuộc/retry.

## 7. Subset estimator

Từ A=1111, B=1100, C=1000, tính macro pass@2 và pass^2 bằng cách liệt kê cặp. Tính khi k=4. Nếu dùng p=7/12 rồi bình phương có đúng macro pass^2 không? Hàm phải xử lý k>n thế nào?

## 8. Không thấy lỗi chưa là an toàn tuyệt đối

Giả định 30 trial Bernoulli i.i.d. có 0 vi phạm. Tính upper bound một phía 95%. Tính Wilson cho 7/12. Vì sao không dùng 41 check hoặc 48 profile cố định như cùng số trial i.i.d. để suy rủi ro model?

## 9. Budget tính tay

Tính units và modeled time happy path. Deadline 310 và 300 ms khác nhau thế nào? Với max_calls=1, read thứ hai có tới backend không? Model turn cuối hoặc proposal sai có tiêu budget không?

## 10. Giới hạn trước effect

Thử max_turns, max_calls, max_writes, max_units, deadline_ms và context_bytes nhỏ. Kiểm counter không vượt cap. Turn limit có thay call limit được không? Context byte có tương đương model tokens không? Đổi deadline giả sang I/O thật cần thêm gì?

## 11. Review có binding

Prepare một draft rồi đổi facts, version, quyền hoặc thời gian. Commit phải bị chặn ra sao? Thử replay với write cap tạm cho phép 2 reservation. Vì sao “đã có một review ở đâu đó trong trace” chưa đủ?

## 12. Final snapshot và journal

Ghi ngoài scope rồi restore world về snapshot đầu. Collateral snapshot có thể đúng không? Safety grader còn thấy vi phạm ở đâu? Nêu giả định tin cậy của journal trong RAM và những gì phải thêm nếu log có thể bị sửa.

## 13. Reset và oracle

Chứng minh hai World không chia sẻ notes. Kiểm view không có expected label và sửa view không đổi task/observations thật. Chỉ tách dict có đủ sandbox cho policy Python tùy ý không? Manifest hash đang bao phủ những gì?

## 14. Baseline và paired comparison

Chạy compare all. Vì sao refuse có 24/48 accepted nhưng positive completion=0/24? Đọc bảng paired adaptive strict/schema_only; đối chiếu Δ với chênh lệch accepted. Vì sao không báo mỗi safety=100% để chọn baseline?

## 15. Split và cách công bố

Nêu khác nhau giữa dev, holdout, regression và traffic sample. Split theo family ID của lab còn hạn chế gì? Thiết kế bộ 10 task riêng, có positive/handoff/attack/error; ghi tiêu chí, nguồn quyền, reset, grader, phiên bản và mẫu số. Nếu trial trong cùng task tương quan, đơn vị suy luận nên là gì?

## 16. Dự án nhỏ

Thêm một miền tổng hợp khác như cập nhật địa chỉ giao hàng, không dùng dữ liệu thật. Có read-only intent, write cần review, cap action, ít nhất hai đường hợp lệ, transient error, stale review, claimed success không effect, effect trước timeout và một lỗi semantic chưa được chấm. Viết report tách harness check, task success, attempted denial và executed violation. Không đổi oracle để che ca trượt.
