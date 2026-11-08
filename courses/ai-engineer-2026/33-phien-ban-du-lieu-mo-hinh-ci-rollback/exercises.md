# Bài 33 — Bài tập

[Bài chính](README.md) · [Lời giải](solutions.md)

Làm phần dự đoán trước khi chạy. Mọi dữ liệu và thao tác release đều tổng hợp/cục bộ; không cần truy cập hệ thống production.

## 1. Cùng code, khác prediction

API cùng commit và weights, nhưng hôm nay một ticket đổi priority. Kể ít nhất bốn thành phần khác có thể gây khác hành vi. Nêu thông tin cần ghi để truy nguyên và vì sao trace ID chưa đủ.

## 2. Data snapshot có thật sự cố định?

Pipeline ghi `dataset='tickets-october'`, query `created_at` trong tháng 10 và chạy lại ngày 5/11. Hôm qua có 1.000 nhãn, hôm nay có 1.080 nhãn và 20 nhãn cũ được sửa.

1. Tên tháng và query có đủ để tái tạo lần đánh giá hôm qua?
2. Đề xuất snapshot/label/cutoff/split metadata cần có.
3. Cần xử lý gì khi dữ liệu thuộc diện phải xóa?

## 3. Hash và canonicalization

Không chạy trước: dự đoán quan hệ digest cho `{'a':1,'b':2}` và `{'b':2,'a':1}`, rồi cho `{'a':1}` và `{'a':1.0}` trong quy ước bài. Viết code kiểm, thử NaN và giải thích vì sao digest không phải chứng nhận tính đúng của model.

## 4. Threshold và feature order

Với `z=4x₁+x₂−2`, x=[.6,0], tính z, sigmoid và label cho threshold .6/.5. Đảo thứ tự feature, tính lại. Vì sao shape test chưa đủ? Đề xuất golden case cho feature thứ hai.

## 5. Bundle và lineage

Sáu thành phần, mỗi thành phần hai version: có bao nhiêu tổ hợp? Nếu chỉ hai bundle đã được kiểm, có thể kết luận gì về các bundle khác? Build baseline/candidate, liệt kê component digest nào đổi và component nào dùng lại. Phân biệt source snapshot thật trong lab với một Git commit ID do người dùng tự gõ.

## 6. CI tổng thể và theo slice

Baseline đúng9/10. Candidate đúng10/10. Slice-regression mobile đúng1/2, web đúng8/8.

1. Tính accuracy overall, macro average giữa hai slice và gain của candidate bằng điểm phần trăm.
2. Áp policy của bài cho từng release, nêu gate thất bại.
3. Vì sao 10/10 validation chưa chứng minh candidate luôn tốt hơn baseline?

## 7. Report giả và artifact khác

Một team sửa `metrics.overall.accuracy=1` trong file report thay vì chạy model. Team khác chấm artifactA rồi build artifactB cùng tên `candidate` để deploy. Lab có thể chặn hai lỗi này ở đâu? Dùng mutation để làm sai stored metrics và kiểm CI không PASS.

## 8. Leakage và validation không cùng bản

Candidate được đánh giá trên validation đã bỏ những ticket khó, baseline dùng validation cũ. Một record validation còn trùng group với train. Giải thích hai loại sai và gate tương ứng. Nêu hai dạng leakage khác mà check ID/group đơn giản chưa bắt.

## 9. CI bằng process thật

Từ thư mục bài, build ra thư mục mới. Chạy `ci_runner.py` với candidate và slice-regression; ghi exit code bằng Python subprocess. Kiểm release manifest trong registry không đổi sau CI; candidate có approval khi dùng --approve, slice-regression không có. Không coi dòng text PASS do một script bất kỳ in ra là bằng chứng đủ.

## 10. Stale controller và ABA

State lần lượt `(baseline,g1)→(candidate,g2)→(baseline,g3)`. Controller cũ giữ expected_generation=1, target=candidate.

1. Tại sao chỉ so release name dễ bỏ sót thay đổi?
2. Dự đoán kết quả activate và state sau lỗi.
3. Chạy hai controller cùng expected1; phải có bao nhiêu commit và audit event mới?

## 11. Fault giữa pointer và audit

Đã active baseline/g1, candidate đã được approve. Inject `after_switch`. Trước khi chạy, dự đoán pointer/audit sau exception. Nếu pointer và audit ở hai file khác nhau, chỉ dùng `os.replace` từng file có đủ giữ cùng tính chất không?

## 12. Request đang chạy và rollback

Pin handle baseline/g1; promote candidate/g2; pin handle candidate; rollback baseline/g3. Cả ba handle/request cùng nhận x=[.6,0]. Liệt kê release/generation/priority của handle cũ, handle candidate và request pin sau rollback. Có nên tự thay weight của handle đang chạy không?

## 13. Corruption sau approval

Approve candidate rồi cố tình sửa bytes object model qua SQL, giữ nguyên object digest key. Dự đoán `load`, `gate` và `activate`. State cũ có được giữ không? Nếu attacker có quyền sửa cả controller và DB thì kiểm hash trong bài còn đảm bảo gì?

## 14. Offline PASS, online suy giảm

Validation candidate10/10. Cohort sau release có năm x=.6,y=0 và năm x=.9,y=1, error_ratio=0.

1. Tính accuracy candidate/baseline trên cohort này.
2. Phác runbook điều tra và rollback, gồm observed version, labels và in-flight.
3. Vì sao không coi đây là canary production hoặc mười mẫu thống kê độc lập?

## 15. Rollback có giới hạn

Ba tình huống: model mới đã gửi email sai; DB migration đã xóa field model cũ cần; policy mới chặn environment của baseline.

Đổi pointer có sửa được mọi thứ không? Nêu compensation/compatibility/fallback hoặc forward-fix phù hợp, quyền cần có và bằng chứng phải kiểm trước khi hành động. Phân biệt rollback software state với hoàn tác external effect.

## 16. Thiết kế release cho RAG và ước lượng canary

1. Đề xuất manifest cho RAG, tối thiểu generator/tokenizer/prompt/embedding/index/reranker/policy, cùng provenance/eval cần thiết. Đổi embedding mà giữ index dimension bằng nhau đã đủ tương thích chưa?
2. 10.000 request, canary5%, candidate error2%: tính kỳ vọng request/lỗi. Traffic120 request/giây, detection+switch15 giây: tính exposure thô. Vì sao các số không là upper bound?
3. Đề xuất ba gate và ba kiểm sau rollout chưa được lab hiện tại thực hiện. Nêu cách kiểm rollback trước khi release thật.
