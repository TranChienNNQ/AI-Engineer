# Bài tập 43 — Dự báo chuỗi thời gian

[← Bài chính](README.md) · [Phụ lục](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Các dữ liệu nhỏ là ví dụ giả định. Ghi rõ index 0-based, origin, dữ liệu có thể dùng và đơn vị khi trả lời. Chạy code từ thư mục bài 43; không cần API/GPU.

## 1. Định nghĩa bài toán capacity

Bạn có log request theo giây và cần thuê thêm máy cho tuần tới. Đề xuất một target, frequency, L, H và origin. Vì sao dự báo tổng request/ngày không đủ để bảo đảm p99 latency? Liệt kê ít nhất hai thông tin thời gian phải lưu.

## 2. Baseline tính tay

Với lịch sử `[10,12,14,16]`, tính naive, mean bốn điểm và drift cho ba bước tới. Vì sao chưa tính được seasonal naive chu kỳ 7? Với tuần gần nhất `[11,21,31,41,51,61,71]`, hãy viết forecast 9 bước của seasonal naive.

## 3. Phân biệt trend, seasonality, missing và zero

Một cửa hàng có tăng trưởng chậm, cuối tuần bán cao hơn, nghỉ toàn bộ ngày T và mất log ngày T+1. Mô tả bốn loại tín hiệu; có nên điền 0 cả hai ngày cuối không? Nếu hết hàng thì sales có bằng demand không?

## 4. Tạo window và tính số hàng

Chuỗi `[1,…,9]`, L=3/H=2, origin 6 có X/Y gì? Với train prefix N=224, L=28/H=7, hãy tính origin đầu/cuối, số hàng và target cuối. Origin 220 có được train không? Sau refitN=280 có bao nhiêu hàng direct và one-step?

## 5. Bắt lỗi leakage

Đánh giá bốn thao tác: random split các window; scaler fit toàn 420 ngày; centered mean dùng y_(T−1:T+1) làm feature ởT; origin 230 dùng y 224…230 để dự báo 231…237. Thao tác nào sai cho protocol bài này, vì sao? Thiết kế một future-target mutation check.

## 6. Covariate biết trước

Origin 223, H 7. Calendar đã biết; kế hoạch ngày 224 công bố ở 220; kế hoạch ngày 225 công bố ở 224; nhiệt độ thực ngày 226; dự báo nhiệt độ ngày 226 phát hành ở 222. Biến nào có thể dùng? Mô tả xử lý khi thiếu biến bắt buộc. Viết code làm guard lab từ chối kế hoạch chưa có ởorigin.

## 7. Anchoring và scaler

Tại origin, y_T=50, ba lag cuối `[47,49,50]`, ba target `[51,48,53]`. Viết lag/residual sau anchoring; nếu output residual `[1,−2,3]` thì forecast mức là gì? Phân biệt anchoring với sai phân. Vì sao std floor 1 làm một số cột không có variance 1?

## 8. Ridge và MLP

Viết objective ridge multi-output với intercept không phạt. Đếm hệ số direct known H 7 và recursive known H 1. Tính parameter của `52→32→7` có bias. Vì sao λ trong sum-SSE không giữ nguyên ý nghĩa khi đổi thành mean-SSE? Nêu một lý do MLP có thể chưa hơn ridge.

## 9. Recursive tính tay và teacher forcing

Model một bước `1+.8*y`, giá trị cuối 10. Tính 3 bước. Nếu actual bước 1=20 rồi dùng 20 để tính bước 2, còn là cùng forecast 3 bước tại origin ban đầu không? Viết rollout không đọc future target.

## 10. Metric và case không xác định

Actual `[0,10,20]`, prediction `[2,8,25]`, scale 2. Tính MAE, RMSE, bias, WAPE, MASE; giải thích MAPE. Với train `[1,2,…,14]`, seasonal scale 7 là bao nhiêu? Train constant gây vấn đề gì? MASE<1 có bảo đảm thắng test seasonal naive không?

## 11. Lịch đánh giá và chọn model

Vì sao có 8 validation origin,12 calibration origin,8 test origin? Model/scaler refit cuối cùng ở mốc nào? MLP chọn epoch 131 thế nào? Trong bảng kết quả, nếu đổi chọn ridge direct vì stable-test MAE nhỏ hơn, bước nào bị vi phạm? Nếu muốn retrain mỗi tuần cần thay protocol ra sao?

## 12. Quantile và calibration tính tay

Pinball τ=.9, y 10, q 8 hoặc 12 có loss gì? Calibration absolute errors 1…9, α.2 cho rank và q gì? Point 20 có interval nào? Với 12 calibration origin trong lab, rankbao nhiêu và mỗi h có bao nhiêu scores? Vì sao không lấy 84 residual gộp rồi nói vẫn là calibration riêng từngh?

## 13. Interval score và coverage

Với y 5, interval[0,4], α.2, tính coverage/width/score. So interval[-100,100] cho cùng y: vì sao coverage riêng không đủ? Nếu coverage mỗi ngày 80%, có được khẳng định 80% cả 7 ngày không? Nêu giới hạn của khoảng lab khi shift xảy ra.

## 14. Đọc ablation và kết quả

Từ report đã chạy, so seasonal naive/ridge_lags/ridge_known/MLP. Kết quả có chứng minh riêng plan tạo toàn bộ cải thiện không? Test shift có width như stable nhưng coverage khác ra sao? Hãy phân biệt dữ kiện lab với giả thuyết cần kiểm tiếp bằng dữ liệu thật.

## 15. Checkpoint và forecast record

Viết một ví dụ save/load ridge/q bằng temporary directory, kiểm dự báo exact và kiểm hash mismatch bị từ chối. Đề xuất schema forecast record đủ tái dựng origin/horizon/target/version. Bundle này có đủ exact resume training hoặc chứng minh artifact được tác giả tin cậy gửi không?

## 16. Thử nghiệm mở rộng có kỷ luật

Đề xuất cách so lookback 14/28/56 hoặc thêm boosting/ETS mà không chọn bằng test hiện tại. Hãy chỉ rõ split/selection/calibration/refit, baseline/metric và chi phí cần đo. Nếu thêm nhiều cửa hàng, cần thay đổi gì để đánh giá cold series, scale khác nhau và tổng vùng? Trình bày kế hoạch có thể thực hiện, không cần tải dữ liệu hay chạy model ngoài phạm vi lab.
