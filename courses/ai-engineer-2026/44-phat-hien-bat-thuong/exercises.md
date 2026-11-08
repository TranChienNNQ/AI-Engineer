# Bài tập 44 — Phát hiện bất thường

[← Bài chính](README.md) · [Phụ lục](deep-dive.md) · [Lời giải — đọc sau khi làm](solutions.md)

Mọi dữ liệu nhỏ dưới đây đều là ví dụ giả định. Với bài thiết kế, ghi rõ giả định, đơn vị, timestamp và population. Chạy Python từ thư mục bài 44.

## 1. Định nghĩa anomaly trong hệ backend

Latency tăng từ 100 lên 180 ms. Đặt hai tình huống: một normal theo context, một cần điều tra. Phân biệt unusual với harmful. Nêu target/entity/frequency/reference và một quyết định mà score chưa đủ để tự thực hiện.

## 2. Z-score, IQR và MAD tính tay

Mean 100/std 10, observation 140 có score gì? Q1=10/Q3=14 cho fence 1.5 IQR nào? Với `[10,11,12,13,14]`, tính median/MAD/modified absolute z của 20. Nếu MAD 0 thì cần làm gì? Một fence ngoài sample có đồng nghĩa probability anomaly 95% không?

## 3. Novelty, outlier và supervision

Phân loại: train chỉ các khoảng vận hành đã review bình thường; train log bị trộn chưa review; train classifier từ nhãn lỗi/không lỗi. Lab thuộc thiết lập nào? Vì sao gọi toàn pipeline hoàn toàn không dùng nhãn là không chính xác? Nêu một cách contamination làm model bỏ lỗi.

## 4. Mahalanobis tính tay

Mean 0, covariance `[[1,.9],[.9,1]]`. Tính score bình phương cho `[1,−1]` và `[1,1]`. Từng feature có cùng magnitude 1, vì sao khác? Regularization covariance giải quyết gì và không giải quyết gì?

## 5. kNN novelty score

Reference `[0,1,2,10]`, query 3, k 2 cho mean-nearest-distance gì? Nếu query là chính một điểm train, self-neighbor làm score thay đổi thế nào? Giải thích cần scaling và vì sao kNN không đồng nhất LOF.

## 6. Bắt lỗi leakage

Chỉ ra vấn đề trong: fit scaler toàn bộ 800 rows; chọn q bằng test F1; thêm mọi test point vào kNN reference trước khi chấm nó; dùng centered smoothing nhìn 2 giờtương lai nhưng ghi timestamp phát hiện ởgiữa window. Thiết kế một mutation check cho fit hoặc score causal.

## 7. PCA và reconstruction

v=`[1,1]/√2`, z=`[2,0]`. Tính projection/residual/squared error. Làm tương tự z=`[10,10]`. Vì sao autoencoder cũng có thể gặp reconstruction thấp cho anomaly? Đề xuất một score bổ sung và cách kiểm nó.

## 8. Protocol của track bảng

Liệt kê fit/calibration/validation/test và quyết định ở mỗi phase. Vì sao calibration lại đứng trước validation ở bài này? Nếu refit trêntrain+validation thì q cũ có dùng luôn được không? Bảng test Mahalanobis tốt hơn kNN có cho phép đổi selected family không?

## 9. Ngưỡng, tie và nominal rate

Calibration scores 1…19, α=.1. Tính rank/q và flag cho score 18/19 với strict`>`. Với 200 calibration và α=.05 có rank gì? Tối đa bao nhiêu calibration points vượt q? Vì sao không suy FPR tương lai chắc chắn 5%? n 1/α=.05 phải xử lý gì theo helper?

## 10. Phát hiện sau quan sát và context

Ở thời điểm t, detector được dùng y_t không? Phân biệt với bài 43. Plan cho t phát hành ởt−48 và actual y_(t+1)được dùng thế nào? Vì sao seasonal-naive anomaly score có thể tạo echo? Có thể coi mọi high residual là incident thật không?

## 11. CUSUM tính tay và trạng thái

r=`[1,1,1,1]`, k=.5, G ban đầu 0: tính G⁺/G⁻/score. Nếu chia thành hai chunk mà reset state thì phần sau khác gì? Vì sao không reset khi alarm có thể giữ score cao lâu? Cần kiểm gì nếu đổi reset policy?

## 12. Confusion matrix, AP và base rate

Labels 0/1/0/1/0, scores.1/.9/.8/.7/.2, q.75: tính confusion/precision/recall/F1/FPR/AP. Nếu mọi score bằng nhau, AP có được phụ thuộc thứ tự rows không? Với prevalence.001, TPR.9,FPR.01, tính precision kỳ vọng. Vì sao FPR 1% không phải precision 99%?

## 13. Point adjustment và event delay

Một incident 10 points, detector chỉ hit điểm cuối, không hit nơi khác. Tính point recall, event recall và delay theo onset. Nếu tô cả 10 points là detected thì kết quả bị thay đổi gì? Nêu ít nhất hai quy tắc phải chốt khi match firing với incident.

## 14. Alert policy tính tay

Với hits`[0,1,1,1,0,0,0,1,1,1,1,1]`, persist 2/clear 3/cooldown 4, tính firing và lúc đóng. Nếu chỉ có spike đơn, policy bỏ lỡ thế nào? Tại sao một alert active trước một event mới không được tự coi là new firing trong hợp đồng lab?

## 15. Bundle, checkpoint và replay

Viết code fit/save/load một detector bảng và so score exact. Chia hits bài 14 thành hai chunk, serialize state sangJSON, kiểm replay giống toàn chuỗi. Còn thiếu state nào cho temporal seasonal/CUSUM? Tại sao digest và checkpoint hiện tại chưa giải quyết authenticated publisher hoặc exactly-once notification?

## 16. Đọc kết quả và thiết kế mở rộng

Giải thích vì sao PCA bắt off tốt nhưng yếu along; vì sao calendar raw có 9 FP nhưng policy có 0 false firing vẫn chưa hoàn hảo; và vì sao CUSUM raw event recall cao vẫn kém hữu ích. Đề xuất thí nghiệm kNN/IsolationForest/AE hoặc adaptive baseline có development/test mới, metric và chi phí rõ ràng. Không cần cài thêm thư viện hoặc tải model.
