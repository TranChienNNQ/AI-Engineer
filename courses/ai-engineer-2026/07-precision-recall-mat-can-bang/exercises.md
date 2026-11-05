# Bài tập — Precision, recall và mất cân bằng

[Về bài học](README.md) · [Lời giải](solutions.md)

Tất cả dữ liệu/chi phí là giả định. Quy ước mặc định: lớp positive=1, hàng thật, cột dự đoán, thứ tự `[0,1]`; score≥ngưỡng thì đoán 1. Tính tay trước, chạy code sau. Python ví dụ chạy từ thư mục bài 7 với NumPy.

## Bài 1. Đếm từ từng ticket

```text
Ticket: A B C D E F G H
y thật: 1 0 1 0 0 1 0 0
y đoán: 1 1 0 0 0 1 0 1
```

1. Ghi TP/FP/FN/TN theo tên ticket, lập ma trận 2×2.
2. Tính accuracy, precision, recall, specificity, FPR và F1.
3. Viết một câu diễn giải mỗi mẫu số precision/recall.
4. Đổi lớp quan tâm: coi “ticket thông thường” là positive, tức đảo cả nhãn thật và dự đoán. Precision/recall mới là bao nhiêu? Accuracy có đổi không?

```python
from lab import binary_confusion, binary_metrics
y = [1,0,1,0,0,1,0,0]
pred = [1,1,0,0,0,1,0,1]
print(binary_confusion(y,pred))
print(binary_metrics(binary_confusion(y,pred)))
```

## Bài 2. Accuracy cao hơn có thắng không?

Dùng 1000 ticket trong bài: 20 positive và 980 negative.

- Mô hình M: TN=950,FP=30,FN=5,TP=15.
- Baseline B: luôn đoán 0.

1. Tính accuracy, recall, F1 và balanced accuracy của hai quy tắc. Precision nào không xác định theo công thức?
2. Với FP=1,FN=10, tính tổng chi phí và số cảnh báo.
3. Nếu FN chỉ tốn 1 và FP vẫn tốn 1, quy tắc nào có chi phí thấp hơn? Điều đó nói gì về mục tiêu đánh giá?
4. Thêm 9000 negative được M dự đoán đúng: accuracy, precision, recall, F1 đổi thế nào? Đây có phải bằng chứng M bắt lớp hiếm tốt hơn không?

## Bài 3. Chia cho 0 và lớp vắng mặt

Với ba bảng `[[TN,FP],[FN,TP]]`:

1. A=`[[3,0],[2,0]]`.
2. B=`[[5,0],[0,0]]`.
3. C=`[[0,0],[2,3]]`.

Tính precision, recall, F1 và balanced accuracy, để `NaN` khi công thức không xác định. Sau đó chạy `binary_metrics(...,zero_division=0)` và giải thích số nào được thay theo quy ước. Vì sao lab vẫn giữ balanced accuracy=NaN khi thiếu một lớp thật?

```python
from lab import binary_metrics
for cm in ([[3,0],[2,0]], [[5,0],[0,0]], [[0,0],[2,3]]):
    print(binary_metrics(cm))
    print(binary_metrics(cm,zero_division=0))
```

## Bài 4. Chọn ngưỡng bằng validation

Dùng `validation_scores()` trong lab và chi phí FP=1,FN=5.

1. Tự lập bảng đếm tại τ=0.7,0.5,0.4, có tính trường hợp hòa score=ngưỡng.
2. Tính precision, recall, F1, accuracy và cost tại mỗi ngưỡng. Chọn ngưỡng theo cost trong ba ứng viên.
3. Vì sao ba accuracy bằng nhau nhưng quyết định không tương đương?
4. Nếu hai ứng viên cùng cost, code chọn theo quy tắc nào? Vì sao quy tắc cần định trước?
5. Sau khi chốt, chạy `python3 lab.py --mode threshold`. Có được đổi lại ngưỡng vì test precision=0.25 không? Nếu đổi, cần làm gì để đánh giá độc lập?

```python
from lab import validation_scores, choose_threshold
best, rows = choose_threshold(*validation_scores())
for row in rows:
    print(row['threshold'],row['tp'],row['fp'],row['fn'],row['tn'],row['cost'])
print('selected',best['threshold'])
```

## Bài 5. Precision có luôn giảm khi hạ ngưỡng?

Cho nhãn `[1,0,1]`, score `[0.9,0.8,0.7]`.

1. Tính precision/recall tại τ=0.9,0.8,0.7.
2. Dùng kết quả để phản biện câu “hạ ngưỡng luôn làm precision giảm”.
3. Đại lượng nào trong TP,FP,recall có tính đơn điệu khi hạ ngưỡng với score/nhãn cố định? Accuracy và F1 thì sao?

## Bài 6. PR, hòa điểm và AP — mở rộng

Cho `(y,score)` là `(1,.9),(0,.8),(1,.8),(0,.1)`.

1. Lập các điểm PR/ROC sau mỗi score khác nhau; tại ngưỡng .8 có mấy mẫu được chọn?
2. Tính AP theo tăng recall; tính diện tích hình thang PR cùng quy ước điểm đầu `(R=0,P=1)`.
3. Tính ROC-AUC bằng các cặp positive–negative, hòa được tính nửa.
4. Tại sao không được dùng thứ tự B/C trong file để cho một mẫu điểm .8 đi vào trước mẫu còn lại?
5. Thay tất cả score s bằng s³. AP/AUC có đổi không? Các score mới có tự được coi là xác suất đã hiệu chuẩn không?

```python
from lab import ranking_example, ranking_curves
y, scores = ranking_example()
for transformed in (scores, scores**3):
    result = ranking_curves(y,transformed)
    print(result['ap'],result['pr_trapezoid'],result['roc_auc'])
```

## Bài 7. Tỷ lệ lớp và precision

Giả định giữ nguyên TPR=0.8,FPR=0.02 ở một ngưỡng, N=10000.

1. Tính số TP,FP,FN,TN kỳ vọng khi π=10% và 1%.
2. Tính precision ở hai quần thể bằng bảng đếm rồi bằng công thức theo π.
3. Precision trên bộ test cân bằng 50/50 là bao nhiêu theo giả định này? Có thể báo đó là precision trên lưu lượng π=1% không?
4. Điều gì có thể sai nếu ngoài tỷ lệ lớp, phân bố đặc trưng điều kiện theo lớp cũng đổi?

## Bài 8. Trung bình đa lớp — mở rộng

Với thứ tự `[billing,technical,account]`, ma trận hàng thật/cột dự đoán:

```text
[[87,2,1],
 [ 4,3,1],
 [ 1,1,0]]
```

1. Tính support, precision, recall, F1 từng lớp.
2. Tính accuracy, macro-F1, weighted-F1, micro-F1 và macro-recall.
3. Chứng minh vì sao micro-F1=accuracy trong trường hợp một nhãn đầy đủ này bằng tổng TP/FP/FN.
4. Weighted-F1 gần 0.895 có đủ kết luận account được nhận diện tốt không? Nếu bỏ account khỏi danh sách khi tính macro để điểm cao hơn thì đã đổi câu hỏi đánh giá thế nào?

```python
from lab import multiclass_counts, multiclass_report
print(multiclass_report(multiclass_counts()))
```

## Bài 9. Class weight và ranh giới sampling

Train gồm 98 negative,2 positive. Mô hình chỉ xuất một xác suất hằng q.

1. Tìm q tối thiểu hóa BCE không trọng số bằng đạo hàm.
2. Cho positive weight=49,negative weight=1; dùng loss chia tổng trọng số. Tìm q mới. Khả năng xếp hạng từng mẫu có được cải thiện nhờ có feature mới không?
3. So sánh hai loss tối ưu từ `python3 lab.py --mode weights`. Có được coi loss số nhỏ hơn chứng minh quy trình tốt hơn nếu chúng dùng mục tiêu có weight khác nhau?
4. Một người tạo bản sao positive toàn bộ dữ liệu rồi mới chia train/test. Nêu hai vấn đề có thể xảy ra.
5. Với CV có scaler, sampler và model, mỗi bước phải fit ở đâu? Vì sao stratification không tự giải quyết user trùng hoặc dữ liệu tương lai?

## Bài 10. Viết báo cáo ra quyết định

Bạn nhận ghi chú: “Mô hình đạt precision 90%, accuracy 99%; đã cân bằng cả test, chỉ gán nhãn những ticket mô hình báo; chọn ngưỡng có test F1 cao nhất. Sẵn sàng triển khai.”

1. Nêu ít nhất bốn thông tin còn thiếu hoặc điểm khiến kết luận chưa có cơ sở.
2. Vì sao chưa thể suy recall toàn bộ lưu lượng từ những ticket đã được gán nhãn?
3. Viết báo cáo 5–8 câu cho **ví dụ thực hành bài 4**: nêu lớp positive, validation/test, ngưỡng/cost, kết quả test, baseline và giới hạn. Không biến dữ liệu tự tạo thành kết quả vận hành.

## Tiêu chí tự đánh giá

Giải thích được “trong những cảnh báo” khác “trong những positive thật”; tự đếm trước khi dùng metric; xử lý mẫu số 0 có chủ ý; chọn ngưỡng bằng development; không dùng cân bằng lại test để né lớp hiếm. PR/AP và trung bình đa lớp là phần mở rộng, có thể luyện lại sau. [Đối chiếu lời giải](solutions.md).
