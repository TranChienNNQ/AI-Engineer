# Bài tập — Cây quyết định, boosting và baseline

[Về bài học](README.md) · [Lời giải](solutions.md)

Dữ liệu đều giả định. Tính tay trước, chạy code sau. Quy ước: gốc depth=0, x≤t sang trái, p≥0.5 gán nhãn 1. Khi split hòa trong 1e−12, lab giữ feature/ngưỡng gặp trước theo thứ tự tăng. Code dưới chạy từ thư mục bài 8, với NumPy.

## Bài 1. Baseline đúng với mục tiêu

Train y=`[1,1,3,5]`.

1. Tìm hằng c tối thiểu train MSE bằng đạo hàm. Tính MSE đó.
2. Dùng median của NumPy, c=2, tính MAE. So với MAE của c=2.5. Vì sao hai c khác nhau vẫn có thể cùng MAE tối ưu?
3. Nếu train có 98 negative,2 positive, baseline xác suất là bao nhiêu? Nhãn majority là gì? Có được lấy tỷ lệ positive test để fit baseline không?
4. Vì sao chỉ so mô hình phức tạp với baseline rất yếu có thể chưa đủ cho quyết định sản phẩm?

## Bài 2. Tính Gini và chọn split

X một feature `[1,2,3,4,5,6]`, y=`[0,0,1,0,1,1]`.

1. Tính Gini gốc, Gini con và điểm có trọng số tại t=1.5,2.5,3.5,4.5,5.5.
2. Split nào được chọn theo quy tắc lab? Tính hai xác suất lá và nhãn dự đoán.
3. Nếu dùng trung bình không trọng số Gini hai con tại t=2.5, thu được số nào? Vì sao khác điểm đúng?
4. Với hai hàng có x giống nhau nhưng nhãn khác nhau, có thể tách riêng chúng bằng một ngưỡng trên x ấy không?

```python
from lab import hand_classification, split_candidates, SmallTree
X,y = hand_classification()
for row in split_candidates(X,y,'binary'):
    print(row['threshold'],row['score'],row['gain'])
print(SmallTree('binary',1,1).fit(X,y).predict(X))
```

## Bài 3. Đi qua cây và giới hạn lá

Dùng stump ở bài 2.

1. Dự đoán p và nhãn cho x=2,2.5,2.6,10.
2. Với p=0.75, nếu đổi ngưỡng xác suất thành 0.8 thì nhãn đổi thế nào? Cây có phải fit lại không?
3. Với `min_leaf=3`, những split nào còn hợp lệ? Với `min_leaf=4`, cây có thể split sáu mẫu không?
4. `max_depth=0` trả p gì? p=1 ở một lá chỉ có một positive có đủ kết luận chắc chắn ngoài mẫu không?

## Bài 4. Cây hồi quy và ngoại suy

Dùng `(1,1),(2,1),(3,3),(4,5)`.

1. Tính mean gốc và train SSE/MSE.
2. Tính mean hai con và SSE tại t=1.5,2.5,3.5; chọn split tốt nhất.
3. Cây depth=1,min_leaf=1 dự đoán gì cho x=0,2.5,3,10?
4. Đối chiếu cây depth=2,min_leaf=1 trên bốn điểm train: MSE có thể về 0 không? Điều đó đã đủ đánh giá ngoài mẫu chưa?
5. Vì sao một cây lá mean không trọng số không dự đoán vượt min/max y_train? Có được áp dụng nguyên phát biểu này cho tổng cây residual không?

```python
from lab import hand_regression, SmallTree, mse
X,y = hand_regression()
for depth in (0,1,2):
    model = SmallTree('regression',depth,1).fit(X,y)
    print(depth,model.predict(X),mse(y,model.predict(X)))
```

## Bài 5. Hai vòng boosting

Dùng bốn điểm bài 4, F₀=mean train, stump min_leaf=1,η=0.5.

1. Tính r₁=y−F₀, split và giá trị lá h₁.
2. Tính F₁ và MSE.
3. Tính r₂=y−F₁. Vì sao không dùng lại residual ban đầu?
4. Tính split và hai giá trị lá h₂, rồi F₂ và MSE.
5. Nếu một người định nghĩa residual=y−F nhưng cập nhật `F -= eta*h`, vấn đề gì xảy ra? Tính MSE sau vòng đầu với dấu sai.

```python
from lab import hand_regression, SquaredBoost, mse
X,y = hand_regression()
model = SquaredBoost(rounds=2,learning_rate=0.5,min_leaf=1).fit(X,y)
for stage,pred in enumerate(model.staged_predict(X)):
    print(stage,pred,mse(y,pred))
```

## Bài 6. Learning rate và số vòng

Vẫn dữ liệu bốn điểm trên, fit **một** stump residual từ F₀.

1. Tính dự đoán và train MSE sau một vòng với η=0.25,0.5,1.
2. Vì sao η=1 tốt nhất trong ba lựa chọn cho đúng một vòng này, nhưng chưa đủ kết luận η=1 tốt nhất trên dữ liệu mới?
3. Giải thích vì sao train MSE không tăng với bước lá mean residual và 0<η≤1 trong số học lý tưởng. Đây có phải bảo đảm validation/test cũng không tăng không?
4. Chọn checkpoint bằng test loss sau mỗi vòng có còn là đánh giá độc lập không? Lab đã có patience early stopping hay chỉ có lưới M định trước?

```python
from lab import hand_regression, SquaredBoost
X,y = hand_regression()
for eta in (0.25,0.5,1.0):
    model = SquaredBoost(1,eta,min_leaf=1).fit(X,y)
    print(eta,model.predict(X),model.train_history[-1])
```

## Bài 7. Đọc so sánh validation

Chạy `python3 lab.py --mode comparison`.

1. Có bao nhiêu ứng viên? Chúng dùng feature và phép chia nào?
2. Ứng viên nào thắng validation MSE? So train MSE của nó với boost M=200.
3. Vì sao train error thấp nhất không đủ chọn mô hình? Tại sao không đổi metric sau khi thấy linear thắng?
4. Những mô hình nào được chấm test trong code? Vì sao không có test của mọi cây/boosting?
5. Mô tả ba giới hạn khi muốn suy từ dữ liệu synthetic này ra một hệ thống latency thật.

## Bài 8. Cây biểu diễn được nhưng greedy chưa chắc tìm được — mở rộng

XOR: `(0,0)→0,(1,0)→1,(0,1)→1,(1,1)→0`.

1. Tính Gini gốc và gain khi chia x₁≤0.5 hoặc x₂≤0.5.
2. Lab `SmallTree('binary',2,1)` có split không? Vì sao tăng max_depth lên 4 chưa giải quyết được điều kiện dừng này?
3. Viết quy tắc depth=2 biểu diễn XOR bằng tay. Nó mâu thuẫn với câu 2 không?
4. Có được khẳng định mọi thư viện cây đều cho cùng kết quả lab ở đây không?

Kiểm tra `python3 lab.py --mode limits`.

## Bài 9. Một cây, forest và boosting phân loại

1. So sánh bagging/random forest điển hình với gradient boosting về dữ liệu fit mỗi cây, sự phụ thuộc giữa cây và cách gộp đầu ra.
2. Vì sao huấn luyện năm cây y hệt nhau rồi lấy mean chưa tự có lợi ích của tính đa dạng?
3. Với binary log loss, F là logit, p=sigmoid(F). Nếu y=`[0,1]`,p=`[0.2,0.7]`, âm gradient theo F là gì? Cây tiếp theo cần là cây hồi quy hay cây trả nhãn rời rạc?
4. Có thể đưa y nhị phân vào `SquaredBoost` rồi gọi đó là boosting tối ưu log loss không? Phân biệt khả năng chạy số học và đúng mục tiêu.

## Bài 10. Rà một đề xuất dùng cây

Một người đề xuất: “Cây không cần preprocessing nên đưa thẳng NaN và mã nhóm 0/1/2 vào; cây sâu nhất có train MSE=0 nên chọn luôn; nó dự đoán tốt cả tải tương lai gấp 10; feature xuất hiện ở root là nguyên nhân gây latency.”

1. Chỉ ra vấn đề trong từng suy luận.
2. Nếu cần impute/encode, fit các bước ấy ở đâu khi có train/validation/test hoặc CV?
3. Dùng cây hồi quy bài 4, đổi đơn vị x thành 100x rồi fit lại. Dự đoán trên input đã đổi tương ứng có giữ nguyên không? Nêu giới hạn số học/phạm vi phát biểu.
4. Viết báo cáo 5–8 câu cho thí nghiệm comparison: baseline, lưới, tiêu chí chọn, test và hạn chế; không gọi điểm synthetic là bằng chứng production.

```python
import numpy as np
from lab import hand_regression, SmallTree
X,y = hand_regression()
grid = np.array([[0.],[2.5],[3.],[10.]])
a = SmallTree('regression',1,1).fit(X,y)
b = SmallTree('regression',1,1).fit(100*X,y)
print(a.predict(grid),b.predict(100*grid))
```

## Tiêu chí tự đánh giá

Tính được split Gini có trọng số, lá mean và residual mới; phân biệt ngưỡng feature với ngưỡng xác suất; chọn mô hình theo development và giữ test ngoài tuning. XOR và nối gradient log loss là phần mở rộng. [Đối chiếu lời giải](solutions.md).
