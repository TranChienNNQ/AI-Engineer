# Bài tập — Hồi quy và phân loại

[Về bài học](README.md) · [Lời giải](solutions.md)

Dữ liệu và chi phí là giả định. Tính tay trước rồi chạy code. Lệnh từ thư mục bài 6 với Python có NumPy. Bài mở rộng có thể quay lại sau; không cần trả lời trong chat để xem lời giải.

## Bài 1. Đích nào, đầu ra nào?

Với mỗi nhu cầu, đề xuất kiểu bài toán và một thiết kế đầu ra/loss cơ bản:

1. Dự đoán thời gian hoàn tất một job, đơn vị giây.
2. Chọn đúng một nhóm routing trong ba nhóm billing/technical/account.
3. Gán nhiều chủ đề cùng lúc cho một ticket.
4. Dự đoán số request trong 5 phút tới.
5. Nhãn mức ưu tiên được lưu bằng mã 0,1,2. Có được mặc định dùng MSE chỉ vì y là số nguyên không?

Nếu đổi nhiệm vụ từ dự đoán latency sang “latency có vượt 100 ms không?”, đích và ý nghĩa phép đo đã thay đổi thế nào?

## Bài 2. Một bước hồi quy với điểm khởi tạo khác

Dùng hai cặp `(1,3),(2,5)`, khởi đầu w=1,b=1, không L2, lr=0.1.

1. Tính dự đoán, MSE, MAE và hai gradient.
2. Cập nhật đồng thời w,b; tính dự đoán, MSE và RMSE mới.
3. Nêu đơn vị nếu y là ms.

```python
import numpy as np
from lab import loss_and_gradients
X=np.array([[1.],[2.]])
y=np.array([3.,5.])
w,b=np.array([1.]),1.0
loss,obj,dw,db=loss_and_gradients('regression',X,y,w,b)
w,b=w-0.1*dw,b-0.1*db
print(loss,dw,db,w,b)
print(loss_and_gradients('regression',X,y,w,b)[0])
```

## Bài 3. Logit, odds, xác suất và nhãn

Cho z=wx+b với w=ln3,b=0, x lần lượt −1,0,1.

1. Tính ba logit, odds và xác suất sigmoid.
2. Gán nhãn với τ=0.5 và τ=0.8, quy tắc p≥τ → 1.
3. Tăng x từ 0 lên 1 rồi từ 1 lên 2: odds nhân mấy lần? Xác suất có tăng cùng một lượng không?
4. Vì sao biết p=0.75 chưa đủ chứng minh calibration của mô hình tốt?

## Bài 4. BCE giảm, accuracy có nhất thiết tăng ngay?

Train gồm `(x,y)=(0,0),(2,1)`. Khởi đầu w=0,b=0, không L2, lr=1.

1. Tính BCE ban đầu, hai gradient và cập nhật đầu tiên.
2. Tính xác suất/BCE sau cập nhật.
3. Với ngưỡng 0.5 và quy tắc hòa gán 1, accuracy trước/sau là bao nhiêu?
4. Vì sao kết quả không mâu thuẫn với việc tối ưu BCE?

```python
import numpy as np
from lab import loss_and_gradients, sigmoid
X=np.array([[0.],[2.]])
y=np.array([0.,1.])
w,b=np.zeros(1),0.0
loss,_,dw,db=loss_and_gradients('binary',X,y,w,b)
w,b=w-dw,b-db
p=sigmoid(X@w+b)
print(loss,w,b,p)
print(loss_and_gradients('binary',X,y,w,b)[0],np.mean((p>=0.5)==y))
```

## Bài 5. Regularization đúng quy ước

Cho data loss=0.7, w=(2,−1), b=3, λ=0.2. Mục tiêu của lab là $J=L_{data}+(\lambda/2)\|w\|^2$, không phạt bias.

1. Tính penalty và objective J.
2. Gradient theo w được cộng thêm vector nào? Gradient bias thêm bao nhiêu?
3. Vì sao không mặc định dùng nguyên λ=0.2 cho thư viện có định nghĩa penalty hoặc tổng loss khác?
4. Khi chọn λ, lab so validation data loss hay objective train có penalty? Vì sao không chọn bằng test?

## Bài 6. Số học ở các logit cực trị

```python
import numpy as np
from lab import sigmoid, data_loss, softmax
z=np.array([-1000.,0.,1000.])
y=np.array([1.,0.,0.])
print(sigmoid(z))
print(data_loss('binary',y,z))
print(softmax(np.log([1.,2.,3.])))
print(softmax(np.log([1.,2.,3.])+1000))
```

1. Vì sao sigmoid có thể in ra 0 hoặc 1 nhưng BCE từ logits vẫn hữu hạn ở ví dụ này?
2. Tính BCE trung bình gần đúng bằng tay.
3. Chứng minh cộng một hằng số vào mọi logit không đổi softmax.
4. Vì sao không chuẩn hóa tất cả mẫu trong batch thành một tổng xác suất duy nhất khi nhiệm vụ là phân loại riêng từng mẫu?

## Bài 7. Quyết định có chi phí

Dùng nhãn validation `[0,1,1,0]`, xác suất `[0.2,0.4,0.6,0.8]`, chi phí FP=1, FN=3.

1. Tính nhãn, FP/FN, chi phí và accuracy với τ=0.3,0.5,0.7.
2. Theo ba ứng viên nên chọn ngưỡng nào và bằng tập nào?
3. Nếu p thật sự đáng tin cậy và chi phí đúng là hằng số trên, suy ra ngưỡng kỳ vọng từ so sánh hai hành động.
4. Ở p=0.4, chi phí kỳ vọng của hai hành động là bao nhiêu? Những giả định nào có thể làm quy tắc không còn tối ưu thực tế?

Kiểm tra `python3 lab.py --mode threshold`.

## Bài 8. Đa lớp khác nhiều nhãn

Thứ tự lớp `[billing,technical,account]`, logits `[0,ln2,ln3]`.

1. Tính softmax và nhãn argmax.
2. Nếu lớp thật là technical, categorical cross-entropy là bao nhiêu?
3. Tính các sigmoid riêng; vì sao tổng lớn hơn 1 không phải lỗi trong nhiệm vụ nhiều nhãn?
4. **Mở rộng:** nếu ba nhãn nhị phân thật là `[1,0,1]`, tính mean BCE theo các sigmoid riêng ấy. Đây có cùng ý nghĩa với categorical cross-entropy ở câu 2 không?

Code để tự đối chiếu:

```python
import numpy as np
from lab import softmax, sigmoid, categorical_loss, data_loss
z=np.log([1.,2.,3.])
print(softmax(z),categorical_loss(z,1))
print(sigmoid(z),data_loss('binary',np.array([1.,0.,1.]),z))
```

## Bài 9. Kiểm tra ranh giới học và đánh giá

Chạy mode `regression` và `classification` với mặc định 3000 bước, lr=0.1.

1. Tập nào cung cấp mean/SD? Tập nào chọn λ? Test được dùng lúc nào?
2. Vì sao `model_scores` của binary chưa phải xác suất? Cần thao tác nào trước khi áp ngưỡng?
3. Hệ số được in là `raw-feature coefficients`; vì sao khác trọng số lưu trên feature đã chuẩn hóa nhưng vẫn mô tả cùng dự đoán?
4. Một người chọn lại λ/ngưỡng vì test accuracy chưa cao. Việc đó thay đổi tính độc lập của đánh giá ra sao?
5. Baseline xác suất của binary lấy tỷ lệ lớp 1 ở đâu? Có được dùng tỷ lệ nhãn test để fit baseline rồi báo là baseline giữ riêng không?

## Bài 10. Giới hạn biểu diễn và kết luận

1. Test logistic trong bài có 4 mẫu và accuracy 100%. Nêu hai kết luận chưa thể rút ra về hệ thống thật.
2. **Mở rộng:** với ngưỡng 0.5, logistic trên đặc trưng gốc x₁,x₂ có thể phân tách hoàn hảo XOR sau không?

| x₁ | x₂ | y |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

Thử suy từ các bất đẳng thức của z=w₁x₁+w₂x₂+b. Sau đó xét feature mới x₁x₂ và logit $z=x_1+x_2-2x_1x_2-0.5$. Nó dự đoán gì? Vì sao không mâu thuẫn với việc logistic có biên tuyến tính trong **không gian feature mà nó nhận**?

## Tiêu chí tự đánh giá

Phân biệt được kiểu đích và đầu ra; tính được cập nhật ở bài 2/4; giải thích được gradient BCE p−y; không nhầm probability với nhãn hay calibration; giữ đúng validation/test. Bài XOR và multi-label BCE là mở rộng, không cần giải hết để bắt đầu bài 7. Đối chiếu [lời giải](solutions.md) theo từng phần đang vướng.
