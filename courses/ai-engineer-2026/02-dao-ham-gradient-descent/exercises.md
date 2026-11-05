# Bài tập — Đạo hàm và gradient descent

[Về bài học](README.md) · [Lời giải](solutions.md)

Đề bài dùng dữ liệu giả định như bài học. Tính tay hoặc viết dự đoán trước khi chạy code. Những bài mở rộng được đánh dấu; không cần trò chuyện với agent để xem đáp án.

## Bài 1. Độ dốc trung bình và đạo hàm

Với $f(w)=(w-2)^2$, xét $w=1$.

1. Tính $f(1)$ và $f(1.1)$.
2. Tính $[f(1.1)-f(1)]/0.1$.
3. Tính đạo hàm đúng $f'(1)$ bằng công thức đã suy ra.
4. Vì sao hai kết quả ở câu 2 và 3 chưa hoàn toàn bằng nhau? Dấu đạo hàm gợi ý thay đổi $w$ theo hướng nào?

## Bài 2. Tự cập nhật một tham số

Vẫn dùng $f$, bắt đầu $w_0=1$, $\alpha=0.1$.

1. Tính $w_1$ và $f(w_1)$.
2. Tính tiếp $w_2$ và $f(w_2)$; phải tính lại đạo hàm tại điểm mới.
3. Một người viết $w_{t+1}=w_t+\alpha f'(w_t)$. Tính một bước từ $w_0=1$ và giải thích lỗi.

## Bài 3. Learning rate và điều kiện hội tụ

Bắt đầu $w_0=5$ với cùng hàm $f$.

1. Với $\alpha=0.5$, cần mấy bước tới nghiệm?
2. Với $\alpha=1$, tính $w_1,w_2$ và hai loss tương ứng.
3. Với $\alpha=1.1$, tính $w_1,w_2$ và hai loss tương ứng.
4. Chạy `python3 lab.py --mode compare` từ thư mục bài 2. Đối chiếu hướng thay đổi sau 12 bước.
5. **Mở rộng:** suy ra khoảng learning rate hội tụ từ $e_{t+1}=(1-2\alpha)e_t$. Vì sao kết quả này không được áp dụng nguyên xi cho hồi quy hai tham số?

## Bài 4. Gradient hồi quy từ một điểm khởi tạo khác

Dùng `(1, 3), (2, 5), (3, 7)` và loss MSE chia cho 3. Bắt đầu $w_0=0,b_0=0$, $\alpha=0.05$.

1. Tính ba sai số, tổng $\sum r_i$, tổng $\sum r_ix_i$ và loss ban đầu.
2. Tính hai đạo hàm riêng.
3. Cập nhật đồng thời $w,b$, tính các dự đoán mới và loss mới.
4. Tự kiểm tra bằng cách nhập `lab.py` trong Python:

```python
from lab import TRAIN, train_linear
print(train_linear(TRAIN, lr=0.05, steps=1, start_w=0, start_b=0))
```

## Bài 5. Tìm lỗi cập nhật

Xét đoạn sau, chạy với dữ liệu bài 1, $w=1,b=0$, `lr=0.1`:

```python
dw, _ = gradients(data, w, b)
w = w - lr * dw
_, db = gradients(data, w, b)
b = b - lr * db
```

1. Đoạn code có thực hiện đúng một bước full-batch gradient descent đang học không?
2. Tính $b$ mà nó tạo ra và so với 0.6 trong bài học.
3. Sửa code để cả hai gradient thuộc cùng trạng thái. Giải thích vì sao tách phép gán thành hai dòng không tự nó gây sai.

## Bài 6. Gradient checking

Chạy `python3 lab.py --mode check`. Sau đó đổi điểm kiểm tra bằng đoạn sau:

```python
from lab import TRAIN, gradients, numerical_gradients
print(gradients(TRAIN, 2.3, -0.7))
print(numerical_gradients(TRAIN, 2.3, -0.7))
```

1. Ghi hai cặp kết quả và độ lệch gần đúng.
2. Nếu công thức giải tích quên hệ số 2 ở cả hai gradient, phép kiểm tra có thể phát hiện không?
3. Vì sao so sánh bằng `==` giữa kết quả giải tích và kết quả số là lựa chọn kém phù hợp?
4. Vì sao kiểm tra thành công ở vài điểm chưa phải chứng minh đúng mọi nơi?

## Bài 7. Chuyển đơn vị đầu vào

**Mở rộng.** Đổi từ số lô sang số bản ghi: $x_i'=100x_i$. Để mô hình biểu diễn cùng dự đoán tại điểm kiểm tra, chọn $w'=w/100$, giữ $b$ nguyên.

1. Chứng minh $w'x_i'+b=wx_i+b$.
2. Với các sai số bằng nhau, gradient theo $w'$ gấp mấy lần gradient theo $w$? Gradient theo $b$ đổi không?
3. Có thể giữ nguyên learning rate và mặc định việc học có hành vi tương đương không?

## Bài 8. Đọc hình và đánh giá kết luận

1. Ở hình 3 của bài học, trục ngang bên trái là gì? Trục ngang bên phải là gì? Vì sao đường đi trên mặt phẳng tham số không nhất thiết là đường thẳng tới nghiệm?
2. Loss cuối rất gần 0 có chứng minh mô hình dự đoán tốt trong mọi điều kiện tải thực tế không?
3. Một đồng nghiệp nói: “MSE hồi quy là hàm lồi, nên chọn learning rate bất kỳ cũng tới nghiệm.” Dùng lệnh dưới để kiểm tra và phản biện:

```bash
python3 lab.py --mode linear --lr 0.2 --steps 20
```

## Tiêu chí tự đánh giá

Cần giải được bài 1–2, tính được các tổng trong bài 4, tìm ra vấn đề ở bài 5 và phân biệt được hội tụ với tổng quát hóa. Bài 3 phần mở rộng và bài 7 giúp hiểu sâu điều kiện hội tụ và thang đo; có thể quay lại sau nếu đại số còn khó. Xem [lời giải](solutions.md) để tự đối chiếu từng bước.
