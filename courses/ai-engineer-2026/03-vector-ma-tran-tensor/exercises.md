# Bài tập — Vector, ma trận và tensor

[Về bài học](README.md) · [Lời giải](solutions.md)

Ghi dự đoán shape và kết quả trước khi chạy code. Các số liệu là giả định. Các lệnh `python3 lab.py ...` dưới đây chạy từ thư mục bài 3 với Python có NumPy.

## Bài 1. Cùng một số phần tử, khác cách biểu diễn

Không chạy code trước, hãy ghi `shape`, `ndim`, `size` của:

```python
import numpy as np

a = np.array(7.0)
b = np.array([7.0])
c = np.array([[7.0]])
v = np.array([1.0, 2.0, 3.0])
row = v[None, :]
col = v[:, None]
```

`v.T` có trở thành `col` không? Vì sao một vector ba thành phần có thể có `ndim=1`? Kiểm tra bằng mode `shapes` rồi tự bổ sung ba trường hợp đầu.

## Bài 2. Các phép tính vector

Với $\mathbf a=(2,1)$, $\mathbf c=(1,2)$, tính:

1. $\mathbf a+\mathbf c$, $3\mathbf a$.
2. Tích từng phần tử và tích vô hướng; nêu loại đầu ra của mỗi phép tính.
3. Độ dài Euclid của $\mathbf a$.
4. Nếu thành phần đầu là ms còn thành phần sau là số request, vì sao không nên mặc định độ dài tính từ số thô là thước đo nghiệp vụ hợp lý?

## Bài 3. Nhân theo hàng–cột

Cho:

$$
X=\begin{bmatrix}1&2\\2&3\\3&2\end{bmatrix},\quad
W=\begin{bmatrix}2&1\\1&-1\end{bmatrix}
$$

1. Tính toàn bộ $XW$ bằng tay, trình bày ít nhất hai ô.
2. `X * W` có thực hiện được theo broadcasting không? Nếu có hoặc không, giải thích từ shape.
3. $WX$ có thực hiện được không?
4. **Mở rộng:** cho $A=\begin{bmatrix}1&2\\0&1\end{bmatrix}$ và $B=\begin{bmatrix}1&0\\3&1\end{bmatrix}$. Tính $AB,BA$ để kiểm tra tính giao hoán.

## Bài 4. Sổ ghi shape

Cho `X.shape=(4,3)`, `w.shape=(3,)`, `W.shape=(3,2)`, `b.shape=(2,)`, `r.shape=(4,)`.

Ghi shape hoặc lý do lỗi của: `X @ w`, `X * w`, `X @ W + b`, `X.T @ r`, `w.T`, `X.mean(axis=0)`, `X.mean(axis=1, keepdims=True)`.

**Mở rộng:** dự đoán shape của `(2,4,3) @ (3,5)` và giải thích mỗi trục còn lại đại diện cho điều gì nếu đầu vào là `(batch, vị trí, đặc trưng)`.

## Bài 5. Lỗi loss âm thầm

```python
import numpy as np

pred = np.array([[5.0], [8.0], [9.0]])
y = np.array([5.0, 8.0, 9.0])
loss = np.mean((pred - y) ** 2)
```

1. Tính shape và ma trận `pred-y` bằng tay.
2. Tính loss. Vì sao dự đoán đúng mà loss vẫn không bằng 0?
3. Đưa ra hai cách sửa có chủ đích về shape.
4. Chạy `python3 lab.py --mode broadcast`. Vì sao kiểm tra shape trước phép trừ có giá trị hơn việc chỉ nhìn loss có phải là một số hữu hạn?

## Bài 6. Vector hóa một bước gradient mới

Dùng $X$ ở bài 3 và $\mathbf y=(5,8,9)$. Khởi đầu **$\mathbf w=(0,0),b=0$**, learning rate 0.01.

1. Tính dự đoán, vector sai số, loss ban đầu.
2. Tính $X^T\mathbf r$, gradient theo hai trọng số và gradient bias.
3. Tính đồng thời $\mathbf w,b$ mới, rồi dự đoán và loss mới.
4. So sánh công thức vector hóa với vòng lặp bằng đoạn sau:

```python
import numpy as np
from lab import example_data, gradients, gradients_loop
X, y = example_data()
w = np.zeros(2)
print(gradients(X, y, w, 0.0))
print(gradients_loop(X, y, w, 0.0))
```

Vì sao công thức chia cho 3 thay vì 2? Dữ liệu giữ riêng có được đưa vào tổng này không?

## Bài 7. Giữ ý nghĩa pixel khi đổi trục

```python
import numpy as np

images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
nchw = images.transpose(0, 3, 1, 2)
wrong = images.reshape(2, 3, 2, 3)
```

1. Ghi ý nghĩa từng trục của `images` và `nchw`.
2. Tìm ba giá trị `images[1,0,2,1]`, `nchw[1,1,0,2]`, `wrong[1,1,0,2]`.
3. Ghi ma trận 2×3 của kênh 1, ảnh 1, trong cách biểu diễn NCHW.
4. `images.reshape(2,-1)` có shape nào? Mỗi hàng đại diện cho gì? Tổng số giá trị có đổi không?
5. `images.T` có chuyển NHWC thành NCHW không? `images.reshape(2,20)` có hợp lệ không?

Kiểm tra với `python3 lab.py --mode tensor` và hình 4 trong bài.

## Bài 8. Đúng shape nhưng sai schema

Mô hình ban đầu dùng thứ tự `[số lô, số truy vấn]`, `w=[2,1]`, `b=1`. Một dịch vụ mới trả `[số truy vấn, số lô]`, vẫn hai số trên một request.

1. Với ba mẫu trong bài, dự đoán sẽ thành gì nếu giữ nguyên trọng số?
2. Vì sao kiểm tra shape không phát hiện lỗi?
3. Đề xuất cách sửa và cách kiểm tra giao diện dữ liệu giữa hai dịch vụ.

## Bài 9. Bộ nhớ của tensor

Một tensor số thực có shape `(32,128,768)`, dtype `float32`.

1. Có bao nhiêu trục và bao nhiêu phần tử?
2. Phần dữ liệu cần bao nhiêu byte và MiB? Dùng $1\text{ MiB}=1024^2$ byte.
3. Đổi sang `float64` thì số phần tử, shape và bộ nhớ thay đổi ra sao?
4. Con số này đã bao gồm tham số mô hình, gradient và các kết quả trung gian khác chưa?

## Tiêu chí tự đánh giá

- Làm được phép nhân hàng–cột và ghi shape trước khi chạy.
- Giải thích được vì sao `.T` của mảng 1-D không tạo cột.
- Tìm ra lỗi broadcasting, không chỉ sửa sao cho hết exception.
- Tính được một gradient cho mỗi đặc trưng, chia đúng số mẫu.
- Phân biệt đổi shape với hoán vị trục và thứ tự đặc trưng.

Các mục mở rộng có thể quay lại sau. Xem [lời giải](solutions.md) để tự kiểm tra lập luận, không chỉ số cuối.
