# Lời giải — Vector, ma trận và tensor

[Về bài học](README.md) · [Về đề bài](exercises.md)

Kết quả số được đối chiếu bằng NumPy; các ví dụ quan trọng có thêm phép tính tay hoặc vòng lặp độc lập. Số thập phân trong bảng đã làm tròn.

## Bài 1

| Biến | Shape | `ndim` | `size` |
|---|---|---:|---:|
| `a` | `()` | 0 | 1 |
| `b` | `(1,)` | 1 | 1 |
| `c` | `(1,1)` | 2 | 1 |
| `v` | `(3,)` | 1 | 3 |
| `row` | `(1,3)` | 2 | 3 |
| `col` | `(3,1)` | 2 | 3 |

`v.T` vẫn có shape `(3,)`. Mảng 1-D không có hai trục hàng/cột để đổi cho nhau. Ba thành phần mô tả số tọa độ trong vector, trong khi `ndim` đếm số trục của cách lưu mảng. Một scalar dạng mảng 0 trục vẫn chứa một giá trị.

## Bài 2

$\mathbf a+\mathbf c=(3,3)$, $3\mathbf a=(6,3)$. Tích từng phần tử là $(2,2)$, một vector. Tích vô hướng bằng $2\times1+1\times2=4$, một scalar. Độ dài $\|\mathbf a\|_2=\sqrt{2^2+1^2}=\sqrt5\approx2.236068$.

Nếu hai thành phần khác đơn vị, tổng bình phương các số thô không tự có ý nghĩa vật lý/nghiệp vụ. Đổi ms thành giây cũng làm thay đổi trị số khoảng cách. Cần xác định cách biểu diễn và thang đo trước khi dùng khoảng cách làm tiêu chí so sánh. Điều này không làm phép tính đại số sai; nó giới hạn cách diễn giải kết quả.

## Bài 3

Hàng đầu nhân cột đầu: $1\times2+2\times1=4$; hàng thứ hai nhân cột thứ hai: $2\times1+3\times(-1)=-1$.

$$
XW=\begin{bmatrix}4&-1\\7&-1\\8&1\end{bmatrix}
$$

`X * W` không hợp lệ: `(3,2)` và `(2,2)` có hai trục cuối bằng 2, nhưng trục còn lại 3 và 2 không bằng nhau và không bên nào bằng 1. $WX$ cũng không hợp lệ: `(2,2) @ (3,2)` có hai kích thước bên trong là 2 và 3.

Mở rộng:

$$
AB=\begin{bmatrix}7&2\\3&1\end{bmatrix},\qquad
BA=\begin{bmatrix}1&2\\3&7\end{bmatrix}
$$

Cả hai phép nhân đều có nghĩa và đều trả ma trận 2×2, nhưng giá trị khác nhau. Không được đổi thứ tự nhân ma trận như đổi thứ tự nhân hai scalar.

## Bài 4

| Biểu thức | Shape | Giải thích |
|---|---|---|
| `X @ w` | `(4,)` | Mỗi hàng cho một tích vô hướng |
| `X * w` | `(4,3)` | Nhân theo phần tử, broadcast trọng số theo hàng |
| `X @ W + b` | `(4,2)` | Hai đầu ra mỗi mẫu; bias dài 2 cộng theo hàng |
| `X.T @ r` | `(3,)` | `(3,4) @ (4,)`, một tổng cho mỗi đặc trưng |
| `w.T` | `(3,)` | Mảng 1-D vẫn 1-D |
| `X.mean(axis=0)` | `(3,)` | Gộp mẫu, giữ các đặc trưng |
| `X.mean(axis=1, keepdims=True)` | `(4,1)` | Gộp đặc trưng, giữ trục độ dài 1 |

Mở rộng: `(2,4,3) @ (3,5)` cho `(2,4,5)`. Batch vẫn có 2 mẫu, mỗi mẫu 4 vị trí; mỗi vector 3 đặc trưng được biến đổi thành 5 đầu ra bằng cùng một ma trận. Trục batch không bị cộng gộp trong phép tính này.

## Bài 5

`pred-y` có shape `(3,3)`, giá trị:

$$
\begin{bmatrix}0&-3&-4\\3&0&-1\\4&1&0\end{bmatrix}
$$

Tổng bình phương bằng 52, có 9 phần tử, nên loss bằng $52/9\approx5.777778$ ms². Ta đã so dự đoán của mẫu này với đích của mọi mẫu, thay vì chỉ mẫu tương ứng.

Hai cách sửa, tùy quy ước được chọn:

```python
# A. Cả hai là 1-D, một phần tử cho mỗi mẫu.
pred_1d = pred[:, 0]
loss_a = np.mean((pred_1d - y) ** 2)

# B. Cả hai là cột (n,1).
y_col = y[:, None]
loss_b = np.mean((pred - y_col) ** 2)
```

Cả hai cho loss 0. Cách A giả định cột duy nhất của `pred` đúng là dự đoán một đầu ra; với nhiều đầu ra không được tự bỏ các cột khác. Một loss hữu hạn vẫn có thể thuộc một phép tính sai ý định. Kiểm tra shape và ý nghĩa từng trục giúp phát hiện trước khi huấn luyện theo mục tiêu sai.

## Bài 6

Dự đoán ban đầu là $(0,0,0)$, sai số $\mathbf r=(-5,-8,-9)$. Loss $=(25+64+81)/3=170/3\approx56.666667$ ms².

$$
X^T\mathbf r=\begin{bmatrix}-5-16-27\\-10-24-18\end{bmatrix}
=\begin{bmatrix}-48\\-52\end{bmatrix}
$$

Gradient theo trọng số là $(-32,-104/3)$; gradient bias là $(2/3)(-22)=-44/3$. Với learning rate 0.01:

$$
\mathbf w_{\text{mới}}=(8/25,26/75)\approx(0.32,0.346667),\quad
b_{\text{mới}}=11/75\approx0.146667
$$

Ba dự đoán mới là $(29/25,137/75,9/5)$, tức $(1.16,1.826667,1.8)$. Sai số tương ứng $(-96/25,-463/75,-36/5)$; loss:

$$
\frac{(96/25)^2+(463/75)^2+(36/5)^2}{3}
=\frac{588913}{16875}\approx34.898548\text{ ms}^2
$$

Hai cách tính gradient trong code phải khớp trong sai số làm tròn. Chia cho 3 vì loss lấy trung bình trên 3 mẫu; 2 là số trọng số/đặc trưng, không phải mẫu số trong định nghĩa loss này. Không đưa dữ liệu giữ riêng vào quá trình chọn tham số.

## Bài 7

`images`: `(N,H,W,C)=(2,2,3,3)`. `nchw`: `(N,C,H,W)=(2,3,2,3)`.

Ba giá trị cần tìm lần lượt là **25, 25, 26**. Chuyển vị trục giữ số 25 ở đúng ảnh, hàng pixel, cột pixel và kênh tương ứng. Reshape với shape đích giống nhau không làm đúng phép đổi trục đó.

Ma trận của kênh 1, ảnh 1, tức `nchw[1,1]`:

$$
\begin{bmatrix}19&22&25\\28&31&34\end{bmatrix}
$$

`images.reshape(2,-1)` có shape `(2,18)`; mỗi hàng là 18 giá trị của một ảnh theo thứ tự đọc mặc định. Tổng số giá trị vẫn 36. `images.T` đảo toàn bộ trục, tạo thứ tự `(C,W,H,N)`, không phải NCHW. `reshape(2,20)` không hợp lệ vì cần 40 giá trị trong khi chỉ có 36.

## Bài 8

Đầu vào mới là `(2,1),(3,2),(2,3)`. Giữ `w=(2,1),b=1` sẽ dự đoán lần lượt **6, 9, 8**, thay vì 5, 8, 9.

Shape vẫn `(3,2)`, nên không có lỗi kích thước. Schema/ý nghĩa của cột đã đổi. Có thể đưa đầu vào về thứ tự cũ trước khi gọi mô hình, hoặc sắp trọng số tương ứng thành `(1,2)` nếu toàn bộ quy trình liên quan được cập nhật nhất quán. Với mô hình phức tạp, không mặc định có thể sửa đơn giản bằng đổi hai trọng số; nên giữ một hợp đồng dữ liệu rõ ràng.

Nên ghi tên và thứ tự đặc trưng, đơn vị, kiểu số, phiên bản biến đổi dữ liệu; dùng vài request biết trước để kiểm tra toàn bộ đường đi từ dịch vụ tới mô hình. Kiểm tra shape là cần thiết nhưng chưa đủ.

## Bài 9

Tensor có 3 trục và $32\times128\times768=3\,145\,728$ phần tử. `float32` dùng 4 byte/phần tử, nên phần dữ liệu có $12\,582\,912$ byte = **12 MiB**.

Đổi sang `float64` không đổi shape/số phần tử, nhưng phần dữ liệu tăng gấp đôi thành 25 165 824 byte = 24 MiB. Đây chưa phải tổng bộ nhớ khi chạy mô hình: chưa cộng tham số, gradient, dữ liệu khác, kết quả trung gian, bộ nhớ tạm và chi phí đối tượng. View dùng chung bộ nhớ cũng cần xét khi cộng nhiều `nbytes`.

## Nếu còn vướng

- Nhầm hàng/cột: viết bảng nhỏ và đọc `X[0]`, `X[:,0]` bằng tay.
- Nhầm `*` với `@`: ghi đầu ra phải là một đóng góp cho mỗi đặc trưng hay một tổng cho mỗi mẫu.
- Nhầm broadcasting: căn shape từ bên phải, thêm trục 1 còn thiếu trước khi tính.
- Nhầm reshape/transpose: theo dõi một phần tử có giá trị khác biệt qua từng bộ chỉ số.
- Nhầm gradient: viết riêng tổng cho từng đặc trưng trước khi ghép thành $X^T\mathbf r$.

Hoàn thành tài liệu không tự chứng minh người học đã đạt các mục tiêu; dùng bài làm và cách giải thích để tự đánh giá.
