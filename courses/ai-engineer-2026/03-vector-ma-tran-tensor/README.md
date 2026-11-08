# Bài 3 — Vector, ma trận và tensor

[Mục lục](../README.md) · [Bài 2](../02-dao-ham-gradient-descent/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Trạng thái: đã biên soạn, bổ sung trực giác hình học, phân tích cạm bẫy kỹ thuật và ứng dụng thực tế cho AI Engineer; mã thực hành và 4 hình minh họa đã kiểm tra trên Python 3.14.6 và NumPy 2.5.4.

---

## 1. Mục tiêu và điều kiện bắt đầu

Sau bài này, bạn sẽ nắm được:

- **Trực giác cốt lõi**: Hiểu rõ vì sao AI không thể thiếu ma trận và tensor, giải phóng bạn khỏi các vòng lặp `for` chậm chạp trong Python.
- **Bản đồ các chiều không gian số**: Phân biệt rành mạch từ Scalar (0D), Vector (1D), Ma trận (2D) đến Tensor đa chiều (3D, 4D, ND).
- **Phân biệt rạch ròi các phép tính**: Hiểu khác biệt bản chất giữa nhân từng phần tử (`*`) và nhân ma trận đại số (`@`), tránh các lỗi cú pháp tai hại.
- **Quy tắc Broadcasting và cạm bẫy "Silent Bug"**: Nhận diện lỗi ngầm nguy hiểm nhất trong lập trình AI — phép toán vẫn chạy êm ru nhưng kết quả Loss sai hoàn toàn mà không báo Exception.
- **Phép biến hình: Reshape vs Transpose**: Hiểu vì sao Reshape bừa bãi sẽ phá nát cấu trúc dữ liệu, và khi nào bắt buộc phải dùng Hoán vị trục (Transpose).
- **Vector hóa Gradient ($X^T \mathbf{r}$)**: Nhìn thấy vẻ đẹp tinh gọn khi gom toàn bộ đạo hàm riêng của hàng nghìn mẫu và hàng trăm đặc trưng vào đúng một phép nhân ma trận duy nhất.
- **Đọc hiểu cấu trúc Tensor trong AI thực tế**: Nắm vững các chuẩn quy ước như `(Batch, Seq_Len, Hidden_Dim)` trong LLM (Transformer) và `(N, C, H, W)` trong Thị giác máy tính (Computer Vision).

**Yêu cầu nền tảng**: Nắm vững phương trình $\hat y = wx + b$ và cách tính đạo hàm/gradient từ [Bài 1](../01-mo-hinh-hoc-tu-du-lieu/README.md) và [Bài 2](../02-dao-ham-gradient-descent/README.md). Cần cài đặt thư viện NumPy cơ bản.

---

## 2. Vì sao AI bắt buộc phải dùng Ma trận thay cho biến đơn lẻ?

Ở Bài 1 và 2, ta chỉ có một đầu vào duy nhất: **Số lô dữ liệu** ($x$).
Nhưng trong thực tế backend, thời gian xử lý (latency) của một request phụ thuộc vào nhiều yếu tố:
1. Số lô bản ghi (Lô).
2. Số truy vấn gọi sang Database (Query).
3. Dung lượng payload (KB).
4. Tải CPU hiện tại (%).

Giả sử ta chỉ xét 2 yếu tố đầu tiên trên 3 request được ghi nhận:

| Request ($i$) | Đặc trưng 1: Số lô | Đặc trưng 2: Số truy vấn DB | Đích thực tế: Latency ($y$, ms) |
|:---:|:---:|:---:|:---:|
| Request 0 | 1 lô | 2 query | 5 ms |
| Request 1 | 2 lô | 3 query | 8 ms |
| Request 2 | 3 lô | 2 query | 9 ms |

*(Dữ liệu giả định được tạo theo quy luật $y = 2 \times \text{lô} + 1 \times \text{truy vấn} + 1$ để phục vụ học tập).*

### Nếu lập trình bằng biến đơn lẻ:
Bạn sẽ phải tạo hàng tá biến: `x0_batch, x0_query, x1_batch, x1_query...` và viết các vòng lặp `for` lồng nhau để duyệt qua từng mẫu, từng đặc trưng. Cách làm này:
- **Code rườm rà, dễ nhầm lẫn chỉ số.**
- **Cực kỳ chậm chạp trong Python**: Vòng lặp `for` của Python xử lý từng con số tuần tự, không thể tận dụng được sức mạnh tính toán song song của CPU đa lõi và GPU.

### Giải pháp của kỹ sư AI:
Ta gom toàn bộ dữ liệu vào một **Ma trận (Matrix)** $X$ duy nhất:
- Mỗi **hàng** là một Request (một mẫu dữ liệu).
- Mỗi **cột** là một Đặc trưng (Feature) theo đúng thứ tự quy ước.

$$
X = \begin{bmatrix} 1 & 2 \\ 2 & 3 \\ 3 & 2 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} 5 \\ 8 \\ 9 \end{bmatrix}
$$

Khi đó, toàn bộ quá trình tính toán dự đoán cho hàng triệu request sẽ được gói gọn trong **đúng một phép nhân ma trận**!

---

## 3. Bản đồ không gian số: Từ Scalar đến Tensor

Hãy xây dựng một hình ảnh trực quan rõ ràng trong tâm trí:

```text
┌──────────────┐     ┌───────────────────────┐     ┌─────────────────────────────┐     ┌───────────────────────────────┐
│ Scalar (0D)  │     │      Vector (1D)      │     │      Ma trận Matrix (2D)    │     │       Tensor Đa chiều (ND)    │
│  Một con số  │ ──> │   Một danh sách số    │ ──> │   Một bảng số (Hàng x Cột)  │ ──> │ Khối hộp chứa nhiều bảng số   │
│  Ví dụ: 5.0  │     │ Ví dụ: [1.0, 2.0, 3.0]│     │ Ví dụ: Bảng dữ liệu (3 x 2) │     │ Ví dụ: Cuốn sổ nhiều trang 2D │
└──────────────┘     └───────────────────────┘     └─────────────────────────────┘     └───────────────────────────────┘
```

### 1. Scalar (Số vô hướng - 0 chiều):
- Là một con số đơn lẻ, ví dụ bias $b = 1.0$ hoặc Loss = $9.66$.
- Trong NumPy: Mảng có `ndim = 0`, `shape = ()`.

### 2. Vector (1 chiều):
- Là một danh sách các con số có thứ tự.
- **Vector đặc trưng của Request 0**: $\mathbf{x}_0 = [1, 2]$ (1 lô, 2 query). Thứ tự mang ý nghĩa sống còn: $[1, 2]$ hoàn toàn khác $[2, 1]$!
- **Vector trọng số**: $\mathbf{w} = [2, 1]$ (nghĩa là: $w_1 = 2\text{ ms/lô}$, $w_2 = 1\text{ ms/query}$).
- Trong NumPy: Mảng có `ndim = 1`, `shape = (d,)` (ví dụ `shape = (2,)`).

![Cộng hai vector trong mặt phẳng, minh họa bằng mũi tên](figures/01-vector.png)

*Hình 1. Trong hình học, vector được biểu diễn bằng mũi tên. Phép cộng hai vector $\mathbf{a}=(2,1)$ và $\mathbf{c}=(1,2)$ ghép đầu-đuôi tạo ra vector tổng $(3,3)$.*

### Tích vô hướng (Dot Product): Trái tim của phép dự đoán

Để tính dự đoán cho Request 0 với đầu vào $\mathbf{x} = [1, 2]$ và bộ trọng số $\mathbf{w} = [2, 1]$:
Ta lấy từng đặc trưng nhân với trọng số tương ứng rồi cộng lại:

$$
\mathbf{x} \cdot \mathbf{w} = (1 \times 2) + (2 \times 1) = 2 + 2 = 4
$$

Cộng thêm số hạng tự do (bias) $b = 1$, ta được dự đoán: $\hat y = 4 + 1 = 5\text{ ms}$.
Kết quả của tích vô hướng giữa 2 vector luôn luôn là **MỘT SỐ VÔ HƯỚNG (Scalar)**.

---

## 4. Bốn thuộc tính căn bản của mảng trong NumPy

Khi debug code AI, 90% lỗi phát sinh từ việc không khớp kích thước mảng. Bạn cần thuộc lòng 4 thuộc tính sau của đối tượng `ndarray`:

| Thuộc tính | Ý nghĩa kỹ thuật | Câu hỏi trả lời | Ví dụ ma trận $X$ ở trên |
|---|---|---|---|
| `shape` | Kích thước từng trục (Tuple) | Mỗi chiều dài bao nhiêu? | `(3, 2)` (3 hàng, 2 cột) |
| `ndim` | Số lượng trục (Number of dimensions) | Có mấy chiều? | `2` (Mảng 2 chiều) |
| `size` | Tổng số lượng phần tử | Có tất cả bao nhiêu số? | `6` ($3 \times 2 = 6$) |
| `dtype` | Kiểu dữ liệu phần tử | Lưu dưới dạng số gì? | `float64` hoặc `float32` |

### Cạm bẫy phân biệt: Vector 1D `(3,)` vs Ma trận hàng `(1, 3)` vs Ma trận cột `(3, 1)`

Đây là điểm gây nhầm lẫn kinh điển nhất cho người mới học:

```python
import numpy as np

v = np.array([1.0, 2.0, 3.0])      # shape (3,)   -> ndim = 1 (Vector thuần túy)
row = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3) -> ndim = 2 (Ma trận 1 hàng, 3 cột)
col = np.array([[1.0], [2.0], [3.0]]) # shape (3, 1) -> ndim = 2 (Ma trận 3 hàng, 1 cột)
```

- Vector thuần túy `v` có shape `(3,)` **không có khái niệm hàng hay cột**! 
- Do đó, phép chuyển vị `v.T` trên mảng 1D **không làm thay đổi shape** (vẫn là `(3,)`).
- Muốn biến `v` thành cột thực sự, bạn phải bổ sung thêm 1 trục: `v[:, None]` hoặc `v.reshape(3, 1)`.

---

## 5. Phép nhân: Phân biệt rạch ròi giữa `*` và `@`

Trong Python / NumPy, ký hiệu `*` và `@` làm hai việc **hoàn toàn khác nhau**:

```text
┌───────────────────────────────────────┬───────────────────────────────────────┐
│        Ký hiệu: *                     │        Ký hiệu: @                     │
│  (Element-wise Multiplication)        │  (Matrix Multiplication - Matmul)     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Nhân từng phần tử tương ứng           │ Nhân ma trận đại số (Hàng nhân Cột)   │
│ Hai mảng phải cùng shape (hoặc broad) │ Trục trong phải khớp: (M, K) @ (K, P) │
│ Kết quả giữ nguyên số chiều           │ Kết quả có shape: (M, P)              │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### 1. Phép dự đoán Vector hóa cho cả Batch (`X @ w + b`):

Thay vì viết vòng lặp, ta tính dự đoán cho cả 3 request cùng lúc:

$$
\hat{\mathbf{y}} = X \mathbf{w} + b = 
\begin{bmatrix} 1 & 2 \\ 2 & 3 \\ 3 & 2 \end{bmatrix} 
\begin{bmatrix} 2 \\ 1 \end{bmatrix} + 1 = 
\begin{bmatrix} (1\times2 + 2\times1) + 1 \\ (2\times2 + 3\times1) + 1 \\ (3\times2 + 2\times1) + 1 \end{bmatrix} = 
\begin{bmatrix} 5 \\ 8 \\ 9 \end{bmatrix}
$$

Trong code Python:

```python
y_hat = X @ w + b  # X shape (3, 2), w shape (2,) -> y_hat shape (3,)
```

Toàn bộ phép tính phức tạp diễn ra chỉ trong **một biểu thức duy nhất**! Đây được gọi là **Vectorization (Vector hóa)**.

### 2. Nhân hai ma trận đại số ($A @ B$):

Để nhân được hai ma trận $A$ kích thước $(m \times k)$ và $B$ kích thước $(k \times p)$, **kích thước ở giữa bắt buộc phải bằng nhau**:

```text
       A               B              Kết quả C
    (m x k)    @    (k x p)    ───>    (m x p)
         ▲             ▲
         └──────┬──────┘
            Bắt buộc
            phải bằng nhau!
```

![Tích hai ma trận, tô màu một hàng và một cột để tính ô đầu tiên](figures/02-nhan-ma-tran.png)

*Hình 2. Lấy Hàng thứ $i$ của ma trận thứ nhất bắn tia (tích vô hướng) với Cột thứ $j$ của ma trận thứ hai để tạo ra phần tử tại vị trí $(i, j)$ của ma trận kết quả.*

---

## 6. Broadcasting: Chiếc đũa thần và Cạm bẫy "Silent Bug"

### Nguyên lý Broadcasting:
Khi bạn thực hiện phép tính giữa hai mảng khác kích thước (ví dụ ma trận `X` shape `(3, 2)` cộng với bias `b = 1.0` là một scalar), NumPy sẽ tự động "kéo giãn" (broadcast) con số 1.0 thành một ma trận toàn số 1 có cùng kích thước `(3, 2)` để cộng vào từng phần tử.

**Quy tắc so khớp (Từ phải sang trái)**:
Hai chiều tương thích nếu chúng **bằng nhau**, hoặc **một trong hai bên có kích thước bằng 1**.

### Cạm bẫy tử thần thầm lặng (The Silent Bug):
Đây là lỗi kinh điển khiến mô hình không học được mà lập trình viên tìm mòn mắt không thấy log lỗi:

Giả sử dự đoán của bạn hoàn toàn chính xác: `pred = np.array([5.0, 8.0, 9.0])` và nhãn `y = np.array([5.0, 8.0, 9.0])`.
Sai số thực tế từng mẫu phải bằng $0$: `pred - y = [0., 0., 0.]`.

Nhưng nếu do sơ suất ở bước tiền xử lý, `pred` bị biến thành dạng cột `(3, 1)` còn `y` là dạng mảng 1D `(3,)`:

```python
pred_col = pred[:, None]  # shape (3, 1)
residual = pred_col - y   # shape (3, 1) trừ (3,) ???
```

NumPy **không hề báo lỗi**! Nó tự động hiểu rằng:
- Chiều 1 của bên trái (3) kết hợp với chiều 1 của bên phải (1) $\to$ thành 3.
- Chiều 2 của bên trái (1) kết hợp với chiều 2 của bên phải (3) $\to$ thành 3.
- Kết quả tạo ra một **ma trận vuông $(3 \times 3)$ so chéo mọi cặp mẫu**:

$$
\begin{bmatrix} 5-5 & 5-8 & 5-9 \\ 8-5 & 8-8 & 8-9 \\ 9-5 & 9-8 & 9-9 \end{bmatrix} = 
\begin{bmatrix} 0 & -3 & -4 \\ 3 & 0 & -1 \\ 4 & 1 & 0 \end{bmatrix}
$$

![So sánh sai số đúng theo từng mẫu và sai số mọi cặp do broadcasting](figures/03-broadcasting.png)

*Hình 3. Bên trái là phép trừ đúng từng cặp mẫu (Loss = 0). Bên phải là cạm bẫy Broadcasting tạo ra ma trận so chéo $3 \times 3$, khiến Loss bị tính nhầm thành $5.78\text{ ms}^2$ dù mô hình đang dự đoán đúng 100%!*

> [!CAUTION]
> **Kinh nghiệm sống còn của AI Engineer**:
> Trước khi tính hàm mất mát (Loss), luôn viết assert kiểm tra shape hoặc đảm bảo cả hai mảng đều là vector 1D:
> ```python
> assert pred.shape == y.shape, f"Lệch shape: {pred.shape} vs {y.shape}"
> ```

---

## 7. Vector hóa Gradient: Vẻ đẹp của $X^T \mathbf{r}$

Hãy nhớ lại ở Bài 2:
Với mỗi đặc trưng, đạo hàm riêng là: $\frac{\partial L}{\partial w_j} = \frac{2}{n}\sum_{i=1}^n r_i \cdot X_{ij}$.

Nếu có 100 đặc trưng, chẳng lẽ ta phải viết 100 dòng code hoặc vòng lặp duyệt từng cột?
Hãy nhìn vào phép nhân kỳ diệu giữa **Ma trận chuyển vị $X^T$** và **Vector sai số $\mathbf{r}$**:

$$
X^T \mathbf{r} = 
\begin{bmatrix} 1 & 2 & 3 \\ 2 & 3 & 2 \end{bmatrix} 
\begin{bmatrix} r_1 \\ r_2 \\ r_3 \end{bmatrix} = 
\begin{bmatrix} 1\cdot r_1 + 2\cdot r_2 + 3\cdot r_3 \\ 2\cdot r_1 + 3\cdot r_2 + 2\cdot r_3 \end{bmatrix}
$$

Hàng thứ nhất của $X^T$ (toàn bộ giá trị của đặc trưng 1 trên mọi mẫu) nhân với vector sai số $\mathbf{r}$ $\implies$ **Ra ngay tổng đạo hàm của $w_1$**!
Hàng thứ hai của $X^T$ nhân với $\mathbf{r}$ $\implies$ **Ra ngay tổng đạo hàm của $w_2$**!

Kiểm tra kích thước (Shape Check):
```text
X.T:          (d, n)      -> (2, 3)
r:               (n,)     -> (3,)
─────────────────────────────────────────────
X.T @ r:      (d,)        -> (2,)   (Một gradient cho mỗi trọng số!)
```

### Toàn bộ thuật toán Gradient Descent đa biến trong 5 dòng code:

```python
# 1. Dự đoán cho cả batch
y_hat = X @ w + b

# 2. Tính vector sai số từng mẫu
r = y_hat - y

# 3. Tính gradient cho toàn bộ trọng số w và bias b
dw = (2.0 / len(y)) * (X.T @ r)
db = 2.0 * np.mean(r)

# 4. Cập nhật đồng thời tất cả tham số
w = w - lr * dw
b = b - lr * db
```

Không còn bất kỳ vòng lặp `for` nào! Đoạn code trên chạy được cho dữ liệu có 2 đặc trưng, 100 đặc trưng, hay 10.000 đặc trưng trên hàng triệu mẫu với tốc độ tối đa của phần cứng.

---

## 8. Tensor trong AI hiện đại: Đọc hiểu ý nghĩa các trục

Khi bước vào Deep Learning, dữ liệu không chỉ dừng lại ở bảng 2 chiều mà được tổ chức thành **Tensor đa chiều**:

### 1. Dữ liệu chuỗi văn bản trong LLM (Transformer) — Tensor 3D `(B, T, D)`:
Khi bạn gửi một câu hỏi vào ChatGPT:
- **`B` (Batch size)**: Số lượng câu hỏi được xử lý song song (ví dụ: 16 câu).
- **`T` (Sequence Length / Tokens)**: Chiều dài tối đa của câu (ví dụ: 512 token).
- **`D` (Hidden Dimension / Embedding)**: Mỗi từ được biểu diễn bằng một vector số thực (ví dụ: 768 hoặc 4096 chiều).
- Tensor đầu vào có shape: `(16, 512, 4096)`.

### 2. Dữ liệu hình ảnh trong Computer Vision — Tensor 4D:
- Chuẩn **NHWC** (TensorFlow / Keras): `(Batch, Height, Width, Channels)` — ví dụ `(32, 224, 224, 3)`.
- Chuẩn **NCHW** (PyTorch / GPU cuDNN): `(Batch, Channels, Height, Width)` — ví dụ `(32, 3, 224, 224)`.

![Tensor ảnh được tách thành hai mẫu và ba kênh để quan sát giá trị](figures/04-tensor.png)

*Hình 4. Minh họa Tensor 4D gồm 2 ảnh, kích thước $2 \times 3$, mỗi ảnh có 3 kênh màu (R, G, B).*

---

## 9. Phép biến hình: Reshape khác hoàn toàn Transpose!

Đây là một trong những hiểu lầm nguy hiểm nhất khi xử lý ảnh hoặc âm thanh:

```text
┌───────────────────────────────────────┬───────────────────────────────────────┐
│        Reshape                        │        Transpose (Hoán vị trục)       │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Bốc dữ liệu theo thứ tự bộ nhớ phẳng  │ Xoay góc nhìn các trục tọa độ         │
│ (flatten) rồi nhồi vào hình dạng mới. │ Bảo toàn vị trí thực tế của dữ liệu.  │
│ KHÔNG BẢO TOÀN cấu trúc không gian!   │ BẢO TOÀN mối quan hệ ngữ nghĩa!       │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### Ví dụ thực tế chứng minh:
Giả sử bạn có một lô ảnh đang ở định dạng NHWC `(2, 2, 3, 3)` và muốn chuyển sang chuẩn NCHW `(2, 3, 2, 3)` để đưa vào mô hình PyTorch:

```python
images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)  # Định dạng NHWC

# CÁCH ĐÚNG: Dùng transpose để đổi chỗ trục Channel (trục 3) lên trước Height (trục 1)
correct_nchw = images.transpose(0, 3, 1, 2)  # shape (2, 3, 2, 3)

# CÁCH SAI: Dùng reshape ép kích thước
wrong_nchw = images.reshape(2, 3, 2, 3)      # shape cũng là (2, 3, 2, 3) nhưng...
```

Hãy kiểm tra giá trị của cùng một pixel tại vị trí cụ thể:
- `images[1, 0, 2, 1]` có giá trị là **25**.
- `correct_nchw[1, 1, 0, 2]` giữ nguyên giá trị **25** (pixel màu đỏ vẫn nằm đúng vị trí đó trên bức ảnh).
- `wrong_nchw[1, 1, 0, 2]` biến thành giá trị **26**! Màu sắc và điểm ảnh đã bị xáo trộn lung tung, làm ảnh bị xé nát cấu trúc không gian.

> [!IMPORTANT]
> Chỉ dùng `reshape` khi bạn muốn trải phẳng dữ liệu (Flatten: ví dụ biến ảnh thành vector `(Batch, -1)`). Khi muốn đổi thứ tự các chiều mang ngữ nghĩa (như chuyển Channel lên đầu), **bắt buộc phải dùng `transpose` / `permute`**.

---

## 10. Thực hành Python với NumPy

Toàn bộ mã nguồn nằm trong tệp [lab.py](lab.py). Bạn có thể chạy từng mode độc lập để quan sát:

### 1. Quan sát thuộc tính mảng và broadcasting:
```bash
python3 lab.py --mode shapes
python3 lab.py --mode broadcast
```

### 2. Kiểm tra Vector hóa Gradient:
```bash
python3 lab.py --mode gradient
```
Kết quả in ra:
```text
initial MSE: 29.333333333333332
dw: [-22.666667 -25.333333] ; db: -10.666666666666666
matches explicit loops: True
after one update, w: [1.226667 0.253333] ; b: 0.10666666666666666
new MSE: 18.013570370370374
```
Dòng `matches explicit loops: True` chứng minh công thức ma trận `X.T @ r` cho kết quả chính xác 100% đến từng số lẻ so với việc chạy vòng lặp lồng nhau!

### 3. Huấn luyện hoàn chỉnh 10.000 bước bằng phép toán ma trận:
```bash
python3 lab.py --mode train --lr 0.01 --steps 10000
```
Kết quả:
```text
w: [2.000157 1.000602] ; b: 0.998232
train MSE: 9.936969557714934e-08
```
Mô hình tự động tìm ra trọng số $w_1 \approx 2.0$, $w_2 \approx 1.0$, $b \approx 1.0$, khôi phục chính xác quy luật ẩn trong dữ liệu backend!

---

## 11. Các ứng dụng thực tế của AI Engineer

1. **Tối ưu hóa thông lượng (Throughput) bằng Batching**:
   - Thay vì gửi từng prompt vào LLM, ta gom 16 hoặc 32 prompt lại thành một Tensor `(Batch, Seq_Len)`. Phép nhân ma trận trên GPU Tensor Core sẽ tính toán song song toàn bộ 32 request cùng lúc, tăng thông lượng xử lý lên gấp hàng chục lần với cùng chi phí phần cứng.
2. **Kỹ thuật Embedding Lookup**:
   - Mỗi token từ vựng được mã hóa thành một ID số nguyên. Việc tra cứu vector embedding thực chất là phép nhân ma trận giữa vector One-Hot và ma trận trọng số $W_{\text{embed}}$.
3. **Debug lỗi kích thước Tensor (Shape Mismatch)**:
   - Trong quá trình xây dựng pipeline RAG hoặc fine-tuning mô hình, kỹ năng đọc và in `tensor.shape` sau mỗi tầng là công cụ số 1 để phát hiện lỗi tràn chiều, lệch trục và sai lệch ngữ nghĩa dữ liệu.

---

## 12. Tổng kết và bước tiếp theo

| Khái niệm | Ý nghĩa kỹ thuật | Trong code NumPy |
|---|---|---|
| **Scalar** | Đại lượng 0 chiều | `np.array(1.0)` (`shape = ()`) |
| **Vector** | Danh sách 1 chiều | `np.array([1, 2])` (`shape = (2,)`) |
| **Matrix** | Bảng số 2 chiều (Hàng x Cột) | `X` (`shape = (n, d)`) |
| **Tensor** | Mảng số đa chiều | `(Batch, Seq_Len, Dim)` hoặc `(N, C, H, W)` |
| **`*`** | Nhân từng phần tử tương ứng | `A * B` (Element-wise) |
| **`@`** | Nhân ma trận đại số (Hàng nhân Cột) | `X @ w` (Matrix Multiplication) |
| **`X.T @ r`** | Vector hóa Gradient cho mọi trọng số | Gom toàn bộ vòng lặp tính đạo hàm vào 1 dòng |
| **Transpose** | Hoán vị trục, bảo toàn vị trí không gian | `img.transpose(0, 3, 1, 2)` |

**Tự kiểm tra kiến thức:**
1. Mở [exercises.md](exercises.md) và làm các bài tập: tính tay tích ma trận (Bài 3), phân tích cạm bẫy broadcasting (Bài 4), phân biệt reshape/transpose (Bài 6).
2. Đối chiếu chi tiết với [solutions.md](solutions.md).

**Học tiếp:**
[Bài 4 — Xác suất, thống kê và bất định](../04-xac-suat-thong-ke-bat-dinh/README.md). Bạn sẽ bước vào thế giới của xác suất: làm sao mô hình biết nó đang tự tin hay phân vân, và cách AI xử lý nhiễu dữ liệu trong thực tế.

---

*Tài liệu tham khảo đối chiếu chuẩn:*
- [NumPy Documentation: Array Attributes](https://numpy.org/doc/stable/user/absolute_beginners.html#array-attributes)
- [NumPy Documentation: Broadcasting Rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch Documentation: Tensor Fundamentals](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
