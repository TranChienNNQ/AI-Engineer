# Bài 3 — Vector, ma trận và tensor

[Mục lục](../README.md) · [Bài 2](../02-dao-ham-gradient-descent/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn và kiểm tra ngày 04/11/2026. Các số liệu là ví dụ giả định, không phải phép đo hiệu năng thực. Bài này dùng NumPy để biểu diễn phép tính; chưa cần PyTorch hoặc GPU.

## 1. Mục tiêu và cách đọc

Sau bài học, người học có thể:

- Phân biệt scalar, vector, ma trận và tensor theo cách biểu diễn dữ liệu trong ML.
- Đọc `shape`, `ndim`, `size`, `dtype` và giải thích ý nghĩa từng trục.
- Tính tay tích vô hướng, tích ma trận–vector và tích hai ma trận nhỏ.
- Phân biệt `*` với `@`, chuyển vị với reshape, và phát hiện broadcasting sai ý định.
- Viết dự đoán và gradient của hồi quy nhiều đặc trưng bằng phép tính vector hóa.
- Giải thích tensor ảnh hoặc chuỗi theo các trục, thay vì chỉ nhớ một bộ số kích thước.

Cần biết Python cơ bản, dự đoán $wx+b$ và gradient MSE từ bài 1–2. Có thể đọc hai lượt: phần 2–7 học biểu diễn và phép tính; phần 8–14 dùng chúng trong dữ liệu, gradient và tensor. Phần cài đặt nằm ở mục 13; vẫn có thể đọc và tính tay trước khi cài thư viện.

## 2. Vì sao không tiếp tục dùng từng biến đơn lẻ?

Ở bài 2, mỗi request có một đầu vào là số lô. Nếu thêm số truy vấn cơ sở dữ liệu, mỗi request cần hai con số. Với nhiều request, ta cần cách tổ chức để biết số nào thuộc request nào, đặc trưng nào.

Quy ước của bài:

- Một **hàng** tương ứng một mẫu/request.
- Một **cột** tương ứng một đặc trưng, theo đúng thứ tự đã thống nhất.
- Đáp án được giữ riêng, không ghép thành đặc trưng đầu vào.

| Mẫu | Số lô dữ liệu | Số truy vấn | Latency đích, ms |
|---|---:|---:|---:|
| 0 | 1 | 2 | 5 |
| 1 | 2 | 3 | 8 |
| 2 | 3 | 2 | 9 |

Mỗi lô vẫn có 100 bản ghi. Dữ liệu này được tạo theo $y=2\times\text{số lô}+1\times\text{số truy vấn}+1$ để có ví dụ tính được bằng tay. Quan hệ đó không được mặc định cho hệ thống thực.

## 3. Scalar và vector

**Scalar — số vô hướng** là một giá trị, chẳng hạn bias $b=1$. Trong code, Python `float` là kiểu số; muốn quan sát thuộc tính mảng 0 chiều, dùng `np.array(1.0)`.

**Vector** là một bộ thành phần có thứ tự. Đầu vào của request đầu tiên là $\mathbf{x}=(1,2)$: thành phần thứ nhất là số lô, thứ hai là số truy vấn. Vector trọng số $\mathbf{w}=(2,1)$ cho biết hệ số tương ứng từng đặc trưng.

Thứ tự là một phần ý nghĩa: $(1,2)$ không tự động tương đương $(2,1)$. Các thành phần có thể mang đơn vị khác nhau; ở đây trọng số tương ứng có đơn vị ms/lô và ms/truy vấn.

### Cộng và nhân với scalar

Với hai vector đại số $\mathbf{a}=(2,1)$, $\mathbf{c}=(1,2)$:

$$
\mathbf{a}+\mathbf{c}=(2+1,1+2)=(3,3),\qquad
2\mathbf{a}=(4,2)
$$

Trong mặt phẳng, vector có thể được vẽ thành một mũi tên chỉ dịch chuyển. Cộng vector tương ứng ghép đầu–đuôi:

![Cộng hai vector trong mặt phẳng, minh họa bằng mũi tên](figures/01-vector.png)

*Hình 1. Đây là hình học của các vector đại số trừu tượng. Không coi trục số lô và trục số truy vấn là cùng đơn vị vật lý để suy khoảng cách thực tế.*

### Tích vô hướng: nhân từng cặp rồi cộng

Với hai vector cùng số thành phần:

$$
\mathbf{x}\cdot\mathbf{w}=\sum_{j=1}^{d}x_jw_j
$$

$d$ là số đặc trưng; $j$ là vị trí đặc trưng. Với request $(1,2)$ và trọng số $(2,1)$:

$$
\mathbf{x}\cdot\mathbf{w}=1\times2+2\times1=4
$$

Thêm bias 1, dự đoán là 5 ms. Kết quả tích vô hướng là **một scalar**, không phải vector hai phần tử.

Độ dài Euclid của vector đại số được tính bằng $\|\mathbf{a}\|_2=\sqrt{\sum_j a_j^2}$; ví dụ $(3,4)$ có độ dài 5. Trong ML, dùng độ dài/khoảng cách trên đặc trưng khác đơn vị cần cân nhắc thang đo. Bài này chưa dùng norm của đầu vào để so độ giống nhau giữa request.

## 4. Ma trận: tổ chức nhiều mẫu, nhiều đặc trưng

Gom ba vector đầu vào theo hàng:

$$
X=\begin{bmatrix}1&2\\2&3\\3&2\end{bmatrix},\qquad
\mathbf{y}=\begin{bmatrix}5\\8\\9\end{bmatrix}
$$

$X$ là ma trận 3 hàng, 2 cột. Viết $X\in\mathbb{R}^{n\times d}$ nghĩa là một ma trận số thực với $n$ mẫu và $d$ đặc trưng; ở đây $n=3,d=2$.

$X_{ij}$ là giá trị ở hàng $i$, cột $j$. Công thức toán thường đánh chỉ số từ 1; **code NumPy đánh từ 0**. Vì thế `X[0, 1]` là số 2 ở hàng đầu, cột thứ hai.

```python
import numpy as np

X = np.array([[1.0, 2.0],
              [2.0, 3.0],
              [3.0, 2.0]])

X[0]       # [1., 2.]: mẫu đầu tiên, shape (2,)
X[:, 0]    # [1., 2., 3.]: đặc trưng số lô, shape (3,)
X[:, 0:1]  # Cùng các giá trị nhưng giữ một trục cột, shape (3, 1)
```

Trong lát cắt `:`, ta lấy tất cả vị trí trên trục đó. Việc dùng chỉ số đơn như `0` thường loại bỏ trục tương ứng, còn lát cắt `0:1` giữ trục có độ dài 1. Vì vậy cùng giá trị chưa chắc cùng shape.

## 5. Shape, số trục và kiểu dữ liệu

Trong NumPy, `ndarray` là kiểu mảng nhiều chiều. Cần phân biệt:

| Thuộc tính | Câu hỏi trả lời | Ví dụ với $X$ |
|---|---|---|
| `shape` | Mỗi trục dài bao nhiêu? | `(3, 2)` |
| `ndim` | Có bao nhiêu trục? | `2` |
| `size` | Tổng số phần tử? | `6` |
| `dtype` | Mỗi phần tử có kiểu số gì? | `float64` trong ví dụ |

Số phần tử bằng tích các kích thước; số trục bằng độ dài tuple `shape`. Đây là các thuộc tính được mô tả trong [NumPy: Array attributes](https://numpy.org/doc/stable/user/absolute_beginners.html#array-attributes).

### Vector 3 thành phần không có nghĩa mảng 3 trục

| Cách tạo | Shape | `ndim` | Cách đọc |
|---|---|---:|---|
| `np.array(3.0)` | `()` | 0 | Scalar dưới dạng mảng |
| `np.array([1., 2., 3.])` | `(3,)` | 1 | Mảng 1 trục, 3 thành phần |
| `np.array([[1., 2., 3.]])` | `(1, 3)` | 2 | Một hàng, 3 cột |
| `np.array([[1.], [2.], [3.]])` | `(3, 1)` | 2 | Ba hàng, một cột |

Trong đại số, vector thuộc $\mathbb R^3$ có ba tọa độ; trong NumPy, cách lưu `(3,)` có một trục. Hai cách dùng từ “chiều” này khác nhau. Dấu phẩy ở `(3,)` là cú pháp tuple một phần tử của Python.

Một số tài liệu gọi số trục tensor là rank. Tuy nhiên **matrix rank — hạng ma trận** là khái niệm đại số khác. Khi nói về shape, bài này dùng `ndim`/số trục để tránh nhầm.

`float32` dùng 4 byte/phần tử, `float64` dùng 8 byte; `uint8` dùng 1 byte. Ví dụ một mảng đặc gồm 6 số `float64` có 48 byte dữ liệu. `nbytes` không bao gồm toàn bộ chi phí đối tượng Python, và nhiều view có thể dùng chung vùng dữ liệu. Việc dùng NumPy không tự tạo đồ thị đạo hàm hay tự chuyển tính toán lên GPU.

## 6. Nhân từng phần tử khác nhân ma trận

Với `w = np.array([2.0, 1.0])`:

```python
X * w    # [[2., 2.], [4., 3.], [6., 2.]]; shape (3, 2)
X @ w    # [4., 7., 8.]; shape (3,)
```

`*` nhân từng thành phần, ở đây có broadcasting vector trọng số lên mỗi hàng. `@` thực hiện tích ma trận–vector: mỗi hàng lấy tích vô hướng với `w`, nên hai đóng góp được cộng lại.

### Dự đoán cho cả batch

Gọi **batch** là một nhóm mẫu được tính cùng nhau. Ba dự đoán là:

$$
\hat{\mathbf y}=X\mathbf w+b\mathbf 1
=\begin{bmatrix}1&2\\2&3\\3&2\end{bmatrix}
\begin{bmatrix}2\\1\end{bmatrix}
+\begin{bmatrix}1\\1\\1\end{bmatrix}
=\begin{bmatrix}5\\8\\9\end{bmatrix}
$$

$\mathbf 1$ là vector toàn số 1 dài $n$; nó diễn đạt việc cộng cùng bias vào mỗi dự đoán. NumPy không cần tạo vector này, dùng `X @ w + b` với broadcasting scalar.

Tính tay từng mẫu:

- Mẫu 0: $1\times2+2\times1+1=5$ ms.
- Mẫu 1: $2\times2+3\times1+1=8$ ms.
- Mẫu 2: $3\times2+2\times1+1=9$ ms.

Quy ước code cho một đầu ra:

```text
X:       (n, d)
w:          (d,)
X @ w:   (n,)
b:       scalar
y_pred:  (n,)
y:       (n,)
```

Đó là **vectorization — vector hóa**: diễn đạt phép tính cả nhóm bằng phép toán mảng thay cho vòng lặp Python ở từng mẫu. Bên dưới vẫn có phép tính và vòng lặp được thư viện thực hiện. Cách viết này giúp kiểm tra cấu trúc và tận dụng thư viện số; không khẳng định mọi ví dụ nhỏ đều chạy nhanh hơn trong mọi môi trường.

### Nhân hai ma trận theo hàng–cột

Nếu $A$ có shape $(m,k)$ và $B$ có shape $(k,p)$ thì $AB$ có shape $(m,p)$:

$$
(AB)_{ij}=\sum_{\ell=1}^{k}A_{i\ell}B_{\ell j}
$$

Hai kích thước bên trong phải bằng nhau, và mỗi ô kết quả cộng $k$ tích. Với dữ liệu $X$ ở trên và một ma trận đại số $W=\begin{bmatrix}2&1\\1&-1\end{bmatrix}$:

$$
XW=\begin{bmatrix}4&-1\\7&-1\\8&1\end{bmatrix}
$$

Ví dụ ô hàng 2, cột 2 trong ký hiệu toán là $2\times1+3\times(-1)=-1$. Hai cột của kết quả có thể được hiểu là hai phép biến đổi có trọng số; không mặc định cả hai là latency vật lý.

![Tích hai ma trận, tô màu một hàng và một cột để tính ô đầu tiên](figures/02-nhan-ma-tran.png)

*Hình 2. Các trục bằng nhau ở phía trong được cộng qua; các trục phía ngoài xác định shape kết quả. `X @ W` không giống `X * W`; phép thứ hai còn không broadcast được với shape `(3,2)` và `(2,2)`.*

Thứ tự cũng quan trọng: nói chung $AB\ne BA$, ngay cả khi cả hai phép nhân đều có nghĩa. Ngữ nghĩa `@` cho mảng 1-D, 2-D và nhiều trục được đối chiếu với [NumPy: matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html). Tài liệu NumPy dùng quy tắc xử lý 1-D đặc biệt, nên hãy đọc shape thay vì áp dụng máy móc quy tắc ma trận 2-D.

## 7. Chuyển vị và phép gộp theo trục

**Transpose — chuyển vị** của ma trận đổi hàng thành cột:

$$
X^T=\begin{bmatrix}1&2&3\\2&3&2\end{bmatrix}
$$

`X.T` có shape `(2, 3)`. Nhưng nếu `v.shape == (3,)`, thì `v.T.shape` vẫn là `(3,)`: không có trục hàng/cột thứ hai để hoán đổi. Muốn một cột, dùng `v[:, None]` hoặc `v.reshape(3, 1)`; muốn một hàng, dùng `v[None, :]`. `None` thêm một trục độ dài 1. [NumPy: transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

Với phép tính trung bình, `axis` xác định trục được gộp:

```python
X.mean(axis=0)  # [2., 2.333333...], shape (2,)
X.mean(axis=1)  # [1.5, 2.5, 2.5], shape (3,)
```

- `axis=0`: gộp qua các hàng/mẫu, giữ một kết quả cho mỗi đặc trưng.
- `axis=1`: gộp qua các cột, giữ một kết quả cho mỗi mẫu.

Cách nhớ là **trục được chỉ định bị gộp lại**, trừ khi dùng `keepdims=True` để giữ trục độ dài 1. Trung bình theo hàng ở ví dụ chỉ minh họa cơ chế; trộn số lô và số truy vấn bằng trung bình không tự tạo ra một đại lượng nghiệp vụ có ý nghĩa.

## 8. Broadcasting: tiện lợi nhưng phải đúng ý định

Broadcasting cho phép phép toán theo phần tử giữa một số mảng khác shape. NumPy so các kích thước **từ phải sang trái**: tương thích nếu bằng nhau hoặc một bên bằng 1; trục còn thiếu được xem như có kích thước 1. [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

| Phép toán | Shape kết quả | Diễn giải |
|---|---|---|
| `(3,2) + scalar` | `(3,2)` | Cộng cùng số vào mọi phần tử |
| `(3,2) * (2,)` | `(3,2)` | Nhân theo từng cột bằng trọng số tương ứng |
| `(3,1) + (2,)` | `(3,2)` | Tạo mọi cặp hàng–cột bằng broadcasting |
| `(3,2) + (3,)` | Lỗi | Kích thước cuối 2 và 3 không tương thích |

Việc hình dung “lặp lại mảng nhỏ” giúp hiểu giá trị, không có nghĩa NumPy luôn sao chép vật lý mảng đó. Ngược lại, kết quả của một phép broadcasting lớn vẫn có thể tốn nhiều bộ nhớ.

### Lỗi không báo exception: loss so mọi cặp mẫu

Giả sử dự đoán đúng hoàn toàn: `pred = np.array([5., 8., 9.])`, `y` giống nó.

```python
pred - y            # shape (3,), giá trị [0., 0., 0.]
pred[:, None] - y   # shape (3,3), không còn so từng cặp tương ứng!
```

Ở dòng thứ hai, `(3,1)` trừ `(3,)` được hiểu thành `(3,1)` trừ `(1,3)`. Kết quả:

$$
\begin{bmatrix}5-5&5-8&5-9\\8-5&8-8&8-9\\9-5&9-8&9-9\end{bmatrix}
=\begin{bmatrix}0&-3&-4\\3&0&-1\\4&1&0\end{bmatrix}
$$

Lấy trung bình bình phương cả ma trận cho $52/9\approx5.777778$ ms², dù dự đoán từng mẫu đang đúng.

![So sánh sai số đúng theo từng mẫu và sai số mọi cặp do broadcasting](figures/03-broadcasting.png)

*Hình 3. Cột bên trái chỉ vẽ dọc ba sai số của mảng 1-D, không có nghĩa mảng đúng có shape `(3,1)`. Bên phải là phép tính khác, không phải lỗi làm tròn.*

Cách xử lý là thống nhất shape có chủ đích. Bài này chọn cả dự đoán và đích là `(n,)` rồi kiểm tra bằng điều kiện rõ ràng. Chọn cả hai là `(n,1)` cũng có thể đúng với một quy ước khác. Không dùng `reshape(-1)` như một cách che mọi lỗi mà chưa biết ý nghĩa trục ban đầu.

## 9. Vector hóa gradient: nối trực tiếp với bài 2

Với $d$ đặc trưng:

$$
\hat{\mathbf y}=X\mathbf w+b\mathbf1,\quad
\mathbf r=\hat{\mathbf y}-\mathbf y,\quad
L=\frac1n\sum_{i=1}^{n}r_i^2
$$

Từ cùng cách suy ra ở bài 2, gradient của trọng số thứ $j$ là $\frac{2}{n}\sum_i r_iX_{ij}$. Mỗi cột của $X$ cần một tổng có trọng số như vậy. Ghép tất cả lại:

$$
\nabla_{\mathbf w}L=\frac{2}{n}X^T\mathbf r,\qquad
\frac{\partial L}{\partial b}=\frac{2}{n}\sum_i r_i
$$

Kiểm tra shape trước khi tính số:

```text
X.T:       (d, n)
r:            (n,)
X.T @ r:   (d,)      ← một gradient cho mỗi trọng số
mean(r):   scalar    ← gradient bias dùng 2 * mean(r)
```

Chia cho số **mẫu $n$**, không phải số đặc trưng $d$. Công thức vẫn dùng MSE chia cho $n$, đúng quy ước bài 2.

### Tính tay một bước với hai đặc trưng

Khởi đầu $\mathbf w=(1,0),b=0$, learning rate $\alpha=0.01$.

| Mẫu | Đầu vào | Dự đoán | Đích | Sai số |
|---|---|---:|---:|---:|
| 0 | (1, 2) | 1 | 5 | −4 |
| 1 | (2, 3) | 2 | 8 | −6 |
| 2 | (3, 2) | 3 | 9 | −6 |

Loss ban đầu là $(16+36+36)/3=88/3\approx29.333333$ ms².

$$
X^T\mathbf r=
\begin{bmatrix}1&2&3\\2&3&2\end{bmatrix}
\begin{bmatrix}-4\\-6\\-6\end{bmatrix}
=\begin{bmatrix}-34\\-38\end{bmatrix}
$$

Do đó $\nabla_{\mathbf w}L=(-68/3,-76/3)$ và $\partial L/\partial b=-32/3$. Cập nhật đồng thời:

$$
\mathbf w_{\text{mới}}=(92/75,19/75)\approx(1.226667,0.253333),\quad
b_{\text{mới}}=8/75\approx0.106667
$$

Dự đoán mới là $(1.84,3.32,4.293333\ldots)$, loss mới xấp xỉ **18.013570 ms²**. Đây là cùng cơ chế gradient descent của bài 2, được viết gọn cho nhiều mẫu và nhiều đặc trưng.

Trong code:

```python
pred = X @ w + b
r = pred - y                  # Chỉ làm sau khi đã kiểm tra shape.
dw = (2.0 / len(y)) * (X.T @ r)
db = 2.0 * np.mean(r)
w, b = w - lr * dw, b - lr * db
```

Lab có cả công thức vector hóa và vòng lặp độc lập để đối chiếu kết quả. Chưa có phép đánh giá ngoài dữ liệu huấn luyện trong ví dụ hai đặc trưng này; train loss giảm không phải bằng chứng tổng quát hóa.

## 10. Tensor: nhiều trục hơn, cùng nguyên tắc đọc shape

Trong lập trình Deep Learning, ta thường làm việc với **tensor dưới dạng mảng số nhiều chiều**: scalar 0 trục, vector 1 trục, ma trận 2 trục và các mảng nhiều trục hơn. Đây là cách dùng thực hành, chưa phải định nghĩa tensor trừu tượng theo quy tắc biến đổi tọa độ trong toán/vật lý.

PyTorch dùng tensor cho dữ liệu, đầu ra và tham số; tensor có các thuộc tính shape, dtype và device. Bài này chỉ dùng NumPy để học phép tính; phần PyTorch và tính gradient tự động sẽ học sau. [PyTorch: Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

Shape cần đi cùng **ý nghĩa trục**, không chỉ các con số:

| Loại dữ liệu / quy ước | Shape ví dụ | Ý nghĩa |
|---|---|---|
| Dữ liệu bảng | `(32, 10)` | 32 mẫu, 10 đặc trưng |
| Batch ảnh NHWC | `(2, 2, 3, 3)` | 2 ảnh, cao 2, rộng 3, 3 kênh |
| Batch ảnh NCHW | `(2, 3, 2, 3)` | Cùng các trục nhưng kênh đứng trước chiều cao/rộng |
| Biểu diễn chuỗi BTD | `(32, 128, 768)` | 32 chuỗi, 128 vị trí, 768 thành phần biểu diễn tại mỗi vị trí |

`N` hoặc `B` là số mẫu, `H/W` là cao/rộng, `C` là kênh; `T` là độ dài chuỗi, `D` là kích thước biểu diễn. Đây là nhãn do người thiết kế quy ước, không phải tên trục tự được NumPy lưu. Số 128 hay 768 chỉ là ví dụ, không phải kích thước bắt buộc của mô hình hiện đại.

Một mảng `(32,128,768)` có 3 trục, không có nghĩa đó là “32 không gian ba chiều”. Các trục có vai trò khác nhau. Chiều dài chuỗi khác nhau còn cần padding/mask hoặc cách biểu diễn khác; sẽ học khi làm dữ liệu chuỗi.

![Tensor ảnh được tách thành hai mẫu và ba kênh để quan sát giá trị](figures/04-tensor.png)

*Hình 4. Dữ liệu là các số 0–35 do `arange` tạo; màu nền chỉ giúp phân biệt kênh, không phải ảnh RGB thực được dựng lại. Mỗi ô vẫn có đúng một chỉ số trên bốn trục N,H,W,C.*

## 11. Reshape khác hoán vị trục

**Reshape** đổi cách chia dãy phần tử theo shape mới, phải giữ tổng số phần tử. Với thứ tự mặc định C, đọc/ghi theo thứ tự chỉ số mà trục cuối thay đổi nhanh nhất. `np.arange(6).reshape(2,3)` tạo `[[0,1,2],[3,4,5]]`.

**Transpose/hoán vị trục** đổi vị trí trục, giữ mối liên hệ giữa giá trị và các chỉ số mang ý nghĩa. Với mảng nhiều trục cần chỉ rõ thứ tự mong muốn; dùng `.T` không đối số sẽ đảo ngược toàn bộ trục, không tự suy ra batch hay channel. [NumPy: reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html), [transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

Ví dụ thực hành:

```python
images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)  # NHWC
nchw = images.transpose(0, 3, 1, 2)                        # NCHW
wrong = images.reshape(2, 3, 2, 3)                        # Cùng shape đích, sai ánh xạ

images[1, 0, 2, 1]  # 25
nchw[1, 1, 0, 2]    # 25: cùng ảnh, pixel và kênh
wrong[1, 1, 0, 2]   # 26: không còn cùng vị trí có ý nghĩa
```

`transpose(0,3,1,2)` nghĩa là trục đầu ra lần lượt lấy từ trục cũ 0, 3, 1, 2. Tổng phần tử và shape kết quả đúng chưa đủ chứng minh dữ liệu được bố trí đúng.

`images.reshape(2, -1)` cho shape `(2,18)`: mỗi ảnh được trải phẳng thành 18 số. `-1` yêu cầu NumPy suy một kích thước còn thiếu, chỉ được có một trục như vậy. Không làm mất giá trị số trong phép reshape này, nhưng thông tin cấu trúc không gian không còn hiện thành các trục riêng. Không thể reshape 36 phần tử thành `(2,20)`.

Reshape và transpose có thể trả view dùng chung dữ liệu; reshape trong một số trường hợp phải sao chép. Đừng mặc định đổi shape là miễn phí, hoặc sửa view luôn độc lập với bản gốc. Phần quản lý bộ nhớ sẽ học sâu khi tối ưu.

## 12. Nối sang mạng neural và batch matmul

Một lớp biến đổi tuyến tính cho $n$ mẫu, $d$ đầu vào và $k$ đầu ra có thể viết:

$$
Z=XW+\mathbf b
$$

Trong code, `X` có shape `(n,d)`, `W` có shape `(d,k)`, bias có shape `(k,)` và `Z` có shape `(n,k)`. Bias được broadcast theo từng hàng. Kết quả có thể đi qua hàm kích hoạt; chưa có activation thì đây vẫn chỉ là phép biến đổi affine.

Với dữ liệu nhiều trục, chẳng hạn `(B,T,D) @ (D,K)`, NumPy có thể tạo `(B,T,K)`, dùng chung ma trận biến đổi cho các vị trí. `@` khi đó không nhân theo mọi trục tùy ý; nó có quy tắc cho các trục cuối và broadcasting trên các trục trước. Viết ý nghĩa và shape trước khi dùng.

## 13. Thực hành và môi trường

Mã: [lab.py](lab.py). Môi trường đã kiểm tra: **Python 3.14.6, NumPy 2.5.4**. Từ bài này, code thực hành cần NumPy; bài 1–2 vẫn chỉ dùng thư viện chuẩn. Không cần GPU hay tài khoản AI.

Nếu chưa có NumPy, từ thư mục bài 3, tạo môi trường ảo. Lệnh cài cần mạng:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode all
```

Ví dụ dùng cú pháp macOS/Linux; trên Windows dùng `.venv\Scripts\python`. Nếu Python đang dùng đã cài NumPy, chạy trực tiếp:

```bash
python3 lab.py --mode shapes
python3 lab.py --mode predict
python3 lab.py --mode gradient
python3 lab.py --mode broadcast
python3 lab.py --mode tensor
```

`--mode all` chạy cả năm phần trên. Nếu terminal ở gốc bộ tài liệu, thay `lab.py` bằng `03-vector-ma-tran-tensor/lab.py`.

Kết quả chính để đối chiếu:

| Mode | Kết quả cần quan sát |
|---|---|
| `shapes` | Scalar `()`, vector `(3,)`, hàng `(1,3)`, cột `(3,1)`; `v.T` vẫn `(3,)` |
| `predict` | `X @ w + 1` bằng `[5,8,9]`; trung bình theo cột `[2, 2.333333...]` |
| `gradient` | `dw≈[-22.666667,-25.333333]`, `db≈-10.666667`; loss sau cập nhật ≈18.013570 |
| `broadcast` | Loss đúng bằng 0; phép broadcasting sai ý định cho ≈5.777778 |
| `tensor` | NHWC `(2,2,3,3)`; NCHW `(2,3,2,3)`; giá trị kiểm tra 25, 25 và 26 |

Để chạy nhiều bước thay vì chỉ một bước tính tay:

```bash
python3 lab.py --mode train --lr 0.01 --steps 10000
```

Kết quả gần `w=[2.000157,1.000602]`, `b=0.998232`, train MSE khoảng $9.94\times10^{-8}$. Chưa đúng tuyệt đối sau số bước hữu hạn; cấu hình này không phải khuyến nghị learning rate/số bước cho mọi dữ liệu. Các mode minh họa ngoài `train` không dùng tùy chọn `--lr`/`--steps`.

## 14. Hình minh họa, hiểu nhầm và bước tiếp theo

Hình đã có PNG và SVG trong [thư mục figures](figures/). Không cần thư viện vẽ chỉ để đọc bài. Muốn tái tạo hình, từ thư mục bài 3 dùng môi trường ảo ở trên:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Mã vẽ: [make_figures.py](make_figures.py); phụ thuộc thực hành: [requirements.txt](requirements.txt); phụ thuộc vẽ: [requirements-figures.txt](requirements-figures.txt). Hình đã kiểm tra với Matplotlib 3.11.2, NumPy 2.5.4. Các gói ghim phiên bản cần một Python/nền tảng tương thích.

Các điểm cần tự kiểm tra:

- Đúng shape chưa chắc đúng thứ tự đặc trưng hoặc đúng ý nghĩa trục.
- Broadcasting hợp lệ không đồng nghĩa phép tính đúng mục tiêu.
- `*`, `@`, reshape và transpose giải quyết các việc khác nhau.
- Số trục mảng khác số thành phần vector và khác hạng ma trận.
- Tensor không mặc định là mảng 3 trục, không mặc định nằm trên GPU và không mặc định đang theo dõi gradient.

Làm [bài tập](exercises.md), rồi đọc [lời giải](solutions.md). Có thể học tiếp khi tự ghi được shape dự đoán/gradient và tìm ra lỗi so cặp mẫu do broadcasting. Chưa cần học nghịch đảo ma trận, eigenvalue hay phân rã ma trận để đạt mục tiêu của bài này.

**Học tiếp:** [Bài 4 — Xác suất, thống kê và bất định](../04-xac-suat-thong-ke-bat-dinh/README.md), có ví dụ tính tay, mô phỏng lấy mẫu và hình minh họa. Nguồn và phần đã đọc: [danh mục nguồn](../SOURCES.md).
