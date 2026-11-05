# Bài 2 — Đạo hàm và gradient descent

[Mục lục](../README.md) · [Bài 1](../01-mo-hinh-hoc-tu-du-lieu/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn và kiểm tra mã ngày 04/11/2026. Các ví dụ, phép tính và đồ thị được tạo cho bài học, không phải dữ liệu benchmark. Bài gồm hai lượt đọc: phần 1–6 xây trực giác một biến; phần 7–12 áp dụng vào mô hình có hai tham số.

## 1. Mục tiêu và điều kiện bắt đầu

Sau bài này, người học có thể:

- Giải thích đạo hàm bằng mức thay đổi đầu ra khi đầu vào thay đổi rất nhỏ.
- Phân biệt giá trị hàm với độ dốc của hàm tại một điểm.
- Tính tay một bước gradient descent, hiểu dấu trừ và learning rate.
- Tính gradient MSE theo $w,b$ và cập nhật hai tham số từ cùng trạng thái cũ.
- Chạy code, kiểm tra gradient bằng sai phân và nhận diện vài dạng hội tụ/phân kỳ.

Cần biết hàm $\hat y=wx+b$, MSE, vòng lặp và hàm Python từ [bài 1](../01-mo-hinh-hoc-tu-du-lieu/README.md). Không cần học trước đạo hàm; bài này giải thích từ phép chia thông thường. Khi gặp công thức dài, đi theo ví dụ số trước rồi quay lại ký hiệu.

## 2. Vấn đề của việc thử mọi ứng viên

Bài 1 thử 41 giá trị cho mỗi tham số, tổng cộng 1681 cặp $(w,b)$. Nếu có $d$ tham số và thử $k$ giá trị cho mỗi tham số, số tổ hợp là $k^d$. Với 10 tham số, mỗi tham số chỉ 10 ứng viên cũng đã là $10^{10}$ tổ hợp.

Ta muốn dùng thông tin ngay tại bộ tham số hiện có để chọn hướng điều chỉnh. Đạo hàm cung cấp thông tin cục bộ ấy. Thuật toán gradient descent lặp việc tính gradient và cập nhật tham số; đây là cơ chế được trình bày trong [Google: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent). Phép tìm kiếm và dữ liệu minh họa dưới đây do bộ tài liệu xây riêng.

## 3. Bắt đầu từ độ dốc: đầu vào tăng thì đầu ra đổi bao nhiêu?

Với đường thẳng $g(w)=3w+1$, tăng $w$ từ 2 lên 4 làm giá trị tăng từ 7 lên 13. Độ dốc là:

$$
\frac{\text{thay đổi đầu ra}}{\text{thay đổi đầu vào}}
=\frac{13-7}{4-2}=3
$$

Đường thẳng này có độ dốc 3 ở mọi điểm. Với đường cong, độ dốc thay đổi theo vị trí.

Để nhìn rõ, tạm dùng một hàm mất mát đơn giản, không có đơn vị vật lý:

$$
f(w)=(w-2)^2
$$

Đây là bài toán một tham số để học toán, **khác MSE hồi quy hai tham số** ở phần sau. Nó có giá trị nhỏ nhất 0 tại $w=2$.

Ở $w=5$, $f(5)=9$. Tăng thêm $h=0.1$ thì $f(5.1)=9.61$. Độ dốc trung bình trên khoảng nhỏ đó là:

$$
\frac{f(5.1)-f(5)}{0.1}=\frac{9.61-9}{0.1}=6.1
$$

Ở đây $h$ là lượng thay đổi đầu vào, không phải learning rate.

| $h$ | $[f(5+h)-f(5)]/h$ |
|---:|---:|
| 1 | 7 |
| 0.1 | 6.1 |
| 0.01 | 6.01 |
| 0.001 | 6.001 |

Các giá trị tiến gần 6. Đạo hàm tại 5 chính là giới hạn của độ dốc này khi khoảng thay đổi tiến về 0:

$$
f'(5)=\lim_{h\to0}\frac{f(5+h)-f(5)}{h}=6
$$

Dấu phẩy trong $f'$ đọc là “đạo hàm”. Ký hiệu $h\to0$ nghĩa là xét những khoảng nhỏ dần, **không thay trực tiếp $h=0$ vào phép chia**.

![Đường cong loss, tiếp tuyến tại w=5 và một cát tuyến gần đó](figures/01-dao-ham.png)

*Hình 1. Đường xanh là hàm, đường cam là tiếp tuyến biểu diễn độ dốc tại đúng một điểm. Cát tuyến xanh lá dùng hai điểm cách nhau 0.8 nên có độ dốc 6.8, chưa bằng đạo hàm 6. Đồ thị tính từ công thức, không phải hình vẽ ước lượng.*

### Suy ra công thức đạo hàm, không cần học thuộc trước

Khai triển tử số với $w$ bất kỳ:

$$
\frac{(w+h-2)^2-(w-2)^2}{h}
=\frac{2(w-2)h+h^2}{h}=2(w-2)+h
$$

Khi $h\to0$, phần còn lại là:

$$
f'(w)=2(w-2)
$$

Trong ngôn ngữ vi phân, cũng có thể viết $df/dw$. Nó là mức nhạy cục bộ của $f$ đối với $w$, không phải giá trị loss.

## 4. Dấu của đạo hàm cho biết hướng nào?

| Vị trí | Giá trị hàm | Đạo hàm | Ý nghĩa khi tăng nhẹ $w$ |
|---|---:|---:|---|
| $w=5$ | 9 | 6 | Hàm tăng; giảm $w$ có thể giúp giảm hàm |
| $w=1$ | 1 | −2 | Hàm giảm; tăng $w$ có thể giúp giảm hàm |
| $w=2$ | 0 | 0 | Độ dốc bằng 0 tại cực tiểu của hàm này |

Một cách đọc chính xác hơn là xấp xỉ tuyến tính:

$$
f(w+\Delta w)\approx f(w)+f'(w)\Delta w
$$

$\Delta w$ là thay đổi nhỏ của tham số. Muốn phần thay đổi dự kiến âm, chọn $\Delta w$ ngược dấu với $f'(w)$.

Từ “nhỏ” quan trọng: đạo hàm mô tả lân cận hiện tại, không bảo đảm một bước rất lớn sẽ làm hàm giảm. Và đạo hàm bằng 0 cũng không luôn là cực tiểu: chẳng hạn $q(w)=-w^2$ có đạo hàm bằng 0 tại 0 nhưng đó là cực đại. Với hàm nhiều biến còn có điểm yên ngựa; các kiến trúc sâu sẽ cần thận trọng hơn ví dụ parabol này.

## 5. Gradient descent một biến: đi ngược độ dốc

Quy tắc cập nhật là:

$$
w_{t+1}=w_t-\alpha f'(w_t)
$$

- $t$ là số thứ tự trạng thái: $t=0$ là trước lần cập nhật đầu tiên.
- $\alpha>0$ là **learning rate — hệ số điều chỉnh độ lớn bước cập nhật**.
- Dấu trừ chọn hướng ngược đạo hàm.
- Kích thước bước thực tế là $|\alpha f'(w_t)|$, không chỉ $\alpha$.

Với $w_0=5$, $\alpha=0.1$:

1. Tính loss cũ: $f(5)=9$.
2. Tính đạo hàm tại điểm cũ: $f'(5)=2(5-2)=6$.
3. Cập nhật: $w_1=5-0.1\times6=4.4$.
4. Tính lại loss: $f(4.4)=(4.4-2)^2=5.76$.

Lần tiếp theo phải tính lại đạo hàm ở **4.4**, không dùng mãi độ dốc tại 5.

| Số lần cập nhật | $w$ | $f(w)$ |
|---:|---:|---:|
| 0 | 5.00000 | 9.00000 |
| 1 | 4.40000 | 5.76000 |
| 2 | 3.92000 | 3.68640 |
| 3 | 3.53600 | 2.359296 |
| 4 | 3.22880 | 1.50994944 |
| 5 | 2.98304 | 0.9663676416 |

Ta không cần thử toàn bộ các giá trị $w$ trong một khoảng. Mỗi vòng dùng đạo hàm ở vị trí hiện tại để tính vị trí tiếp theo.

## 6. Learning rate: nhỏ quá, vừa đủ, lớn quá

![So sánh đường đi của tham số và loss khi thay đổi learning rate](figures/02-learning-rate.png)

*Hình 2. Cùng hàm một biến và điểm bắt đầu $w_0=5$, chạy 12 cập nhật. Hình phải dùng trục logarit để nhìn được các mức loss khác nhau; khoảng cách bằng nhau trên trục đứng biểu diễn tỷ lệ, không phải chênh lệch tuyệt đối.*

Với đúng hàm $f(w)=(w-2)^2$:

- $\alpha=0.01$: tiến chậm trong 12 bước.
- $\alpha=0.25$: khoảng cách tới 2 giảm một nửa sau mỗi bước.
- $\alpha=1$: tham số luân phiên $5,-1,5,-1,\ldots$, loss luôn bằng 9.
- $\alpha=1.1$: nhảy qua nghiệm và ngày càng xa, loss tăng.

Để tự kiểm chứng, đặt $e_t=w_t-2$ là khoảng cách có dấu tới nghiệm. Thay vào công thức cập nhật:

$$
e_{t+1}=(1-2\alpha)e_t
$$

Muốn độ lớn khoảng cách giảm, cần $|1-2\alpha|<1$, tương đương $0<\alpha<1$. Điều kiện này dành cho **hàm một biến cụ thể đang xét**, với điểm khởi đầu khác nghiệm. Không dùng nó làm ngưỡng chung cho hồi quy hoặc mạng neural. Riêng $\alpha=0.5$ đưa hàm này tới nghiệm chỉ sau một bước; điều đó cũng không tạo ra một learning rate tối ưu cho mọi bài toán.

Learning rate là **hyperparameter — siêu tham số** do người thiết kế cấu hình, khác $w,b$ được cập nhật trong huấn luyện. Vai trò và đánh đổi này được đối chiếu với [Google: Hyperparameters](https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters). Giá trị phù hợp phụ thuộc hàm loss và thang đo dữ liệu.

## 7. Từ đạo hàm sang đạo hàm riêng và gradient

Quay về dữ liệu của bài 1: `(1, 3), (2, 5), (3, 7)`; $x$ là số lô và $y$ là latency giả định, ms.

$$
\hat y_i=wx_i+b,\qquad
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2
$$

Loss phụ thuộc **hai biến $w,b$**. Ta cần trả lời riêng:

- Nếu thay đổi nhẹ $w$ và giữ $b$ cố định, loss thay đổi thế nào?
- Nếu thay đổi nhẹ $b$ và giữ $w$ cố định, loss thay đổi thế nào?

Đó là hai **đạo hàm riêng**, viết $\partial L/\partial w$ và $\partial L/\partial b$. Ký hiệu $\partial$ dùng ở đây để nhấn mạnh việc chỉ đổi một biến.

Gộp chúng thành một vector là **gradient**:

$$
\nabla L=\left(\frac{\partial L}{\partial w},\frac{\partial L}{\partial b}\right)
$$

Đọc $\nabla L$ là “gradient của L”. Về mặt hình học trong hệ tọa độ đang dùng, gradient chỉ hướng tăng cục bộ nhanh nhất theo chuẩn Euclid; ta dùng hướng ngược lại để giảm loss khi bước đủ nhỏ.

### Tự suy ra gradient của MSE

Đặt $r_i=wx_i+b-y_i$ là sai số của mẫu $i$. Chỉ tăng $w$ thêm $h$ sẽ làm sai số thành $r_i+hx_i$. Bình phương sai số thay đổi:

$$
\frac{(r_i+hx_i)^2-r_i^2}{h}=2r_ix_i+hx_i^2
$$

Khi $h\to0$, ta được $2r_ix_i$. Tương tự, chỉ tăng $b$ thêm $h$ sẽ làm sai số thành $r_i+h$, cho đạo hàm $2r_i$. Lấy trung bình các mẫu:

$$
\frac{\partial L}{\partial w}=\frac{2}{n}\sum_{i=1}^{n}r_ix_i,
\qquad
\frac{\partial L}{\partial b}=\frac{2}{n}\sum_{i=1}^{n}r_i
$$

Hệ số $x_i$ xuất hiện ở gradient của $w$ vì thay $w$ một lượng nhỏ làm dự đoán đổi theo $x_i$; thay $b$ cùng lượng ấy chỉ dịch mọi dự đoán như nhau. Bài về chain rule/backpropagation sẽ tổng quát hóa cách truyền thay đổi này.

Một số tài liệu dùng loss $\frac{1}{2n}\sum r_i^2$ nên công thức gradient không có hệ số 2. Phải đối chiếu **định nghĩa loss** trước khi so công thức hoặc learning rate. Bài này luôn dùng MSE chia cho $n$.

## 8. Tính tay một bước cho cả w và b

Khởi đầu như ứng viên đầu bài 1: $w_0=1,b_0=0$, learning rate $\alpha=0.1$.

| $x_i$ | $y_i$ | Dự đoán | $r_i$ | $r_ix_i$ |
|---:|---:|---:|---:|---:|
| 1 | 3 | 1 | −2 | −2 |
| 2 | 5 | 2 | −3 | −6 |
| 3 | 7 | 3 | −4 | −12 |
| Tổng | | | −9 | −20 |

Loss ban đầu là $29/3\approx9.666667$ ms². Hai đạo hàm riêng:

$$
g_w=\frac{2}{3}(-20)=-\frac{40}{3},\qquad
g_b=\frac{2}{3}(-9)=-6
$$

Cập nhật từ cùng điểm $(1,0)$:

$$
w_1=1-0.1\left(-\frac{40}{3}\right)=\frac{7}{3}\approx2.333333
$$

$$
b_1=0-0.1(-6)=0.6
$$

Cả hai tăng lên vì cả hai gradient đều âm. Dấu trừ trong công thức không có nghĩa luôn giảm giá trị tham số.

Tại điểm mới, dự đoán là $44/15,79/15,114/15$; sai số là $-1/15,4/15,9/15$. Vì vậy:

$$
L(w_1,b_1)=\frac{1+16+81}{225\times3}=\frac{98}{675}\approx0.145185\text{ ms}^2
$$

Loss giảm mạnh, nhưng chưa bằng 0. $w_1$ vượt quá nghiệm 2 cũng không có nghĩa bước cập nhật sai: cả $w$ lẫn $b$ đang cùng thay đổi; phải xét loss tại **cặp tham số**.

### Cập nhật đồng thời, không tính lại gradient giữa chừng

Đúng với thuật toán đang học:

```python
dw, db = gradients(data, w, b)
w, b = w - lr * dw, b - lr * db
```

Nếu đổi `w` rồi tính lại `db` bằng `w` mới, hai gradient không còn thuộc cùng điểm. Khi đó bạn đang thực hiện quy tắc cập nhật khác. Chỉ viết hai phép gán ở hai dòng vẫn đúng nếu `dw, db` đã được tính từ cùng trạng thái cũ.

## 9. Code thực hành và kết quả mong đợi

Mã đầy đủ: [lab.py](lab.py). Đã chạy bằng **Python 3.14.6**, chỉ dùng thư viện chuẩn. Hàm lõi:

```python
def gradients(data, w, b):
    n = len(data)
    dw = 2.0 / n * sum((w * x + b - y) * x for x, y in data)
    db = 2.0 / n * sum(w * x + b - y for x, y in data)
    return dw, db
```

Mỗi lần cập nhật sử dụng toàn bộ tập huấn luyện, gọi là **full-batch gradient descent**. Vì thế trong ví dụ này một lần cập nhật cũng tương ứng một lượt qua toàn bộ dữ liệu; điều đó không còn đúng khi dùng mini-batch.

### A. Quan sát hàm một biến

Từ thư mục bài 2:

```bash
python3 lab.py --mode toy --lr 0.1 --steps 5
python3 lab.py --mode compare
```

Lệnh đầu tái hiện bảng ở phần 5. Lệnh `compare` luôn so bốn learning rate cố định trong 12 bước; không dùng `--lr` và `--steps` để thay bảng này.

### B. Huấn luyện mô hình của bài 1

```bash
python3 lab.py --mode linear --lr 0.1 --steps 1000
```

Nếu terminal ở gốc bộ tài liệu, thay `lab.py` bằng `02-dao-ham-gradient-descent/lab.py`. Chương trình chỉ in trạng thái 0–5 và trạng thái cuối, vẫn tính đủ mọi bước ở giữa.

| Bước | $w$ | $b$ | Train MSE, ms² |
|---:|---:|---:|---:|
| 0 | 1.000000 | 0.000000 | 9.666667 |
| 1 | 2.333333 | 0.600000 | 0.145185 |
| 2 | 2.182222 | 0.546667 | 0.030038 |
| 3 | 2.193481 | 0.564444 | 0.027318 |
| 5 | 2.182810 | 0.584482 | 0.024769 |
| 1000 | ≈2 | ≈1 | ≈$2.31\times10^{-23}$ |

Kết quả cuối gần nghiệm của grid search bài 1, nhưng được tìm bằng các bước gradient. Train loss rất nhỏ là sai số số học và sai số hội tụ còn lại, không cần đúng bằng 0. Chương trình này chạy số bước cố định; chưa có cơ chế tự dừng theo ngưỡng hội tụ.

![Đường cập nhật hai tham số và đường loss giảm qua các bước](figures/03-hoi-quy.png)

*Hình 3. Trái: mỗi đường đồng mức nối các cặp $(w,b)$ có cùng MSE; số trên đường là MSE, ms². Đường xanh là hành trình tham số, ngôi sao là nghiệm $(2,1)$. Phải: MSE theo bước cập nhật, trục đứng logarit. Hai hình cùng dùng dữ liệu và thuật toán trong lab.*

Mẫu giữ riêng `(4, 9)` chỉ dùng **sau huấn luyện**. Nó không tham gia tính gradient. Kết quả dự đoán xấp xỉ 9 ms không thay thế một phép đánh giá thực tế với nhiều dữ liệu.

### C. Thử learning rate quá lớn cho hồi quy

```bash
python3 lab.py --mode linear --lr 0.2 --steps 20
```

Với dữ liệu và khởi tạo này, loss tăng mạnh. Trong khi đó $\alpha=0.2$ vẫn làm hàm một biến ở phần 3 hội tụ. Đây là ví dụ trực tiếp rằng cùng learning rate không đảm bảo cùng hành vi giữa hai bài toán.

## 10. Kiểm tra gradient bằng sai phân số

Có thể kiểm tra công thức đạo hàm bằng cách đo thay đổi loss qua hai điểm gần nhau. Với $h>0$ nhỏ:

$$
g_w^{\text{số}}\approx\frac{L(w+h,b)-L(w-h,b)}{2h}
$$

Tương tự cho $b$, giữ nguyên $w$. Đây là **central difference — sai phân trung tâm**, khác công thức một phía dùng để giới thiệu đạo hàm ở phần 3.

```bash
python3 lab.py --mode check
```

Tại $(w,b)=(1,0)$, cả công thức giải tích lẫn sai phân đều cho xấp xỉ `dw=-13.33333333`, `db=-6.00000000`; sai khác tuyệt đối ở môi trường đã kiểm tra nhỏ hơn $10^{-8}$. Đây là kiểm tra số hữu ích, không phải chứng minh công thức đúng ở mọi điểm.

Không chọn $h$ nhỏ vô hạn trong máy tính: trừ hai số rất gần nhau có thể làm sai số dấu phẩy động nổi bật. `h=1e-5` chỉ là giá trị thực hành phù hợp cho ví dụ này. Sai phân cũng phải đánh giá loss thêm theo từng tham số nên không phải cách ta sẽ huấn luyện mạng lớn; sau này dùng đạo hàm tự động.

## 11. Giới hạn và các hiểu nhầm cần tránh

| Hiểu nhầm | Cách hiểu đúng |
|---|---|
| Gradient descent luôn giảm loss | Chỉ có bảo đảm dưới điều kiện phù hợp; learning rate lớn có thể làm loss tăng |
| Gradient bằng 0 nghĩa là mô hình tốt | Có thể là cực đại, điểm yên ngựa hoặc cực tiểu của mục tiêu chưa phù hợp |
| Loss nhỏ nghĩa là tổng quát hóa tốt | Đang tối ưu train loss; vẫn cần đánh giá trên dữ liệu mới |
| 1000 bước ở đây là mặc định cho mọi mô hình | Chỉ là cấu hình để quan sát ví dụ này |
| Gradient lớn luôn có nghĩa lỗi lớn | Độ lớn còn phụ thuộc thang đo đặc trưng và cách định nghĩa loss |
| Mọi mạng có thể khởi tạo giống ví dụ này | Khởi tạo mạng nhiều neuron có vấn đề đối xứng; sẽ học riêng |

MSE của hồi quy tuyến tính là hàm lồi theo các tham số. Với bộ dữ liệu này có nghiệm tối ưu duy nhất $(2,1)$. Tính lồi không cứu được một learning rate gây phân kỳ; còn mạng neural nói chung không có hình dạng loss đơn giản như vậy.

Nếu đổi đầu vào từ số lô sang số bản ghi, giá trị số của $x$ tăng 100 lần, trọng số biểu diễn cùng hàm dự đoán phải đổi và gradient theo trọng số cũng đổi. Đó là một lý do cần học chuẩn hóa đặc trưng. Code đang dùng các giá trị số theo đơn vị đã quy ước; không chuyển learning rate sang hệ đơn vị khác một cách máy móc.

## 12. Tự học, tái tạo hình và bước tiếp theo

Làm [bài tập](exercises.md), sau đó đối chiếu [lời giải](solutions.md). Có thể chuyển tiếp khi tự tính được một bước cập nhật, giải thích được dấu của đạo hàm, nhận ra lỗi cập nhật không đồng thời và đọc được đường loss.

Hình đã được lưu sẵn dạng PNG và SVG nên đọc tài liệu không cần cài thư viện vẽ. Nếu muốn tự tái tạo hình, dùng môi trường ảo; chỉ bước này cần tải thư viện:

```bash
# Chạy từ thư mục bài 2. Không cần làm bước này để chạy lab.py.
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Các lệnh trên dùng cú pháp macOS/Linux; trên Windows dùng Python trong `.venv\Scripts`. Phiên bản vẽ đã kiểm tra: Matplotlib 3.11.2, NumPy 2.5.4; cần môi trường Python có gói tương thích. Mã vẽ: [make_figures.py](make_figures.py), phụ thuộc: [requirements-figures.txt](requirements-figures.txt). SVG có thể dùng khi xuất bản cần hình co giãn không vỡ nét.

**Học tiếp:** [Bài 3 — Vector, ma trận và tensor](../03-vector-ma-tran-tensor/README.md), biểu diễn phép dự đoán và gradient cho nhiều mẫu, nhiều đặc trưng. Có tính tay, code NumPy và hình minh họa.

Nguồn và phạm vi đối chiếu: [danh mục nguồn](../SOURCES.md). Các suy diễn đại số, số liệu và hình của bài này có thể tái kiểm tra từ code đi kèm.
