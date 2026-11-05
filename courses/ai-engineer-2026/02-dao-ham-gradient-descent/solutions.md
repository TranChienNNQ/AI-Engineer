# Lời giải — Đạo hàm và gradient descent

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các phép tính được đối chiếu với code; kết quả gần đúng được làm tròn để dễ đọc. Độ lệch rất nhỏ giữa các môi trường có thể do số thực dấu phẩy động.

## Bài 1

$f(1)=1$, $f(1.1)=0.81$. Độ dốc trung bình là $(0.81-1)/0.1=-1.9$. Đạo hàm đúng tại 1 là $f'(1)=2(1-2)=-2$.

Độ dốc trung bình dùng khoảng hữu hạn, trong khi đạo hàm là giới hạn khi khoảng thu nhỏ về 0. Với hàm này, công thức độ dốc một phía là $2(w-2)+h$, nên độ lệch đúng bằng $h=0.1$. Đạo hàm âm: tăng $w$ một lượng đủ nhỏ làm hàm giảm tại vùng đang xét.

## Bài 2

Bước 1: $f'(1)=-2$, nên $w_1=1-0.1(-2)=1.2$ và $f(w_1)=0.64$.

Bước 2: $f'(1.2)=-1.6$, nên $w_2=1.2-0.1(-1.6)=1.36$ và $f(w_2)=0.4096$.

Nếu dùng dấu cộng, bước đầu cho $w_1=1+0.1(-2)=0.8$, loss thành $1.44$, lớn hơn 1 ban đầu. Ta đã đi theo hướng tăng cục bộ thay vì hướng giảm. Nếu gradient bằng 0 thì đổi dấu không tạo chuyển động, nên không dùng trường hợp ấy để kiểm tra dấu.

## Bài 3

| Learning rate | $w_1$ | $w_2$ | $f(w_1)$ | $f(w_2)$ |
|---:|---:|---:|---:|---:|
| 0.5 | 2 | 2 | 0 | 0 |
| 1.0 | −1 | 5 | 9 | 9 |
| 1.1 | −1.6 | 6.32 | 12.96 | 18.6624 |

Với $\alpha=0.5$ chỉ cần một bước. Với 1, khoảng cách tới nghiệm giữ nguyên và đổi dấu; với 1.1, độ lớn khoảng cách tăng 1.2 lần mỗi bước.

Sau 12 bước, mode `compare` cho xấp xỉ:

| $\alpha$ | $w_{12}$ | Loss |
|---:|---:|---:|
| 0.01 | 4.354150 | 5.542023 |
| 0.25 | 2.000732 | $5.364418\times10^{-7}$ |
| 1.00 | 5.000000 | 9.000000 |
| 1.10 | 28.748301 | 715.471625 |

Mở rộng: $e_t=(1-2\alpha)^te_0$. Với $e_0\ne0$, điều kiện $|1-2\alpha|<1$ làm khoảng cách tiến về 0, suy ra $0<\alpha<1$. Hai biên cho không giảm khoảng cách; bên ngoài làm khoảng cách tăng. Nếu khởi đầu đúng nghiệm, ví dụ này đứng yên dù learning rate lớn.

Kết luận phụ thuộc độ cong của đúng hàm $f$. Hồi quy hai tham số có mặt loss khác; ví dụ `linear --lr 0.2` phân kỳ dù 0.2 thuộc khoảng trên. Tính lồi không thay thế điều kiện bước cập nhật phù hợp.

## Bài 4

Ở $(0,0)$, ba dự đoán đều bằng 0. Các sai số là −3, −5, −7. Tổng sai số bằng −15; tổng sai số nhân đầu vào bằng $-3-10-21=-34$.

Loss ban đầu $=(9+25+49)/3=83/3\approx27.666667$ ms².

Hai gradient:

$$
g_w=\frac{2}{3}(-34)=-\frac{68}{3},\qquad g_b=\frac{2}{3}(-15)=-10
$$

Cập nhật $\alpha=0.05$:

$$
w_1=\frac{17}{15}\approx1.133333,\qquad b_1=0.5
$$

Ba dự đoán là $49/30,83/30,117/30$, tương ứng khoảng 1.633333, 2.766667, 3.9. Sai số là $-41/30,-67/30,-93/30$.

$$
L_1=\frac{41^2+67^2+93^2}{900\times3}
=\frac{14819}{2700}\approx5.488519\text{ ms}^2
$$

Điểm mới vẫn dự đoán thấp hơn cả ba đích, nhưng loss đã giảm. Dữ liệu giữ riêng không xuất hiện trong bất kỳ tổng tính gradient nào.

## Bài 5

Code tính $g_w$ tại $(1,0)$, cập nhật $w$ thành $7/3$, rồi tính $g_b$ tại $(7/3,0)$. Đây không phải quy tắc cập nhật đồng thời trong bài.

Ở điểm thứ hai, ba sai số là $-2/3,-1/3,0$, tổng −1, nên $g_b=-2/3$. Kết quả $b=0-0.1(-2/3)=1/15\approx0.066667$, khác 0.6.

Cách sửa:

```python
dw, db = gradients(data, w, b)
w = w - lr * dw
b = b - lr * db
```

Hai phép gán nằm ở hai dòng vẫn đúng vì cả `dw` và `db` đã được tính trước khi đổi `w`. Cũng có thể dùng phép gán tuple như trong lab. Vấn đề là trạng thái dùng để tính gradient, không phải số dòng.

## Bài 6

Tại $(2.3,-0.7)$, ba dự đoán là 1.6, 3.9, 6.2 và các sai số là −1.4, −1.1, −0.8.

$$
g_w=\frac{2}{3}(-1.4-2.2-2.4)=-4,\qquad
g_b=\frac{2}{3}(-3.3)=-2.2
$$

Sai phân trung tâm với $h=10^{-5}$ trả gần $(-4,-2.2)$, sai lệch nhỏ hơn $10^{-8}$ trong môi trường đã kiểm tra. Nếu quên hệ số 2, công thức giải tích cho $(-2,-1.1)$, lệch rõ. Cần so cả công thức với đúng định nghĩa MSE, không tùy tiện bỏ hệ số để khớp kết quả.

Hai đường tính có sai số làm tròn khác nhau nên không nên yêu cầu `==`. Có thể dùng `math.isclose` với ngưỡng tuyệt đối và tương đối phù hợp. Kiểm tra vài điểm chỉ kiểm tra các điểm ấy; cũng nên tránh chỉ thử tại nghiệm, nơi gradient đúng và nhiều công thức sai đều có thể bằng 0.

## Bài 7

$w'x_i'+b=(w/100)(100x_i)+b=wx_i+b$, nên các sai số $r_i$ giữ nguyên tại hai cách biểu diễn tương ứng.

Gradient theo $w'$ là:

$$
\frac{2}{n}\sum r_ix_i'=100\frac{2}{n}\sum r_ix_i
$$

Nó lớn gấp 100 lần về giá trị số; gradient theo $b$ không đổi tại điểm tương ứng. Dùng cùng learning rate làm bước cập nhật của $w'$ lớn hơn, trong khi để biểu diễn cùng thay đổi mô hình thì bước của $w'$ cần nhỏ hơn bước của $w$ 100 lần. Vì vậy cùng learning rate không tạo hành vi tương đương. Việc có thể cần điều chỉnh theo từng tọa độ cũng cho thấy vì sao chuẩn hóa và lựa chọn optimizer đáng học tiếp.

## Bài 8

Hình trái có trục ngang là $w$, trục đứng là $b$; hình phải có trục ngang là số lần cập nhật và trục đứng là MSE trên thang logarit. Gradient phụ thuộc vị trí hiện tại và độ cong theo các hướng, nên đường cập nhật không bắt buộc là đoạn thẳng nối tới nghiệm.

Train loss gần 0 chỉ cho thấy khớp tập huấn luyện giả định này. Nó chưa cho biết lỗi khi tải cao, cache khác hoặc số lô xa khoảng quan sát. Mẫu giữ riêng cũng quá ít để chứng minh độ tin cậy thực tế.

Với `linear --lr 0.2 --steps 20`, loss tăng từ khoảng 9.67 lên khoảng 26 190 ms². Đó là phản ví dụ cho kết luận “hàm lồi thì learning rate nào cũng hội tụ”. Đặc tính của mục tiêu và đặc tính của thuật toán với cấu hình cụ thể phải xét cùng nhau.

## Nếu chưa đạt tiêu chí

- Nhầm giá trị hàm và đạo hàm: làm lại bài 1 và ghi hai cột riêng.
- Nhầm dấu cập nhật: làm lại bài 2, kiểm tra loss trước/sau.
- Nhầm gradient nhiều biến: giữ $b$ cố định, chỉ đổi $w$ trong sai phân rồi làm ngược lại.
- Chạy code được nhưng chưa hiểu cập nhật: chỉ chạy 1 bước và đối chiếu từng tổng ở bài 4.

Không có kết luận về mức hiểu của người học chỉ từ việc tài liệu hoặc code đã hoàn thiện.
