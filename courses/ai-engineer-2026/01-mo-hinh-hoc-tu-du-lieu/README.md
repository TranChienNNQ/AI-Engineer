# Bài 1 — Mô hình học từ dữ liệu như thế nào?

[Mục lục](../README.md) · [Đề cương](../ROADMAP.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Trạng thái: đã biên soạn, mã thực hành đã chạy kiểm tra ngày 04/11/2026. Bài này xây nền cho Deep Learning; chưa cần biết mạng neural hoặc đạo hàm.

## 1. Mục tiêu và kiến thức cần trước

Sau bài này, người học có thể:

- Phân biệt dữ liệu đầu vào, đáp án quan sát, dự đoán và tham số.
- Tính dự đoán và trung bình bình phương sai số bằng tay.
- Giải thích việc tìm tham số từ dữ liệu và chạy chương trình minh họa.
- Phân biệt huấn luyện, suy luận và đánh giá trên dữ liệu giữ riêng.
- Nêu vì sao khớp dữ liệu nhỏ chưa chứng minh mô hình hữu ích ngoài thực tế.

Cần biết biến, hàm, vòng lặp Python, phép cộng/nhân và bình phương. Chưa cần NumPy, PyTorch hoặc GPU.

## 2. Từ một nhu cầu backend đến bài toán học

Giả sử một API xử lý bản ghi theo lô, mỗi lô gồm 100 bản ghi. Ta muốn dự đoán thời gian xử lý từ số lô của request.

**Dữ liệu sau là giả định được tạo cho bài học**, không phải số đo benchmark hay quy luật hiệu năng hệ thống thực:

| Số lô | Thời gian xử lý |
|---:|---:|
| 1 | 3 ms |
| 2 | 5 ms |
| 3 | 7 ms |

Nếu viết ứng dụng theo quy tắc thủ công, lập trình viên có thể tự đặt công thức tính thời gian. Với học máy, ta vẫn chọn cách biểu diễn dự đoán, nhưng để một thuật toán tìm giá trị tham số dựa trên các cặp đầu vào–đáp án.

Đây là một ví dụ **học có giám sát (supervised learning)**: dữ liệu huấn luyện có cả đầu vào và giá trị đích. Đích là một số nên bài toán thuộc loại **hồi quy (regression)**. Một bài toán như phân loại ticket thành billing/technical/account có đích là nhóm, gọi là phân loại (classification).

## 3. Mô hình là hàm dự đoán có tham số

Ta chọn dạng:

$$
\hat y = wx+b
$$

Đọc thành lời: thời gian dự đoán bằng hệ số nhân với số lô, cộng một số hạng tự do.

| Ký hiệu | Thuật ngữ | Ý nghĩa và đơn vị |
|---|---|---|
| $x$ | Feature — đặc trưng đầu vào | Số lô, đơn vị lô |
| $y$ | Label/target — giá trị đích | Thời gian quan sát, ms |
| $\hat y$ | Prediction — dự đoán | Thời gian mô hình tính ra, ms |
| $w$ | Weight — trọng số | Thời gian tăng dự đoán trên mỗi lô thêm vào, ms/lô |
| $b$ | Bias/intercept — số hạng tự do | Giá trị mô hình dự đoán ở $x=0$, ms |

Ở bài này, bias là một tham số số học, khác khái niệm thiên lệch xã hội của hệ thống AI. Có thể hình dung $b$ như phần thời gian nền trong mô hình đơn giản; dữ liệu chưa có phép đo tại $x=0$ nên chưa chứng minh nó chính là overhead thực.

Nếu $w=2$ và $b=1$, hàm trở thành $\hat y=2x+1$. Với $x=4$, dự đoán là $2\times4+1=9$ ms.

Trong Python:

```python
def predict(x, w, b):
    return w * x + b
```

Ba thứ cần tách rõ:

1. **Dạng mô hình**: ở đây do người thiết kế chọn là `w * x + b`.
2. **Tham số**: `w` và `b` là các số được tìm trong quá trình huấn luyện.
3. **Đầu vào**: `x` được đưa vào ở mỗi lần dự đoán.

Dạng hồi quy này và cách hiểu weight/bias được đối chiếu với [Google: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression). Các con số latency là ví dụ riêng của bài học.

## 4. Cần một thước đo để biết tham số tốt đến đâu

Giả sử bắt đầu bằng $w=1$, $b=0$. Khi đó $\hat y=x$.

| $x$ | $y$ | $\hat y$ | Sai số $\hat y-y$ | Bình phương sai số |
|---:|---:|---:|---:|---:|
| 1 | 3 | 1 | −2 | 4 |
| 2 | 5 | 2 | −3 | 9 |
| 3 | 7 | 3 | −4 | 16 |

Lấy trung bình cột cuối:

$$
L=\frac{4+9+16}{3}=\frac{29}{3}\approx9.6667
$$

Đó là **Mean Squared Error (MSE)** — trung bình bình phương sai số. Khi dùng để hướng dẫn việc tìm tham số, nó đóng vai trò **loss — hàm mất mát**:

$$
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2
$$

- $n$ là số mẫu; trong ví dụ $n=3$.
- $i$ là số thứ tự mẫu; $x_i,y_i$ là cặp giá trị của mẫu đó.
- $\sum$ có nghĩa cộng kết quả của mọi mẫu.
- Viết $L(w,b)$ nhấn mạnh loss thay đổi khi chọn tham số khác.

Công thức tương ứng với bốn thao tác: dự đoán từng mẫu → trừ giá trị đích → bình phương → lấy trung bình.

### Vì sao bình phương?

Nếu chỉ lấy trung bình sai số có dấu, hai sai số −2 và +2 triệt tiêu thành 0 dù cả hai dự đoán đều sai. Bình phương loại bỏ dấu và phạt sai số lớn mạnh hơn. Lấy trị tuyệt đối cũng loại bỏ dấu; chọn bình phương là một lựa chọn thiết kế, không phải cách duy nhất. [Google: Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)

Đơn vị của MSE là **ms²**, không phải ms. MSE bằng 9.6667 không có nghĩa sai lệch trung bình là 9.6667 ms. Muốn một chỉ số cùng đơn vị đầu ra, có thể dùng căn bậc hai MSE (RMSE); sự khác nhau giữa các chỉ số sẽ học kỹ ở phần đánh giá.

Trong code, thử đọc theo đúng bốn thao tác:

```python
def mse(data, w, b):
    squared_errors = [
        (predict(x, w, b) - y) ** 2
        for x, y in data
    ]
    return sum(squared_errors) / len(data)
```

Hàm này giả định dữ liệu không rỗng.

## 5. Huấn luyện là tìm tham số theo một mục tiêu

Một số ứng viên:

| $w$ | $b$ | MSE trên dữ liệu huấn luyện |
|---:|---:|---:|
| 1 | 0 | 9.6667 |
| 1 | 1 | 4.6667 |
| 2 | 0 | 1.0000 |
| 2 | 1 | 0.0000 |

Ta có thể thử một tập ứng viên và giữ bộ có loss nhỏ nhất. Thuật toán hôm nay là **grid search — tìm kiếm trên lưới**: liệt kê một lưới giá trị cho tham số, chấm điểm rồi chọn.

Ba thành phần của hệ thống học lúc này là:

| Thành phần | Trong bài học |
|---|---|
| Mô hình | Hàm $wx+b$ |
| Mục tiêu tối ưu | Giảm MSE trên tập huấn luyện |
| Thuật toán tìm tham số | Thử lần lượt các cặp $(w,b)$ trên lưới |

Ta sẽ thử $w$ từ 0 đến 4, $b$ từ −2 đến 2, bước 0.1: có $41\times41=1681$ cặp. Khoảng và bước tìm kiếm là cấu hình do người thiết kế chọn, không phải số mô hình tự học được trong chương trình này.

Thuật toán chỉ tìm được bộ tốt nhất **trong các ứng viên đã thử**. Nếu đáp án tốt nhất nằm ngoài khoảng hoặc giữa hai bước, nó có thể bỏ lỡ. Đây là phương pháp minh họa dễ hiểu; không dùng để thử mọi trọng số của một mạng lớn. Bài kế tiếp sẽ dùng đạo hàm và gradient để tìm hướng cập nhật hiệu quả hơn.

## 6. Thực hành Python

Mã đầy đủ: [lab.py](lab.py). Môi trường đã kiểm tra: **Python 3.14.6**, chỉ dùng thư viện chuẩn; không cần cài gói hay tải dữ liệu. Có thể đọc mã trước, dự đoán đầu ra, rồi chạy.

Từ thư mục gốc của bộ tài liệu (thư mục chứa mục lục này ở cấp cha), chạy:

```bash
python3 01-mo-hinh-hoc-tu-du-lieu/lab.py
```

Hoặc nếu terminal đang ở chính thư mục bài 1:

```bash
python3 lab.py
```

Kết quả mong đợi:

```text
Variant: baseline
Initial train MSE: 9.6667
Best w: 2.00; best b: 1.00
Train MSE: 0.0000
Held-out prediction at x=4: 9.00 ms
Held-out MSE: 0.0000
```

Đọc hàm `train_grid` và tìm ba chỗ: tạo ứng viên → chấm bằng `mse` → giữ kết quả tốt nhất. Đáp án $w=2,b=1$ không được gán sẵn trong thuật toán; chỉ dữ liệu và lưới tìm kiếm được cung cấp.

## 7. Huấn luyện, suy luận và đánh giá là ba việc khác nhau

```text
Huấn luyện: dữ liệu có đích → tìm w, b → lưu tham số
Suy luận:   đầu vào mới + w, b đã chọn → dự đoán
Đánh giá:   dự đoán + đích giữ riêng → đo chất lượng
```

**Inference — suy luận** ở đây chỉ tính `w * x + b`. Mỗi request không chạy lại toàn bộ vòng tìm kiếm. Có hệ thống cập nhật mô hình liên tục, nhưng đó là quy trình được thiết kế riêng.

Chương trình giữ mẫu `(4, 9)` riêng, không dùng nó trong `train_grid`. Đây là phép minh họa rất nhỏ cho việc đánh giá ngoài dữ liệu huấn luyện. Trong dự án lớn, thường cần tập validation để điều chỉnh lựa chọn và tập test giữ cho đánh giá cuối. Nếu liên tục nhìn kết quả test để sửa lựa chọn, nó không còn là phép đánh giá cuối độc lập. [scikit-learn: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

## 8. Phản ví dụ và giới hạn

### Khớp dữ liệu cũ không bảo đảm dự đoán tốt

Hàm tra bảng có thể trả chính xác 3, 5, 7 cho đầu vào 1, 2, 3 nhưng chưa có cách hợp lý xử lý 4. Điều ta cần là khả năng **generalization — tổng quát hóa**: hoạt động tốt trên dữ liệu mới trong bối cảnh sử dụng.

Trong ví dụ, $x=4$ còn nằm ngoài khoảng huấn luyện $[1,3]$, gọi là ngoại suy. Đúng ở một điểm ngoài khoảng không chứng minh đúng ở $x=100$. Một mẫu giữ riêng cũng không đủ đo độ tin cậy thực tế.

### Loss bằng 0 không phải mục tiêu luôn đạt được

Nếu một đầu vào $x=2$ có hai phép đo lần lượt 4.8 và 5.2 ms, một hàm tất định chỉ nhận $x$ sẽ dự đoán cùng một giá trị cho cả hai. Nó không thể khớp cả hai phép đo. Latency còn phụ thuộc tải, cache và các yếu tố chưa đưa vào mô hình.

### Chọn dạng mô hình cũng là giả định

Hàm $wx+b$ giả định mức tăng dự đoán cố định theo $x$. Hệ thống có ngưỡng bão hòa hay thay đổi chiến lược truy vấn có thể không tuân theo dạng đó. Giảm loss không sửa được mọi thiếu sót của dữ liệu hoặc mọi giả định sai về mô hình.

## 9. Nối sang Deep Learning

Khi có nhiều đầu vào, phép tính mở rộng thành:

$$
z=w_1x_1+\dots+w_dx_d+b
$$

$d$ là số đặc trưng; mỗi đầu vào có một trọng số. Một neuron thường bắt đầu bằng tổng có trọng số này, rồi áp dụng hàm kích hoạt. Mạng neural kết hợp các phép tính qua nhiều tầng. Ta sẽ học tensor, hàm kích hoạt và cách tính gradient trước khi viết mạng nhiều lớp.

Bài hôm nay là mô hình nền tảng, chưa phải một mạng Deep Learning.

## 10. Tự kiểm tra và bước tiếp theo

Làm [bài tập](exercises.md), sau đó đọc [lời giải](solutions.md). Có thể chuyển tiếp khi tự giải thích được: thứ gì thay đổi trong huấn luyện, loss được tính ra sao, và vì sao cần kiểm tra trên dữ liệu mới. Nếu chưa làm được, quay lại ví dụ số ở phần 3–5; không cần học lại toàn bộ Python.

**Học tiếp:** [Bài 2 — Đạo hàm và gradient descent](../02-dao-ham-gradient-descent/README.md), tìm hướng cập nhật tham số thay vì thử mọi ứng viên. Có ví dụ tính tay, code và hình minh họa.

Nguồn và phạm vi đã đọc: [danh mục nguồn](../SOURCES.md). Bố cục, ví dụ backend, mã và bài tập được biên soạn cho bộ tài liệu này.
