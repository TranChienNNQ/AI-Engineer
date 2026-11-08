# Bài 1 — Mô hình học từ dữ liệu như thế nào?

[Mục lục](../README.md) · [Đề cương](../ROADMAP.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Trạng thái: đã biên soạn, bổ sung trực giác kỹ thuật và ứng dụng AI thực tế; mã thực hành đã kiểm tra trên Python 3.14.6. Bài này xây nền trực giác cốt lõi cho toàn bộ Machine Learning và Deep Learning; chưa cần biết mạng neural phức tạp hay đạo hàm.

---

## 1. Mục tiêu và kiến thức cần trước

Sau bài này, bạn sẽ nắm được:

- **Trực giác cốt lõi**: Hiểu mô hình học máy như một "cỗ máy có các núm vặn tham số", không còn bị ngợp bởi các công thức toán học.
- **Phân biệt các thành phần**: Dữ liệu đầu vào ($x$), đáp án quan sát thực tế ($y$), dự đoán của máy ($\hat y$), và các tham số điều khiển ($w, b$).
- **Bản chất của hàm mất mát (Loss/MSE)**: Hiểu vì sao phải trừ sai số, vì sao phải bình phương, và cách tính bằng tay.
- **Cơ chế huấn luyện (Training)**: Bản chất việc "học" thực ra là quá trình tự động tìm và vặn các tham số sao cho sai số là nhỏ nhất.
- **Tách bạch vòng đời hệ thống AI**: Phân biệt rành mạch giữa *Huấn luyện (Training)*, *Suy luận (Inference)* và *Đánh giá độc lập (Evaluation)*.
- **Ứng dụng thực tế của AI Engineer**: Biết cách mô hình hóa các bài toán đời sống và kỹ thuật thực tế (dự đoán chi phí gọi LLM API, tính timeout cho microservices, ước lượng VRAM GPU).

**Yêu cầu nền tảng**: Chỉ cần biết lập trình Python cơ bản (biến, vòng lặp, hàm), các phép toán cộng, nhân và lũy thừa. Chưa cần thư viện NumPy, PyTorch hay GPU.

---

## 2. Từ tư duy lập trình truyền thống đến học máy

Hãy bắt đầu bằng một bài toán kỹ thuật mà bất kỳ lập trình viên backend nào cũng từng gặp: **Dự đoán thời gian xử lý của API để thiết lập timeout hợp lý**.

Giả sử một worker service nhận các lô bản ghi dữ liệu (batch). Ta quan sát thấy:

| Số lô dữ liệu ($x$) | Thời gian xử lý thực tế ($y$) |
|---:|---:|
| 1 lô | 3 ms |
| 2 lô | 5 ms |
| 3 lô | 7 ms |

*(Dữ liệu trên là ví dụ giả định phục vụ học tập, biểu diễn quan hệ đơn giản).*

### Cách tiếp cận 1: Lập trình truyền thống (Software 1.0)

Kỹ sư phần mềm nhìn vào bảng dữ liệu, tự phân tích bằng mắt và tự tay viết code logic (Hardcoded Rule):

```python
def estimate_latency_manual(batch_count: int) -> float:
    # Lập trình viên tự suy ra: mỗi lô tốn thêm 2 ms, cộng thêm 1 ms chuẩn bị
    return 2.0 * batch_count + 1.0
```

Cách này hoạt động tốt nếu bài toán đơn giản, dữ liệu sạch và chỉ có 1 biến đầu vào. Nhưng nếu có 50 yếu tố ảnh hưởng (kích thước payload, số lượng kết nối mạng, tải CPU, kích thước cache...), con người **không thể** tự ngồi viết hàng nghìn dòng `if-else` hay công thức thủ công được nữa.

### Cách tiếp cận 2: Học máy (Software 2.0)

Thay vì lập trình viên tự nghĩ ra quy luật:
1. Ta đưa **Dữ liệu đầu vào** và **Kết quả quan sát thực tế** vào máy tính.
2. Ta cung cấp một **khung mô hình có các núm vặn** (tham số).
3. Thuật toán sẽ tự động chạy để "dò tìm" giá trị của các núm vặn đó sao cho khớp với dữ liệu nhất.

```text
[Lập trình truyền thống]
Dữ liệu đầu vào + Quy tắc thủ công (Code/Rules)  ──>  Kết quả đầu ra

[Học máy / Machine Learning]
Dữ liệu đầu vào + Kết quả thực tế quan sát được  ──>  Thuật toán tự sinh ra Quy tắc (Mô hình & Tham số)
```

Đây là bài toán **Học có giám sát (Supervised Learning)**: ta dạy mô hình bằng các ví dụ mẫu đã có sẵn đáp án (cặp đầu vào - kết quả). Vì giá trị cần dự đoán là một số liên tục (thời gian tính bằng ms), bài toán thuộc loại **Hồi quy (Regression)**. Nếu bài toán yêu cầu đoán một nhãn danh mục (ví dụ email này là *Spam* hay *Ham*, giao dịch là *Hợp lệ* hay *Gian lận*), nó được gọi là **Phân loại (Classification)**.

---

## 3. Trực giác mô hình: Cỗ máy và các núm vặn

Mô hình đơn giản nhất mà chúng ta chọn có dạng:

$$
\hat y = wx + b
$$

*(Đọc là: y-mũ bằng w nhân x cộng b)*

Đừng nhìn vào ký hiệu toán học như một công thức xa lạ. Hãy tưởng tượng mô hình này như một chiếc hộp đen với **hai núm vặn**:

```text
       Đặc trưng đầu vào: x (Số lô)
                    │
                    ▼
           ┌─────────────────┐
  Núm w ──>│  Nhân: w * x    │
(Độ nhạy)  └────────┬────────┘
                    ▼
           ┌─────────────────┐
  Núm b ──>│  Cộng: (w*x) + b│
(Mức nền)  └────────┬────────┘
                    │
                    ▼
       Dự đoán: y_hat (Thời gian dự kiến)
```

### Ý nghĩa trực quan của từng đại lượng

| Ký hiệu | Tên gọi chuyên môn | Tên gọi trực quan / Dễ hiểu | Ý nghĩa trong ví dụ |
|---|---|---|---|
| $x$ | Feature (Đặc trưng) | Dữ liệu đầu vào | Số lô bản ghi cần xử lý (ví dụ: 1, 2, 3 lô). |
| $y$ | Label / Ground Truth | Thực tế đo được | Thời gian đo được bằng đồng hồ đo hệ thống (ms). |
| $\hat y$ | Prediction (Dự đoán) | Con số máy đoán | Thời gian mà mô hình tính ra (ms). |
| $w$ | Weight (Trọng số) | **Núm vặn "Độ nhạy" / Tỷ lệ biến thiên** | Cứ thêm 1 lô dữ liệu thì thời gian xử lý tăng thêm bao nhiêu ms? |
| $b$ | Bias (Số hạng tự do) | **Núm vặn "Mức nền khởi điểm"** | Khi chưa có lô nào ($x=0$), hệ thống đã tốn sẵn bao nhiêu ms chi phí cố định (overhead khởi động connection, TLS)? |

> [!NOTE]
> Trong học máy, thuật ngữ **Bias** ở đây là một con số đại số (số hạng tự do/độ lệch gốc tọa độ). Nó hoàn toàn khác với khái niệm "thiên kiến xã hội" (social bias) hay định kiến phân biệt đối xử của các hệ thống AI.

### Ba ví dụ thực tế tương tự trong đời sống và kỹ thuật

Để thấy công thức $\hat y = wx + b$ xuất hiện khắp mọi nơi:

1. **Dự toán cước xe công nghệ (Grab / Taxi)**:
   - $x$: Quãng đường di chuyển (km).
   - $w$: Giá cước mỗi km (ví dụ 10.000 VNĐ/km).
   - $b$: Giá mở cửa cố định dù chỉ đi 10 mét (ví dụ 12.000 VNĐ).
   - $\text{Tiền cước} = 10.000 \times x + 12.000$.

2. **Dự đoán số Token / Chi phí gọi API LLM (OpenAI / Anthropic)**:
   - $x$: Số từ trong câu hỏi của người dùng.
   - $w$: Tỷ lệ chuyển đổi từ sang token (khoảng 1.3 token/từ).
   - $b$: Số token cố định của System Prompt và hướng dẫn định dạng JSON (ví dụ 150 token).
   - $\text{Total Tokens} = 1.3 \times x + 150$.

3. **Dự đoán dung lượng VRAM GPU khi chạy Inference LLM**:
   - $x$: Chiều dài ngữ cảnh (Context length - số lượng token).
   - $w$: Bộ nhớ KV Cache tiêu hao thêm cho mỗi token bổ sung.
   - $b$: Dung lượng bộ nhớ cố định để nạp Model Weights vào GPU.
   - $\text{VRAM} = w \times x + b$.

Trong Python, hàm dự đoán được viết vô cùng đơn giản:

```python
def predict(x: float, w: float, b: float) -> float:
    return w * x + b
```

Nếu ta vặn núm thử tại $w = 2.0$ và $b = 1.0$:
- Khi có request mới gửi đến với $x = 4$ lô:
- Dự đoán $\hat y = 2.0 \times 4 + 1.0 = 9.0$ ms.

---

## 4. Thước đo Loss (Hàm mất mát): Bảng chấm điểm sai số

Khi bắt đầu, máy tính chưa biết phải vặn núm $w$ và $b$ về số mấy. Giả sử máy tính vặn bừa một cặp số ban đầu: $w = 1.0$, $b = 0.0$ (tức là đoán $\hat y = 1.0 \times x + 0.0 = x$).

Làm sao máy biết được bộ số này là "tốt" hay "tồi"? Ta cần một **thước đo sai số**, chuyên môn gọi là **Hàm mất mát (Loss Function)**.

Hãy theo dõi bảng tính sai số cho từng mẫu:

| Số lô ($x$) | Thực tế ($y$) | Máy đoán ($\hat y$) | Sai lệch $(\hat y - y)$ | Bình phương sai số $(\hat y - y)^2$ |
|---:|---:|---:|---:|---:|
| 1 | 3 ms | 1 ms | $1 - 3 = -2$ | $(-2)^2 = 4$ |
| 2 | 5 ms | 2 ms | $2 - 5 = -3$ | $(-3)^2 = 9$ |
| 3 | 7 ms | 3 ms | $3 - 7 = -4$ | $(-4)^2 = 16$ |

Lấy trung bình cộng của toàn bộ cột bình phương sai số:

$$
\text{Loss} = \frac{4 + 9 + 16}{3} = \frac{29}{3} \approx 9.6667 \text{ ms}^2
$$

Giá trị $9.6667$ này chính là **Mean Squared Error (MSE)** — Sai số bình phương trung bình.

### Giải mã công thức toán học tổng quát

Công thức toán học đầy đủ thường được viết trong các tài liệu nghiên cứu:

$$
L(w,b) = \frac{1}{n}\sum_{i=1}^{n}\Big(\underbrace{w x_i + b}_{\text{Máy đoán } \hat y_i} - \underbrace{y_i}_{\text{Thực tế}}\Big)^2
$$

Đừng hoảng sợ trước dấu Sigma ($\sum$). Công thức này chỉ là cách viết ngắn gọn của **bốn bước tư duy lập trình**:

```text
Dữ liệu mẫu (x_i, y_i)
        │
        ▼
[1. Dự đoán]: Tính y_hat = w * x_i + b
        │
        ▼
[2. Đo độ lệch]: Hiệu số = (y_hat - y_i)
        │
        ▼
[3. Phạt sai số]: Bình phương = Hiệu số ** 2
        │
        ▼
[4. Trung bình]: Cộng tất cả lại rồi chia cho số lượng mẫu n (1/n * Sum)
        │
        ▼
Điểm phạt Loss (MSE)
```

### Tại sao lại phải bình phương sai số?

1. **Triệt tiêu dấu âm**: Nếu chỉ lấy trung bình độ lệch thông thường, một mẫu đoán thừa $+5$ ms và một mẫu đoán thiếu $-5$ ms sẽ cộng lại thành $0$, khiến máy tưởng lầm là mô hình dự đoán hoàn hảo! Bình phương làm mọi sai số đều thành số dương.
2. **Phạt nặng sai số lớn**: Một lỗi lệch $1$ đơn vị chỉ bị phạt $1^2 = 1$. Nhưng một lỗi lệch $5$ đơn vị sẽ bị phạt tới $5^2 = 25$! Điều này thúc đẩy thuật toán ưu tiên hạn chế tối đa các dự đoán sai nghiêm trọng.
3. **Đơn vị của MSE**: Lưu ý đơn vị của MSE là $\text{ms}^2$ (bình phương đơn vị đo). Nếu muốn đo sai lệch theo đơn vị gốc ms, người ta lấy căn bậc hai: $\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{9.6667} \approx 3.11$ ms.

Trong code Python, hàm MSE phản ánh chính xác 4 bước trên:

```python
def mse(data, w: float, b: float) -> float:
    squared_errors = [
        (predict(x, w, b) - y) ** 2
        for x, y in data
    ]
    return sum(squared_errors) / len(data)
```

---

## 5. Quá trình huấn luyện (Training): Tự động xoay núm vặn

**Huấn luyện (Training)** thực chất là gì? 
Nó không có gì huyền bí: **Đó là quá trình tìm kiếm bộ tham số $(w, b)$ sao cho hàm mất mát (Loss) đạt giá trị nhỏ nhất có thể**.

Hãy tưởng tượng ta có một danh sách các ứng viên $(w, b)$ và kiểm tra bảng điểm lỗi của chúng:

| Ứng viên $(w, b)$ | Hàm dự đoán | Loss (MSE) trên dữ liệu học | Nhận xét |
|:---:|---|---:|---|
| $w = 1.0, b = 0.0$ | $\hat y = 1.0x$ | $9.6667$ | Quá tệ (dự đoán thiếu trầm trọng) |
| $w = 1.0, b = 1.0$ | $\hat y = 1.0x + 1$ | $4.6667$ | Khá hơn một chút |
| $w = 2.0, b = 0.0$ | $\hat y = 2.0x$ | $1.0000$ | Khá sát, nhưng vẫn còn lệch |
| **$w = 2.0, b = 1.0$** | $\mathbf{\hat y = 2.0x + 1}$ | **$0.0000$** | **Hoàn hảo! Sai số bằng 0 tuyệt đối** |

### Thuật toán tìm kiếm trên lưới (Grid Search)

Trong bài thực hành đầu tiên này, ta dùng thuật toán đơn giản và trực quan nhất: **Grid Search (Dò quét trên lưới)**:
- Ta chuẩn bị một dải các giá trị cho $w$: từ $0.0$ đến $4.0$ (bước nhảy 0.1).
- Ta chuẩn bị một dải các giá trị cho $b$: từ $-2.0$ đến $2.0$ (bước nhảy 0.1).
- Máy tính sẽ kiên nhẫn thử lần lượt tất cả $41 \times 41 = 1681$ cặp kết hợp, tính MSE cho từng cặp, và ghi nhận cặp $(w, b)$ có MSE nhỏ nhất.

```python
def train_grid(data, w_values, b_values):
    best_w, best_b = None, None
    best_loss = float("inf")
    
    for w in w_values:
        for b in b_values:
            current_loss = mse(data, w, b)
            if current_loss < best_loss:
                best_w, best_b, best_loss = w, b, current_loss
                
    return best_w, best_b, best_loss
```

> [!TIP]
> **Nhược điểm của Grid Search**: Dò quét toàn bộ lưới rất trực quan nhưng cực kỳ chậm khi mô hình có nhiều tham số. Nếu có 10 tham số, thử 100 nấc cho mỗi tham số sẽ cần $100^{10} = 10^{20}$ phép tính!
> Ở [Bài 2](../02-dao-ham-gradient-descent/README.md), bạn sẽ học một phương pháp thông minh hơn nhiều: dùng **đạo hàm và gradient** để chỉ rõ *cần vặn núm sang trái hay sang phải* thay vì phải mò mẫm tất cả các nấc.

---

## 6. Thực hành Python

Toàn bộ mã nguồn minh họa nằm trong tệp [lab.py](lab.py), viết bằng thư viện chuẩn Python (không cần cài đặt thêm bất kỳ thư viện ngoài nào).

### Chạy thử nghiệm

Mở terminal tại thư mục bài học và chạy lệnh:

```bash
python3 lab.py --variant baseline
```

Kết quả in ra:

```text
Variant: baseline
Initial train MSE: 9.6667
Best w: 2.00; best b: 1.00
Train MSE: 0.0000
Held-out prediction at x=4: 9.00 ms
Held-out MSE: 0.0000
```

### Điều gì vừa diễn ra?

1. Ban đầu với $w=1, b=0$, sai số MSE là $9.6667$.
2. Sau khi chạy `train_grid`, thuật toán đã tự động tìm ra bộ tham số tối ưu: $w=2.00$ và $b=1.00$, kéo MSE về $0.0000$.
3. Thuật toán **không hề được lập trình sẵn đáp án $w=2, b=1$**, mà hoàn toàn tự tìm thấy thông qua việc tối thiểu hóa hàm mất mát.

---

## 7. Vòng đời hệ thống AI: Huấn luyện, Suy luận và Đánh giá

Một kỹ sư AI cần phân biệt rạch ròi 3 giai đoạn độc lập:

```text
┌───────────────────────────────────────────────────────────────┐
│ 1. HUẤN LUYỆN (Training / Offline)                            │
│    Dữ liệu có đáp án ──> Thuật toán tối ưu ──> Tìm ra w, b    │
│    (Chạy trên máy huấn luyện/GPU, tốn thời gian tính toán)     │
└──────────────────────────────┬────────────────────────────────┘
                               │ Xuất bộ trọng số (w, b)
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 2. TRIỂN KHAI & SUY LUẬN (Serving / Inference / Production)   │
│    Request x mới (4 lô) + (w=2, b=1) ──> Trả về y_hat = 9 ms  │
│    (Chỉ tốn 1 phép nhân + 1 phép cộng, phản hồi trong vài µs) │
└──────────────────────────────┬────────────────────────────────┘
                               │ Giám sát kết quả
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 3. ĐÁNH GIÁ ĐỘC LẬP (Evaluation / Test Set)                   │
│    Dự đoán trên tập dữ liệu giữ riêng (Held-out) chưa từng học│
│    Đo lường năng lực tổng quát hóa (Generalization)           │
└───────────────────────────────────────────────────────────────┘
```

1. **Huấn luyện (Training)**: Quá trình tìm kiếm tham số $w, b$. Việc này thường làm offline trên cụm máy server/GPU.
2. **Suy luận (Inference)**: Khi người dùng hoặc hệ thống khác gọi API, ta **không chạy lại quá trình huấn luyện**. Ta lấy bộ số $(w=2, b=1)$ đã lưu sẵn và tính nhanh `w * x + b`. Đây là lý do vì sao một mô hình khổng lồ huấn luyện mất hàng triệu USD nhưng khi phục vụ một truy vấn API chỉ mất vài chục mili-giây.
3. **Đánh giá trên tập kiểm tra độc lập (Held-out / Test Evaluation)**:
   - Trong `lab.py`, mẫu dữ liệu `(x=4, y=9)` được giấu đi, không đưa vào hàm `train_grid`.
   - Sau khi huấn luyện xong, ta lấy mẫu `x=4` ra hỏi mô hình xem nó đoán bao nhiêu, rồi đối chiếu với đáp án thực tế $y=9$.
   - Mục đích: kiểm tra xem mô hình có thực sự "hiểu quy luật" để áp dụng cho dữ liệu mới hay chỉ đơn giản là "học vẹt" dữ liệu cũ. Hiện tượng rò rỉ dữ liệu test vào tập train (data leakage) là lỗi tối kỵ trong kỹ thuật AI.

---

## 8. Các ứng dụng thực tế của AI Engineer

Một mô hình hồi quy tham số đơn giản có ứng dụng gì trong công việc thực tế của một kỹ sư phần mềm / AI Engineer?

### Ứng dụng 1: Dự đoán ngân sách và quản lý chi phí gọi LLM API

Khi xây dựng các ứng dụng như RAG (Retrieval-Augmented Generation) hoặc xử lý hàng loạt tài liệu lớn qua OpenAI/Claude:
- Nếu để người dùng tải lên tài liệu 50.000 từ mà không cảnh báo chi phí trước, hệ thống có thể bị thâm hụt ngân sách hoặc người dùng bị trừ tiền ngoài ý muốn.
- Kỹ sư AI xây dựng một bộ ước lượng nhanh:
  $$\text{Chi phí dự kiến} = w \times (\text{Số ký tự input}) + b$$
- Bộ tham số $w, b$ được học tự động từ 10.000 lượt gọi API thực tế trong lịch sử. Hệ thống có thể từ chối ngay lập tức các request vượt quá hạn mức trước khi gửi đi.

### Ứng dụng 2: Dynamic Timeout cho hệ thống Microservices

- Trong kiến trúc phân tán, việc đặt một giá trị timeout cố định (ví dụ 5 giây cho mọi request) thường dẫn đến hai tai họa:
  - Nếu request nhỏ bị lỗi treo mạng, client phải chờ mòn mỏi suốt 5 giây mới báo lỗi.
  - Nếu request lớn cần 6 giây để xử lý bình thường, hệ thống lại ngắt ngang (timeout oan).
- Bằng cách thu thập kích thước payload ($x$) và thời gian xử lý thực tế ($y$), mô hình tuyến tính học ra độ trễ dự kiến:
  $$\text{Latency} = w \times \text{Payload\_Size} + b$$
- Client tính toán `timeout = Latency_dự_đoán * 1.5` để cấp phát thời gian chờ linh hoạt và chính xác cho từng request.

### Ứng dụng 3: Tự động co giãn tài nguyên đám mây (Auto-scaling Pods)

- Dựa trên số lượng request mỗi giây (RPS) đổ về gateway, hệ thống học mô hình để dự đoán dung lượng RAM và số lượng Pod Kubernetes cần bật thêm:
  $$\text{Số lượng Pod} = \lceil w \times \text{RPS} + b \rceil$$
- Giúp hệ thống chủ động mở rộng tài nguyên trước khi cụm máy chủ bị nghẽn (Overload).

---

## 9. Phản ví dụ và những cạm bẫy thực tế

Mô hình học máy không phải là phép màu giải quyết được mọi việc. Hãy cảnh giác với 3 cạm bẫy sau:

### Cạm bẫy 1: Khớp tốt dữ liệu cũ không có nghĩa là hoạt động tốt ngoài thực tế

Giả sử bạn viết một hàm tra cứu (Lookup table / Dictionary): nếu $x=1 \to 3$, $x=2 \to 5$, $x=3 \to 7$. Hàm này đạt Loss bằng $0$ tuyệt đối trên 3 mẫu cũ. Nhưng khi gặp $x=4$, từ điển hoàn toàn bất lực!
Khả năng mô hình phán đoán chính xác trên dữ liệu hoàn toàn mới được gọi là **Tính tổng quát hóa (Generalization)**. Đúng ở $x=4$ cũng không bảo đảm sẽ đúng khi người dùng gửi một request khổng lồ với $x=1000$ lô (nguy cơ ngoại suy sai lầm).

### Cạm bẫy 2: Dữ liệu thực tế luôn có nhiễu (Loss bằng 0 là điều không tưởng)

Trong hệ thống thực, mạng máy tính có độ trễ dao động, CPU lúc bận lúc rỗi. Khi bạn gửi $x=2$ lô, lần thứ nhất đo được $4.8$ ms, lần thứ hai lại đo được $5.2$ ms.
Một hàm tất định $\hat y = wx+b$ chỉ có thể dự đoán **một số duy nhất** cho $x=2$. Do đó, mô hình thực tế sẽ **không bao giờ đạt được Loss = 0**.

Chạy thử biến thể dữ liệu có nhiễu:

```bash
python3 lab.py --variant noisy
```

Mô hình tìm được $w=2.00, b=1.00$ với MSE tối ưu trên lưới là $0.0467\text{ ms}^2$. Dù Loss lớn hơn 0, đường thẳng này vẫn nắm bắt xuất sắc xu hướng biến thiên tổng thể của dữ liệu.

### Cạm bẫy 3: Giả định mô hình bị sai dạng (Model Underfitting)

Mô hình $\hat y = wx + b$ giả định mối quan hệ là đường thẳng đều đặn. Nhưng trong thực tế backend, khi số lô vượt quá ngưỡng chịu đựng của RAM hoặc pool kết nối database, thời gian xử lý sẽ tăng vọt theo dạng hàm mũ ($O(n^2)$ hoặc nghẽn cổ chai). Khi đó, dù có vặn $w$ và $b$ khéo đến đâu, một đường thẳng cũng không thể mô tả đúng thực tế.

---

## 10. Chiếc cầu nối sang Deep Learning và các mô hình lớn (LLM)

Đến đây, bạn có thể tự hỏi: *"Học $\hat y = wx + b$ cơ bản này thì liên quan gì đến ChatGPT, Claude hay nhận diện hình ảnh trong Deep Learning?"*

Sự thật thú vị là: **Toàn bộ các mô hình AI hiện đại nhất đều được xây dựng trên chính nền móng này!**

```text
Bài 1: 1 Đặc trưng, 2 Tham số               ──>  Mô hình AI lớn: Hàng tỷ Tham số
───────────────────────────────────────────────────────────────────────────────────
Đầu vào: 1 số thực x                         ──>  Đầu vào: Vector đặc trưng nghìn chiều (x_1, x_2, ..., x_d)
Tham số: 1 số thực w, 1 số thực b            ──>  Tham số: Hàng tỷ ma trận trọng số W, vector độ lệch B
Tính toán: y = w * x + b                     ──>  Neuron: z = w_1*x_1 + w_2*x_2 + ... + w_d*x_d + b
Học tập: Vặn w, b để giảm MSE                ──>  Học tập: Cập nhật W, B để giảm Loss qua hàng nghìn GPU
```

1. **Từ 1 đầu vào đến nhiều đầu vào**: Khi dự đoán thời gian xử lý không chỉ dựa vào số lô ($x_1$) mà còn kích thước bản ghi ($x_2$), băng thông mạng ($x_3$)... công thức mở rộng thành:
   $$z = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$
2. **Từ 1 phép tính đến 1 Neuron**: Công thức trên chính là phép tính lõi bên trong **một Neuron nhân tạo**.
3. **Từ 1 Neuron đến Mạng nhiều tầng (Deep Learning)**: Người ta ghép hàng nghìn neuron lại thành các tầng, đặt thêm hàm kích hoạt phi tuyến (activation) để giải quyết các quan hệ phức tạp.
4. **Các mô hình như GPT-4 hay DeepSeek**: Thực chất là một mạng neuron khổng lồ với hàng trăm tỷ tham số $W$ và $B$. Nhưng nguyên lý căn bản của quá trình huấn luyện vẫn không hề thay đổi:
   - Cho dữ liệu chạy qua mạng để dự đoán.
   - So sánh dự đoán với nhãn thực tế bằng một hàm mất mát (Loss).
   - Tinh chỉnh các tham số trọng số để kéo giá trị Loss xuống thấp nhất có thể.

---

## 11. Tự kiểm tra và bước tiếp theo

Hãy thực hành củng cố kiến thức:
1. Mở tệp [exercises.md](exercises.md) và tự giải 5 bài tập nhỏ (có bài tính tay và bài chạy lệnh code).
2. Sau khi làm xong, đối chiếu lập luận với [solutions.md](solutions.md).
3. Đạt yêu cầu khi bạn có thể tự trả lời:
   - $w$ và $b$ đóng vai trò gì?
   - Tại sao phải tính Loss và vì sao dùng bình phương sai số?
   - Vì sao không dùng dữ liệu kiểm tra (test) để huấn luyện mô hình?

**Bước tiếp theo:**
Chuyển sang [Bài 2 — Đạo hàm và gradient descent](../02-dao-ham-gradient-descent/README.md). Bạn sẽ khám phá cách các mô hình AI thực tế dùng độ dốc đạo hàm để tự động "trượt dốc" tìm ngay ra tham số tối ưu trong chớp mắt mà không cần phải đoán mò từng nấc trên lưới.

---

*Tham khảo chuẩn kiến thức:*
- [Google Machine Learning Crash Course: Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression)
- [Google Machine Learning Crash Course: Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)
- [scikit-learn: Common Pitfalls and Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html)
