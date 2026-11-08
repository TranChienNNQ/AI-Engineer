# Bài 15 — Dữ liệu chuỗi và ý tưởng RNN

[Mục lục](../README.md) · [Bài 14](../14-cnn-du-lieu-anh/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Ví dụ tự tạo và PyTorch thật trên CPU; không tải dataset hay model. Bài học cơ chế chuỗi qua số thực trước khi học embedding ở bài 16. Không coi RNN là lựa chọn mặc định cho mọi bài toán có thời gian hoặc văn bản.

> [!NOTE]
> **Định nghĩa các khái niệm cốt lõi của bài học:**
> 1. **Dữ liệu chuỗi (Sequential Data):** Là loại dữ liệu mà thứ tự xuất hiện của các phần tử mang thông tin quyết định ý nghĩa (như chuỗi từ ngữ trong câu văn, chuỗi giá cổ phiếu theo ngày, hoặc chuỗi tín hiệu nhịp tim ECG theo mili-giây).
> 2. **Recurrent Neural Network (RNN — Mạng nơ-ron hồi quy):** Là kiến trúc mạng nơ-ron chuyên biệt cho dữ liệu chuỗi. Khác với mạng truyền thẳng (Feedforward) xử lý các mẫu độc lập, RNN sở hữu một "vòng lặp hồi quy" cho phép thông tin từ các bước thời gian quá khứ được lưu giữ và truyền sang các bước thời gian tương lai.
> 3. **Trạng thái ẩn (Hidden State $h_t$):** Là một vector số thực đóng vai trò như "cuốn sổ tay ghi nhớ" của mạng nơ-ron. Tại mỗi bước $t$, $h_t$ nén lại toàn bộ thông tin quan trọng của các bước quá khứ kết hợp với dữ liệu đầu vào hiện tại.
> 4. **Trải mạng theo thời gian (Unrolling through time):** Cách hình dung đồ thị tính toán của RNN: một ô mạng nơ-ron (cell) duy nhất được sao chép và tái sử dụng lặp đi lặp lại qua từng bước thời gian $t = 1, 2, \dots, T$.
> 5. **BPTT (Backpropagation Through Time — Lan truyền ngược qua thời gian):** Thuật toán tính đạo hàm cho RNN bằng cách mở rộng quy tắc dây chuyền (chain rule) ngược dòng lịch sử từ bước cuối cùng trở về bước đầu tiên.

## 1. Mục tiêu và kiến thức cần trước

Sau bài này, bạn có thể phân biệt mẫu/bước thời gian/feature, tính RNN ba bước và gradient qua thời gian, tạo batch chuỗi không cùng độ dài, chọn đúng hidden state cuối, huấn luyện bộ phân loại chuỗi và nhận diện một số lỗi leakage/state.

> [!TIP]
> **Tại sao kỹ sư AI cần hiểu sâu cơ chế RNN cổ điển dù đã có Transformer?**
> Mặc dù Transformer và LLM hiện đang thống trị xử lý ngôn ngữ tự nhiên, tư duy về **trạng thái ẩn (hidden state)**, **sự phụ thuộc thời gian (temporal dependency)**, **lan truyền ngược qua thời gian (BPTT)**, và các kỹ thuật xử lý chuỗi độ dài biến thiên (**padding, packing, masking**) là nền tảng cốt tử trong mọi hệ thống xử lý tín hiệu thời gian thực (real-time streaming), mô hình dự báo chuỗi thời gian (time series forecasting), và các kiến trúc mô hình thế hệ mới (như State Space Models / Mamba).

Cần bài 3 về shape, bài 10–11 về graph/autograd/DataLoader, bài 12–13 về validation và clipping. Bài 14 chia sẻ weight theo vị trí ảnh; bài này chia sẻ weight theo bước thời gian. Đọc phần 2–7 rồi chạy `hand`, `padding`, `gradients`, `windows`; tiếp theo đọc thí nghiệm và chạy `train`.

## 2. Chuỗi khác một tập số ở điểm nào?

Chuỗi có thứ tự: `[1,−1,2]` và `[2,−1,1]` chứa cùng các giá trị nhưng trình tự khác nhau. Tổng/mean không phân biệt được chúng. Với dữ liệu cảm biến, chiều hướng thay đổi có thể quan trọng; với văn bản, đổi thứ tự từ có thể đổi ý nghĩa. Không phải mọi nhãn đều phụ thuộc thứ tự: tổng chi tiêu là ví dụ có thể chỉ cần aggregation.

> [!NOTE]
> **Ẩn dụ trực quan về vai trò của thứ tự:**
> - Trong ngôn ngữ tự nhiên: Câu *"Người cắn chó"* và *"Chó cắn người"* chứa chính xác các từ giống hệt nhau, nhưng ý nghĩa thông điệp hoàn toàn đảo ngược!
> - Trong bảo mật: Mật khẩu mở khóa `"1234"` và `"4321"` có cùng các chữ số, nhưng chỉ duy nhất một thứ tự đúng mới mở được khóa.
> - Trong y tế: Nhịp tim từ $60 \to 80 \to 100$ bpm (đang tăng khi vận động) mang ý nghĩa hoàn toàn khác với $100 \to 80 \to 60$ bpm (đang hồi phục về trạng thái nghỉ), dù giá trị trung bình là như nhau. Nếu chỉ tính tổng hay trung bình (mean), bạn đã làm mất sạch chiều thông tin quý giá này.

Chỉ số bước không tự biểu diễn khoảng thời gian thực. Nếu các phép đo cách nhau không đều, cần quyết định resampling hoặc thêm thông tin khoảng cách thời gian theo bài toán; pipeline này chỉ dùng thứ tự bước, chưa xử lý irregular sampling.

Một sample có T bước, mỗi bước có F feature. Ví dụ 4 thời điểm, mỗi thời điểm có nhiệt độ và độ ẩm: shape `(4,2)`, không phải 8 bước thời gian. Batch B chuỗi có độ dài chung T được viết `(B,T,F)` trong bài này; dùng `batch_first=True` khi gọi RNN. Mặc định API có thể dùng thứ tự khác, nên phải đọc hợp đồng shape.

> [!IMPORTANT]
> **Quy ước trục tensor chuỗi: `(B, T, F)` vs `(T, B, F)`**
> - **Kích thước một mẫu (Single Sequence):** Có shape $(T, F)$, trong đó $T$ (Time steps) là số bước thời gian (số từ trong câu, số giây đo), và $F$ (Features) là số đặc trưng tại mỗi bước (ví dụ: nhiệt độ, áp suất, độ ẩm $\implies F=3$).
> - **Khi gom thành Batch:**
>   - **Mặc định PyTorch (`batch_first=False`):** Shape là **`(T, B, F)`** (Time-major). Đây là định dạng tối ưu lịch sử cho GPU khi xử lý vòng lặp bước thời gian tuần tự.
>   - **Quy ước hiện đại (`batch_first=True`):** Shape là **`(B, T, F)`** (Batch-major: Batch size $\times$ Time steps $\times$ Features). Định dạng này tự nhiên và trực quan hơn nhiều với tư duy dữ liệu bảng thông thường. Hãy luôn đặt `batch_first=True` một cách tường minh để tránh nhầm lẫn tai hại giữa số lượng mẫu $B$ và số bước thời gian $T$.

| Bài toán             | Input/target minh họa                   | Điểm cần chú ý                                         |
| -------------------- | --------------------------------------- | ------------------------------------------------------ |
| Phân loại toàn chuỗi | Đoạn tín hiệu → một class               | Many-to-one, có thể dùng hết đoạn input                |
| Dự báo bước tiếp     | Các giá trị đã biết → giá trị tương lai | Target phải sau các feature có sẵn tại lúc dự báo      |
| Gán nhãn từng bước   | Mỗi bước input → nhãn tương ứng         | Many-to-many, cần mask phần padding khi tính loss      |
| Sinh chuỗi           | Lịch sử → phân bố bước kế               | Cách dùng target/train và inference cần thiết kế riêng |

> [!NOTE]
> **Minh họa 4 cấu hình xử lý chuỗi trong thực tế:**
> 1. **Many-to-One (Nhiều vào - Một ra):** Phân tích cảm xúc đoạn văn bản (đọc cả câu đánh giá $\to$ nhãn "Tích cực" / "Tiêu cực"), hoặc phát hiện bất thường nhịp tim (chuỗi 100 nhịp ECG $\to$ "Bình thường" / "Loạn nhịp").
> 2. **One-to-Many (Một vào - Nhiều ra):** Sinh phụ đề cho ảnh (Image Captioning: nạp 1 bức ảnh $\to$ sinh ra chuỗi từ mô tả ảnh).
> 3. **Many-to-Many đồng bộ (Nhiều vào - Nhiều ra từng bước):** Gán nhãn từ loại (POS Tagging: mỗi từ trong câu tương ứng với 1 nhãn Danh từ/Động từ/Tính từ ngay tại bước đó).
> 4. **Many-to-Many không đồng bộ (Seq2Seq / Encoder-Decoder):** Dịch máy (đọc hết câu tiếng Anh rồi mới bắt đầu dịch sang câu tiếng Việt) hoặc tóm tắt văn bản.

Lab huấn luyện **phân loại toàn chuỗi**, không huấn luyện mô hình dự báo hoặc sinh văn bản. Ví dụ windows ở phần 9 chỉ kiểm cách căn input/target.

![Chuỗi và vai trò thứ tự](figures/01-sequences.png)

_Hai chuỗi trong một cặp có cùng tập giá trị và mean, nhưng nhãn tăng/giảm khác nhau. Độ dài không phải số feature._

Trộn thứ tự các **sample độc lập** trong train loader không tự phá thứ tự bên trong mỗi sample. Đảo/trộn các bước thời gian bên trong một sample lại có thể đổi nhãn. Với stream cần nối state liên tục, cả thứ tự sample cũng cần giữ theo thiết kế.

## 3. RNN: cập nhật một trạng thái theo bước

Hidden state hₜ là vector tính từ input hiện tại và state trước đó. Nó là tóm tắt học được, không phải bản sao đầy đủ của mọi dữ liệu đã đi qua. Với vector hàng và một lớp tanh:

\[
a_t=x_tW_{ih}^{\top}+h_{t-1}W_{hh}^{\top}+b,\qquad h_t=\tanh(a_t).
\]

> [!NOTE]
> **Giải phẫu phương trình trạng thái ẩn của RNN:**
> Tại mỗi bước thời gian $t$:
> - $x_t \in \mathbb{R}^{1 \times F}$: Vector đầu vào tại thời điểm hiện tại.
> - $h_{t-1} \in \mathbb{R}^{1 \times H}$: Vector trạng thái ẩn được truyền lại từ thời điểm quá khứ ngay trước đó.
> - $W_{ih} \in \mathbb{R}^{H \times F}$: Ma trận trọng số biến đổi đầu vào (Input-to-Hidden weights).
> - $W_{hh} \in \mathbb{R}^{H \times H}$: Ma trận trọng số biến đổi trạng thái hồi quy (Hidden-to-Hidden weights).
> - $b \in \mathbb{R}^{1 \times H}$: Vector hệ số điều chỉnh (bias).
> - $a_t$: Giá trị kích hoạt thô trước phi tuyến (pre-activation).
> - $\tanh(\cdot)$: Hàm kích hoạt phi tuyến đưa toàn bộ giá trị của $h_t$ về khoảng đối xứng $(-1, 1)$, giữ cho độ lớn của bộ nhớ không bị bùng nổ vô hạn qua thời gian.

Tanh áp theo phần tử, output nằm trong (−1,1). Cùng W và b được dùng ở mọi bước. Khởi tạo h₀=0 là lựa chọn của lab. Một head có thể biến h_T thành logits nếu cần một class cho cả chuỗi. [D2L: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)

> [!IMPORTANT]
> **Nguyên lý Chia sẻ Trọng số qua Thời gian (Weight Sharing across Time):**
> Trong RNN, dù chuỗi của bạn dài 3 bước, 10 bước hay 1.000 bước, **chỉ có duy nhất một bộ tham số $\{W_{ih}, W_{hh}, b\}$ tồn tại**! Cùng một công thức toán học này được áp dụng lặp đi lặp lại ở mọi thời điểm. Số bước thời gian $T$ hoàn toàn không làm tăng số lượng tham số của mô hình.

Đây là mạng được “trải theo thời gian” trên graph: x₁→h₁, x₂ cùng h₁→h₂, v.v. T không làm tăng số parameter của cell, nhưng làm tăng số phép tính và lượng thông tin trung gian cần cho backprop. Giữ kích thước state cố định không có nghĩa nhớ vô hạn hoặc không tốn bộ nhớ khi train.

### Ví dụ tính tay ba bước

Dùng một input feature, một hidden unit: w_x=0.5,w_h=0.25,b=0,h₀=0; x=`[1,−1,2]`.

|   t |  xₜ |                  aₜ=0.5xₜ+0.25hₜ₋₁ |  hₜ=tanh(aₜ) |
| --: | --: | ---------------------------------: | -----------: |
|   1 |   1 |                                0.5 |  0.462117157 |
|   2 |  −1 | −0.5+0.25×0.462117157≈−0.384470711 | −0.366583715 |
|   3 |   2 |  1+0.25×(−0.366583715)≈0.908354071 |  0.720341322 |

> [!NOTE]
> **Giải thích chi tiết 3 bước tính tay:**
> - Khởi tạo: $h_0 = 0$.
> - **Bước 1 ($t=1, x_1=1$):**
>   $a_1 = 0.5 \times 1 + 0.25 \times 0 = 0.5 \implies h_1 = \tanh(0.5) \approx \mathbf{0.462117157}$.
> - **Bước 2 ($t=2, x_2=-1$):**
>   Nhận $h_1$ từ bước 1: $a_2 = 0.5 \times (-1) + 0.25 \times 0.462117157 = -0.5 + 0.115529289 = -0.384470711$.
>   $h_2 = \tanh(-0.384470711) \approx \mathbf{-0.366583715}$.
> - **Bước 3 ($t=3, x_3=2$):**
>   Nhận $h_2$ từ bước 2: $a_3 = 0.5 \times 2 + 0.25 \times (-0.366583715) = 1.0 - 0.091645929 = 0.908354071$.
>   $h_3 = \tanh(0.908354071) \approx \mathbf{0.720341322}$.
> Trạng thái cuối cùng $h_3 \approx 0.72034$ là vector đại diện cho toàn bộ chuỗi $[1, -1, 2]$.

Đảo chuỗi thành `[2,−1,1]` với cùng parameter/h₀ cho state cuối≈**0.401118659**. RNN này phân biệt được thứ tự dù mean không đổi; không suy mọi parameter RNN đều phân biệt mọi cặp chuỗi.

> [!NOTE]
> **Minh chứng toán học về khả năng ghi nhớ thứ tự:**
> Khi nạp chuỗi đảo $[2, -1, 1]$:
> - $t=1, x_1=2 \implies a_1 = 0.5 \times 2 = 1.0 \implies h_1 = \tanh(1.0) \approx 0.761594$.
> - $t=2, x_2=-1 \implies a_2 = -0.5 + 0.25 \times 0.761594 = -0.309601 \implies h_2 \approx -0.300109$.
> - $t=3, x_3=1 \implies a_3 = 0.5 + 0.25 \times (-0.300109) = 0.424973 \implies h_3 \approx \mathbf{0.401118659}$.
> Vì $0.720341 \neq 0.401119$, mạng RNN hoàn toàn phân biệt được hai chuỗi khác nhau về trật tự dù chúng có tập hợp phần tử y hệt nhau.

```python
from lab import scalar_reference
a = scalar_reference([1.,-1.,2.])
b = scalar_reference([2.,-1.,1.])
print(a['states'], b['states'][-1])
```

![RNN trải theo thời gian](figures/02-unrolled-rnn.png)

_Ba lần sử dụng cùng cell, không phải ba bộ weight riêng. State thay đổi; parameter chỉ đổi khi optimizer update._

## 4. Chuyển sang nn.RNN: shape và hai bias

Trong PyTorch, một bước RNN có hai bias cộng lại: `b_ih+b_hh`. Ví dụ trên đặt cả hai bằng 0. Nếu muốn bias tổng b khác 0, không đặt cả hai bằng b rồi vô tình có tổng2b. Weight lưu theo layout `(out,in)` giống Linear bài 11. [RNN API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNN.html)

> [!NOTE]
> **Tại sao PyTorch lại tách thành hai bias `b_ih` và `b_hh`?**
> Trong thiết kế phần cứng GPU (đặc biệt là thư viện cuDNN của NVIDIA), các phép tính được chia làm hai luồng song song độc lập: luồng chuyển đổi từ input $x_t W_{ih}^\top + b_{ih}$ và luồng chuyển đổi từ trạng thái ẩn $h_{t-1} W_{hh}^\top + b_{hh}$. Sau đó kết quả của hai luồng được cộng gộp lại.
> Do đó, bias tổng thực tế tác động lên pre-activation là:
> $$b_{\text{tổng}} = b_{ih} + b_{hh}$$
> Nếu bạn muốn gán bias tổng là $0.5$, bạn nên gán $b_{ih} = 0.5$ và $b_{hh} = 0$, tránh việc gán cả hai bằng $0.5$ khiến bias tổng bị nhân đôi thành $1.0$.

Trong cấu hình một lớp, một hướng, batch_first=True:

| Thành phần | Shape với B mẫu,T bước,F feature,H hidden      |
| ---------- | ---------------------------------------------- |
| x          | B×T×F                                          |
| weight_ih  | H×F                                            |
| weight_hh  | H×H                                            |
| Hai bias   | mỗi vector H                                   |
| output     | B×T×H, nếu input là tensor padded thông thường |
| h₀ và h_n  | 1×B×H                                          |

`batch_first` **không đổi trục hidden state**. `output` chứa hidden theo từng bước, không tự là class logits. `h_n` chứa state cuối; với tensor padded thông thường, đó là cuối phần tensor đã đọc, có thể gồm cả padding. Khi input packed, output là PackedSequence; hidden cuối tương ứng các độ dài thật.

> [!WARNING]
> **Phân biệt `output` và `h_n` trong PyTorch RNN:**
> - **`output` (shape `(B, T, H)`):** Chứa toàn bộ các vector trạng thái ẩn tại **mọi bước thời gian** $t = 1, 2, \dots, T$. Nó dùng cho các bài toán gán nhãn từng bước (many-to-many) hoặc tầng chú ý (attention).
> - **`h_n` (shape `(num_layers, B, H)`):** Chỉ chứa duy nhất vector trạng thái ẩn ở **bước thời gian cuối cùng**. Trục đầu tiên của `h_n` luôn là số tầng (num_layers), **ngay cả khi bạn đã bật `batch_first=True`**. Với mô hình 1 tầng, `h_n[0]` (hoặc `h_n[-1]`) chính là vector đại diện cho toàn bộ chuỗi để đưa vào bộ phân loại Linear (many-to-one).

```python
import torch
from torch import nn
rnn = nn.RNN(input_size=2, hidden_size=4, batch_first=True)
x = torch.zeros(3,5,2)
output, h_n = rnn(x)
print(output.shape, h_n.shape)  # (3,5,4), (1,3,4)
```

Số parameter cell một lớp một hướng có bias là HF+H²+2H. Với F=1,H=8: 8+64+16=88. Head Linear8→2 thêm18, tổng **106**. Chuỗi dài4 hay9 bước vẫn106 parameter; tăng H làm phần H² tăng nhanh. Lab không dùng stacked/bidirectional RNN. RNN một hướng dùng input đến bước hiện tại; mô hình hai hướng có thể dùng cả phía sau trong đoạn được cung cấp, cần cân nhắc thông tin sẵn có lúc dự báo.

> [!NOTE]
> **Công thức đếm số tham số của PyTorch RNN:**
> $$\text{Parameters} = \underbrace{H \times F}_{\text{weight\_ih}} + \underbrace{H \times H}_{\text{weight\_hh}} + \underbrace{H}_{\text{bias\_ih}} + \underbrace{H}_{\text{bias\_hh}} = H(F + H + 2)$$
> Với $F=1$ (1 feature đầu vào), $H=8$ (chiều ẩn):
> - $W_{ih}: 8 \times 1 = 8$.
> - $W_{hh}: 8 \times 8 = 64$.
> - Hai bias: $8 + 8 = 16$.
> - Tổng tham số cell RNN: $8 + 64 + 16 = \mathbf{88}$ tham số.
> - Tầng phân loại Linear ($8 \to 2$): $W \in \mathbb{R}^{2 \times 8}$ ($16$ weights) $+ b \in \mathbb{R}^2$ ($2$ biases) $= \mathbf{18}$ tham số.
> - Tổng toàn mạng: $88 + 18 = \mathbf{106}$ tham số.

## 5. Backpropagation through time là chain rule

BPTT là backprop trên graph trải theo thời gian. Weight dùng chung nên gradient nhận tổng đóng góp từ mọi lần sử dụng, như kernel chia sẻ ở CNN. Không cần một thuật toán đạo hàm hoàn toàn khác. [D2L: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)

> [!NOTE]
> **Bản chất của BPTT (Backpropagation Through Time):**
> Về mặt toán học, BPTT không phải là một thuật toán mới kỳ bí. Khi bạn mở rộng (unroll) mạng RNN qua $T$ bước thời gian, nó biến thành một mạng truyền thẳng nhiều tầng sâu $T$ lớp. Áp dụng quy tắc dây chuyền (Chain Rule - Bài 10), gradient sai số được truyền giật lùi từ bước $t=T$ về $t=T-1, \dots, t=1$. Vì cùng một ma trận trọng số $W$ được dùng ở mọi bước, gradient cập nhật cuối cùng là **tổng tích lũy đạo hàm từ tất cả các bước thời gian**:
> $$\frac{\partial L}{\partial W} = \sum_{t=1}^T \frac{\partial L}{\partial a_t} \cdot \frac{\partial a_t}{\partial W}$$

Tiếp ví dụ scalar, chọn loss **chỉ tại state cuối**: L=½(h₃−0.5)²≈0.0242751492. Đặt δₜ=∂L/∂aₜ. Vì tanh′(aₜ)=1−hₜ²:

\[
\delta_3=(h_3-0.5)(1-h_3^2),\qquad
\delta_t=\delta_{t+1}w_h(1-h_t^2)\quad(t=2,1).
\]

Kết quả δ₃≈0.106008057, δ₂≈0.022940578, δ₁≈0.004510391. Cộng gradient về cùng parameter:

\[
\frac{\partial L}{\partial w_x}=\sum_{t=1}^3\delta_tx_t
\approx0.004510391-0.022940578+2(0.106008057)
=0.193585927,
\]
\[
\frac{\partial L}{\partial w_h}=\sum_t\delta_th_{t-1}\approx-0.028259593,
\qquad \frac{\partial L}{\partial b}=\sum_t\delta_t\approx0.133459025.
\]

> [!NOTE]
> **Giải thích từng bước tính tay BPTT:**
> 1. **Tại bước cuối ($t=3$):**
>    - Loss $L = \frac{1}{2}(h_3 - 0.5)^2 \implies \frac{\partial L}{\partial h_3} = h_3 - 0.5 = 0.720341322 - 0.5 = 0.220341322$.
>    - $\delta_3 = \frac{\partial L}{\partial h_3} \cdot \tanh'(a_3) = (h_3 - 0.5)(1 - h_3^2) = 0.220341322 \times (1 - 0.720341322^2) \approx \mathbf{0.106008057}$.
> 2. **Truyền ngược về bước 2 ($t=2$):**
>    - $\delta_2 = \delta_3 \times w_h \times (1 - h_2^2) = 0.106008057 \times 0.25 \times (1 - (-0.366583715)^2) \approx \mathbf{0.022940578}$.
> 3. **Truyền ngược về bước 1 ($t=1$):**
>    - $\delta_1 = \delta_2 \times w_h \times (1 - h_1^2) = 0.022940578 \times 0.25 \times (1 - 0.462117157^2) \approx \mathbf{0.004510391}$.
> 4. **Tổng hợp gradient các tham số:**
>    - $\frac{\partial L}{\partial w_x} = \delta_1 x_1 + \delta_2 x_2 + \delta_3 x_3 = 0.00451 \times 1 + 0.02294 \times (-1) + 0.10601 \times 2 \approx \mathbf{0.193586}$.
>    - $\frac{\partial L}{\partial w_h} = \delta_1 h_0 + \delta_2 h_1 + \delta_3 h_2 = 0 + 0.02294 \times 0.46212 + 0.10601 \times (-0.36658) \approx \mathbf{-0.028260}$.
>    - $\frac{\partial L}{\partial b} = \delta_1 + \delta_2 + \delta_3 \approx \mathbf{0.133459}$.

Mode `hand` so các giá trị với nn.RNN float64. Trong API hai bias, gradient **mỗi bias** bằng ∂L/∂b trên; không cộng hai gradient rồi gọi đó là gradient của một bias khái niệm. Nếu tối ưu cả hai bias bằng SGD, tổng bias hiệu dụng đổi theo cả hai update, khác mô hình chỉ có một bias cùng learning rate.

Công thức δ ở đây chỉ đúng cho loss đặt tại h₃ và cell scalar này. Nếu mỗi bước đều có loss/head, gradient tại hₜ còn nhận nhánh loss trực tiếp ở bước đó. Không lấy ví dụ cuối chuỗi áp nguyên xi cho mọi bài many-to-many.

## 6. Vì sao gradient có thể mất dần hoặc bùng nổ?

Trong scalar tanh cell, ∂hₜ/∂hₜ₋₁=(1−hₜ²)w_h. Qua nhiều bước phải nhân nhiều đạo hàm. Tanh gần bão hòa làm hệ số1−h² nhỏ; weight và trạng thái quyết định tích cụ thể. Với vector, đó là tích các Jacobian, không thể chỉ nhìn một weight rồi kết luận toàn mạng chắc chắn mất/bùng gradient.

Để cô lập phép nhân, xét **mô hình tuyến tính giả định** hₜ=ahₜ₋₁, không tanh, h₀=1. Sau20 bước, ∂h₂₀/∂h₀=a²⁰:

- a=0.5 → khoảng9.5367×10⁻⁷.
- a=1 →1.
- a=1.5 → khoảng3.325,25673.

> [!WARNING]
> **Hai căn bệnh kinh niên của RNN cổ điển (Vanishing & Exploding Gradients):**
> Hãy hình dung việc lan truyền tín hiệu qua thời gian như trò chơi "truyền tin":
> 1. **Biến mất Gradient (Vanishing Gradient):** Nếu mỗi lần truyền qua 1 bước thời gian, tín hiệu bị giảm đi một nửa ($a = 0.5$). Sau 20 bước, độ lớn gradient chỉ còn $0.5^{20} \approx 0.00000095$ (gần bằng 0 tuyệt đối!). Tín hiệu sai số từ hiện tại không thể truyền ngược về quá khứ xa. Hậu quả là **mạng nơ-ron bị "mất trí nhớ ngắn hạn"**, nó chỉ học được những từ ngữ ngay sát cạnh và hoàn toàn quên mất ngữ cảnh ở đầu câu.
> 2. **Bùng nổ Gradient (Exploding Gradient):** Nếu tín hiệu bị khuếch đại lên 1.5 lần ($a = 1.5$). Sau 20 bước, gradient vọt lên $1.5^{20} \approx 3.325$ lần! Bước nhảy cập nhật tham số trở nên khổng lồ, phá nát toàn bộ trọng số và làm hàm loss tràn số thành `NaN`.

Mode `gradients` đối chiếu với autograd. Đây là minh họa độ nhạy qua chuỗi dài, chưa là đo gradient của RNN phân loại ở cuối bài.

![Tích đạo hàm và giới hạn gradient](figures/03-gradients.png)

Clipping bài 13 có thể giới hạn gradient lớn trước step; không khôi phục tín hiệu đã mất vì gradient rất nhỏ. Lab clip global norm ở1, ghi max norm **trước** clipping theo epoch. Với Adam, norm gradient bị giới hạn không có nghĩa norm update bị giới hạn đúng learning_rate×1.

LSTM đưa thêm cell state và các gate điều tiết luồng thông tin; GRU là một hướng gated recurrent khác. Chúng nhằm cải thiện việc truyền/lưu thông tin nhưng không bảo đảm nhớ mọi lịch sử. Bài chỉ giới thiệu động cơ, không huấn luyện LSTM/GRU. [LSTM API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html)

> [!NOTE]
> **Động cơ ra đời của LSTM (Long Short-Term Memory) và GRU:**
> - Để giải quyết triệt để bệnh biến mất gradient, Hochreiter & Schmidhuber (1997) phát minh ra kiến trúc **LSTM**.
> - Bí quyết của LSTM nằm ở **Đường truyền cao tốc Trạng thái Ô (Cell State $C_t$)**: Thay vì liên tục nhân ma trận làm co cụm tín hiệu như RNN thường, Cell State truyền dọc thời gian chủ yếu thông qua **phép cộng tuyến tính** ($C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$).
> - Nhờ các cổng thông minh (Cổng quên — Forget Gate, Cổng nạp — Input Gate, Cổng xuất — Output Gate), gradient có thể chảy thông suốt ngược dòng hàng trăm bước thời gian mà không bị suy giảm, giúp mô hình nhớ được ngữ cảnh dài lâu hơn rất nhiều.

## 7. Chuỗi dài/ngắn khác nhau: padding không phải “không làm gì”

Một batch cần cùng T để thành tensor. Ví dụ A=`[1]`,B=`[2,3,4]`: right-padding bằng0 cho `[[1,0,0],[2,3,4]]`, lengths=`[1,3]`. Số0 cũng có thể là dữ liệu thật; không đoán length bằng cách đếm phần tử khác0. Sau chuẩn hóa, padding0 của lab là0 trên thang đã chuẩn hóa, không phải raw value0.

[pad_sequence](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.rnn.pad_sequence.html) gộp chuỗi và thêm đệm; nó không tự làm RNN bỏ qua phần đệm.

> [!NOTE]
> **Vấn đề Chuỗi độ dài biến thiên (Variable-length Sequences):**
> Trong thực tế, các câu văn hoặc đoạn âm thanh hầu như không bao giờ có độ dài bằng nhau (ví dụ câu A có 3 từ, câu B có 15 từ). Nhưng GPU chỉ có thể tính toán ma trận song song khi dữ liệu được đóng thành một khối hộp chữ nhật vuông vức có kích thước cố định $(B, T_{max}, F)$.
> Do đó, ta buộc phải đệm thêm các giá trị giả lập (thường là số 0 — Zero Padding) vào đuôi các chuỗi ngắn để kéo dài chúng bằng chuỗi dài nhất trong batch ($T_{max}$).

**Tính tay để thấy lỗi:** cell ReLU có w_x=w_h=1,b=0.5,h₀=0, mọi pre-activation đều dương:

| Chuỗi | Các state khi đọc tensor padded | State đúng tại bước thật cuối |
| ----- | ------------------------------- | ----------------------------: |
| A     | 1.5 →2.0 →2.5                   |                           1.5 |
| B     | 2.5 →6.0 →10.5                  |                          10.5 |

Lấy `output[:,-1]` hoặc hidden cuối của tensor padded cho A=2.5, sai nếu mục đích biểu diễn phần thật. Với **một hướng**, gather `output[batch_index,lengths−1]` lấy được state đúng ở các bước thật; vẫn tốn compute cho padding. Trong RNN hai hướng hoặc cấu hình khác, không tự mở rộng lập luận này.

> [!WARNING]
> **Cạm bẫy ngây thơ: Đệm số 0 KHÔNG CÓ NGHĨA là mạng RNN sẽ đứng yên!**
> Hãy quan sát chuỗi A=`[1]` được đệm thành `[1, 0, 0]` với cell ReLU:
> - $t=1$: Input thật $x_1 = 1 \implies h_1 = \text{ReLU}(1 \times 1 + 1 \times 0 + 0.5) = \mathbf{1.5}$. (Đây là bộ nhớ đúng của chuỗi A!).
> - $t=2$: Gặp số đệm $x_2 = 0 \implies h_2 = \text{ReLU}(1 \times 0 + 1 \times 1.5 + 0.5) = \mathbf{2.0}$. (Bộ nhớ đã bị ô nhiễm bởi số đệm!).
> - $t=3$: Gặp số đệm $x_3 = 0 \implies h_3 = \text{ReLU}(1 \times 0 + 1 \times 2.0 + 0.5) = \mathbf{2.5}$.
> Nếu bạn ngây thơ lấy `h_last = output[:, -1]`, bạn sẽ nhận được $h = 2.5$ hoàn toàn sai lệch! Giá trị đúng tại bước thật cuối cùng của A phải là $h_1 = 1.5$.

Lab dùng [pack_padded_sequence](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.rnn.pack_padded_sequence.html): lengths là vector CPU long, các bước thật đứng ở đầu mỗi sample, enforce_sorted=False để nhận lengths chưa sắp. RNN xử lý các bước hợp lệ và trả hidden theo các mẫu; mode `padding` kiểm cả thứ tự trả về với lengths `[1,3]`. Nếu lengths sai quá ngắn, thông tin thật có thể bị bỏ mất; lab chặn length≤0 hoặc lớn hơnT.

```python
from lab import padding_demo
print(padding_demo())  # naive [2.5,10.5]; valid/packed [1.5,10.5]
```

Mode này còn đổi padding của A thành99 và kiểm packed state không đổi. Không sửa giá trị thật: pack dựa **lengths**, không dựa vào giá trị0. Batch packing trong lab chỉ dùng right-padding; left-padding chưa được hỗ trợ bởi pipeline này.

> [!TIP]
> **Giải pháp chuẩn kỹ thuật: `pack_padded_sequence`**
> Hàm `pack_padded_sequence` nén tensor padded lại thành một cấu trúc đặc biệt gọi là `PackedSequence`. Nó thông báo tường minh cho tầng PyTorch RNN biết chính xác độ dài thực sự của từng chuỗi (`lengths`). Nhờ vậy:
> 1. RNN sẽ dừng cập nhật trạng thái ẩn ngay khi đọc hết phần dữ liệu thật của từng chuỗi.
> 2. Các vị trí đệm padding hoàn toàn bị bỏ qua, không gây tốn tài nguyên GPU và không làm biến dạng trạng thái ẩn cuối cùng.
> 3. Trả về đúng giá trị $[1.5, 10.5]$ bất kể bạn đệm số $0$ hay số $99$ vào đuôi.

![Padding và hidden state cuối](figures/04-padding.png)

### Nếu loss có ở từng bước

Cần mask các vị trí đệm khi đo loss/metric. Ví dụ ba loss thật là1,4,9, được bố trí trong tensor2×3 có ba ô padding: mean trên6 ô sau gán padding0 là14/6, còn mean theo bước thật là **14/3**. Nếu muốn mỗi chuỗi nặng như nhau, mean per-sequence là((1+4)/2+9)/2=5.75, một mục tiêu khác. Phải khai báo denominator. Lab phân loại toàn chuỗi có một target/mẫu nên CE được cộng theo số chuỗi, không tính token loss ở phần padding.

> [!NOTE]
> **Kỹ thuật Mặt nạ (Loss Masking) trong bài toán gán nhãn từng bước:**
> Khi tính mất mát cho từng từ trong câu (Many-to-Many), các vị trí padding không mang nhãn thật. Nếu tính loss trung bình trên toàn bộ tensor $(B \times T)$, mẫu số sẽ bị chia thừa cho cả các ô padding vô nghĩa:
> - Tính ngây thơ trên cả 6 ô: $\frac{1 + 4 + 9 + 0 + 0 + 0}{6} = \frac{14}{6} \approx 2.33$ (Loss bị pha loãng sai lệch).
> - Tính đúng có mặt nạ (Masking): Tạo một ma trận boolean `mask` (bằng 1 ở vị trí thật, bằng 0 ở padding). Tổng loss chỉ chia cho số lượng token thật: $\frac{1 + 4 + 9}{3} = \frac{14}{3} \approx 4.67$.

## 8. Hidden state có vòng đời

Các sample trong lab là các chuỗi độc lập. Mỗi forward không truyền hx nên nn.RNN bắt đầu bằng zeros; không đưa state của người A sang người B chỉ vì nằm cùng vị trí batch. Model không tự lưu lịch sử các lần gọi như một bộ nhớ xuyên request.

> [!NOTE]
> **Vòng đời của Hidden State: Không trạng thái (Stateless) vs Có trạng thái (Stateful)**
> - **Stateless (Mặc định trong đa số bài toán):** Mỗi mẫu trong batch là một câu chuyện độc lập. Khi bắt đầu duyệt một câu mới hoặc một tệp âm thanh mới, ta bắt buộc phải khởi tạo lại $h_0 = 0$. Tuyệt đối không được giữ lại hidden state của batch trước, nếu không thông tin của người dùng A sẽ vô tình bị "nhớ nhầm" sang người dùng B!
> - **Stateful Streaming (Dòng dữ liệu dài liên tục):** Khi bạn đọc một cuốn sách dài 100.000 từ hoặc đo cảm biến máy bay liên tục trong 10 tiếng, ta không thể nạp toàn bộ vào một batch. Lúc này, chuỗi dài được cắt thành các đoạn nhỏ (chunks). Hidden state ở cuối đoạn 1 được truyền làm $h_0$ cho đoạn 2.

Nếu cố ý chia **cùng một stream** thành nhiều chunk theo thứ tự, có thể carry hidden sang chunk sau. Khi truncated BPTT, `hidden=hidden.detach()` giữ giá trị nhưng cắt đường gradient về các chunk trước; khác reset hidden về0. Cần biết sample nào nối với sample nào và reset ở ranh giới stream. Lab chỉ chạy full BPTT cho chuỗi4–9 bước, chưa thực nghiệm stateful streaming/truncated BPTT.

> [!IMPORTANT]
> **Truncated BPTT và lệnh `hidden.detach()`:**
> Nếu bạn tiếp tục truyền hidden state qua 1.000 chunk mà không cắt đồ thị tính toán, bộ nhớ đồ thị autograd sẽ phình to khủng khiếp và làm sập RAM/VRAM. Lệnh `h_detach = h.detach()` thực hiện một nhiệm vụ sống còn: **Giữ nguyên giá trị số học của vector bộ nhớ để truyền tiếp sang tương lai, nhưng chặt đứt nhánh lan truyền ngược đồ thị về quá khứ**. Điều này cho phép mô hình duy trì trí nhớ liên tục mà chi phí tính toán backprop chỉ bị giới hạn trong phạm vi 1 chunk.

Trong nhiều lớp RNN, state tensor còn có trục layer/direction. Không dùng cách chỉ số `h_n[-1]` của model một lớp một hướng trong bài rồi suy áp được nguyên xi cho bidirectional hoặc LSTM có thêm cell state.

## 9. Chuỗi thời gian: căn window/target trước khi chọn model

Dãy `[10,12,11,13,15]`, lookback3, horizon1. Nếu index bắt đầu0:

| Input đã biết | Index input | Target | Index target |
| ------------- | ----------- | -----: | -----------: |
| [10,12,11]    | [0,1,2]     |     13 |            3 |
| [12,11,13]    | [1,2,3]     |     15 |            4 |

Mode `windows` tạo đúng hai dòng này. Chỉ số target phải sau input cuối; thêm giá trị15 vào feature để dự đoán13 sẽ nhìn tương lai trong ví dụ này.

> [!NOTE]
> **Kỹ thuật Cửa sổ trượt (Sliding Window) trong Dự báo chuỗi thời gian:**
> - **Lookback Window ($L$ — Độ dài cửa sổ nhìn lại):** Số bước thời gian quá khứ được phép quan sát để làm đặc trưng đầu vào ($x$).
> - **Horizon ($H$ — Tầm nhìn dự báo):** Khoảng thời gian tương lai cần dự đoán ($y$).
> - Ví dụ trên: $L=3, H=1$. Dùng 3 ngày liên tiếp $[10, 12, 11]$ (ngày 0, 1, 2) để dự báo giá ngày thứ 3 ($13$). Sau đó trượt cửa sổ sang phải 1 bước: dùng $[12, 11, 13]$ (ngày 1, 2, 3) để dự báo giá ngày thứ 4 ($15$).

Với forecasting, chia theo thời điểm **target** và thông tin có sẵn ở lúc dự báo, không random split các window chồng lấn rồi coi là độc lập. Ví dụ train target đến index5, validation target6–7, test target8–9. Window validation có thể dùng lịch sử đã quan sát thuộc đoạn train: điều đó có thể hợp lệ. Không phải mọi overlap input đều là leakage; phải kiểm horizon, thời điểm dữ liệu/nhãn có sẵn và cách đánh giá một bước hay nhiều bước. Khi nhãn bao phủ khoảng tương lai hoặc có nhiều nguồn tương quan, có thể cần gap/purge/group split riêng.

> [!WARNING]
> **Lỗi nhìn trước tương lai (Lookahead Bias / Data Leakage trong Time Series):**
> Trong bài toán chuỗi thời gian, tuyệt đối không bao giờ được dùng hàm `train_test_split(..., shuffle=True)` ngẫu nhiên!
> Nếu bạn xáo trộn ngẫu nhiên các cửa sổ trượt:
> - Cửa sổ ngày $[1, 2, 3] \to \text{ngày } 4$ nằm ở tập Train.
> - Cửa sổ ngày $[2, 3, 4] \to \text{ngày } 5$ nằm ở tập Test.
> Mô hình ở tập Test sẽ dự đoán ngày 5 dựa vào ngày 4 — một giá trị mà nó đã được nhìn thấy và ghi nhớ trực tiếp từ tập Train!
> **Quy tắc bắt buộc:** Luôn chia dữ liệu theo **trục thời gian tuyến tính (Chronological Split)**: Dùng quá khứ để huấn luyện (ví dụ tháng 1 - tháng 8), dùng đoạn giữa để thẩm định validation (tháng 9 - tháng 10), và dùng đoạn tương lai xa nhất làm tập kiểm thử test độc lập (tháng 11 - tháng 12).

Scaler/vocabulary fit từ train, không từ cả timeline. Nếu feature là ID danh mục/token, giá trị ID không tự mang khoảng cách số học; embedding ở bài 16 sẽ xử lý chủ đề đó. Bài này dùng một feature số thực, không giả lập token ID thành tín hiệu đo lường.

## 10. Thực hành: phân loại tăng/giảm với chuỗi4–9 bước

Generator data_seed17 tạo96 group. Mỗi group có một dãy gần tăng gồm center ngẫu nhiên trong[−1,1], amplitude[0.6,1.2], độ dài nguyên4–9 và nhiễu Gaussian std0.06. Hai sample là dãy đó và bản đảo chính xác, nhãn0=tăng,1=giảm. Nhiễu không bảo đảm mọi cặp điểm kề tăng tuyệt đối; nhãn là hướng sinh của chuỗi, không phải kiểm toán strict monotonic.

Split giữ cả hai sample của group cùng tập:64/16/16 group → **128/32/32 chuỗi**. Cân bằng lớp và cùng phân bố độ dài giữa hai lớp nhờ ghép cặp. Fit mean/std chỉ từ **các bước thật của train**, mỗi time point có trọng số ngang nhau, không thêm padding: mean≈0.0246224512,std≈0.7806527695. Mỗi sample chuẩn hóa trước khi collate/pad.

Một layer RNN(1,8,tanh), head Linear8→2, tổng106 parameter. Input batch `(B,T_max,1)`, lengths `(B,)`, label `(B,)` long. Forward dùng pack và head trên h_n[-1]. Config định trước: Adam lr=.01, batch16,30 epoch, seed2026, CPU float32 một thread, clip norm1. Có8 update/epoch và240 update toàn run.

Collate chuẩn bị batch; shuffle chỉ đổi thứ tự sample, không đổi bước thời gian. Đánh giá dùng eval/no_grad, cộng CE reduction=sum rồi chia số chuỗi. Chọn minimum validation CE, hòa giữ epoch sớm hơn; test chỉ chấm sau lựa chọn. Checkpoint lưu weight, config, preprocessing, class mapping, data hash và epoch; là artifact inference, **chưa có optimizer/RNG để resume** như last.pt bài 12.

Kết quả thực chạy:

| Đại lượng                                            | Kết quả                  |
| ---------------------------------------------------- | ------------------------ |
| Best epoch                                           | 30                       |
| Validation CE / accuracy                             | ≈0.002202466 /32/32=100% |
| Test CE / accuracy                                   | ≈0.002150564 /32/32=100% |
| Confusion test, hàng=true,cột=pred, thứ tự tăng/giảm | `[[16,0],[0,16]]`        |
| Baseline majority: validation accuracy               | 50%                      |
| Quy tắc dấu điểm cuối−đầu: validation accuracy       | 100%                     |

Baseline endpoint được khai báo trước: cuối>đầu thì tăng, ngược lại giảm. Kết quả nhắc rằng bài đơn giản này **không cần RNN mới giải được**. Mean-only không có thông tin phân biệt trong cặp đảo: về toán học cùng mean mà nhãn đối nhau. Đây là giải thích mất thông tin thứ tự, không phải một mô hình mean-only đã được fit/benchmark trong lab.

> [!IMPORTANT]
> **Bài học sâu sắc về Baseline & Nguyên lý Dao cạo Occam (Occam's Razor):**
> Mô hình RNN trong thí nghiệm đạt độ chính xác hoàn hảo $100\%$ trên tập Test. Nhưng hãy chú ý dòng cuối cùng của bảng kết quả:
> **Một quy tắc đơn giản chỉ so sánh 2 số: Nếu $x_{\text{cuối}} > x_{\text{đầu}}$ thì kết luận là "Tăng", ngược lại là "Giảm" cũng đạt chính xác $100\%$!**
> Đây là lời cảnh tỉnh lớn cho mọi kỹ sư AI: Đừng vội vã triển khai các mô hình học sâu phức tạp (RNN, LSTM, Transformer) khi một luật heuristic hoặc một mô hình tuyến tính đơn giản đã đủ để giải quyết triệt để bài toán kinh doanh với chi phí vận hành bằng 0. Hãy luôn xây dựng baseline đơn giản trước tiên để đối chiếu năng lực thực sự của mô hình học sâu.

Một seed/split synthetic không chứng minh chất lượng trên sensor thật hoặc chuỗi dài hơn. Test32 sample đến từ16 group; không coi là32 nguồn độc lập khi suy khoảng tin cậy. Không huấn luyện/so sánh LSTM, GRU hoặc Transformer ở đây.

![Run thật và baseline](figures/05-training.png)

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import Config, train, predict_sequence
with TemporaryDirectory() as directory:
    out = Path(directory)/'run'
    report = train(out, Config(epochs=2))  # Kiểm pipeline ngắn, không kỳ vọng số bảng 30 epoch
    print(report['parameters'])  # 106
    print(predict_sequence(out/'best.pt', [0.,.4,.8,1.2]))
```

`prediction_roundtrip=True` trong report kiểm đường predict_sequence và forward trên cùng chuỗi test cho xác suất giống nhau sau tải checkpoint; không dùng phép kiểm này để chọn lại model. Output gồm data.json có values/group/split/nhãn, config.json, preprocess.json, metadata.json, metrics.csv, best.pt và report.json. Hash hỗ trợ phát hiện dữ liệu/code đổi, không tự thay thế bản dữ liệu hoặc version control.

## 11. Chẩn đoán những lỗi thường gặp

- Shape đúng số phần tử nhưng sai trục: kiểm ý nghĩa B,T,F bằng một sample cụ thể, không sửa bằng reshape tùy ý.
- Model dùng padding như dữ liệu: so packed với chạy từng chuỗi riêng, thử thay giá trị chỉ trong vùng padding.
- Label hoặc lengths bị tráo sau sort: kiểm batch chưa sắp và đối chiếu từng sample; đừng tự sort một tensor mà quên các tensor đi kèm.
- Nhớ sang sample khác: kiểm vòng đời hx, ranh giới stream và thứ tự batch; state cùng index batch không có nghĩa cùng đối tượng.
- Gradient nhỏ/lớn: đọc norm trước clip, độ dài, saturation và scale feature; clipping không thay thế kiểm dữ liệu.
- Điểm tốt nhưng baseline cũng tốt: xem task có cần thứ tự/model phức tạp không; báo baseline trước khi kết luận năng lực.

> [!TIP]
> **Bảng tóm tắt chẩn đoán lỗi mô hình chuỗi:**
>
> | Triệu chứng | Nguyên nhân tiềm ẩn | Cách kiểm tra & Khắc phục |
> |---|---|---|
> | **Loss không giảm / hội tụ cực chậm** | Gradient bị mất dần (Vanishing gradient) do chuỗi quá dài hoặc hàm kích hoạt bị bão hòa. | Đổi sang LSTM/GRU; kiểm tra scale dữ liệu đầu vào; rút ngắn độ dài chuỗi. |
> | **Loss vọt lên `NaN` đột ngột** | Gradient bùng nổ (Exploding gradient) do tích lũy trọng số hồi quy quá lớn. | Bật Gradient Clipping (`torch.nn.utils.clip_grad_norm_`, ngưỡng 1.0); giảm learning rate. |
> | **Độ chính xác tụt khi đổi độ dài batch** | Hidden state cuối bị đọc nhầm ở vùng số 0 của padding thay vì bước thật. | Dùng `pack_padded_sequence` hoặc trích xuất đúng tọa độ `lengths - 1`. |
> | **Kết quả test cao giả tạo** | Trộn ngẫu nhiên các cửa sổ trượt dẫn đến nhìn trước tương lai (Lookahead bias). | Chia split theo trục thời gian (Time-based split); fit Scaler chỉ trên tập Train. |

## 12. Chạy thực hành và học tiếp

Đã chạy Python 3.14.6, torch 2.15.0, NumPy 2.5.4, macOS arm64. Model dùng CPU float32; hand/padding/gradient toy dùng float64. [requirements.txt](requirements.txt) ghim phụ thuộc. Từ thư mục bài 15:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode padding
.venv/bin/python lab.py --mode gradients
.venv/bin/python lab.py --mode windows
.venv/bin/python lab.py --mode all --out /tmp/lesson15-run-01
```

Windows dùng `.venv\Scripts\python` và output phù hợp. Chọn output mới; bỏ --out để tạo thư mục tạm và in đường dẫn, chép ra nơi lưu lâu dài nếu cần giữ. `--mode train` chỉ chạy phân loại chuỗi. Không yêu cầu mạng tải dataset, GPU hay thư viện NLP. Bản ghim cần wheel phù hợp; đối chiếu [cài PyTorch](https://pytorch.org/get-started/locally/) nếu nền tảng khác. Seed không bảo đảm giống từng bit giữa các phiên bản/backend.

Năm hình PNG/SVG có sẵn. Tái tạo bằng [make_figures.py](make_figures.py), Matplotlib 3.11.2 và run tạm:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Làm [10 bài tập](exercises.md), đối chiếu [lời giải](solutions.md). Cần tính được h₁–h₃, giải thích gradient shared weight, phân biệt padding/length/mask và kiểm input/target không nhìn tương lai. **Bài 16 dự kiến:** embedding và biểu diễn, hiện mới là đề cương. [Nguồn và phạm vi đã đọc](../SOURCES.md).
