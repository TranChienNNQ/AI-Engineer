# Bài 17 — Attention: từ Q–K–V đến multi-head và mô hình học được

[Mục lục](../README.md) · [Bài 16](../16-embedding-bieu-dien/README.md) · [Đào sâu toán và API](deep-dive.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Bài này được viết kỹ hơn vì attention là nền cho các bài Transformer sau. Toàn bộ ví dụ số, hình và dữ liệu là tự tạo; PyTorch chạy thật trên CPU, không tải model hay gọi API AI.

> [!NOTE]
> **Định nghĩa các khái niệm cốt lõi của cơ chế Attention:**
> 1. **Cơ chế Chú ý (Attention Mechanism):** Là cơ chế toán học cho phép mô hình học sâu tập trung "chú ý" vào những phần quan trọng nhất của dữ liệu đầu vào tùy thuộc vào ngữ cảnh hiện tại, thay vì ép toàn bộ thông tin vào một vector nén cố định như mạng RNN hay Mean Pooling.
> 2. **Bộ ba Query — Key — Value (Q–K–V):** Bộ ba khái niệm trừu tượng mô phỏng hệ thống tra cứu thông tin (tương tự như công cụ tìm kiếm hoặc cơ sở dữ liệu):
>    - **Query ($Q$ — Câu hỏi / Nhu cầu truy vấn):** Đại diện cho mục tiêu hiện tại: *"Tôi đang tìm kiếm loại thông tin gì?"*.
>    - **Key ($K$ — Khóa / Tiêu đề nhận diện):** Đại diện cho nhãn định danh của từng mục dữ liệu trong kho lưu trữ: *"Tôi chứa đựng thông tin về chủ đề nào?"*.
>    - **Value ($V$ — Giá trị / Nội dung thực tế):** Đại diện cho chính nội dung chi tiết cần được trích xuất nếu Key khớp với Query.
> 3. **Điểm số Tương đồng (Attention Score):** Thước đo mức độ phù hợp giữa Query và Key (thường tính bằng tích vô hướng có chia tỷ lệ: $QK^\top / \sqrt{d_k}$).
> 4. **Trọng số Chú ý (Attention Weights $\alpha$):** Phân phối xác suất chuẩn hóa qua hàm Softmax từ các điểm số tương đồng. Trọng số này quy định tỷ lệ phần trăm đóng góp của từng Value vào kết quả cuối cùng.
> 5. **Self-Attention (Tự chú ý):** Cơ chế trong đó các token trong cùng một câu tự so sánh với nhau ($Q, K, V$ đều sinh ra từ cùng một chuỗi đầu vào) để hiểu được mối liên kết ngữ pháp và ngữ nghĩa nội tại.
> 6. **Cross-Attention (Chú ý chéo):** Cơ chế trong đó Query đến từ một chuỗi (ví dụ: câu đang dịch ở Decoder), còn Key và Value đến từ một chuỗi khác (ví dụ: câu gốc ở Encoder).
> 7. **Multi-Head Attention (Chú ý đa đầu):** Kiến trúc chia nhỏ không gian biểu diễn thành nhiều "đầu" (heads) độc lập, cho phép mô hình đồng thời quan sát và kết nối dữ liệu ở nhiều khía cạnh khác nhau (cú pháp, ngữ nghĩa, đại từ thay thế).

## 1. Mục tiêu và cách đọc

Sau bài này, bạn có thể tự tính score→softmax→weighted sum; suy shape Q, K, V và output; giải thích scaling/mask; phân biệt self-attention, cross-attention, multi-head; kiểm một implementation và nhận ra giới hạn khi diễn giải attention weight.

> [!TIP]
> **Tại sao Attention là bước ngoặt vĩ đại nhất của AI hiện đại?**
> Trước năm 2017, các mô hình xử lý chuỗi (như RNN, LSTM) bị giới hạn nghiêm trọng bởi việc phải truyền thông tin tuần tự qua từng bước thời gian, dẫn đến hiện tượng nghẽn cổ chai thông tin (bottleneck) và đãng trí khi gặp câu dài. Cơ chế Attention (đặc biệt là bài báo lịch sử *"Attention Is All You Need"* của Google) đã phá bỏ hoàn toàn vòng lặp tuần tự: nó cho phép mọi từ trong văn bản kết nối trực tiếp với nhau với khoảng cách đường truyền chỉ là $O(1)$. Đây chính là nền móng trực tiếp sinh ra Transformer, BERT, GPT-4, Claude, Gemini và toàn bộ làn sóng GenAI ngày nay.

Cần hiểu vector/matmul bài 3, softmax/cross-entropy bài 6, gradient bài 10, padding bài 15 và embedding bài 16. Ký hiệu và shape được nhắc lại trong bài; không yêu cầu đã biết Transformer.

Nên học theo ba lượt, không cố nhớ hết tên ngay:

1. **Cơ chế một query:** phần 2–6, chạy `hand`, `matrix`; tự tính được1/7,2/7,4/7 trước khi xem code.
2. **Đúng khi dùng thật:** phần 7–12 và [phụ lục](deep-dive.md), chạy `masks`, `multihead`, `invariance`.
3. **Học và kiểm:** phần 13–15, chạy `train`, làm14 bài tập rồi đọc lời giải.

Bài đã huấn luyện một single-head cross-attention nhỏ và kiểm multi-head API; **chưa phải Transformer hoàn chỉnh**. Residual, LayerNorm, FFN và positional representation trong một block sẽ được kết nối ở bài 18.

## 2. Từ mean pooling đến “đọc theo câu hỏi”

Ở bài 16, mean pooling cho mọi token thật cùng trọng số1/T. Nếu một ticket có nhiều chi tiết mà câu hỏi chỉ cần một chi tiết, một vector mean cố định có thể trộn chúng lại. RNN bài 15 giữ state cập nhật theo thứ tự; attention cho phép tạo một kết quả **phụ thuộc query hiện tại** bằng cách gán trọng số cho các value được phép đọc.

> [!NOTE]
> **Ẩn dụ trực quan: Tìm sách trong thư viện lớn**
> Hãy hình dung bạn bước vào một thư viện có 10.000 cuốn sách:
> - **Mean Pooling (Bài 16):** Giống như việc xay nhuyễn tất cả 10.000 cuốn sách thành một cốc sinh tố rồi uống! Mọi thông tin bị hòa lẫn vào nhau, cuốn sách quan trọng nhất bị pha loãng bởi hàng ngàn cuốn sách râu ria.
> - **RNN (Bài 15):** Giống như việc bạn phải đọc lướt tuần tự từng cuốn sách từ cuốn thứ 1 đến cuốn thứ 10.000 và cố gắng ghi nhớ trong đầu. Khi đọc đến cuốn thứ 10.000 thì bạn đã quên sạch chi tiết của những cuốn sách đầu tiên.
> - **Attention:** Bạn cầm trong tay một tờ giấy ghi câu hỏi: *"Làm thế nào để cấu hình DNS?"* (**Query**). Bạn lướt mắt nhìn qua các tiêu đề trên gáy sách (**Keys**). Cuốn sách nào có tiêu đề khớp với câu hỏi sẽ nhận được ánh nhìn chăm chú $95\%$ (**Attention Weight**), trong khi các cuốn sách về nấu ăn nhận $0\%$. Sau đó, bạn chỉ mở đúng cuốn sách mạng máy tính ra để đọc nội dung bên trong (**Values**) và tổng hợp thành câu trả lời (**Output**).

Hãy nghĩ đến bảng dịch vụ và trạng thái. Câu hỏi cần dịch vụ nào là query; đặc trưng nhận diện mỗi dòng là key; nội dung trạng thái cần lấy là value. So query với từng key để tính mức phù hợp, chuẩn hóa các mức đó rồi trộn value.

| Thành phần | Vai trò | Không nên hiểu nhầm |
|---|---|---|
| Query q | Nhu cầu đọc của một vị trí/câu hỏi | Không nhất thiết là câu chữ người dùng nhập |
| Key kⱼ | Đặc trưng dùng để so với query | Không nhất thiết bằng raw token ID hoặc raw embedding |
| Value vⱼ | Nội dung được lấy/trộn từ dòng j | Key và value cùng dòng nhưng vector có thể khác |
| Weight αⱼ | Mức đóng góp tương đối của value j cho query này | Không tự là xác suất “dòng này đúng sự thật” |
| Output z | Tổng có trọng số của các value | Không phải chính bảng weight hoặc một token được chọn cứng |

Phép so sánh có thể học được; attention không mặc định thực hiện equality lookup chính xác như dict. Query khác có thể cho trọng số khác trên cùng memory. Thêm một key mới cũng thay đổi denominator softmax và có thể thay đổi các trọng số cũ.

> [!NOTE]
> **Khác biệt cốt tử giữa Soft Attention và Hard Dictionary Lookup:**
> - **Dictionary Lookup trong Python (`dict[key]`):** Là phép tra cứu cứng (Hard Equality). Key phải trùng khớp tuyệt đối $100\%$ (`query == key`), trả về đúng 1 Value duy nhất. Phép toán này có đạo hàm bằng 0 ở khắp mọi nơi (bậc thang rời rạc), do đó **mạng nơ-ron không thể học được bằng Gradient Descent**!
> - **Soft Attention trong Deep Learning:** Là phép tra cứu mềm khả vi (Differentiable Soft Lookup). Mọi Value đều đóng góp một phần tỷ trọng $\alpha_j \in [0, 1]$. Vì hàm Softmax và phép nhân ma trận hoàn toàn liên tục và trơn tru, gradient có thể lan truyền ngược xuyên suốt qua $Q, K, V$ để tối ưu trọng số.

Attention không bắt đầu từ Transformer năm2017. Bài [Bahdanau, Cho, Bengio](https://arxiv.org/abs/1409.0473) được đưa lên arXiv năm2014 mô tả soft alignment trong dịch máy để tránh chỉ nén nguồn vào một vector cố định. Bài [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) phát triển kiến trúc Transformer với scaled dot-product và multi-head; ở đây tập trung cơ chế, không tái hiện kết quả dịch máy của các nguồn.

## 3. Một query: ví dụ tính tay có kết quả chính xác

Chọn q=[ln2], ba key k₁=[0], k₂=[1], k₃=[2], và:

```text
v1 = [1,0]
v2 = [0,2]
v3 = [3,1]
```

Query/key có d_k=1, value có d_v=2. Chúng **không bắt buộc có cùng số chiều**. Số key phải bằng số value để giữ cặp theo dòng.

> [!NOTE]
> **Tính độc lập về số chiều giữa ($Q, K$) và $V$:**
> - Query và Key bắt buộc phải có cùng số chiều $d_k$ (ở đây $d_k = 1$) để thực hiện được phép nhân vô hướng so khớp $q \cdot k_j$.
> - Value hoàn toàn có thể có số chiều $d_v$ khác biệt (ở đây $d_v = 2$). Số chiều đầu ra của Attention ($z$) được quyết định $100\%$ bởi số chiều của Value ($d_v$), hoàn toàn không phụ thuộc vào $d_k$!

### Bước A — So query với từng key

Scaled score sⱼ=q·kⱼ/√d_k. Vì √1=1:

| j | q·kⱼ | exp(score) |
|---:|---:|---:|
| 1 | 0 | 1 |
| 2 | ln2 | 2 |
| 3 | 2ln2=ln4 | 4 |

> [!NOTE]
> **Giải thích chi tiết Bước A (Tính điểm tương đồng):**
> Ta có $q = \ln 2 \approx 0.69315$.
> - Dòng 1: $s_1 = q \cdot k_1 = \ln 2 \times 0 = 0 \implies \exp(s_1) = e^0 = \mathbf{1}$.
> - Dòng 2: $s_2 = q \cdot k_2 = \ln 2 \times 1 = \ln 2 \implies \exp(s_2) = e^{\ln 2} = \mathbf{2}$.
> - Dòng 3: $s_3 = q \cdot k_3 = \ln 2 \times 2 = 2\ln 2 = \ln(2^2) = \ln 4 \implies \exp(s_3) = e^{\ln 4} = \mathbf{4}$.
> Các giá trị $\exp(\cdot)$ được chọn khéo léo để ra đúng các số nguyên $1, 2, 4$ giúp bạn dễ dàng theo dõi phép tính tay mà không bị rối bởi số thập phân lẻ!

### Bước B — Softmax qua các key

Tổng exp=7 nên α=[1/7,2/7,4/7]. Ba trọng số không âm và tổng1. Softmax được tính **trên trục key**, nghĩa là với query này, phân phối mức đọc nằm trên ba dòng memory.

> [!NOTE]
> **Giải thích chi tiết Bước B (Chuẩn hóa Softmax thành trọng số chú ý):**
> Tổng các giá trị mũ: $\sum_{j=1}^3 \exp(s_j) = 1 + 2 + 4 = \mathbf{7}$.
> - $\alpha_1 = \frac{\exp(s_1)}{\text{Tổng}} = \mathbf{\frac{1}{7}} \approx 14.29\%$.
> - $\alpha_2 = \frac{\exp(s_2)}{\text{Tổng}} = \mathbf{\frac{2}{7}} \approx 28.57\%$.
> - $\alpha_3 = \frac{\exp(s_3)}{\text{Tổng}} = \mathbf{\frac{4}{7}} \approx 57.14\%$.
> Trọng số $\alpha = [1/7, 2/7, 4/7]$ thỏa mãn đầy đủ tính chất của một phân phối xác suất: $\alpha_j > 0$ và $\sum \alpha_j = 1$. Query này thể hiện mức độ chú ý mạnh nhất vào dòng thứ 3 (chiếm hơn một nửa sự tập trung).

### Bước C — Trộn value, không trộn key

\[
z=\frac17[1,0]+\frac27[0,2]+\frac47[3,1]
=\left[\frac{13}{7}, \frac87\right]
\approx[1.857143,1.142857].
\]

Output có2 chiều vì value có2 chiều. Nó không bằng một value nguyên vẹn: attention này là phép đọc mềm. Nếu muốn chọn cứng key thứ3 bằng argmax thì output sẽ là[3,1], một phép toán khác.

> [!NOTE]
> **Giải thích chi tiết Bước C (Tổng hợp thông tin có trọng số):**
> Vector đầu ra $z$ là tổ hợp tuyến tính lồi (convex combination) của các vector Value:
> $$z = \frac{1}{7} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \frac{2}{7} \begin{bmatrix} 0 \\ 2 \end{bmatrix} + \frac{4}{7} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{1 \times 1 + 2 \times 0 + 4 \times 3}{7} \\ \frac{1 \times 0 + 2 \times 2 + 4 \times 1}{7} \end{bmatrix} = \begin{bmatrix} \frac{1 + 0 + 12}{7} \\ \frac{0 + 4 + 4}{7} \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{13}{7} \\ \frac{8}{7} \end{bmatrix}}$$
> Hãy lưu ý: **Key chỉ dùng để định hướng điểm số chú ý $\alpha$; nội dung thực sự được hòa trộn để tạo nên $z$ là Value!**

![Một query đọc ba cặp key–value](figures/01-qkv.png)

```python
from lab import hand_demo
print(hand_demo())  # weights=[1/7,2/7,4/7]; output=[13/7,8/7]
```

Nếu giữ K, V mà đổi q=0 thì mọi score=0, weight đều1/3, output=[4/3,1]. Nếu giữ q, K mà đổi V thì weight không đổi nhưng output có thể đổi. Q/K điều khiển cách đọc; V quyết định nội dung được trộn. Trong self-attention, chúng có thể cùng bắt nguồn từ X và cùng được cập nhật khi X đổi, nên phép tách ở đây là để hiểu vai trò toán học.

## 4. Nhiều query cùng lúc: ma trận và shape

Gọi L là số query, S là số key/value. Không cần L=S:

\[
S_{score}=\frac{QK^\top}{\sqrt{d_k}}, \qquad
A=\operatorname{softmax}(S_{score}+M, \text{axis=key}), \qquad Z=AV.
\]

M là mask cộng vào score, sẽ học ở phần 8. Đừng nhầm chữ S dùng chỉ số lượng key với ma trận score; trong code tên biến là `scores`.

| Tensor | Một sample | Batch một head |
|---|---|---|
| Q | L×d_k | B×L×d_k |
| K | S×d_k | B×S×d_k |
| V | S×d_v | B×S×d_v |
| QKᵀ và A | L×S | B×L×S |
| Z | L×d_v | B×L×d_v |

Output có **một hàng cho mỗi query**, không một hàng cho mỗi key. Code dùng `k.transpose(-2, -1)` để đổi hai trục cuối, không đảo nhầm batch/head.

> [!IMPORTANT]
> **Giải phẫu kích thước ma trận trong Attention:**
> Hãy ghi nhớ quy tắc nhân ma trận để không bao giờ bị nhầm shape:
> 1. $Q \in \mathbb{R}^{L \times d_k}$ nhân với $K^\top \in \mathbb{R}^{d_k \times S} \implies S_{score} \in \mathbb{R}^{L \times S}$.
>    - Hàng thứ $i$, cột thứ $j$ của ma trận score chính là điểm tương đồng giữa Query thứ $i$ và Key thứ $j$.
> 2. Hàm Softmax được tính dọc theo từng hàng (trục $S$ — `dim=-1`):
>    - Ma trận trọng số $A \in \mathbb{R}^{L \times S}$. Mỗi hàng của $A$ là một phân phối xác suất có tổng bằng $1$.
> 3. $A \in \mathbb{R}^{L \times S}$ nhân với $V \in \mathbb{R}^{S \times d_v} \implies Z \in \mathbb{R}^{L \times d_v}$.
>    - Mỗi Query thứ $i$ sẽ nhận được đúng một vector biểu diễn tổng hợp $z_i$ có số chiều $d_v$.

### Ví dụ2×3 có scaling khác1

Lấy Q=`[[1,0], [0,1]]`, K=`[[1,0], [0,1], [1,1]]`, V như phần 3. d_k=2 nên:

```text
scores = [[1/√2, 0,    1/√2],
          [0,    1/√2, 1/√2]]
```

Đặt r=exp(1/√2), a=r/(2r+1)≈0.401112093, b=1/(2r+1)≈0.197775815. Khi đó:

```text
A = [[a,b,a],         Z = [[4a,    2b+a],
     [b,a,a]]             [b+3a,  3a  ]]
```

Z≈`[[1.604448371,0.796663722], [1.401112093,1.203336278]]`. Mỗi hàng A cộng1; mỗi cột không cần cộng1. A không cần đối xứng, kể cả khi số query bằng số key; softmax theo từng hàng và Q, K khác nhau có thể phá đối xứng.

> [!NOTE]
> **Giải thích tính chất không đối xứng của ma trận Attention ($A$):**
> Trong ma trận $A$, hàng $i$ biểu thị *"Query $i$ chú ý vào những Key nào"*.
> - Việc từ "ngân hàng" chú ý mạnh vào từ "tiền tệ" không có nghĩa là từ "tiền tệ" cũng phải chú ý ngược lại với trọng số tương đương vào từ "ngân hàng".
> - Softmax chuẩn hóa độc lập theo từng hàng (tổng mỗi hàng $= 1$). Tổng các phần tử theo cột hoàn toàn không bắt buộc phải bằng 1!

![Score → trọng số → output theo ma trận](figures/02-matrix.png)

```python
from lab import matrix_demo
print(matrix_demo())
```

Lõi attention viết từ phép cơ bản trong [lab.py](lab.py):

```python
import torch
from lab import attention
Q=torch.tensor([[[1.,0.],[0.,1.]]])
K=torch.tensor([[[1.,0.],[0.,1.],[1.,1.]]])
V=torch.tensor([[[1.,0.],[0.,2.],[3.,1.]]])
Z,A=attention(Q,K,V)
print(Z.shape,A.shape)  # (1,2,2), (1,2,3)
```

## 5. Vì sao chia √d_k? Nó không phải cosine

Theo giả định minh họa: các thành phần q, k độc lập, mean0, variance1 thì dot là tổng d_k tích, có variance d_k. Chia √d_k đưa variance của score về1 theo giả định này. Score quá chênh có thể làm softmax gần one-hot và đạo hàm theo score rất nhỏ. [D2L: Attention scoring](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)

> [!WARNING]
> **Bản chất toán học: Tại sao phải chia cho $\sqrt{d_k}$ (Scaled Factor)?**
> Giả sử các phần tử của $q$ và $k$ là các biến ngẫu nhiên độc lập có giá trị trung bình bằng 0 và phương sai bằng 1: $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$.
> Tích vô hướng là tổng của $d_k$ số hạng:
> $$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$
> Phương sai của tích hai biến độc lập là $\operatorname{Var}(q_i k_i) = 1$. Vì các số hạng độc lập, phương sai của tổng bằng tổng các phương sai:
> $$\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \implies \text{Độ lệch chuẩn } \sigma = \sqrt{d_k}$$
> - **Khi $d_k$ lớn (ví dụ $d_k = 64$ hoặc $128$):** Độ lớn của tích vô hướng sẽ dao động rất mạnh, dễ đạt tới các giá trị lớn như $+30$ hoặc $-30$.
> - **Hậu quả tai hại đối với Softmax:** Khi nạp các số quá lớn vào hàm Softmax, giá trị lớn nhất sẽ chiếm trọn xác suất $\approx 1.0$, trong khi toàn bộ các giá trị còn lại bị đè bẹp về $0$ (phân phối biến thành one-hot sắc nhọn). Tại vùng bão hòa này, **đạo hàm của Softmax tiệm cận 0 tuyệt đối $\implies$ Mạng nơ-ron bị tê liệt vì biến mất gradient (Vanishing Gradient)**!
> - **Cách giải quyết:** Chia cho $\sqrt{d_k}$ đưa phương sai của điểm số trở về đúng $1.0$, giữ cho phân phối Softmax ở vùng mượt mà, giúp gradient chảy thông suốt.

Đây là trực giác cho lựa chọn scaling, không phải chứng minh Q, K sau huấn luyện luôn độc lập hoặc có variance1. Trong self-attention, query/key cùng nguồn và score đường chéo có thể có đặc tính khác. Scaling không bảo đảm không bao giờ saturation.

Chia theo **chiều key/query mỗi head**, không theo số token, S, L hoặc số class. Ví dụ q=k=[1,1,1,1], dot=4, d_k=4, scaled score=2. Cosine của hai vector này là1: cosine chia thêm cho cả hai norm, nên khác scaled dot-product. Norm Q, K vẫn ảnh hưởng score của attention đang học.

> [!NOTE]
> **Phân biệt Scaled Dot-Product vs Cosine Similarity:**
> - **Cosine Similarity:** $\frac{q \cdot k}{\|q\|_2 \|k\|_2}$ chia cho độ dài của cả hai vector, triệt tiêu hoàn toàn thông tin về độ lớn (norm).
> - **Scaled Dot-Product:** $\frac{q \cdot k}{\sqrt{d_k}}$ chỉ chia cho một hằng số cố định $\sqrt{d_k}$. Nếu một từ khóa rất quan trọng được mô hình học để có vector norm lớn, điểm số chú ý của nó vẫn sẽ được nâng cao tương ứng.

### Softmax ổn định và độ sắc của phân phối

Softmax(s)=exp(s−max(s))/Σexp(s−max(s)), cùng kết quả toán học nhưng tránh exp của số dương quá lớn. Cộng cùng một hằng số vào cả hàng không đổi weight. Nhân score với một hằng số thì thường đổi weight.

Với nhiệt độτ>0, softmax(s/τ):τ nhỏ hơn làm phân phối sắc hơn khi score chênh, τ lớn hơn làm phẳng hơn. Với score[0, ln2, ln4], τ=.5 cho[1,4,16]/21; τ=1 cho[1,2,4]/7; τ=2 cho trọng số tỷ lệ[1, √2,2]. Nhiệt độ là cách phân tích ở đây; lab train dùng scale chuẩn1/√d_k, không thêm một temperature được tune.

> [!TIP]
> **Ý nghĩa của Nhiệt độ ($\tau$ — Temperature) trong Softmax:**
> - $\tau \to 0$ (Nhiệt độ lạnh): Phân phối trở nên cực kỳ sắc nhọn, tiệm cận phép chọn cứng $\operatorname{argmax}$ (chỉ dồn toàn bộ $100\%$ sự chú ý vào duy nhất 1 key có score cao nhất).
> - $\tau \to \infty$ (Nhiệt độ nóng): Phân phối trở nên phẳng lì, tiệm cận phân phối đều $1/S$ (chú ý cào bằng vào tất cả các key).
> - Phép chia $\sqrt{d_k}$ trong attention thực chất đóng vai trò tương tự như việc đặt một mức nhiệt độ cố định $\tau = \sqrt{d_k}$ để giữ cho sự chú ý ở trạng thái cân bằng lý tưởng.

![Scaling và temperature](figures/03-softmax.png)

Trước dropout, weight tạo một phân phối trên các **key được phép** và z là một tổ hợp lồi của V. Vì V có thể chứa số âm nên z không cần dương. Sau dropout trên A hoặc projection W_O, đừng mặc định các tính chất “tổng weight1” hoặc “output ở trong convex hull value gốc” vẫn giữ nguyên.

## 6. Q, K, V từ đâu đến? Projection và tham số học được

Các số Q, K, V trong phần 3–4 được đặt tay để thấy phép tính. Với representation X theo quy ước hàng:

\[
Q=XW_Q, \quad K=XW_K, \quad V=XW_V.
\]

W_Q, W_K, W_V là parameter học được, có thể khác nhau. Dùng chung X không có nghĩa Q=K=V. Trong PyTorch, Linear lưu weight theo(out, in), nên công thức forward tương ứng `X @ linear.weight.T + bias`.

> [!NOTE]
> **Tại sao cần 3 ma trận chiếu độc lập $W_Q, W_K, W_V$?**
> Nếu không có các ma trận chiếu này mà dùng trực tiếp $Q = K = V = X$:
> - Tích $X X^\top$ sẽ luôn có các phần tử trên đường chéo chính là lớn nhất (vì $x_i \cdot x_i = \|x_i\|^2 > 0$). Kết quả là mỗi từ sẽ luôn luôn chú ý mạnh nhất vào chính nó!
> - Việc nhân với 3 ma trận trọng số học được $W_Q, W_K, W_V$ cho phép mô hình linh hoạt "nhìn nhận" cùng một từ dưới 3 lăng kính hoàn toàn khác nhau:
>   - $W_Q$ học cách đặt câu hỏi: *"Từ này đang cần tìm kiếm thông tin gì?"*.
>   - $W_K$ học cách tạo nhãn: *"Từ này có thể trả lời cho loại câu hỏi nào?"*.
>   - $W_V$ học cách đóng gói thông tin: *"Nếu được chọn, từ này sẽ đóng góp nội dung gì?"*.

Lõi `softmax(QKᵀ/√d_k)V` không có một bảng parameter riêng nếu đã cho Q, K, V; khả năng học đến từ các projection/embedding/head và những phần khác của model. `F.scaled_dot_product_attention` không tự tạo projection; `nn.MultiheadAttention` là module có projection và output projection.

Embedding lookup ở bài 16 lấy một hàng theo ID; attention trộn nhiều value theo query và memory. Attention không thay tokenizer, vocabulary hoặc biến raw ID thành ý nghĩa số học. Representation trước/sau attention cũng không tự là output xác suất class; classification cần head/loss riêng.

## 7. Attention học bằng gradient — một phép tính không bị giấu

Ở ví dụ phần 3, đặt loss kiểm toán **L=z₁=13/7**, chỉ lấy tọa độ thứ nhất output. Đây là loss tuyến tính để thấy đạo hàm, không phải objective phân loại của thí nghiệm cuối.

Gọi g=∂L/∂z=[1,0]. Vì z=Σαⱼvⱼ:

- ∂L/∂vⱼ=αⱼg.
- ∂L/∂αⱼ=g·vⱼ nên gradient theo A là[1,0,3].
- Softmax khiến tăng một score ảnh hưởng **cả hàng**, không chỉ một weight.

Đạo hàm theo score là `g_s = α * (g_α − Σ α*g_α)` theo phần tử. Σαg_α=13/7, nên:

```text
dL/dscores = [-6/49, -26/49, 32/49]
```

Tổng gradient score bằng0, phù hợp với việc cộng một hằng số vào hàng score không đổi softmax. Do score=qk/√1:

```text
dQ = 38/49 ≈ 0.775510204
dK = ln2 × [-6/49, -26/49, 32/49]ᵀ
dV = [[1/7,0],[2/7,0],[4/7,0]]
```

Mode `hand` đối chiếu các đạo hàm với autograd float64. [Phụ lục toán](deep-dive.md) suy Jacobian softmax và gradient ma trận đầy đủ, để bạn tự kiểm implementation chứ không chỉ học thuộc công thức forward.

> [!NOTE]
> **Giải thích tính chất Tổng Gradient Score luôn bằng 0:**
> Tổng các phần tử của vector gradient theo score:
> $$\sum_j \frac{\partial L}{\partial s_j} = -\frac{6}{49} - \frac{26}{49} + \frac{32}{49} = \mathbf{0}$$
> Đây là một tính chất toán học cơ bản và đẹp đẽ của hàm Softmax: Vì việc cộng thêm một hằng số $C$ vào toàn bộ các phần tử trong một hàng score không hề làm thay đổi phân phối xác suất đầu ra ($\operatorname{softmax}(s + C) = \operatorname{softmax}(s)$), nên hướng thay đổi đồng đều dọc theo vector $\mathbf{1}$ có đạo hàm bằng đúng 0!

![Gradient score và giới hạn diễn giải weight](figures/06-gradient-interpretation.png)

Không có nhãn attention trực tiếp trong thí nghiệm cuối: classification loss truyền gradient qua head→Z→V/A→Q/K→parameter. V hoặc score khác nhau đều có thể ảnh hưởng loss; weight cao không tự là một giải thích nhân quả đầy đủ.

## 8. Mask: chặn đường đọc trước softmax

Mask không phải cách gán label, nó quy định cặp(query, key) nào được phép nối. Để chặn một ô, đặt score ô đó thành−∞ trước softmax; exp(−∞)=0. Với hàng còn ít nhất một key hợp lệ, các key còn lại được chuẩn hóa lại.

Ví dụ chặn key3 ở phần 3:score=[0, ln2, −∞], A=[1/3,2/3,0], Z=[1/3,4/3]. Nếu chỉ tính A cũ rồi set α3=0, hai weight còn lại tổng3/7, chưa phải attention đã mask đúng. Nếu score cần chặn đang âm, nhân score với0 còn có thể làm ô đó hấp dẫn hơn! Dùng mask theo đúng API.

> [!WARNING]
> **Cạm bẫy tai hại: Tại sao phải gán $-\infty$ mà không được gán bằng $0$?**
> Một sai lầm rất phổ biến của người mới là nhân điểm số score cần chặn với $0$:
> - Giả sử score của các key là $[-5, -2, -10]$. Key thứ 3 bị cấm.
> - Nếu bạn nhân với 0: Score thứ 3 biến thành $0$. Trong khi đó $e^0 = 1 > e^{-2} \approx 0.135 > e^{-5} \approx 0.0067$! Ô bị cấm bỗng nhiên trở thành **ô có điểm số cao nhất và nhận được nhiều sự chú ý nhất**!
> - **Cách làm chuẩn xác duy nhất:** Cộng thêm $-\infty$ vào trước khi tính Softmax:
>   $$s_{\text{sau mask}} = s + M, \quad \text{với } M_{ij} = \begin{cases} 0 & \text{nếu được phép} \\ -\infty & \text{nếu bị cấm} \end{cases}$$
>   Khi đó $e^{-\infty} = 0$, trọng số chú ý tại vị trí bị cấm sẽ bằng $0$ tuyệt đối.

### Padding mask và causal mask

| Loại | Câu hỏi trả lời | Ví dụ |
|---|---|---|
| Key padding mask | Dòng memory nào chỉ là đệm? | Bỏ key có ID0, giữ cả value0 nếu đó là class thật |
| Causal mask | Query i có được đọc key tương lai j>i không? | Self-attention vuông:cho j≤i |
| Query validity/loss mask | Output query nào là mẫu/bước thật cần đo? | Bỏ loss ở query PAD, không chỉ bỏ PAD key |

Causal allowed mask cho3 vị trí:

```text
[[1,0,0],
 [1,1,0],
 [1,1,1]]
```

Giữ đường chéo không tự là leakage:trong next-token training, representation tại token i thường dùng để dự đoán token i+1; cần căn input/target lệch đúng. Nếu vô tình đưa token target vào input trước đó, causal mask không sửa được lỗi căn dữ liệu.

> [!NOTE]
> **Ý nghĩa của Causal Mask (Mặt nạ nhân quả) trong các mô hình GPT/LLM:**
> Khi huấn luyện mô hình ngôn ngữ tự hồi quy (Autoregressive Language Models như GPT-4, LLaMA), mô hình phải học cách dự đoán từ tiếp theo mà **không được nhìn trộm tương lai**:
> - Token thứ 1 ($i=1$) chỉ được phép nhìn chính nó (hàng 1: `[1, 0, 0]`).
> - Token thứ 2 ($i=2$) được phép nhìn token 1 và token 2 (hàng 2: `[1, 1, 0]`).
> - Token thứ 3 ($i=3$) được phép nhìn cả token 1, 2, và 3 (hàng 3: `[1, 1, 1]`).
> Toàn bộ nửa trên của ma trận vuông bị chặn bằng $-\infty$, tạo thành ma trận tam giác dưới đặc trưng.

![Causal, padding và trọng số sau mask](figures/04-masks.png)

Key padding mask chỉ cấm đọc **cột** PAD, nó không tự biến output ở **hàng query PAD** thành0. Mode `masks` cho một padded query vẫn có output norm≈1.300537 trong toy; cần bỏ các query đó khi pooling/loss, hoặc xử lý output theo mục tiêu. Không tạo một hàng bị chặn toàn bộ rồi mặc định mọi kernel xử lý giống nhau.

Lõi attention của lab chủ động báo lỗi khi một hàng không có key hợp lệ. Softmax của hàng toàn−∞ không tạo phân phối thông thường. Một API/backend có thể trả0 hoặc có hành vi khác; bài không dựa vào hành vi đó. Đừng thay−∞ bằng−1e9 rồi coi đã giải quyết:hàng bị chặn toàn bộ vẫn có thể thành phân phối đều sai ý định.

## 9. Self-attention và cross-attention

Hai tên này nói **nguồn Q và K/V**, không nói số head hoặc bắt buộc có causal mask.

| Cơ chế | Nguồn query | Nguồn key/value | Shape điển hình |
|---|---|---|---|
| Self-attention | Các representation của cùng chuỗi X | Cũng từ X, thường qua projection khác | L=S=T, output T vị trí |
| Cross-attention | Chuỗi/câu hỏi X_q | Memory/chuỗi khác X_m | L và S có thể khác nhau |

> [!NOTE]
> **Phân biệt trực quan Self-Attention vs Cross-Attention trong kiến trúc Transformer:**
> - **Self-Attention (Tự chú ý):** Cả $Q, K, V$ đều được sinh ra từ cùng một câu đầu vào $X$.
>   - *Ví dụ:* Trong câu *"Con mèo không thể trèo lên cây vì **nó** quá mệt"*, cơ chế Self-Attention giúp từ **"nó"** nhận diện và liên kết chặt chẽ với từ **"con mèo"** thay vì từ "cái cây".
> - **Cross-Attention (Chú ý chéo):** $Q$ đến từ một câu/luồng này, còn $K$ và $V$ đến từ một câu/luồng khác.
>   - *Ví dụ trong Dịch máy (Anh $\to$ Việt):* Câu tiếng Việt đang được sinh ra ở Decoder đóng vai trò là $Q$ (*"Tôi đang cần dịch từ tiếp theo là gì?"*). Toàn bộ câu tiếng Anh gốc ở Encoder đóng vai trò là $K$ và $V$ (*"Kho ngữ cảnh nguồn chứa những từ nào?"*). Cross-Attention đóng vai trò là cầu nối tra cứu giữa hai ngôn ngữ.

Ví dụ self:token trong một câu cập nhật theo các token được phép trong chính câu đó. Ví dụ cross:một query dịch vụ đọc một memory các trạng thái; hoặc decoder đọc output encoder. Cùng một cặp K/V có thể dùng cho nhiều query; với cross-attention, không cần ghép hai chuỗi bằng cách cộng tensor khác chiều.

Các quy tắc mask phải theo thông tin được phép có ở thời điểm dự đoán. Encoder self-attention có thể đọc cả đoạn input đã biết; decoder tự hồi quy cần hạn chế tương lai ở chuỗi đích. Không đánh đồng “self-attention” với “causal attention”.

## 10. Multi-head: nhiều phép đọc ở những không gian con

Một head tạo một A trên query/key đã projection. Multi-head dùng các projection khác nhau để mỗi head có thể tính score và trộn V khác nhau:

\[
H_r=\operatorname{Attention}(X_qW_Q^{(r)}, X_mW_K^{(r)}, X_mW_V^{(r)}),
\quad Y=\operatorname{Concat}(H_1, \ldots, H_h)W_O.
\]

Đây là kiến trúc cho phép các quan hệ khác nhau, không phải bảo đảm head1 học cú pháp, head2 học ngữ nghĩa. Head có thể dư thừa; thêm head không tự làm chất lượng tăng. Không lấy trung bình các head rồi mặc định tương đương concat→projection.

> [!NOTE]
> **Tại sao cần Multi-Head (Nhiều đầu chú ý)? — Ẩn dụ Hội đồng chuyên gia:**
> Nếu chỉ có 1 đầu chú ý duy nhất (Single-Head), phân phối Softmax chỉ có thể tập trung dồn sự chú ý vào một nhóm từ duy nhất. Nhưng một từ trong câu có thể đồng thời có nhiều mối quan hệ phức tạp:
> - **Head 1:** Tập trung vào quan hệ cú pháp chủ ngữ - vị ngữ (*"Ai làm hành động này?"*).
> - **Head 2:** Tập trung vào quan hệ đại từ thay thế (*"'Nó' trỏ về danh từ nào phía trước?"*).
> - **Head 3:** Tập trung vào quan hệ thời gian, địa điểm (*"Hành động xảy ra ở đâu, khi nào?"*).
> Multi-Head Attention cho phép mô hình đồng thời "nhìn" bức tranh văn bản ở nhiều không gian con khác nhau. Sau đó, kết quả của tất cả các head được ghép nối lại (Concat) và chiếu qua ma trận $W_O$ để tổng hợp thông tin.

### Theo shape từng bước: D=4, h=2

Trong cấu hình thông dụng d_k=d_v=D/h=2, B=2, L=S=3:

| Bước | Shape |
|---|---|
| Input X | (2,3,4) |
| Projection Q, K, V toàn chiều D | mỗi tensor(2,3,4) |
| Reshape B, T, h, d rồi transpose | (2,2,3,2) |
| Score/weight từng head | (2,2,3,3) |
| Mỗi head sau nhân V | (2,2,3,2) |
| Đổi về B, T, h, d rồi concat | (2,3,4) |
| Output projection | (2,3,4) |

Trong `B, h, L, d_k`, softmax vẫn trên **S ở trục cuối của score**, không qua head hoặc batch. Cần transpose trước/sau reshape đúng ý nghĩa; số phần tử khớp chưa chứng minh các head đúng.

> [!TIP]
> **Bí quyết phân chia không gian: Tại sao Multi-Head KHÔNG LÀM TĂNG số tham số?**
> Nhiều người lầm tưởng $h$ head sẽ làm tăng khối lượng tính toán và số tham số lên $h$ lần. Nhưng trong kiến trúc chuẩn của Transformer:
> - Thay vì giữ nguyên kích thước $D$ cho từng head, người ta chia nhỏ số chiều của mỗi head thành:
>   $$d_k = d_v = \frac{D}{h}$$
> - Ví dụ: Với $D = 512$ và $h = 8$ head $\implies$ mỗi head chỉ xử lý không gian $d_k = 64$ chiều.
> - Tổng kích thước các ma trận chiếu: $h \times (D \times \frac{D}{h}) = D \times D = D^2$.
> Số lượng tham số và tổng khối lượng tính toán FLOPs của Multi-Head Attention hoàn toàn tương đương với một Single-Head Attention kích thước đầy đủ $D$, nhưng năng lực biểu diễn và sự đa dạng thông tin tăng lên vượt trội!

![Chia head, tính attention, ghép và projection](figures/05-multihead.png)

Module PyTorch D=4 có Q/K/V projection tổng3D²+3D và output D²+D: **4D²+4D=80 parameter** khi dùng bias mặc định. Ở cấu hình giữ tổng D, mỗi head dùng D/h, không nhân số parameter toàn bộ thêm h lần. `embed_dim` phải chia hết cho `num_heads` trong module này. [MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)

Mode `multihead` lấy weight/bias của nn.MultiheadAttention, tự chạy projection→split→attention→concat→output Linear và đối chiếu API. Output sai khác max≈1.11×10⁻¹⁶ trên CPU float64. Cả weight theo head cũng được đối chiếu; xem[phụ lục API](deep-dive.md) để đọc công thức bằng code.

## 11. Attention không tự có vị trí hoặc thành một Transformer

Với self-attention không position, không mask cố định, hoán vị đồng thời hàng input sẽ hoán vị tương ứng hàng output: **permutation equivariance**. Nó không phân biệt thứ tự như RNN chỉ nhờ matmul. Nếu sau đó mean-pool output, representation toàn chuỗi sẽ không đổi theo hoán vị.

> [!WARNING]
> **Điểm yếu chí mạng của Attention thuần túy: Không có khái niệm về Thứ tự (Order-agnostic)!**
> Trong phép tính $S = Q K^\top / \sqrt{d_k}$, mỗi cặp từ được so khớp độc lập bằng tích vô hướng.
> Nếu bạn xáo trộn ngẫu nhiên vị trí các từ trong câu (ví dụ biến *"Tôi yêu bạn"* thành *"Bạn yêu tôi"*), các vector đầu ra vẫn y hệt như cũ, chỉ bị tráo đổi vị trí tương ứng (Permutation Equivariance). Khác với RNN có vòng lặp thời gian, **Attention thuần túy hoàn toàn không biết từ nào đứng trước, từ nào đứng sau**!
> Đó là lý do tại sao Transformer bắt buộc phải cộng thêm **Mã hóa Vị trí (Positional Encoding)** vào vector embedding trước khi đưa vào các tầng Attention (sẽ học chi tiết ở Bài 18).

Với cross-attention không position, giữ Q và hoán vị đồng thời K, V theo cùng thứ tự sẽ giữ output; đó là điều phù hợp nếu memory là tập các dòng không có thứ tự. Hoán vị K mà không hoán vị V làm hỏng cặp key/value, có thể đổi ý nghĩa dù shape đúng.

Các kết luận trên cần giữ điều kiện:causal mask cố định tạo cấu trúc hướng, position encoding thêm thông tin vị trí. Nếu có mask riêng, muốn suy equivariance phải biến đổi mask phù hợp. Mode `invariance` kiểm trường hợp không position, không mask; không tuyên bố mọi Transformer bất biến thứ tự.

Transformer còn kết nối attention với residual, normalization, FFN, position và thiết kế encoder/decoder/loss. Bài 18 sẽ ghép những phần này. Bài này chưa train language model, LLM, KV cache hoặc flash kernel trên GPU.

## 12. Hai API PyTorch: mask polarity và dropout

Đừng chuyển nguyên boolean mask giữa các API chỉ vì đều gọi attention:

| Giao diện | Bool True | Float mask |
|---|---|---|
| `attention` của lab | Được đọc | Lab không nhận float mask |
| `F.scaled_dot_product_attention` | Được đọc | Cộng trực tiếp vào score |
| `nn.MultiheadAttention.attn_mask` | Bị chặn | Cộng trực tiếp vào score |
| `nn.MultiheadAttention.key_padding_mask` | Key bị bỏ qua | Mask cộng theo key |

> [!CAUTION]
> **Cạm bẫy đảo cực Mask (Mask Polarity Trap) trong PyTorch:**
> Đây là một trong những lỗi gây đau đầu nhất cho các kỹ sư AI khi làm việc với PyTorch:
> - Trong hàm cấp thấp `F.scaled_dot_product_attention`: Mask kiểu boolean quy ước `True` là **ĐƯỢC PHÉP ĐỌC**, `False` là **BỊ CHẶN**.
> - Nhưng trong module cấp cao `nn.MultiheadAttention`: Tham số `key_padding_mask` quy ước ngược lại $180^\circ$: `True` là **BỊ BỎ QUA (BỊ CHẶN)**, `False` là **ĐƯỢC PHÉP ĐỌC**!
> Nếu bạn chuyển nhầm boolean mask giữa hai API này mà không đảo dấu `~mask`, mô hình của bạn sẽ cấm đọc toàn bộ dữ liệu thật và chỉ chăm chăm đọc vào các ô đệm padding!

Module MHA thường nhận input B×T×D khi batch_first=True; functional SDPA thường dùng B×h×L×d_k và trả B×h×L×d_v. Chỉ SDPA không có projection output và không trả A trong return chuẩn của hàm. MHA có thể trả A; `average_attn_weights=False` để thấy B×h×L×S, thay vì mặc định đã trung bình head. [SDPA API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

SDPA áp dropout theo `dropout_p` truyền vào, kể cả module bọc đang eval. Khi evaluation, cần truyền0, hoặc trong forward dùng `self.p if self.training else 0.0`. Code đối chiếu của bài luôn dropout0. Hình weight tổng1 trong bài là trước dropout.

Mode `masks` kiểm SDPA bằng mask cho phép, so với MHA nhận mask chặn có polarity ngược. Kiểm causal square bằng cả explicit mask và is_causal=True. Non-square causal alignment, export, GQA, CUDA và mixed precision chưa được thực nghiệm; đừng tự suy mọi layout/combination được hỗ trợ như nhau.

## 13. Chi phí và cách đọc attention map

Cách triển khai explicit lưu score có B×h×L×S phần tử. Self-attention L=S=T làm phần này tăng theo T²; nếu T tăng gấp2 thì số ô score gấp4. Ví dụ B=1, h=8, T=4096, float32:chỉ một tensor score đã có8×4096²×4 byte=**512MiB**, chưa gồm A, gradient, QKV và phần khác. Đây là ước lượng cho **tensor explicit**, không phải số RAM bắt buộc của mọi kernel tối ưu.

> [!NOTE]
> **Độ phức tạp bậc hai $O(T^2)$ và cuộc cách mạng FlashAttention:**
> Trong Self-Attention thông thường, việc tính ma trận $Q K^\top$ đòi hỏi phải lưu một ma trận kích thước $T \times T$ vào bộ nhớ VRAM của GPU:
> - Chuỗi $T = 1.000$ token $\implies 1.000.000$ ô tính toán.
> - Chuỗi $T = 128.000$ token (ngữ cảnh dài trong LLM hiện đại) $\implies 16.384.000.000$ ô tính toán! Bộ nhớ VRAM sẽ bị tràn ngay lập tức (OOM).
> Để giải quyết nghẽn cổ chai này, các kỹ thuật tối ưu phần cứng hiện đại như **FlashAttention (Dao et al., 2022)** sử dụng thuật toán xếp khối (tiling) để tính toán Attention trực tiếp trên bộ nhớ SRAM siêu nhanh của GPU mà không cần tạo ra ma trận $T \times T$ khổng lồ trong bộ nhớ HBM.

QK và AV có chi phí tương ứng bậc B h L S d_k và B h L S d_v, ngoài projection. Bài train chỉ L=1, S≤5, nên không minh họa tốc độ/memory của long-context self-attention. Fused attention có thể tránh lưu đầy đủ một số tensor trung gian; code dễ đọc của bài không phải implementation tối ưu.

Attention map cho biết tỷ trọng trộn V trong một phép tính. Nó không tách được mọi tác động của projection, V, residual hoặc layer sau. Phản ví dụ cụ thể:ba value đều[1,2]; A=[1/3,1/3,1/3] và A′=[.99, .005, .005] đều cho z=[1,2]. Vì vậy chỉ nhìn weight sắc/nhạt chưa đủ suy đầu ra phải khác hoặc đã có giải thích nhân quả.

Mode `invariance` kiểm phản ví dụ đó. Trong task cuối, các dòng khác nhau còn có thể cùng value class; nên chọn đúng value và argmax attention trúng key là **hai metric khác nhau**, dù run hiện tại đều đạt100% trên tập giữ riêng.

> [!IMPORTANT]
> **Cảnh báo: Attention Weight KHÔNG ĐỒNG NGHĨA với Giải thích Nhân quả (Attention is NOT Explanation)!**
> Nhiều bài báo và ứng dụng thường vẽ biểu đồ nhiệt (Heatmap) của Attention Map rồi khẳng định: *"Từ nào có màu đậm nhất là nguyên nhân chính khiến mô hình đưa ra quyết định"*.
> Thực tế toán học đã chứng minh điều này không hoàn toàn đúng (Jain & Wallace, 2019):
> 1. Nếu các vector Value $V$ của các từ gần như giống hệt nhau, thì dù Attention dồn $99\%$ vào từ A hay cào bằng $33\%$ cho cả 3 từ, kết quả đầu ra $z$ vẫn y hệt nhau!
> 2. Các tầng tuyến tính phía sau ($W_O$, FFN) và kết nối tắt (Residual Connections) có thể khuếch đại hoặc triệt tiêu tín hiệu của một từ bất kể trọng số attention của nó lớn hay nhỏ.
> Hãy cẩn trọng khi sử dụng Attention Map để giải trình quyết định mô hình cho người dùng cuối.

## 14. Thực hành học cross-attention để truy xuất bộ nhớ

Mỗi sample có một query ID dịch vụ1–6 và2–5 dòng memory, key không trùng, value class0/1/2 gán ngẫu nhiên. Target là value ở dòng có key trùng query. Value cần trả đã nằm trong input memory **theo thiết kế bài truy xuất**, không phải dự báo một giá trị tương lai chưa có. Model cần học cách chọn dòng, hệ thống dict có thể giải chính xác.

Tạo192 memory mapping khác nhau, so theo canonical sorted(key, value) để không lặp group. Mỗi group tạo2 query khác nhau và xáo thứ tự dòng riêng. Split group128/32/32 → **256/64/64 sample**, giữ các query cùng memory một tập. Các ID dịch vụ đều là danh mục quen thuộc, đánh giá không phải unseen-ID generalization.

Model một head:

```text
Embedding ID (7×8, key0=PAD)
  query → Linear8→8 không bias → Q (B,1,8)
  keys  → Linear8→8 không bias → K (B,S,8)
Embedding value class (3×8)    → V (B,S,8)
  mask theo keys != 0 → Attention → Z (B,1,8)
  Linear8→3 → logits (B,3)
```

Tổng parameter=56+64+64+24+27=**235**. Không có W_V riêng trong model này vì vector V đã do embedding value học; không có W_O riêng trước head. Đây là thiết kế toy, cần phân biệt với MHA đầy đủ ở phần 10.

Padding theo keyID0; valueID0 là **class hợp lệ** nên không mask mọi value0. Train Adam lr=.01, batch32, seed2026, data_seed17,40 epoch=8×40=320 update, clip global grad norm1. Classification CE là loss duy nhất, không thêm loss buộc attention trúng key.

Chọn minimum validation CE, hòa giữ epoch sớm hơn.Test chỉ chấm sau lựa chọn; một probe test sau đó kiểm đường inference/forward nhất quán, không dùng để đổi model.

| Đại lượng | Kết quả thực chạy |
|---|---|
| Best epoch / validation CE | 40 /≈0.001039705 |
| Validation accuracy / argmax attention trúng query key | 100% /100% trên64 sample |
| Test CE / accuracy / key-match rate | ≈0.001068935 /100% /100% trên64 sample |
| Confusion test, hàng=true, cột=pred, value0/1/2 | `[[17,0,0], [0,20,0], [0,0,27]]` |
| Majority từ train:validation accuracy | 15/64=23.4375% |
| Dict lookup rule:validation accuracy | 100% |

Lớp không được ép cân bằng; không mặc định majority validation luôn≥1/3 khi nhãn tập giữ riêng khác train. Rule dict đã giải được bài, nên không coi attention là cần thiết cho tra cứu dịch vụ thực tế. Kết quả chỉ chứng minh cơ chế học được trong toy; chưa là benchmark Transformer hoặc năng lực ngôn ngữ.

Probe thật:query6, keys=[3,2,4,6,1], values=[0,2,0,1,0]. Weight≈[.00000836, .00000765, .01191081, .98360163, .00447158], trọng số lớn nhất ở key6, target/predict class1. Các ô khác vẫn có trọng số nhỏ, không phải equality lookup cứng.

> [!NOTE]
> **Phân tích kết quả thực tế của Probe:**
> Khi đưa vào Query ID 6 và danh sách Keys `[3, 2, 4, 6, 1]`:
> - Trọng số attention phân bổ: $\alpha \approx [0.0008\%, 0.0008\%, 1.19\%, \mathbf{98.36\%}, 0.45\%]$.
> - Mô hình tự động dồn hơn $98.36\%$ sự chú ý vào đúng vị trí thứ 4 (nơi có Key = 6), trích xuất chính xác Value tương ứng là Class 1 với xác suất tự tin $99.86\%$.
> Thí nghiệm này chứng minh cơ chế Cross-Attention đã tự học thành công khả năng tra cứu mềm tương tự như một cuốn sổ danh bạ thông minh.

![Đường học và attention thực sự được học](figures/07-training.png)

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import Config,train,predict_memory
with TemporaryDirectory() as directory:
    out=Path(directory)/'run'
    report=train(out,Config(epochs=2))  # Pipeline ngắn,không kỳ vọng bảng40 epoch
    print(report['parameters'])  # 235
    print(predict_memory(out/'best.pt',2,[1,2,3],[0,1,2]))
```

Output gồm data.json, config.json, metadata.json, metrics.csv, best.pt, report.json. Checkpoint lưu model/config, domain ID và epoch/data hash để inference; chưa có optimizer/RNG để resume như last.pt bài 12. Không có vocabulary ngôn ngữ hoặc tokenizer trong task này.

## 15. Lệnh chạy, tự đánh giá và bước tiếp theo

Đã kiểm Python 3.14.6, torch 2.15.0, NumPy 2.5.4, macOS arm64. Hand/mask/MHA checks dùng CPU float64; model train dùng CPU float32 một thread. Phụ thuộc ở [requirements.txt](requirements.txt). Từ thư mục bài 17:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode matrix
.venv/bin/python lab.py --mode masks
.venv/bin/python lab.py --mode multihead
.venv/bin/python lab.py --mode invariance
.venv/bin/python lab.py --mode all --out /tmp/lesson17-run-01
```

Windows dùng `.venv\Scripts\python` và output phù hợp.Chọn output mới; bỏ --out để tạo thư mục tạm, in đường dẫn rồi chép ra nơi lưu lâu dài nếu muốn giữ.`--mode train` chỉ chạy recall model.Không cần GPU, dataset hoặc pretrained model tải ngoài.Bản ghim cần wheel phù hợp; đối chiếu[cài PyTorch](https://pytorch.org/get-started/locally/)khi nền tảng khác.Seed không bảo đảm giống từng bit giữa phiên bản/backend.

Bảy hình đã có PNG/SVG, tái tạo bằng [make_figures.py](make_figures.py), Matplotlib 3.11.2 và run tạm:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Hoàn thành [14 bài tập](exercises.md), rồi đối chiếu[lời giải](solutions.md). Nếu chưa tự suy được A có L hàng, S cột hoặc vì sao mask trước softmax, hãy quay lại phần 3–4 và8 trước khi ghép Transformer. [Đào sâu toán và API](deep-dive.md) cung cấp phép suy gradient và code theo head. **Bài tiếp theo:** [Bài 18 — Transformer](../18-transformer/README.md), ghép attention với vị trí, residual, LayerNorm và FFN để xử lý/sinh chuỗi. [Nguồn và phạm vi đã đọc](../SOURCES.md).
