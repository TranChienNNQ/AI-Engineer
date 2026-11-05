# Bài 18 — Đào sâu: gradient, tính nhân quả và đọc implementation

[Bài học](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md) · [Mã thực hành](lab.py)

Đọc sau phần block và shift của bài chính. Phần này suy luận từ công thức và kiểm bằng ví dụ tự tạo; không phải bản chép mô hình lớn hoặc benchmark hiệu năng.

## A. Gradient qua LayerNorm đi đâu?

Xét một token D chiều. Đặt c=x−μ, s=√(σ²+ε), x̂=c/s, y=γ⊙x̂+β. Gọi g=∂L/∂y và u=g⊙γ. Khi mean tính trên D feature:

\[
\frac{\partial L}{\partial x_i}
=\frac1s\left[u_i-\operatorname{mean}(u)-\hat x_i\operatorname{mean}(u\odot\hat x)\right].
\]

Hai số hạng trừ xuất hiện vì μ và σ² cũng phụ thuộc x; không thể coi chúng là hằng khi backward. Công thức vẫn dùng được với ε>0. Theo từng token, ∂L/∂γ=g⊙x̂ và ∂L/∂β=g; vì γ,β dùng chung cho các token, khi tính parameter gradient phải cộng đóng góp từ B,T.

Ví dụ x=[1,2,3,4], ε=0, γ=1, L=y₀. Khi đó g=[1,0,0,0], s=√5/2, x̂=[−3,−1,1,3]/√5:

\[
\nabla_x L=\frac1{\sqrt5}[0.6,-0.8,-0.2,0.4]
\approx[0.268328,-0.357771,-0.089443,0.178885].
\]

Dù loss chỉ đọc một feature output, các feature input còn lại vẫn nhận gradient qua thống kê. Nhưng gradient không lan sang token khác chỉ vì LN(D). Với ε=0 và phương sai khác 0, phép dịch đồng đều mọi feature không đổi output; tổng gradient bằng 0 phù hợp điều đó.

```python
import math
import torch
from torch import nn
x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64, requires_grad=True)
ln = nn.LayerNorm(4, eps=0.).double()  # Chỉ ví dụ có variance > 0.
ln(x)[0].backward()
expected = torch.tensor([.6, -.8, -.2, .4], dtype=torch.float64) / math.sqrt(5)
torch.testing.assert_close(x.grad, expected)
print(x.grad)
```

Đừng nhầm gradient công thức trên với BatchNorm hoặc LayerNorm([T,D]); chúng có các trục chuẩn hóa khác. Nếu normalize cả T và D trong một decoder, thống kê của prefix có thể phụ thuộc future token dù attention đã có causal mask.

## B. Residual và pre/post-norm không hoán đổi được

Bỏ chỉ số và xét một sublayer:

\[
J_{pre}=I+J_F(\operatorname{LN}(X))J_{LN}(X),
\qquad
J_{post}=J_{LN}(X+F(X))(I+J_F(X)).
\]

Đó là Jacobian của output theo input; backward dùng chuyển vị các Jacobian. Pre-norm giữ một hạng I nằm ngoài nhánh chuẩn hóa trong sublayer này. Post-norm đưa cả tổng qua LN. Điều đó giải thích khác biệt đường gradient, nhưng không chứng minh mọi eigenvalue hoặc mọi lớp đều ổn định trong mọi cấu hình.

Để so pre/post, cần cùng trọng số, cùng input, cùng mask và tắt dropout; không lấy hai module khởi tạo độc lập rồi quy toàn bộ khác biệt output cho vị trí norm. `block_demo()` chép trọng số đúng từng module trước khi so với PyTorch reference. Nó kiểm cả forward, input gradient và parameter gradient cho hai biến thể. Trong môi trường đã chạy, max absolute error forward bằng 0 cho cả hai; không hứa các backend khác luôn cho sai số đúng 0.

## C. Vì sao nhân quả được giữ khi chồng nhiều tầng?

Chứng minh bằng phụ thuộc dữ liệu:

1. Embedding+PE tại vị trí t chỉ phụ thuộc ID tại t và vị trí t.
2. LN(D), FFN và residual hoạt động theo từng token; nếu input tại t chỉ phụ thuộc prefix ≤t, output của chúng cũng vậy.
3. Causal self-attention query t chỉ đọc K,V ở j≤t; nếu các biểu diễn đó chỉ phụ thuộc prefix của chính chúng, toàn bộ output vẫn chỉ phụ thuộc ID≤t.
4. Lặp theo tầng, final LN và head theo token cũng giữ phụ thuộc prefix.

Vì nhãn tại t là token t+1, không có đường từ ID t+1 đến logits t. Nhãn có thể dùng trong loss bên ngoài forward, nhưng không đưa vào feature tại t. Một module trộn toàn chuỗi ngoài mask, như mean pooling của cả future rồi broadcast ngược về mọi token, sẽ phá lập luận này.

Kiểm trên mô hình nhiều tầng, gồm PE và final norm:

```python
import torch
from lab import Config, TinyTransformer, BOS, SEP
torch.manual_seed(18)
model = TinyTransformer(Config()).double().eval()
a = torch.tensor([[BOS, 4, 5, SEP, 4, 5]])
b = a.clone()
b[:, 4:] = torch.tensor([[7, 6]])
with torch.no_grad():
    torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4])
    short = model(a[:, :4])
    torch.testing.assert_close(short, model(a)[:, :4])
print('Prefix logits không đọc future token')
```

Phép kiểm short/full còn là điều kiện quan trọng khi đối chiếu teacher-forced prefix với bước generate. Nó không tự kiểm chất lượng dự đoán; model random cũng phải vượt qua nếu cấu trúc nhân quả đúng.

## D. Padding và số hạng loss là hai hệ thống riêng

Giả sử source dài 3 và 5 trong batch. Với task copy, input có độ dài thật 8 và 12; mẫu đầu được thêm 4 PAD query. `key_padding_mask` ngăn mọi query đọc các PAD key, nhưng các PAD query vẫn chạy LayerNorm, attention, residual và FFN. Lab bỏ nhãn ở đó bằng IGNORE, không chấm output đó.

Vì dùng right padding, các vị trí thật của mẫu đầu vẫn là 0..7, không bị dịch. Nếu chèn PAD bên trái, PE của token thay đổi và các hàng causal đầu có thể không còn key thật; code không hỗ trợ cách đó. Việc chặn PAD trong loss cũng không buộc PAD embedding mãi bằng 0 ở mọi thiết kế; ở đây `padding_idx=0` giữ hàng embedding đó khỏi gradient lookup, còn PE vẫn khác 0.

Ví dụ phép chia loss: mẫu A có tổng CE=2 trên 2 token, mẫu B tổng CE=12 trên 6 token. Mean theo token là 14/8=1.75. Trung bình hai mean là (1+2)/2=1.5, tương ứng cho mỗi mẫu trọng số bằng nhau, không phải metric token-weighted của lab.

`ignore_index` bỏ vị trí target khỏi loss, không loại một class khỏi denominator softmax. Dù PAD không bao giờ là nhãn được chấm, PAD vẫn là một trong 8 output logits và model học giảm xác suất của nó ở những vị trí có nhãn thật. Hàm generate dừng nếu gặp PAD/BOS/SEP bất thường để tính là lỗi, không tự sửa thành đáp án hợp lệ.

## E. Đọc `Block.forward` từng nhánh

Pre-norm trong lab:

```text
z = norm1(x)                       # [B,T,D]
a = attn(z,z,z, mask, padding)[0]   # [B,T,D], không lấy attention weight để cộng
x = x + a                         # residual vẫn là x chưa norm
f = ff(norm2(x))                   # [B,T,D] qua hidden [B,T,F]
y = x + f                         # chưa final norm tại đây
```

`need_weights=False` ở MHA vì task không cần xuất bảng trọng số. Điều đó không bỏ phép attention. Không dùng attention weight làm lời giải thích nhân quả cho dự đoán; hạn chế đã học ở bài 17 vẫn áp dụng.

`ModuleList([Block(...) for ...])` tạo module mới mỗi lần. Viết `[block] * N` sẽ dùng chung đúng một object N lần; đó là weight sharing, không phải N tầng có tham số độc lập. `TinyTransformer` tự tạo từng block, không dựa vào clone một block rồi ngầm giả định trọng số khác nhau ngay từ khởi tạo.

Output head không tied với embedding. Nếu chủ động tying, phải phân biệt số parameter object duy nhất với số lượt một weight được dùng; không cộng hai lần số tham số dùng chung. Lab giữ hai weight riêng để việc đếm rõ ràng.

## F. Encoder–decoder: những gì API thực sự làm

`seq2seq_demo()` dùng source ID `[A,B,EOS,PAD]` và `[C,EOS,PAD,PAD]`; target đầy đủ `[BOS,A,B,EOS]` và `[BOS,C,EOS,PAD]`. Sau shift:

```text
tgt input: [BOS,A,B]     labels: [A,B,EOS]
           [BOS,C,EOS]           [C,EOS,PAD]
```

Tgt input hàng thứ hai không chứa PAD; label cuối mới là PAD và bị bỏ loss. Cần phân biệt padding mask của **input** với loss mask của **labels**, vì sau shift chúng không cùng bố trí PAD.

Source embedding và target embedding đều cộng PE riêng với vị trí bắt đầu 0. `nn.Transformer` nhận các vector đó, trả `[2,3,8]` hidden; head 8→8 trong demo giữ shape số học giống nhau nhưng đổi ý nghĩa từ feature sang vocabulary logit. CE chấm 5 token; đã chạy ra 2.443548 với seed 18 và môi trường ghi trong bài. Đây là model chưa train, không phải kết quả chất lượng.

Source PAD được chặn ở encoder và cross-attention. Sửa feature tại source PAD thành số lớn phải không đổi logits; sửa target input tương lai phải không đổi logits prefix. Demo kiểm cả hai trước backward. Không tự đặt causal mask cho memory khi source đã có đầy đủ; cross score có T hàng và S cột, không phải mặc định tam giác T×T.

## G. Checkpoint và phép đo phải đi cùng cấu hình

Best checkpoint lưu config, vocabulary, data hash, epoch và `state_dict`. Buffer PE được lưu cùng model nên `load_state_dict` cũng phục hồi nó. `load_model` dựng đúng kiến trúc rồi tải weight, gọi eval và dùng CPU. Chỉ nạp checkpoint do mình tạo hoặc nguồn tin cậy; lab dùng `weights_only=True` và checkpoint chứa tensor cùng metadata cơ bản.

Kiểm lại bảng kết quả từ run của mình:

```python
# Chạy sau: python lab.py --mode train --out /tmp/lesson18-run-01
import json
from pathlib import Path
from lab import Config, evaluate, loader, load_model, greedy_metrics
run = Path('/tmp/lesson18-run-01')
model, checkpoint = load_model(run / 'best.pt')
records = json.loads((run / 'data.json').read_text())
report = json.loads((run / 'report.json').read_text())
metrics = evaluate(model, loader(records['test'], Config(**checkpoint['config'])))
metrics.update(greedy_metrics(model, records['test']))
assert metrics == report['test']
print(metrics)
```

Đây là phép tái kiểm checkpoint đã chọn, không dùng kết quả test để quay lại chọn epoch hoặc đổi cấu hình. Để so kiến trúc mới, phải thiết kế trước cách dùng validation và giữ một test phù hợp cho kết luận cuối; không coi một tập test đã xem nhiều lần là mãi “chưa được dùng”.

## H. Phạm vi học tiếp

Bạn đã có một causal Transformer nhỏ học end-to-end và một encoder–decoder forward/backward có kiểm mask. Chưa có tokenizer ngôn ngữ thật, pretraining, instruction tuning, beam search, KV cache, distributed training hoặc đánh giá an toàn LLM. Chúng là các lớp kiến thức tiếp theo của đề cương; không cần giả định chúng đã có chỉ vì class mang tên Transformer.
