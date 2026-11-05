# Bài 18 — Lời giải và tiêu chí đối chiếu

[Bài học](README.md) · [Đề bài](exercises.md) · [Phụ lục](deep-dive.md)

Các số tính tay dưới đây là ví dụ tự tạo. Số thực nghiệm lấy từ run được ghi trong bài; thay môi trường/config thì dùng số liệu run của mình.

## 1. Shape

ID `[2,5]`; embedding `[2,5,12]`; Q/K/V mỗi tensor sau tách head `[2,3,5,4]`; score `[2,3,5,5]`; output mỗi head `[2,3,5,4]`; ghép và chiếu `[2,5,12]`; FFN hidden `[2,5,24]`; logits `[2,5,8]`. D=H×dₖ=3×4. Trong block này T không đổi; số token không bằng số feature hoặc số lớp dự đoán.

## 2. Residual

X+F(X)=[2.3,−0.5]. Scalar F(x)=2x → Y=3x=3, dY/dx=1+2=3. L=Y²/2 nên dL/dx=Y×3=9. Nếu chỉ tính qua nhánh F sẽ được 6 và bỏ mất đường identity. Hai nhánh gradient phải cộng như bài 10.

## 3. LayerNorm

μ=2.5, variance=(2.25+0.25+0.25+2.25)/4=1.25. Output=[−3,−1,1,3]/√5. Với x hằng và ε>0, x−μ=0, output=β=0. Chia D là định nghĩa phương sai dùng trong LN của lab/API, không phải bài toán ước lượng phương sai tổng thể từ mẫu cần correction=1. Dùng chia D−1 sẽ tạo output khác module tham chiếu.

## 4. Gradient LN

Từ công thức phụ lục, gradient=[0.6,−0.8,−0.2,0.4]/√5≈[0.268328,−0.357771,−0.089443,0.178885]. μ và σ² phụ thuộc mọi feature nên gradient lan qua các feature cùng token. LN(D) không trộn token nên không tự gây gradient sang token khác; attention hoặc tầng trộn token mới có thể tạo đường đó. Code ở phụ lục kiểm autograd. Tổng gradient bằng 0 trong ví dụ, phù hợp bất biến khi cộng cùng một hằng vào mọi feature.

## 5. FFN

u=[2,−1]: uW₁=[2,−3,−1]; ReLU=[2,0,0]; nhân W₂ được FFN=[2,0]; residual=[4,−1]. Cùng W₁,W₂ được áp dụng từng token. Nếu u đã chứa ngữ cảnh từ attention, FFN dùng ngữ cảnh ấy nhưng không đọc thêm trực tiếp các hàng lân cận.

## 6. Causal tính tay

A=[[1,0],[1/2,1/2]]. Attention output Z=[[1,3],[1.5,1.5]], khác trường hợp không mask ở hàng đầu. R=X+Z=[[2,6],[3.5,1.5]], norm thứ nhất vẫn U=[[-1,1],[1,−1]]. FFN và norm thứ hai do đó vẫn trả Y=[[-1,1],[1,−1]].

Đây là ví dụ output cuối có thể che khuất khác biệt trung gian. D=2 và LN khiến nhiều vector khác độ lớn có cùng output. Không kết luận mask không quan trọng từ một mẫu; cần inspect attention, kiểm nhiều input và kiểm phụ thuộc future token, không chỉ so hai output cuối tình cờ bằng nhau.

## 7. Nhánh bằng 0

Pre-norm block trả X qua hai residual. Post-norm block trả LN₂(LN₁(X)); không được ngầm thay bằng X hoặc luôn bằng LN₁(X), vì có ε,γ,β riêng. Final norm sau pre-norm stack không làm mọi cấu hình tương đương post-norm; khi nhánh khác 0, input từng nhánh và đường gradient đều khác.

## 8. PE và hoán vị

D=4: PE₀=[0,1,0,1], PE₁=[sin1,cos1,sin0.01,cos0.01]≈[0.841471,0.540302,0.010000,0.999950].

- Block không PE, không mask: f(PX)=Pf(X), output hoán vị theo input.
- Token hoán vị, PE giữ theo vị trí: input là PX+PE, thường khác P(X+PE); không còn yêu cầu output chỉ hoán vị.
- Cả token lẫn PE cùng hoán vị: input P(X+PE), lại có tính tương đương của block không mask.

Causal mask giữ cố định tạo cấu trúc j≤t; một hoán vị tùy ý không giữ cấu trúc đó. Không kéo kết luận không mask sang decoder causal. `position_demo` kiểm trường hợp không mask, sai số lần lượt khoảng 5.55e−17 và 1.53838 khi giữ PE theo vị trí trong run đã ghi; con số sau không phải metric chất lượng.

## 9. Shift

```text
full:   BOS A B C SEP A B C EOS
x:      BOS A B C SEP A B C
y:        · · · ·   A B C EOS
index:    0 1 2 3   4 5 6  7
```

Query index 4 tại SEP dự đoán A; có 4 nhãn được chấm. Causal blocked mask cho prefix 4×4 là `[[F,T,T,T],[F,F,T,T],[F,F,F,T],[F,F,F,F]]`. Đường chéo mở vì input t được phép dùng để dự đoán t+1. Nếu labels không dịch, đường chéo và residual đều có thể tiết lộ token cần đoán.

## 10. Loss

Mean CE=−(ln(1/2)+ln(1/4))/2=1.5ln2≈1.039721; không chia 5 vì ba vị trí IGNORE không thuộc mẫu số. Hai batch: tổng CE=14, tổng token=8 →1.75, không phải mean batch mean=1.5. `ignore_index` chỉ bỏ vị trí; PAD logit vẫn tham gia denominator ở những vị trí được chấm, cùng 7 class khác của lab.

## 11. Tham số

MHA=4D²+4D=624; FFN=2DF+F+D=612; hai LN=4D=48. Một block=1,284; hai block=2,568. Embedding=VD=96; final LN=2D=24; head=DV+V=104. Tổng **2,792**. Sinusoidal buffer không là parameter; T thay đổi số activation, không đổi số tham số cấu hình này. Learned absolute PE hữu hạn sẽ có tham số, nhưng đó là mô hình khác.

## 12. Kiểm nhân quả, padding và gradient

Code độc lập từ thư mục bài 18:

```python
import torch
from torch import nn
from lab import Config, TinyTransformer, Block, causal_blocked, BOS, PAD

torch.set_num_threads(1)
torch.manual_seed(18)
model = TinyTransformer(Config()).double().eval()
a = torch.tensor([[BOS, 4, 5, 6, 7]])
b = a.clone()
b[:, 3:] = torch.tensor([[5, 4]])
with torch.no_grad():
    torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])
    padded = torch.cat([a, torch.full((1, 3), PAD)], dim=1)
    torch.testing.assert_close(model(a), model(padded)[:, :a.shape[1]])

block = Block(8, 2, 16).double().eval()
x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
block(x, causal_blocked(5))[:, :3].square().sum().backward()
torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))

# Một LN có trục thời gian sẽ tạo đường phụ thuộc future không qua attention.
wrong_ln = nn.LayerNorm([5, 8]).double()
z = x.detach().clone()
z[:, 3:] += 100
assert not torch.allclose(wrong_ln(x.detach())[:, :3], wrong_ln(z)[:, :3])

small = TinyTransformer(Config(dim=12, heads=3, ff=24))
assert sum(p.numel() for p in small.parameters()) == 2792
print('PASS: causal, padding, future gradient, LN axis, parameter count')
```

Padded query vẫn có PE, residual/bias và quyền đọc các key thật nên có thể khác 0. Chỉ cần không làm ảnh hưởng token thật và không bị chấm loss. Với left padding, phép kiểm trên không còn đúng như hiện tại; lab từ chối cách đó.

## 13. Encoder–decoder

Memory `[2,4,8]`; cross Q `[2,2,3,4]`, K/V `[2,2,4,4]`; cross score `[2,2,3,4]`; tgt causal mask `[3,3]`; source padding mask `[2,4]`. Encoder không đọc PAD key chưa đủ: encoder output tại PAD query có thể khác 0, nên decoder vẫn phải chặn các memory key ấy. Trong task source đã cho đầy đủ, không cần tam giác causal trên source memory.

`nn.Transformer` nhận vector feature, không tự có embedding, PE hay vocabulary head của tác vụ. `seq2seq_demo()` đặt chúng bên ngoài, shift target rồi kiểm source PAD và target future trước backward. Mask bool True=chặn; không dùng polarity của functional SDPA.

## 14. Đánh giá

814/855≈95.2047% teacher-forced token accuracy; 94/128=73.4375% greedy exact. Exact yêu cầu cả thứ tự, token và EOS đúng; model phải tự dùng output trước đó, thay vì có prefix thật. Không lấy tích token accuracy để suy chính xác sequence accuracy vì lỗi không độc lập, độ dài khác nhau và điều kiện đánh giá khác.

Baseline quy tắc 100% cho biết copy là tác vụ đã có thuật toán chính xác; lab vẫn hữu ích để học forward/backward và generation, nhưng không chứng minh nên dùng neural model cho copy. Best checkpoint dùng **epoch 58**, chọn theo validation CE; epoch 60 là trạng thái cuối chứ không mặc định tốt nhất. Xem `metrics.csv` và epoch trong checkpoint để đối chiếu.

## 15. Sửa pipeline

1. Tạo full sequence có BOS/EOS và shift x/y một vị trí; kiểm vài mẫu bằng tay.
2. Causal mask chặn future, key padding mask chặn PAD, loss mask bỏ prompt/PAD theo mục tiêu học; không dùng một mask thay ba mask.
3. Tính CE raw logits, chia theo số nhãn thật; kiểm với ví dụ có kết quả ln2 và ln4.
4. Chọn checkpoint bằng validation; giữ test cho phép đánh giá cuối.
5. Viết generate tự nối token, dừng EOS/budget; báo riêng token accuracy teacher-forced và exact match tự sinh.
6. Kiểm sửa future không đổi prefix, thêm PAD không đổi token thật, gradient future bằng 0, checkpoint roundtrip. Sau đó mới tin đường cong loss.

Nếu đã dùng test để chọn epoch, tập đó đã tham gia phát triển; cần tập test mới phù hợp khi muốn công bố đánh giá độc lập. Không xóa lịch sử rồi coi split cũ là chưa được nhìn.

## 16. Thực nghiệm có kiểm soát

Một lời giải tốt có bảng cấu hình, dữ liệu giữ cố định, tiêu chí chọn best xác định trước, số seed và phân biệt metric. Không có một con số “đúng” bắt buộc cho cấu hình chưa chạy. Bảng tham chiếu baseline 2 tầng nằm ở bài chính; các biến thể phải đo thật.

Có thể chạy thử **validation-only** bằng đoạn độc lập dưới đây. Nó train 2 epoch để kiểm quy trình; đổi `epochs=60` khi làm thí nghiệm đầy đủ. Nó không truy cập nhãn hoặc metric test:

```python
import torch
from torch.nn import functional as F
from lab import Config, TinyTransformer, make_records, loader, evaluate, greedy_metrics, IGNORE

torch.set_num_threads(1)
cfg = Config(layers=1, epochs=2)
torch.manual_seed(cfg.seed)
records = make_records(cfg.data_seed)
model = TinyTransformer(cfg)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
batches = loader(records['train'], cfg, True)
validation = loader(records['validation'], cfg)
best_ce, best_state, best_epoch = float('inf'), None, None
for epoch in range(1, cfg.epochs + 1):
    model.train()
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=IGNORE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
    score = evaluate(model, validation)
    if score['ce'] < best_ce:
        best_ce = score['ce']
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_epoch = epoch
model.load_state_dict(best_state)
print({'parameters': sum(p.numel() for p in model.parameters()), 'epoch': best_epoch,
       'updates': cfg.epochs * len(batches), **evaluate(model, validation),
       **greedy_metrics(model, records['validation'])})
```

Với N=1, số tham số là 17,672−8,544=9,128. Số update vẫn 960 nếu giữ 60 epoch, batch size 32 và 512 mẫu. Một seed chỉ cho quan sát trong điều kiện đó; muốn kết luận vững hơn cần nhiều seed và kiểm soát ngân sách/tính toán. Không chọn kết luận dựa một run đẹp.

Để thử n=7–8 cần sinh source mới, giữ riêng ngoài miền, điều chỉnh `generate` vì nó đang giới hạn n≤6, bảo đảm `max_len` chứa prefix và toàn output, và báo metric riêng theo length. Input tối đa cho n=8 dài 18, nằm trong buffer 32 nhưng code guard vẫn phải được đổi rõ ràng. Nếu train thêm các mẫu dài này thì đó là thí nghiệm khác; không còn cùng câu hỏi tổng quát hóa ngoài miền. PE có thể tính không đồng nghĩa model đã học thao tác copy cho các độ dài đó.

## Gợi ý cho phần mở rộng

Thêm dropout phải xử lý riêng dropout attention weight, hidden FFN và output residual; so trong eval khi kiểm tương đương. Learned PE cần embedding vị trí có domain chỉ số đủ dài. Đối chiếu SDPA phải đổi polarity mask đúng, tách/ghép head đúng và giữ cùng weight. Thống kê lỗi EOS/ký hiệu/hết budget giúp chẩn đoán, nhưng không thay thế việc giữ metric exact đã định nghĩa trước. Các phần mở rộng là đề xuất tự làm, chưa được coi là implementation đã kiểm của bài.
