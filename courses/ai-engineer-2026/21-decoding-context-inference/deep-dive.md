# Bài 21 — Đào sâu: phân phối, KV cache và kiểm inference

[Bài học](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## A. Phân phối sinh và likelihood gốc

Với tập candidate S, qᵢ=pᵢ/Σⱼ∈S pⱼ nếu i thuộc S, ngược lại bằng 0. Khi có temperature, dùng softmax(z/τ) trước khi chọn S. Một token có probability model 0.5 có thể có probability sampling 2/3 hoặc 1 sau lọc. Vì vậy cần ghi rõ sequence logprob đang dùng phân phối model gốc hay policy sau processors.

Ví dụ k=2 rồi p=0.6 chỉ giữ A, còn p=0.6 trực tiếp trên phân phối gốc giữ A và B. Cùng seed trên hai phân phối khác nhau cũng không bảo đảm cùng token. Khi kiểm cache, nên so logits trước khi so output: một argmax giống nhau có thể che sai lệch ở các score còn lại.

```python
import torch
from lab import distribution, choose
p = torch.tensor([.5,.25,.15,.1],dtype=torch.float64)
z = p.log()
for c in [0., 1000., -1000.]:
    torch.testing.assert_close(distribution(z+c),p)
a = torch.Generator().manual_seed(21)
b = torch.Generator().manual_seed(21)
seq_a = [choose(z,'sample',a,top_p=.7) for _ in range(20)]
seq_b = [choose(z,'sample',b,top_p=.7) for _ in range(20)]
assert seq_a == seq_b
assert set(seq_a) <= {0,1}
```

Đây là bất biến dịch logit và tái lập trong cùng môi trường. Không suy mọi giá trị cực lớn đều an toàn với mọi dtype; code từ chối NaN, +∞ hoặc all-blocked khi không còn phân phối hợp lệ.

## B. Vì sao cache K/V cũ vẫn đúng?

Trong causal block, hidden tại vị trí j chỉ phụ thuộc prefix đến j. Thêm token mới sau j không làm thay đổi hidden, K, V tại j nếu weight, vị trí và mask giữ đúng. Qua nhiều tầng, lập luận lặp theo layer; mỗi layer cần cache riêng.

Điều kiện cần giữ:

- Model ở eval, dropout và các nguồn ngẫu nhiên được xử lý đúng; weight không cập nhật giữa request.
- Prefix cũ không bị sửa; token mới tiếp nối đúng ID và position scheme.
- Cache chứa projection, shape, dtype và layer tương ứng với model hiện tại.
- Query mới chỉ đọc key hợp lệ; không nhìn future trong một chunk mới.

Nếu đổi system instruction ở đầu prompt, đổi adapter hoặc retokenize làm prefix ID thay đổi, cache cũ không còn mặc định hợp lệ. Một implementation khác có thể dùng lại phần prefix chung và tính lại suffix, nhưng lab chưa hỗ trợ truncate, rollback hay prefix sharing.

Nếu position ID của token decode bị reset về 0, embedding vị trí không còn bằng full forward. Nếu feed lại toàn prompt cùng cache đã chứa prompt, vừa nhân đôi K/V vừa sai vị trí. Những bug này vẫn có thể sinh chữ nhìn hợp lý nên phải kiểm tensor.

## C. Đọc `cached_forward` theo shape

Với past=S, new=L, D=32, H=4, dₕ=8:

```text
input mới                 [1,L]
embedding + position      [1,L,32]
LN₁ + QKV linear          [1,L,96]
Q mới                     [1,4,L,8]
K,V sau nối cache         [1,4,S+L,8]
allowed mask              [L,S+L]
attention output          [1,4,L,8]
ghép / W_O / residual     [1,L,32]
FFN / final LN / head     [1,L,V]
```

MHA của core lưu QKV weight trong tensor `[3D,D]`. `F.linear(...).chunk(3,-1)` tách output feature; chuyển `[B,L,H,dₕ]` sang `[B,H,L,dₕ]` cần transpose, không chỉ reshape. Cache nối theo length ở trục -2, không theo head hoặc feature.

SDPA mặc định scale theo √dₕ; không chia thêm lần nữa. Output projection, residual, LayerNorm và FFN vẫn phải đúng công thức của core; tính QKV đúng chưa đủ để Transformer output khớp.

```python
import torch
from core import LanguageModel, Config, BOS
from lab import cached_forward, allowed_mask

torch.manual_seed(21)
model = LanguageModel(259,Config()).double().eval()
ids = torch.tensor([[BOS,100,101,102,103,104]])
with torch.no_grad():
    full = model(ids)
    first, cache = cached_forward(model,ids[:,:3])
    tail, cache = cached_forward(model,ids[:,3:],cache)
    torch.testing.assert_close(torch.cat([first,tail],1),full,atol=1e-10,rtol=1e-10)
assert allowed_mask(3,2).tolist() == [[True,True,True,True,False],[True,True,True,True,True]]
assert cache['k'].shape == (1,4,6,8)
```

Lab dùng no_grad cho cache. [Autograd 2.15](https://docs.pytorch.org/docs/2.15/notes/autograd.html) phân biệt eval và grad modes: eval không tự ngăn ghi graph; no_grad không tự tắt dropout. Inference mode có thêm tối ưu nhưng tensor tạo trong đó có hạn chế khi dùng lại với autograd. Bài không dùng hoặc benchmark inference mode.

## D. Query một token và chunk nhiều token

Nếu L=1 và mọi key hợp lệ, query mới được đọc cả S+1 key. Một implementation chuyên biệt có thể không cần mask ở bước ấy. Nhưng với L>1, query đầu chunk không được đọc token cuối chunk; bỏ mask toàn bộ sẽ sai.

Lab luôn tạo offset mask cho mọi L, không dùng is_causal như cách đoán past length. Nó cũng không truyền đồng thời explicit mask và is_causal=True. Upper-left alignment ở SDPA không vuông được kiểm bằng Q=K=0, V=[1,2,3,4] trong bài chính. [SDPA 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).

Batch có padding hoặc past length khác nhau còn cần mask phù hợp theo từng hàng. Lab từ chối PAD và chỉ hỗ trợ batch 1 để phạm vi dễ kiểm. Không áp nguyên implementation này cho một serving batch dài ngắn khác nhau.

## E. Lifetime cache là một phần request state

Cache là dict K/V, length và owner ID; không lưu toàn prompt. Owner tag chặn dùng cache trên object model khác nhưng không phát hiện weight bị sửa tại chỗ. Người gọi phải giữ đúng model, prefix, vị trí và phiên bản trong lifetime request.

Trong beam search, mỗi prefix candidate có history riêng. Khi chọn beam con, cache phải đi theo parent tương ứng; khi reorder batch beam phải reorder K/V theo cùng chỉ số. Dùng một cache bị mutate chung cho mọi beam sẽ trộn history. Beam của lab chạy trên cây hữu hạn, chưa có neural beam cache.

Response cache cũng khác KV cache: nó có thể trả lại cả response đã tính cho một request tương thích, còn KV giúp tiếp tục prefix. Cache key cần xét model, tokenizer, template, config và dữ liệu liên quan. Câu trả lời phụ thuộc dữ liệu mới cần chính sách hết hạn phù hợp. Đây là suy luận thiết kế, chưa có implementation HTTP cache hoặc cache chia sẻ trong lab.

## F. Stop, streaming và tính đúng tác vụ

EOS là kết thúc được model dự đoán; budget hết là quyết định của ứng dụng. Model có thể EOS sớm hoặc không EOS, nên phải báo khác nhau. Stop string có thể nằm qua hai chunk stream; matcher phải giữ suffix đủ dài để phát hiện. Bài không triển khai matcher đó.

UTF-8 incremental decoder giữ byte chưa hoàn chỉnh. Đừng ghép các chunk đã bị replacement rồi mong khôi phục byte gốc. Output ID, raw bytes và chuỗi hiển thị là ba đối tượng khác nhau. Với JSON cần parse/schema sau khi có output đủ điều kiện; stream HTTP kết thúc không tự chứng minh tác vụ thành công.

Model random trong lab có thể sinh UTF-8 sai và được báo cờ. Model SFT bài 20 sinh NET ở cả bốn strategy trong một ví dụ nhỏ, không là cam kết mọi input đều đúng. Muốn chọn decoding cho sản phẩm cần đo accuracy, format, latency và chi phí theo ngân sách.

## G. Kiểm artifact và phạm vi

Sau lệnh có --out, report.json chứa kết quả các mode đã chạy. Không có weight mới vì bài không train. --run 20 cần tokenizer.json và full-sft.pt của bài 20; code kiểm hash trước khi dùng tokenizer với checkpoint.

```python
# Sau lệnh all trong bài chính.
from pathlib import Path
import json
run = Path('/tmp/lesson21-run-01')
report = json.loads((run/'report.json').read_text())
assert report['cache']['max_abs_error'] < 1e-10
assert report['cache']['kv_bytes'] == 4096
for result in report['inference']['strategies'].values():
    assert result['cached']['ids'] == result['uncached']['ids']
print(report['inference']['origin'])
```

Nếu bổ sung benchmark thật, cần ghi warmup, số lần lặp, thiết bị, dtype, concurrency, chiều dài input/output và boundary đo. GPU cần synchronization phù hợp để không nhầm thời gian enqueue trên CPU với thời gian hoàn thành tính toán. Bài hiện chỉ kiểm độ đúng cache và đếm vị trí tính lại, không báo tốc độ GPU hoặc server throughput.
