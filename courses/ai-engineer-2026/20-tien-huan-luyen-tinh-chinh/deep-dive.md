# Bài 20 — Đào sâu gradient, dữ liệu và thực nghiệm

[Bài học](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md) · [Code](lab.py)

## A. Prompt bị bỏ loss vẫn có thể nhận gradient

Nếu logits zₜ không được chấm thì ∂L/∂zₜ=0 trực tiếp. Nhưng hidden hₜ của prompt còn đi vào key/value mà answer query đọc; có đường hₜ→attention answer→logits answer→L. Vì vậy không suy ∂L/∂hₜ=0 từ loss mask tại t. Causal chỉ ngăn đọc tương lai, không ngăn answer đọc prompt quá khứ.

Ví dụ kiểm trên model random, không cần train xong:

```python
import torch
from core import ByteBPE, LanguageModel, Config, IGNORE
from lab import batch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(20)
tok = ByteBPE()
record = {'prompt':'ab ', 'answer':'X', 'text':'ab X'}
x, y = batch([record], tok, sft=True)
model = LanguageModel(tok.size, Config()).double()
captured = {}
def keep(module, args, output):
    output.retain_grad()
    captured['embedding'] = output
hook = model.embedding.register_forward_hook(keep)
logits = model(x)
logits.retain_grad()
loss = F.cross_entropy(logits.flatten(0,1), y.flatten(), ignore_index=IGNORE)
loss.backward()
hook.remove()
assert logits.grad[0,:3].eq(0).all()
assert captured['embedding'].grad[0,:3].abs().sum() > 0
print('Direct prompt-logit gradient = 0; prompt embedding vẫn nhận gradient')
```

Test lưu gradient **output embedding theo vị trí**, không chỉ xem gradient cả bảng embedding. Một token ID có thể xuất hiện nhiều lần; gradient hàng embedding cộng từ các lần xuất hiện, dễ gây nhầm đường nào tạo gradient.

## B. Gradient LoRA theo vector cột

Đặt c=α/r, u=Ax, y=Wx+cBu, g=∂L/∂y. Khi W frozen:

\[
\frac{\partial L}{\partial B}=cgu^\top, \qquad
\frac{\partial L}{\partial A}=cB^\top g x^\top, \qquad
\frac{\partial L}{\partial x}=W^\top g+cA^\top B^\top g.
\]

W không được optimizer cập nhật không có nghĩa nhánh Wx biến mất khỏi forward hoặc gradient theo x. Nếu adapter ở lớp trước cần gradient truyền qua W frozen ở lớp sau, vẫn cần giữ graph phù hợp. Freeze parameter khác với detach activation.

Với B=0, A≠0, ∂L/∂A=0 lúc đầu, nhưng ∂L/∂B có thể khác 0 vì u≠0. Sau khi B đổi, A mới nhận tín hiệu. Nếu A=B=0 thì cả hai gradient đều 0 với cấu trúc này; optimizer không tự thoát điểm khởi tạo đó chỉ nhờ một bước gradient thông thường. Đây là suy trực tiếp từ công thức, không là cam kết về mọi biến thể adapter.

Khi α/r cố định và weight dense tương thích, có thể viết W'=W+cBA cho inference của phép tuyến tính này. Merge vào weight không có nghĩa được áp cùng adapter lần nữa; làm vậy sẽ cộng update hai lần. Lab chỉ kiểm số forward/gradient, chưa có adapter injection, train/merge/save adapter trên Transformer.

## C. Token-weighted loss và gradient accumulation

Giả sử microbatch 1 có 2 answer token, loss sum=2; microbatch 2 có 6 token, loss sum=12. Mean gộp là 14/8=1.75. Mean của hai batch mean là(1+2)/2=1.5; đây là cách weighting khác. Nếu muốn gradient tương đương batch gộp có độ dài khác nhau, cần tích lũy loss sum chia đúng tổng số valid target, không mặc định mỗi microbatch mean có trọng số bằng nhau.

Lab chưa gradient accumulation; mỗi update dùng mean trên valid token của batch hiện tại. Các batch khác nhau có số token answer khác nhau vì AUTH/DATA dài 4 byte còn NET dài 3 byte trong tokenizer này, cộng EOS. Epoch train CE được log bằng tổng loss nhân token count chia tổng count; giá trị đó là metric trong quá trình weight cập nhật, không phải eval model cố định cuối epoch.

Trong SFT nhiều lượt hội thoại, cần chọn có chấm mọi assistant turn hay chỉ lượt cuối; user/system token thường là context theo mục tiêu đã chọn. Không tự tạo mask bằng tìm substring `assistant:` trong raw text vì người dùng có thể viết đúng chuỗi đó. Hãy dùng template và metadata vai trò để xác định span, kiểm sau tokenization/truncation. Khái niệm assistant-only trong [TRL](https://huggingface.co/docs/trl/sft_trainer) cần template hỗ trợ; bài không chạy TRL nên không cung cấp một config được hứa dùng được cho mọi model.

## D. Phân biệt task data, corpus pretrain và test contamination

Nếu một incident có record LM, record route và nhiều đoạn hội thoại hỗ trợ, hãy split theo incident trước khi tạo các bản biểu diễn đó. Nếu split từng record riêng, base có thể đã đọc incident test dù SFT split nhìn không trùng. Không có nhãn target trong pretraining cũng chưa đủ chứng minh test độc lập: input test hoặc các bản sao gần trùng có thể đã có trong corpus.

Lab grouping theo cặp service–operation là lựa chọn tự tạo để kiểm được; event và từ vựng vẫn dùng chung. Nó không kiểm unseen-event, không kiểm tiếng Việt tự do, nhiều lỗi đồng thời hoặc event thiếu/trái quy tắc. Khi mở rộng, cần bộ ngoài miền/ambiguous với nhãn phù hợp thay vì ép tất cả vào 3 lớp.

Tokenizer cũng là artifact đã học. Ở lab, fit merge trên LM train rồi freeze cho base và mọi SFT branch. Nếu định thay tokenizer cho pretrained model thật, phải xử lý mapping/embedding/head và kiểm lại; không đổi vocab size là xong. Các vấn đề đó nối trực tiếp bài 19.

## E. Freezing không giữ nguyên output

Với head-only, hidden cố định cho mỗi input khi backbone ở chế độ deterministic như lab, nhưng y=W_head h+b_head đổi. Một head vừa bị dạy sinh label ngắn có thể giảm xác suất token log cũ. Vì vậy backbone weight bằng nhau không đồng nghĩa output phân phối bằng nhau.

Lab có 10659 head parameter trên 33699 tổng; backbone 23040 được so bit-for-bit trước/sau stage. Không lấy requires_grad của model load về để suy stage trước đã train gì: state_dict lưu weight/buffer, không lưu quy trình chọn optimizer/freeze. Report ghi trainable count và changed list tại stage, cùng kiểm frozen_unchanged.

Đo frozen weight bằng code này trước khi bắt đầu training mới:

```python
from core import ByteBPE, LanguageModel, Config
from torch import optim
model = LanguageModel(ByteBPE().size, Config())
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith('head.'))
optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 259*33
assert len(optimizer.param_groups[0]['params']) == 2  # Weight và bias head.
```

Giữ cấu hình trong công thức để kiểm đúng số: với vocabulary 259 ở đoạn độc lập này, head count=8547; với vocab 323 của run chính là 10659. Giữ V trong công thức D×V+V để tránh bê số từ cấu hình khác.

## F. So sánh ngân sách và điểm bắt đầu

Mỗi stage 300 update, nhưng best base ở epoch 18 ứng với 216 update. Ta vẫn thực hiện đủ 300 update để tìm best theo validation, sau đó quay về checkpoint 216 update ấy. Chi phí tìm/select vẫn là 300, không được chỉ báo 216 như toàn chi phí pretrain đã dùng.

Full SFT và head-only khởi đầu cùng hash base; scratch khởi đầu cùng initial random hash trước pretraining. Mọi branch dùng seed shuffle 2026 và optimizer mới, nên cùng thứ tự SFT; khác trainable parameter và điểm khởi tạo. Full đã có thêm kiến thức/compute từ pretraining, không kết luận “cùng tổng tài nguyên” với scratch.

Một seed và một cấu hình cho bằng chứng cục bộ. Muốn nghiên cứu lợi thế pretraining chắc hơn cần nhiều seed, ngân sách tương xứng, task đa dạng, cách chọn hyperparameter được xác định trước và test độc lập. Không cần tăng test nhiều lần chỉ để tìm kết quả đẹp.

## G. Artifact, rollback và regression

Một gói inference cần model/config, tokenizer, template/task protocol, decoding policy và provenance (parent/stage/version). Lab lưu model/config/tokenizer hash; protocol prompt trong code cùng report, chưa có deployment registry. Đừng chỉ gửi file head hoặc adapter rồi quên base/tokenizer cần thiết.

Trước khi đưa thay đổi ra môi trường dùng thật, đo tác vụ mục tiêu cùng một suite khả năng cần giữ. Định nghĩa giới hạn regression theo sản phẩm; có thể cần dữ liệu trộn, giới hạn update hoặc phương án thích nghi khác, nhưng phải đo hiệu quả chứ không coi biện pháp nào chắc chắn giải quyết forgetting. Giữ base checkpoint cho rollback và so lại cùng template/tokenizer.

Tái kiểm run của mình:

```python
# Sau: python lab.py --mode train --out /tmp/lesson20-run-01
from pathlib import Path
import json
from core import ByteBPE
from lab import load_checkpoint, evaluate, task_metrics, model_hash
out = Path('/tmp/lesson20-run-01')
tok = ByteBPE.load(out/'tokenizer.json')
records = json.loads((out/'data.json').read_text())
report = json.loads((out/'report.json').read_text())
base, _ = load_checkpoint(out/'base.pt',tok)
full, cp = load_checkpoint(out/'full-sft.pt',tok)
assert cp['parent_hash'] == model_hash(base)
assert evaluate(full,records['test'],tok,True) == report['stages']['full-sft']['task_ce_test']
assert task_metrics(full,records['test'],tok) == report['stages']['full-sft']['task_test']
print(cp['stage'],cp['epoch'])
```

Đây là kiểm artifact đã chọn, không dùng kết quả test để chọn lại epoch. Checkpoint không đủ exact resume vì không có optimizer/RNG/trạng thái stage. Cách lưu resume có ở bài 12; không dùng fine-tuning như tên khác cho resume.
