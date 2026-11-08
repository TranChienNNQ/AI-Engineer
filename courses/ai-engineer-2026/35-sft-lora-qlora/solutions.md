# Bài 35 — Lời giải và hướng đối chiếu

[Bài chính](README.md) · [Bài tập](exercises.md)

Các đáp án thiết kế có thể khác nếu objective, trust boundary và giới hạn được nêu rõ. Số liệu dưới đây thuộc toy CPU experiment.

## 1. Ba trục quyết định

SFT là objective/dataset; LoRA là parameterization của update; quantization là representation của base. Ba ví dụ: full SFT FP32; SFT LoRA trên FP32 base; SFT LoRA trên frozen4bit base với compute floating point. Không lựa chọn SFT để “thay thế” LoRA.

Rule là baseline mạnh khi mapping đơn giản/ổn định; prompt khi chưa đủ dữ liệu hoặc chỉ cần ít thay đổi format; retrieval/tool khi cần fact hiện hành/provenance/authorization. Fine-tuning cần chứng minh cải thiện task/quality/cost theo scope, không tự thay backend policy.

## 2. Dữ liệu và leakage

Paraphrase của cùng ticket ở hai split có thể làm test quá giống train. Group theo ticket/source/incident/template family hoặc time tùy deployment; giữ annotation revision, extraction cutoff, label policy, tokenizer/template và consent/retention phù hợp dữ liệu. Không coi chép cùng nội dung qua ID mới là nguồn độc lập.

Fixture tách service–operation groups nhưng cùng6 event và deterministic mapping. Rule lookup giải100%. Chưa kiểm event mới, policy mới, câu mơ hồ, ngôn ngữ thật hoặc distribution shift.

## 3. Shift và mask tính tay

Input=[BOS,USER,p1,p2,ASSISTANT,NET]. Label trước mask=[USER,p1,p2,ASSISTANT,NET,EOS]; sau mask=[IGNORE,IGNORE,IGNORE,IGNORE,NET,EOS]. Prefix dài5, answer ở label index4. Mask5 vị trí đầu làm mất targetNET, chỉ cònEOS.

```python
from data import records,Tokenizer,example,IGNORE,EOS
r=records();tok=Tokenizer.fit(r['train']);x,y=example(r['train'][0],tok)
assert sum(v!=IGNORE for v in y)==2
assert y[-2:]==tok.encode(r['train'][0]['answer'])+[EOS]
assert all(v==IGNORE for v in y[:-2])
```

## 4. Gradient qua prompt và accumulation

Prompt là context cho answer attention nên embedding có thể nhận indirect gradient. Direct gradient theo ignored prompt logits0 chưa đồng nghĩa gradient mọi hidden state/embedding tại prompt0. Frozen embeddings không nhận parameter gradient nhưng context vẫn ảnh hưởng forward.

Global mean=(2+18)/(2+6)=2.5. Mean-of-means=(2/2+18/6)/2=2.0, objective khác. Code single-process, không dropout/batchnorm:

```python
from copy import deepcopy
import torch
from data import records,Tokenizer,batch
from model import TinyLM
from training import loss_sum
torch.manual_seed(35);rows=records()['train'][:2];rows[1]=dict(rows[1],answer='NET NET')
tok=Tokenizer.fit(records()['train']);a=TinyLM(tok.size);b=deepcopy(a)
x,y=batch(rows,tok);raw,n=loss_sum(a(x),y);(raw/n).backward()
for row in rows:
    xx,yy=batch([row],tok);r,_=loss_sum(b(xx),yy);(r/n).backward()
assert all(torch.allclose(p.grad,q.grad,atol=3e-6,rtol=3e-6) for p,q in zip(a.parameters(),b.parameters()))
```

## 5. Forward/backward LoRA

Ax5, W₀x[−1,5], scaleBAx[2,−1], y[1,4], g[0,2], L2. gradB=[[0],[20]], gradA=[[-.8,-.4,-1.2]]. W₀ frozen bỏ parameter gradient/update của W₀; input gradient có thể vẫn cần để học lớp trước.

```python
import torch
from demos import hand
h=hand();assert h['output']==[1.,4.] and h['loss']==2
torch.testing.assert_close(torch.tensor(h['grad_A']),torch.tensor([[-.8,-.4,-1.2]]))
assert h['grad_B']==[[0.],[20.]]
```

## 6. Rank và parameter budget

4096²=16.777.216 so8×8192=65.536, tỷ lệ256. Matrix2×3 full6 so adapter1×(2+3)=5. Với d×d,2rd<d² nghĩa r<d/2. Update low-rank không bắt toàn W₀+BA/network low-rank; activations/base/optimizer/storage cũng không co theo cùng ratio.

```python
from demos import hand
h=hand()['large_matrix']
assert h['full_params']//h['lora_params']==256
assert h['lora_params']==65536
assert 1*(2+3)==5 and 2*3==6
```

## 7. Initialization

A random/B0: initial update0, gradA0, gradB có thể khác0. A0/B random: update0, gradB0, gradA có thể khác0. Cả hai0 thì cả hai gradient có thể0. Alpha/r và LR không tương đương trong mọi optimizer, nhất là moments/clipping/weight decay hoặc khi đổi factor initialization.

```python
import torch
from torch import nn
from model import LoRALinear
m=LoRALinear(nn.Linear(3,2,bias=False),rank=1,alpha=2)
with torch.no_grad():m.A.zero_()
m(torch.tensor([[2.,1.,3.]])).square().sum().backward()
assert m.A.grad.eq(0).all() and m.B.grad.eq(0).all()
```

## 8. NF4 và nibble packing

Byte=0 | (2<<4)=32=0x20. Low nibble0, high2. Với scale1,−.5 gần NF4 level≈−.525073 hơn level≈−.394917, code2. NF4 mức không đều, không phải integer×một step đều. Zero block dequantize0; padding chỉ phục vụ block/pack và bị cắt khi khôi phục shape.

```python
import torch
from quantization import pack,unpack,quantize,dequantize
c=torch.tensor([0,2],dtype=torch.uint8)
assert pack(c).tolist()==[32] and torch.equal(unpack(pack(c)),c)
w=torch.zeros(3,5);p,s=quantize(w,8)
assert torch.equal(dequantize(p,s,w.shape,8),w)
assert p.numel()==8 and s.numel()==2
```

## 9. Frozen quantized base và autograd

No_grad có thể cắt gradient theo input nên adapter trước không được học. Base codes/scales frozen không cần gradient qua argmin/rounding; matmul theo x vẫn phải ở graph. B0 giữ W_hat, không original W₀.

```python
import torch
from torch import nn
from quantization import NF4Linear
q=NF4Linear(nn.Linear(3,2,bias=False),8)
x=torch.tensor([[1.,2.,3.]],requires_grad=True);q(x).sum().backward()
assert x.grad is not None and x.grad.abs().sum()>0
assert not list(q.parameters())
```

## 10. Memory theo scope

BF16 weights14 tỷ byte≈13.04 GiB;4bit3.5 tỷ≈3.26 GiB. Scale overhead.5 bit/weight, nested total≈4.126953125 bit/weight theo giả định64/256 của bài. Còn scales/metadata, unquantized layers, adapters, gradients, optimizer/master states, activations, workspace/dequant temporary và allocator.

LoRA giữ base rồi thêm adapters nên resident weight state tăng. Nhưng base không có optimizer moments/parameter grads trong setup đúng; chỉ adapter có phần trainable state ấy. Đó là scope khác với static state bytes, file size hoặc peak VRAM. Không suy OOM fit từ weight payload.

## 11. Target modules và kiểm trainables

q/v mỗi4×(24+24)=192;up/down mỗi4×(24+48)=288;head4×(24+30)=216; tổng2×192+2×288+216=1176. k/o không có adapter. TinyLM head untied embedding; architecture khác có target/bias/tied weights khác.

Kiểm tên/shape/trainable count, optimizer groups, Bgrad ban đầu, Agrad sau Bđổi, frozen tensor hash và tiny-batch loss/generation. Chỉ in “LoRA enabled” chưa kiểm được injection đúng. Layer name không tồn tại phải fail rõ, không chạy tiếp như thành công.

## 12. Đọc kết quả thí nghiệm

Không thể kết luận QLoRA hơn full: CPU quantizer chưa bitsandbytes QLoRA, một seed, LR khác, task đơn giản và exact metric bão hòa. Rule cũng24/24. NF4 validationCE thấp nhất có thể do trajectory/setup, không là causal proof quantization giúp quality.

Retention giảm rõ: old-textCE base≈.933, full≈1.412, LoRA≈9.269, NF4adapter≈9.948. Frozen base chỉ bảo toàn stored weights, không bảo toàn effective function khi adapter bật. Cần dataset/hyperparameter/seeds rộng hơn, matched budgets và task/retention/latency eval để quyết định.

## 13. Merge và re-quantization

LoRA FP32 dùng W₀+sBA. Quant branch dùng W_hat+sBA. Merge nhầm W₀ thay W_hat thay baseline term, nên không còn cùng model. Quantize lại merged weight thêm approximation; cần task/retention/generation/latency kiểm trên export/runtime mới.

Core merged() chỉ dense hóa layer có adapter; k/o quantized chưa adapter vẫnNF4Linear. Tolerance2e−5 cho logit parity không là chứng minh mọi sample/generation bất biến; gần argmax tie, sai số nhỏ vẫn có thể đổi token. Check token output và downstream metrics ở bundle thực.

## 14. Adapter checkpoint và nhầm base

Load base đúng khớp output; base tensor khác bị từ chối bởi metadata hash. Cần identity/tokenizer/template/config, targets/rank/alpha, quant scheme/codebook và eval lineage. Adapter-only inference checkpoint chưa resume đủ optimizer/RNG; hashes chưa xác thực actor.

```python
import torch
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from data import records,Tokenizer,data_hash
from model import TinyLM,adapt
from training import adapter_metadata,save_adapter,load_adapter
torch.manual_seed(35);r=records();tok=Tokenizer.fit(r['train']);base=TinyLM(tok.size);m=adapt(base)
with TemporaryDirectory() as tmp:
    path=Path(tmp)/'a.pt';save_adapter(path,m,adapter_metadata(base,tok,'lora',data_hash(r)))
    load_adapter(path,base,tok,'lora',data_hash(r))
    wrong=deepcopy(base)
    with torch.no_grad():wrong.embedding.weight[1,0]+=.01
    try:load_adapter(path,wrong,tok,'lora',data_hash(r))
    except ValueError:pass
    else:raise AssertionError('wrong base accepted')
```

## 15. Thư viện và hardware

Config object chỉ kiểm kwargs/schema. PEFT Linear CPU train/save/load/merge kiểm một đường library nhỏ. Load4bit trên hardware là bằng chứng khác, còn actual backward/optimizer/frozen-state/eval ở target GPU là bước tiếp theo. Cài được package chưa chứng minh kernels/backend/model tương thích.

Máy tác giả CUDA=False; preflight trả trạng thái ấy. Recipe không tự chuyển sang CPU rồi gọi là GPU success. Đường GPU chưa được chạy; chuẩn bị template/collate/config và syntax test chưa thay được integration run trên GPU. Không cần thuê tài nguyên chỉ để làm core của bài.

## 16. Thiết kế một vòng SFT có thể phát hành

Xác định target behavior, giữ rule/prompt/RAG baseline, dữ liệu được phép dùng và nhãn đủ policy/escalation. Split source/group/time, pin template/tokenizer/base/revision/quant, kiểm mask/EOS/truncation. Chọn rank/targets/compute/batch/sequence budget, checkpoint/validation trước test.

Failure cần chặn: mask chỉ cònEOS hoặc tất cảIGNORE; nhầm base/tokenizer/template; target injection sai; nonfinite gradients; data leakage; retention/quality suy giảm; export/requantization đổi hành vi. PAD=EOS phải mask theo độ dài/attention và kiểm EOS thật vẫn được supervise.

Đánh giá exact/task quality, format, abstention/security, retention, cost/latency và compatibility. Release cần metadata/provenance/approval và rollback target load được. Smoke2 update/8 row chỉ để tìm lỗi plumbing, không đủ chất lượng để phát hành. Chưa có pipeline GPU smoke pass trong bài này.
