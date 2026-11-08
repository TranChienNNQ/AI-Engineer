# Lời giải bài 37

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

Mười snippet Python độc lập bên dưới chạy từ thư mục bài này với requirements đã cài. Các câu mở không có duy nhất một thiết kế; cần giải thích trade-off và cách kiểm chứng.

## 1. Token đầu tiên đến từ đâu?

Forward1 nhận4token prompt, tạo cache dài4 và dự đoán y1. Forward2 nhận y1, cache dài5, dự đoán y2. Forward3 nhận y2, cache dài6, dự đoán y3 rồi kết thúc. Cache dàiP+N−1=6, không cần forwardy3. KV là trạng thái attention, khác việc lưu câu trả lời hoàn chỉnh hoặc lưu memory hội thoại bằng văn bản.

## 2. Đếm công việc bằng tay

Full xử lý4+5+6=15vị trí; cache xử lý4+1+1=6. Causal pairs full10+15+21=46; cache10+5+6=21. Kernel đặc có thể tính cả ô mask; overhead, tensor layout và memory traffic không nằm trong hai phép đếm này.

```python
from arithmetic import work
r=work(4,3)
assert (r['uncached_positions'],r['cached_positions'])==(15,6)
assert (r['uncached_causal_pairs'],r['cached_causal_pairs'])==(46,21)
assert work(4,1)['cached_positions']==4
print(r)
```

## 3. Một query nhìn ba key

Softmax `[e,1,e]/(2e+1)`≈`[.422319,.155362,.422319]`; output4.844638. Tính Q/K/V của token mới, append K/V mới rồi query nhìn K/V toàn context hợp lệ. Q cũ không đủ vì output mới cần values cũ và similarity giữa query mới với keys cũ. Các layer đều cần cache riêng.

```python
import math
import numpy as np
scores=np.array([1.,0.,1.]);p=np.exp(scores-scores.max());p/=p.sum()
out=p@np.array([2.,4.,8.])
assert abs(out-(10*math.e+4)/(2*math.e+1))<1e-12
print(p.tolist(),out)
```

## 4. Bẫy causal non-square

Mask là11110/11111. Với query cuối ở prefix dài3, cả ba key phải được nhìn. `is_causal=True` upper-left với shape1×3 chỉ mở key0 trong API đã kiểm. Ví dụ bên dưới đúng ra4.844638 nhưng cách sai cho2.0.

```python
import torch
from torch.nn import functional as F
from model import allowed_mask
mask=allowed_mask(torch.tensor([3]),torch.tensor([2]),2)
assert mask[0].int().tolist()==[[1,1,1,1,0],[1,1,1,1,1]]
q=torch.tensor([[[[1.]]]],dtype=torch.float64)
k=torch.tensor([[[[1.],[0.],[1.]]]],dtype=torch.float64)
v=torch.tensor([[[[2.],[4.],[8.]]]],dtype=torch.float64)
good=F.scaled_dot_product_attention(q,k,v,attn_mask=torch.ones(1,1,1,3,dtype=torch.bool),dropout_p=0.)
bad=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
assert abs(bad.item()-2)<1e-12 and abs(good.item()-4.8446375965)<1e-10
print(good.item(),bad.item())
```

## 5. Dự toán KV

MHA2GiB, GQA.5GiB/request; tám GQA request cùng độ dài4GiB nếu không sharing. Chưa tính weights, activations/scratch, allocator/block metadata, tensor tạm và bản copy. Bản4bit của weights chưa thay dtype cache.

```python
from arithmetic import kv_bytes
mha=kv_bytes(32,32,128,[4096],2)
gqa=kv_bytes(32,8,128,[4096],2)
assert mha==2*1024**3 and gqa==512*1024**2
assert kv_bytes(32,8,128,[4096]*8,2)==4*1024**3
print(mha,gqa)
```

## 6. Padding và lấy logit cuối

Input order34ô/22ô thật, efficiency64.7%; bucketed24ô,91.7%. Right padding lengths3/5 lấy logit index2/4. Decode append vào position3/5. Không dùng common padded width làm position của mọi hàng.

```python
import torch
from arithmetic import padding
r=padding([2,9,3,8]);assert (r['valid'],r['input_order_dense'],r['bucketed_dense'])==(22,34,24)
logits=torch.arange(2*5*4).reshape(2,5,4)
lengths=torch.tensor([3,5])
last=logits[torch.arange(2),lengths-1]
assert torch.equal(last[0],logits[0,2]) and torch.equal(last[1],logits[1,4])
print(22/34,22/24)
```

## 7. Kiểm logits trước argmax

Một lỗi có thể làm xác suất đổi nhưng top1 vẫn như cũ. Vì thế kiểm logits valid, tolerance và generation cùng hữu ích. Code kiểm một batch ragged với single, rồi các chuỗi greedy cache/full. Các đầu ra token tổng hợp không chứng minh chất lượng ngôn ngữ.

```python
import torch
from model import make_model,pack
from generation import generate
torch.set_num_threads(1);m=make_model();prompts=[[1,7,9],[1,4,6,8,10]]
x,n=pack(prompts);out=m(x,n)
for i,p in enumerate(prompts):
    one=m(*pack([p]))
    assert torch.allclose(out[i,:len(p)],one[0],atol=2e-6)
a=generate(m,prompts,5,True);b=generate(m,prompts,5,False)
assert a['outputs']==b['outputs']
assert a['outputs'][0]==generate(m,[prompts[0]],5)['outputs'][0]
print('logits and greedy parity')
```

## 8. Request kết thúc sớm

Active IDs: `[r0,r1,r2]`, `[r1,r2]`, `[r1,r2]`, `[r2]`. Outputs `[EOS]`, `[7,7,EOS]`, `[7,7,7,7]`; reasons EOS/EOS/length. Prompt lengths3+5+2=10, output đã feed lại0+2+3=5, tổng15vị trí. Fixture ép token, chỉ kiểm lifecycle/mapping, chưa đo model stop quality.

```python
from model import make_model,EOS
from generation import generate
def fixture(logits,step,active):
    return [EOS if (i==0 or (i==1 and step==2)) else 7 for i in active]
a=generate(make_model(),[[1,4,6],[1,7,8,9,10],[1,3]],4,selector=fixture)
assert a['outputs']==[[2],[7,7,2],[7,7,7,7]]
assert a['stop_reasons']==['eos','eos','length']
assert a['valid_positions_processed']==15
assert [r['active_ids'] for r in a['steps']]==[['r0','r1','r2'],['r1','r2'],['r1','r2'],['r2']]
print(a['outputs'])
```

## 9. Lịch batching tính tay

Static finish A4/B1/C6/D5, TTFT C4/D3; continuous finish A4/B1/C3/D4, TTFT C1/D2. Active fraction2/3 và1. World cho mọi iteration cùng thời lượng, bỏ prefill/KV pressure/overhead; chưa là dự báo phần cứng.

```python
from scheduling import demo
s=demo();a=s['static'];b=s['continuous']
assert [r['finish'] for r in a['records']]==[4,1,6,5]
assert [r['finish'] for r in b['records']]==[4,1,3,4]
assert [r['ttft'] for r in a['records']]==[1,1,4,3]
assert [r['ttft'] for r in b['records']]==[1,1,1,2]
assert a['active_fraction']==2/3 and b['active_fraction']==1
print(a['makespan'],b['makespan'])
```

## 10. Cấp phát khác sử dụng

Used2048byte, buffer reserved8192byte. Page4: ceil(5/4)+ceil(9/4)=2+3=5page, capacity20, used14, waste6slot. Các block cuối còn chỗ trống; tổng hệ thống còn metadata, weight, activation và allocator. Tên một kỹ thuật quản lý memory không bỏ mọi overhead.

## 11. Đọc throughput đúng nghĩa

14/700s=.02s=20ms. Tám token/step cho400aggregate token/s; mỗi request nhận1token/20ms nên ITL vẫn20ms trong mô hình lý tưởng. 2.5ms là amortized time trên tổng output token, không là khoảng giữa hai token của cùng chuỗi. Chờ đủ batch, prefill khác dài hoặc cạnh tranh memory/compute đều có thể tăng TTFT.

## 12. Cache identity và compaction

Tensor K/V, lengths và request IDs phải được select/reorder cùng nhau. `cache.select([0])` giữ A; `select([1,0])` đổi thứ tự B/A. Không đổi IDs bằng tay rồi coi trạng thái đã di chuyển. Guard model owner/version kiểm nhiều lỗi identity, nhưng không phân tích văn bản prompt: caller phải append đúng suffix hoặc tạo cache mới.

```python
import torch
from model import make_model,pack
m=make_model();c=m.new_cache(['A','B'],16)
m(*pack([[1,7],[1,9,8]]),c,['A','B'])
d=c.select([1,0]);assert d.request_ids==('B','A') and d.lengths.tolist()==[3,2]
x,n=pack([[4],[5]])
try:m(x,n,c,['B','A'])
except ValueError:pass
else:raise AssertionError('row mismatch accepted')
with torch.no_grad():m.head.weight[0,0]+=1
try:m(x,n,c,['A','B'])
except ValueError:pass
else:raise AssertionError('stale weights accepted')
print('row/weight identity rejected')
```

## 13. Prefix reuse và cập nhật model

Không dùng chung KV của từ timeout chỉ vì cùng string/token: K/V ở các layer phụ thuộc prefix và vị trí. Cần cùng token prefix, model/revision/adapter, position/mask convention và KV representation; khi share còn cần namespace/quyền hợp lệ. Temperature không tự thay hidden state của một prefix y hệt; output sampling khác làm prefix về sau khác. DPO đổi weights, cần invalidate cache cũ. Cache không cấp thêm quyền truy cập dữ liệu.

## 14. Thiết kế và kiểm benchmark

Công thức token/s=`64/(median_ms/1000)`. Table là makespan bốn request tổng64token, không là latency percentile từng request. Ép độ dài tránh một nhánh nhanh vì stop sớm; lifecycle EOS vẫn cần test riêng. Probe bỏ initial pack/cache allocation và mọi queue/tokenizer/network, nên chỉ phản ánh một local request đã sẵn tensor/cache.

```python
from benchmark import run
r=run(repeats=3)
assert r['outputs_identical'] and r['requests']==4
for row in r['rows']:
    assert row['output_tokens']==64 and len(row['samples_ms'])==3
    assert row['min_ms']<=row['median_ms']<=row['max_ms']
    assert abs(row['output_tokens_per_second']-64000/row['median_ms'])<1e-8
assert len(r['token_probe']['itl_ms'])==15
print('fixed-work timing structure checked')
```

## 15. Chọn hướng tối ưu theo bottleneck

Thử giới hạn prefill token/lượt hoặc chunked prefill để giảm decode bị chặn; thử admission dựa KV/token budget để không nhận quá khả năng. Đo TTFT/ITL theo request, queue wait, KV usage, failure/cancellation và quality/context retention. Không âm thầm cắt context chỉ để metric nhanh hơn.

KV cache tái sử dụng trạng thái; FlashAttention tổ chức attention giảm I/O; paging quản lý layout/block; compile giảm overhead khi graph phù hợp; weight quantization giảm payload trọng số và có trade-off số học/chất lượng. Lab đã chạy manual/SDPA CPU, dense KV, batch compaction và simulation schedule. Chưa chạy FlashAttentionGPU, paged kernel, compile, quantization/offload hay distributed inference.

## 16. Đọc kết quả có giới hạn

Cached31logical/36dense và full164logical/204dense là các ô token đi qua forward trong demo. Chúng không đếm FLOPs/tổng byte hoặc số token chất lượng. Benchmark TinyDecoder CPU không chọn được batch size cho7B GPU.

Một thiết kế tiếp theo hợp lý: cố định một checkpoint đã được phép dùng, workload gồm prompt/output lengths đại diện, cùng decoding và quality gate. Kiểm logits/output trên tập nhỏ trước; warm-up, quét batch/token budget có kiểm memory, ghi TTFT/ITL/end-to-end từ client và throughput/goodput cùng queue/failure. Dừng tăng tải khi chạm memory budget hoặc SLO đã đặt cho bối cảnh, không chọn mức có token/s đẹp nhất mà mất yêu cầu công việc. Những việc này là đề xuất, chưa phải kết quả đã chạy.
