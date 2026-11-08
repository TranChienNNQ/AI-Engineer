# Lời giải bài 38

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

Mười snippet Python dưới đây chạy độc lập từ thư mục bài này với requirements đã cài. Ví dụ code nhỏ là kiểm cơ chế, không tự trở thành benchmark chất lượng/kernel production.

## 1. Storage, compute và đối tượng quantization

Cần hỏi: weights nào được quantize; codes có pack thật không; group/scale/zero-point/padding; phần không quantize; activation/KVdtype; compute/accumulator/kernel; workload/context/batch; peak lúc load/forward. Weight-only chưa làm activations/KVthấpbit. CastBF16là đổi float format, khác uniform integer quantization với scale/zero-point.

## 2. Symmetric quantization tính tay

Scale.5, q=`[-7,-2,0,1,5,7]`, reconstruction=`[-3.5,-1,0,.5,2.5,3.5]`; MSE.015/max.2. Zero tensor cần tránh scale0; lab dùng scale1,q0. Bound s/2cho round-nearest không bao phần vượt range bị clipping.

```python
import torch
from quantization import quantize_weight,dequantize_weight
w=torch.tensor([[-3.5,-1.2,0.,.7,2.4,3.5]])
d,s,sp=quantize_weight(w);r=dequantize_weight(d,s,sp)
assert abs(float(((r-w)**2).mean())-.015)<1e-7
assert torch.allclose(r,torch.tensor([[-3.5,-1.,0.,.5,2.5,3.5]]))
z=torch.zeros(2,3);d,s,sp=quantize_weight(z)
assert torch.equal(dequantize_weight(d,s,sp),z) and torch.isfinite(s).all()
print('symmetric and zero matrix checked')
```

## 3. Affine và thứ tự round

s=.2,z=5, code[0,3,5,8,15], reconstruction[−1,−.4,0,.6,2]. Với tie, round(.5)+1=1 nhưng round(1.5)=2. Vì vậy giữ đúng thứ tự của quantizer; không tự đổi công thức ở biên tie.

```python
import torch
from quantization import affine
q,r,m=affine(torch.tensor([-1.,-.35,0.,.65,2.]),-1.,2.)
assert q.tolist()==[0,3,5,8,15] and m['zero_point']==5
v=torch.tensor(.5);z=1
assert (torch.round(v)+z).item()==1
assert torch.round(v+z).item()==2
print(q.tolist(),r.tolist())
```

## 4. Packing và padding

Offset codes[1,6,8,9,13,15]cho bytes0x61/0x98/0xFD. Ba unsigned codes[1,6,8]cần hai byte[0x61,0x08]; nibble cao cuối là padding. Count3giúp bỏ nibble đó. Còn cần biết mapping code, scale, shape và group để tái tạo số thực.

```python
import torch
from quantization import pack_nibbles,unpack_nibbles
c=torch.tensor([1,6,8],dtype=torch.uint8);p=pack_nibbles(c)
assert p.tolist()==[0x61,0x08]
assert torch.equal(unpack_nibbles(p,3),c)
assert p.numel()==2
print(p.tolist())
```

## 5. Dot integer và zero-point

qx=[2,−4,6],qw=[2,1,−4],acc=−24; giải scale cho−3. Affine cần tổng `(qx−zx)(qw−zw)`. Offset nibble8của lab phải được giải về signed q trước khi dùng công thức symmetric.

```python
import torch
x=torch.tensor([1.,-2.,3.]);w=torch.tensor([.5,.25,-1.])
qx=torch.round(x/.5).to(torch.int32);qw=torch.round(w/.25).to(torch.int32)
acc=int((qx*qw).sum());assert acc==-24
assert acc*.5*.25==float(x@w)==-3.
a=torch.tensor([5,6,7]);b=torch.tensor([3,4,5]);za,zb=5,3
correct=int(((a-za)*(b-zb)).sum())
expanded=int((a*b).sum()-zb*a.sum()-za*b.sum()+len(a)*za*zb)
assert correct==expanded
print(acc,correct)
```

## 6. Granularity và byte metadata

Group3: mỗi hàng pad5→6, tổng12slot; codes6byte,4scale×4byte=16, tổng22byte. Per-row g5:10slot→5byte codes,2scale→8byte, tổng13byte. Chưa bias/header. Trong MLP, g32pad hàng8/16lên32, làm số code slot tăng; tổngW4-g32là1288byte nhưW8-row.

```python
import torch
from quantization import quantize_weight
w=torch.ones(2,5)
d,s,sp=quantize_weight(w,4,3)
assert d.numel()==6 and s.numel()==4
assert d.numel()+s.numel()*4==22
d,s,sp=quantize_weight(w,4)
assert d.numel()+s.numel()*4==13
print('22 versus 13 bytes')
```

## 7. Weight MSE và metric công việc

`Δy=xΔWᵀ`; `|Δyj|≤||x||₂||Δwj||₂`. Input norm lớn tăng bound. Margin top1/top2nhỏ làm argmax nhạy. Accuracy bằng nhau chỉ là tổng số đúng bằng nhau; agreement mới đo dự đoán giống nhau, còn xác suất/logits vẫn có thể khác dù agreement100%.

## 8. Calibration shift

Static reconstruct[−1,0,1], clip2/3. Dynamic per-row có range[−4,4], giữ hai đầu trong ví dụ. Calibration lab lấy128trainrow; stress4×dùng cùng testrowbiến đổi nên không độc lập. Values hash giúp nhận ra dữ liệu thay đổi dù indices giữ nguyên; cũng cần preprocessing/model identity.

```python
import torch
from quantization import fake_activation
x=torch.tensor([[-4.,0.,4.]])
a,s=fake_activation(x,'static8',torch.tensor(1/127))
b,d=fake_activation(x,'dynamic8')
assert a.tolist()==[[-1.,0.,1.]] and s['clipped']==2
assert torch.allclose(b,x) and d['clipped']==0
print(a.tolist(),b.tolist())
```

## 9. PTQ, QAT và STE

TrainFP32→chọn validation best→thu calibration train→convert configs đã định→đánh giá test/stress. Round có đạo hàm0hầu khắp; STE thay đạo hàm trong backward. Biểu thức sau có forward=qhat, gradient1theo x; đây là identity STE, chưa là gradient-clipped QATrecipe và chưa trainQAT. PackedLinear giảiFP32trước matmul.

```python
import torch
x=torch.tensor([-.7,.2,1.4],requires_grad=True)
qhat=torch.round(x/.5).clamp(-2,2)*.5
ste=x+(qhat-x).detach()
assert torch.equal(ste.detach(),qhat.detach())
ste.sum().backward()
assert torch.equal(x.grad,torch.ones_like(x))
print(ste.detach().tolist(),x.grad.tolist())
```

## 10. Memory ledger7B

Codes3.43GB,scales.214375GB,phầnFP16.28GB,tổng3.924375GBthập phân, effective4.485bit/parameter. FP16toàn bộ14GB. Trần payload1vs20request là kết quả giả định; chưa trừ mọi temporary/alignment/loadpeak, workspace có thể đổi theo batch, chưa kiểm latency/SLO hay quality/context.

```python
from arithmetic import plan
p=plan()
assert p['int4_codes_bytes']==3_430_000_000
assert p['scales_bytes']==214_375_000
assert p['mixed_weight_bytes']==3_924_375_000
assert p['effective_bits']==4.485
assert (p['payload_ceiling_fp16'],p['payload_ceiling_mixed4'])==(1,20)
print(p['mixed_weight_bytes'],p['effective_bits'])
```

## 11. Đọc bảng chất lượng

W4-g8khác FP32ở2prediction: một mẫu đúng→sai và một mẫu sai→đúng trong lượt đã kiểm, nên cùng126đúng nhưng agreement130/132. W4-row880byte, RMSE.591692;g81084byte, RMSE.436954;g321288bytevà cùng reconstructed weights như row trong kiến trúc này. So total bytes, không chỉ bit label.

Static clip6/7392→657/7392và accuracy124→106khi stress4×cho thấy range không đủ đại diện ở thí nghiệm này. Dynamic không clip theo absmaxvừa chọn, vẫn có rounding và overhead. Chưa chứng minh luôn robust hoặc luôn nhanh/tốn ít peak memory hơn static; fake quant activations đang làFP32.

## 12. Artifact và identity

Cần format version, kiến trúc, bit/group/axis/offset/range, feature/tokenizer/template/preprocessing, state và source/calibration/data identity. Artifact thí nghiệm lưu các thông tin cần cho model nhỏ. Hash chỉ so với expected identity đáng tin, chưa là chữ ký tác giả. File còn container/header/metadata, không bằng tensor payload.

Snippet sau tạo bundle tối thiểu phục vụ kiểm loader; không giả làm checkpoint đã train. Thư mục tạm được dọn khi kết thúc.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import torch
from model import MLP,QuantMLP
from training import state_hash,load_bundle
from quantization import FORMAT
q=QuantMLP(MLP(),4,8,'static8')
m=dict(format=FORMAT,spec=q.spec,payload_hash=state_hash(q.state_dict()))
b=dict(metadata=m,state=q.state_dict())
with TemporaryDirectory() as tmp:
    p=Path(tmp)/'q.pt';torch.save(b,p);loaded=load_bundle(p,m)
    assert state_hash(loaded.state_dict())==m['payload_hash']
    for kind in ['metadata','payload']:
        bad=copy.deepcopy(b)
        if kind=='metadata':bad['metadata']['spec']['bits']=8
        else:bad['state']['layers.0.data'][0]^=1
        torch.save(bad,p)
        try:load_bundle(p,m)
        except ValueError:pass
        else:raise AssertionError('mismatch accepted')
print('roundtrip and two mismatches checked')
```

## 13. Persistent memory và peak memory

Codes/scales/bias còn tồn tại khi dequant weightFP32tạm xuất hiện, kèm input/output/activation. Process còn giữ baseline và runtime overhead, nên1084/1096byte không là tổngRAM. Có thể bỏ hidden-state outputs không cần, inference_mode, giảm in-flight/context có chủ đích hoặc offload. Cần đo mất ngữ cảnh/chất lượng, latency/băng thông và tổng memory trên backend thực; offload không miễn phí.

## 14. K/V lượng tử hóa

Weights và KVlà tensor khác. K/Vcó2×2×17×8×2=1088phần tử, FP32→4352byte. INT4codes544byte; có136group8cho cảK/V, scales136×4=544byte; tổng1088byte. Sai sốKđổi score/softmax, Vđổi vector trộn. Diagnostic nhìn một attention, per-token/head như nhau cho K/V, chưaKIVI hoặc engine sinh chuỗi.

```python
from attention import run
r=run()
assert r['fp32_kv_bytes']==4352
assert [row['packed_payload_bytes'] for row in r['rows']]==[1632,1088]
assert all(row['rmse']>0 for row in r['rows'])
assert r['rows'][1]['rmse']>r['rows'][0]['rmse']  # chỉ kết quả fixture seed này
print([(v['bits'],v['packed_payload_bytes'],v['rmse']) for v in r['rows']])
```

## 15. Gọi đúng tên phương pháp

Absmax RTNchọn scale theo max và làm tròn; GPTQdùng xấp xỉ bậc hai; AWQdùng activation statistics để bảo vệ kênh weight; SmoothQuantdùng biến đổi tương đương để phân phối khó khăn quantization; NF4là codebookkhông đều. Các mô tả nghiên cứu chỉ ở mức abstract đã đọc trong bài, không làimplementation tương đương.

Vì Dchéo nên Dᵀ=D, `(WD)ᵀ=DWᵀ`; suy ra `XD⁻¹DWᵀ=XWᵀ`. Rounding không giao hoán với scale và phép nhân nên sau quantization hai đường có thể sai số khác. Cần kiểm chất lượng, không chỉ đẳng thức float.

## 16. Thiết kế phép chọn triển khai

Khóa model/runtime/device và baseline trước. Calibration đại diện task/ngôn ngữ/context; dùng validation để chọn config, giữ testcuối cho xác nhận. So metric công việc/slices, factuality/citation nếu RAG, logits/decode và format compatibility. Đo peakload/forward, KVtheo workload, TTFT/ITL/goodput trên chính backend, có artifact identity và rollback khi không đạt yêu cầu.

Không kết luận tốc độGPU, chất lượngLLMlong-context, concurrentcapacity hay ưu thếGPTQ/AWQ/QATtừ model867parameter. Cũng không coi fakeA8là đã giảm peakactivationRAM. Đây là đề xuất thí nghiệm tiếp theo, chưa phải một deployment/benchmark đã thực hiện.
