# Lời giải — Attention

[Về bài học](README.md) · [Về đề bài](exercises.md) · [Đào sâu toán/API](deep-dive.md)

## Bài 1

Query biểu diễn dịch vụ cần hỏi, key biểu diễn ID/đặc trưng của dòng, value là nội dung trạng thái cần đọc. Giữ Q, K, mask thì A giữ; đổi V có thể đổi Z. Đổi Q có thể đổi score/weight/output, nhưng không bắt buộc output đổi:ví dụ mọi V cùng vector. Soft attention trộn theo weight; argmax chọn một dòng; dict equality lookup theo khóa là cơ chế xác định khác, không tự được bảo đảm chỉ nhờ softmax.

## Bài 2

Scores=[0, ln2, ln4], exp=[1,2,4], A=[1,2,4]/7. Z=[(1+12)/7, (4+4)/7]=[13/7,8/7]. q=0 cho A đều1/3, Z=[4/3,1]. Số chiều output theo d_v=2, không theo chiều query1. Mỗi key có một value cùng dòng; không lấy weighted sum key thay cho value.

## Bài 3

QKᵀ=[[1,0,1], [0,1,1]], chia√2. Đặt a≈.401112092680, b≈.197775814640; A=[[a, b, a], [b, a, a]]. Z=[[4a,2b+a], [b+3a,3a]]≈[[1.604448370719, .796663721961], [1.401112092680,1.203336278039]]. Softmax theo từng hàng trên3 key. A2×3 không phải ma trận vuông; ngay cả vuông cũng không mặc định đối xứng hoặc cột cộng1.

## Bài 4

Giả định minh họa là thành phần q/k độc lập, mean0, variance1; dot có variance d_k do cộng d_k tích độc lập có variance1. q=k toàn1 cho dot4, scaled2, cosine1. Những vector này không thỏa giả định mean0 để suy thống kê, nhưng phép tính vẫn hợp lệ. Cộng1000 cùng cả hàng không đổi softmax toán học; nhân2 làm tỷ lệ exp đổi, nói chung weight đổi. Trừ max tránh exp số dương lớn gây overflow; không thay đổi phân phối đúng.

## Bài 5

G=[1,0], G_V=[[1/7,0], [2/7,0], [4/7,0]], G_A=[1,0,3]. Weighted mean G_A là13/7. G_S=[−6, −26,32]/49, tổng0; G_Q=38/49≈.775510204082; G_K=ln2×G_Sᵀ. Tổng0 liên quan tính bất biến khi cộng hằng số vào cả score row. Đổi V có thể đổi G_A và gradient score dù QK hiện tại không đổi. Nếu muốn loss CE ở head thật, cần thêm gradient của head/CE, không dùng G=[1,0] tùy ý.

## Bài 6

Mask key3:exp còn[1,2,0], A=[1/3,2/3,0], Z=[1/3,4/3]. Chỉ gán weight3 cũ về0 cho[1/7,2/7,0], output[1/7,4/7], tổngweight3/7, không phải kết quả mask chuẩn. Nhân score với0 không tạo−∞; score bị cấm vẫn có exp1 và có thể chiếm trọng số. Lab báo ValueError nếu hàng không có key hợp lệ; không dựa vào hành vi NaN/zero khác nhau theo API/backend.

## Bài 7

Allowed causal4×4 là[[1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]]. Padding key chặn cột; query PAD vẫn có thể đọc key thật và tạo output khác0, nên cần query/loss mask riêng. Next-token training có input token i dự đoán token i+1 nên được đọc chính token i, nếu input/target đã căn đúng. Kiểm:đổi token ở vị trí cuối, giữ prefix, forward causal rồi so output prefix không đổi. Không thử bằng cách đổi luôn query prefix rồi yêu cầu output giữ nguyên.

## Bài 8

SDPA True=được đọc; MHA attn_mask và key_padding_mask True=bị chặn. SDPA valid key broadcast là valid_keys[:, None, None, :], AND với causal[None, None, :, :] nếu cần. MHA key_padding_mask=~valid_keys. MHA mask2D causal bị chặn là ~causal. SDPA không tự đọc mode eval của module bọc để tắt dropout_p=.2; phải truyền0 ở eval. Các phép đối chiếu trong bài dùng dropout0 để kết quả xác định.

## Bài 9

Q5×4, K7×4, V7×6, score/A5×7, Z5×6. Self cùng nguồn X nhưng projection khác nên Q, K, V không cần bằng nhau hoặc có cùng width trong dạng tổng quát. Self không đồng nghĩa causal; quy tắc cho phép đọc phụ thuộc task. Cross khác nguồn query/memory, không nhất thiết là dịch máy hoặc LLM.

## Bài 10

Head1 weight[1/3,2/3], output1/3+2=7/3. Head2 weight[2/3,1/3], output20/3. Concat[7/3,20/3], mean là9/2 một scalar nên mất khả năng giữ hai tọa độ riêng trước W_O. Giữ tổngD và d_head=D/h thì các projection tổng chiềuD không tự tăng h lần parameter. Nếu đổi thiết kế để mỗi head giữ toànD thì đó là kiến trúc và chi phí khác, cần đếm lại.

## Bài 11

Projection mỗi tensor(3,5,12). Split(3,3,5,4); score/weight(3,3,5,5); head output(3,3,5,4); concat(3,5,12); W_O trả(3,5,12). Parameter4D²+4D=576+48=624. Reshape trực tiếp từ layoutB, h, T, d có thể ghép các vị trí khác nhau thay vì các head của cùng vị trí; phải transpose vềB, T, h, d rồi reshape. Số phần tử khớp là điều kiện cần, chưa đủ đúng ý nghĩa.

## Bài 12

Self không position/mask có Z′=PZ khi X′=PX. Cross giữ Q và hoán vị K, V cùng nhau thì Z′=Z. Fixed causal mask/position có thể phá điều kiện của lập luận; không mở rộng tùy ý. Score explicit8×4096²×4 byte=536.870.912 byte=512MiB, chưa tính phần khác. Memory của kernel tối ưu không bắt buộc bằng cách lưu explicit này. Nếu V gồm3 hàng đều[1,2] thì A=[1/3,1/3,1/3] và[.99, .005, .005] đều cho[1,2]. Weight khác chưa đủ chứng minh output khác hoặc giải thích nhân quả.

## Bài 13

Sample256/64/64. Parameter:ID embedding7×8=56; Q projection64; K projection64; value embedding3×8=24; head8×3+3=27; tổng235. Train256/batch32=8 update/epoch,40 epoch320. Key-match kiểm key ở argmax A trùng query; accuracy kiểm class value dự đoán. Nhiều dòng có thể cùng value nên hai metric không tương đương. Rule dict dùng key equality giải chính xác, nên task chỉ kiểm cơ chế học được, không chứng minh attention cần thiết hoặc vượt thuật toán phù hợp.

## Bài 14

Báo cáo mẫu:Corpus gồm192 memory mapping độc nhất, mỗi group2 query khác nhau, split group128/32/32 cho256/64/64 sample. Mỗi memory có2–5 key khác nhau từID1–6, value class0/1/2 ngẫu nhiên và target là value của query key có trong memory. Target xuất hiện trong input theo định nghĩa truy xuất, không phải dự báo tương lai. Model single-head cross-attention dim8, head3 có235 parameter, mask theo key0=PAD. Train Adam lr=.01, batch32,40 epoch320 update, clip norm1, loss chỉ là classification CE. Chọn epoch40 theo validation CE≈.001039705; test CE≈.001068935, accuracy và key-match100% trên64 sample. Test confusion diagonal[17,20,27], lớp không ép cân bằng. Majority từ train cho validation15/64≈23.4375%, dict lookup rule100%. Probe query6 có weight≈.983602 ở slot key6 và dự đoán value1. Kết quả một seed/synthetic chưa là bằng chứng ngôn ngữ, long-context hoặc full Transformer.

Value0 hợp lệ, không được mask theo value0; mask nguồn từ keyID0. best.pt là inference artifact có model/config/domain ID/metric/data hash, thiếu optimizer/RNG và trạng thái để resume đầy đủ. Khi triển khai cần hợp đồng input/mask cùng phiên bản; không chỉ weight shape đúng.

## Thực hành mở rộng — Code đối chiếu

Từ thư mục bài 17, dùng môi trường đã cài. Các output mới tạo trong thư mục tạm:

```python
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lab import attention,numpy_attention,hand_tensors,Config,train,load_best

torch.manual_seed(11)
Q=torch.randn(2,2,3,4,dtype=torch.float64)
K=torch.randn(2,2,5,4,dtype=torch.float64)
V=torch.randn(2,2,5,2,dtype=torch.float64)
allowed=torch.tensor([True,True,True,False,False])[None,None,None,:]
Z,A=attention(Q,K,V,allowed)
NZ,NA=numpy_attention(Q.numpy(),K.numpy(),V.numpy(),allowed.numpy())
np.testing.assert_allclose(Z.numpy(),NZ,atol=1e-13,rtol=1e-12)
torch.testing.assert_close(Z,F.scaled_dot_product_attention(Q,K,V,attn_mask=allowed,dropout_p=0),atol=1e-13,rtol=1e-12)

# Sai phân số từng phần tử Q,K,V cho loss z[0,0].
q,k,v=hand_tensors(True)
attention(q,k,v)[0][0,0].backward()
base=[q.detach(),k.detach(),v.detach()]
for which,actual in enumerate([q.grad,k.grad,v.grad]):
    for index in np.ndindex(*actual.shape):
        high=[t.clone() for t in base];low=[t.clone() for t in base]
        high[which][index]+=1e-6;low[which][index]-=1e-6
        numeric=(attention(*high)[0][0,0]-attention(*low)[0][0,0]).item()/2e-6
        assert abs(numeric-actual[index].item())<1e-9

# Forward và parameter gradient của MHA cùng được kiểm.
m=nn.MultiheadAttention(4,2,batch_first=True,dropout=0).to(dtype=torch.float64)
x=torch.randn(2,3,4,dtype=torch.float64)
api,_=m(x,x,x,need_weights=True,average_attn_weights=False)
q,k,v=F.linear(x,m.in_proj_weight,m.in_proj_bias).chunk(3,-1)
split=lambda t:t.reshape(2,3,2,2).transpose(1,2)
heads,_=attention(split(q),split(k),split(v))
manual=F.linear(heads.transpose(1,2).contiguous().reshape(2,3,4),m.out_proj.weight,m.out_proj.bias)
torch.testing.assert_close(api,manual,atol=1e-13,rtol=1e-12)
g_api=torch.autograd.grad(api.square().sum(),tuple(m.parameters()))
g_manual=torch.autograd.grad(manual.square().sum(),tuple(m.parameters()))
for a,b in zip(g_api,g_manual):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)

with TemporaryDirectory() as directory:
    root=Path(directory)/'run';report=train(root,Config(epochs=2,batch_size=30))
    assert report['global_step']==18  # 256=8*30+16 =>9 update/epoch
    model,_=load_best(root/'best.pt')
    query=torch.tensor([2]);keys=torch.tensor([[1,2,3]]);values=torch.tensor([[0,1,2]])
    with torch.no_grad():
        reference,_=model(query,keys,values)
        padded,_=model(query,F.pad(keys,(0,2),value=0),F.pad(values,(0,2),value=2))
        perm=torch.tensor([2,0,1]);reordered,_=model(query,keys[:,perm],values[:,perm])
        torch.testing.assert_close(reference,padded,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(reference,reordered,atol=1e-6,rtol=1e-6)
print('Extension checks passed')
```

Phép so gradient không có dropout và dùng float64. Nếu đổi mask, loss hoặc state giữa hai lần dịch epsilon thì không còn là gradient check cùng một hàm. Mô hình train2 epoch chỉ kiểm plumbing; không yêu cầu đạt số của bảng 40 epoch. Việc giữ output khi hoán vị memory là đúng với task tập dòng không có vị trí của lab, chưa phải thuộc tính của attention có positional information.
