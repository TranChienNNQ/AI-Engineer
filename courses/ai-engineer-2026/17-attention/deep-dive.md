# Phụ lục — Đào sâu gradient, mask và multi-head API

[Về bài 17](README.md) · [Code lab](lab.py) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Phụ lục này không yêu cầu học trước phần mới. Nó mở các bước toán và tensor được dùng trong bài chính. Các công thức gradient dưới đây giả định attention không dropout, loss scalar, và mỗi query có ít nhất một key hợp lệ.

## A. Softmax: một score ảnh hưởng cả hàng

Với một hàng s có S phần tử, a_i=exp(s_i)/Σ_j exp(s_j). Lấy đạo hàm theo s_j bằng quy tắc thương:

\[
\frac{\partial a_i}{\partial s_j}=a_i(\mathbf{1}[i=j]-a_j).
\]

Hai trường hợp:

- i=j: a_i(1−a_i), tăng score của chính ô thường tăng weight ô đó.
- i≠j: −a_i a_j, tăng score một ô lấy bớt khối lượng khỏi những ô khác.

Nếu biết gradient upstream g_i=∂L/∂a_i, không cần tạo ma trận Jacobian S×S:

\[
\frac{\partial L}{\partial s_j}
=\sum_i g_i a_i(\mathbf{1}[i=j]-a_j)
=a_j\left(g_j-\sum_i g_i a_i\right).
\]

Đó là phép trừ **cùng một weighted mean** khỏi mọi g_j rồi nhân a_j. Tổng gradient score theo j bằng0. Nếu a gần one-hot, nhiều đạo hàm rất nhỏ; scaling giúp kiểm soát score nhưng không bảo đảm tránh saturation sau mọi lần train.

## B. Backward toàn ma trận

Một sample Q:L×d_k, K:S×d_k, V:S×d_v. Đặt c=√d_k, S_score=QKᵀ/c, A=softmax(S_score), Z=AV. Biết G=∂L/∂Z có shape L×d_v:

\[
G_V=A^\top G\quad(S\times d_v), \qquad
G_A=GV^\top\quad(L\times S).
\]

Với từng hàng, r_i=Σ_j A_ij(G_A)_ij. Broadcast r về L×1 rồi:

\[
G_S=A\odot(G_A-r), \qquad
G_Q=G_SK/c, \qquad G_K=G_S^\top Q/c.
\]

Tất cả shape khớp parameter/tensor gốc. Với mask cố định, A tại ô cấm=0 nên đóng góp gradient tại score cấm=0. Mask không phải một parameter số nguyên cần tối ưu. Với self-attention, Q, K, V đều sinh từ X nên gradient về X là tổng các nhánh projection, không chỉ nhánh Q.

### Tính tay lại ví dụ 1×3

Lấy q=ln2, K=[0,1,2]ᵀ, V=[[1,0], [0,2], [3,1]], L=z_1. A=[1,2,4]/7, G=[1,0]:

```text
G_V = [[1/7,0],[2/7,0],[4/7,0]]
G_A = [1,0,3]
r   = 13/7
G_S = [-6,-26,32]/49
G_Q = (-26 + 64)/49 = 38/49
G_K = ln2 * [-6,-26,32]ᵀ/49
```

Code dưới chạy độc lập từ thư mục bài 17; nó tính lại backward bằng phép toán ma trận và so autograd:

```python
import math
import torch
from lab import attention, hand_tensors
Q,K,V=hand_tensors(requires_grad=True)
Z,A=attention(Q,K,V)
Z[0,0].backward()
with torch.no_grad():
    G=torch.tensor([[1.,0.]],dtype=torch.float64)
    GA=G@V.T
    GS=A*(GA-(A*GA).sum(-1,keepdim=True))
    dV=A.T@G
    dQ=GS@K/math.sqrt(Q.shape[-1])
    dK=GS.T@Q/math.sqrt(Q.shape[-1])
    for actual,expected in [(V.grad,dV),(Q.grad,dQ),(K.grad,dK)]:
        torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-13)
print('Manual matrix backward matches autograd')
```

Đối chiếu chỉ một case chưa là chứng minh implementation đúng mọi shape. [Lời giải](solutions.md) có thêm sai phân trung tâm và masked case; lab còn có NumPy reference độc lập cho forward.

## C. Mask đúng polarity và broadcast đúng chiều

Ví dụ B=2, h=2, L=S=3. SDPA nhận Q/K/V theo B×h×T×d. `valid_keys` là B×S, còn causal là L×S. Kết hợp cho phép đọc:

```python
import torch
from torch.nn import functional as F
from lab import attention
B,H,T,D=2,2,3,2
torch.manual_seed(7)
Q=torch.randn(B,H,T,D,dtype=torch.float64)
K=torch.randn(B,H,T,D,dtype=torch.float64)
V=torch.randn(B,H,T,D,dtype=torch.float64)
valid_keys=torch.tensor([[True,True,False],[True,True,True]])
causal=torch.ones(T,T,dtype=torch.bool).tril()
allowed=valid_keys[:,None,None,:] & causal[None,None,:,:]
manual,A=attention(Q,K,V,allowed)
api=F.scaled_dot_product_attention(Q,K,V,attn_mask=allowed,dropout_p=0.0)
torch.testing.assert_close(manual,api,atol=1e-14,rtol=1e-13)
float_mask=torch.zeros_like(allowed,dtype=Q.dtype).masked_fill(~allowed,float('-inf'))
api_float=F.scaled_dot_product_attention(Q,K,V,attn_mask=float_mask,dropout_p=0.0)
torch.testing.assert_close(manual,api_float,atol=1e-14,rtol=1e-13)
print(allowed.shape,A.shape)  # allowed(2,1,3,3), weights(2,2,3,3)
```

Mask chia sẻ giữa các head bằng broadcast, nhưng mỗi head vẫn có score riêng. Query cuối sample0 là query PAD trong ví dụ: key mask không tự vô hiệu output hàng đó. Code trên chỉ kiểm phép đọc; khi task có loss theo token phải mask loss/query tương ứng.

Không dùng cả explicit mask và `is_causal=True` tùy tiện; đối chiếu hợp đồng API/phiên bản. Trong bài chỉ so `is_causal=True` với explicit causal mask vuông khi không truyền attn_mask. Non-square causal alignment chưa được kiểm.

### Bảng chuyển mask sang MHA

Giữ `valid_keys` như trên, MHA dùng `key_padding_mask=~valid_keys`. Causal mask MHA là `blocked=~causal`, khác với SDPA `allowed=causal`. Nếu dùng cả padding và attn_mask trong MHA, cần thống nhất loại bool/float theo API.

- SDPA mask có thể broadcast tới B×h×L×S.
- MHA attn_mask 2D là L×S, hoặc 3D B*h×L×S; không tự đưa B×L×S rồi cho là đúng vì trông gần giống.
- MHA key_padding_mask là B×S.
- Bool True của MHA là bị chặn; float−∞ luôn mang vai trò bias làm score cấm trong các ví dụ này.

Nguồn: [SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) và [MHA.forward](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward). Lab không dựa vào mask heuristic hoặc kernel GPU.

## D. Tự bóc module MultiheadAttention

Một module tự làm projection. Một functional SDPA chỉ làm phần attention trên tensor được cung cấp. Đừng tự project QKV rồi đưa vào MHA và mặc định module không project thêm.

Với self-attention, PyTorch lưu ba Linear input trong `in_proj_weight` shape(3D, D), bias(3D, ). Split **sau projection**, không chia X thành head trước rồi dùng cùng một matrix nhỏ cho tất cả head. Hai cách chỉ tương đương nếu bố trí parameter khớp đúng.

```python
import torch
from torch import nn
from torch.nn import functional as F
from lab import attention
B,T,D,H=2,3,4,2
torch.manual_seed(13)
mha=nn.MultiheadAttention(D,H,batch_first=True,dropout=0).to(dtype=torch.float64)
X=torch.randn(B,T,D,dtype=torch.float64)
api,weights=mha(X,X,X,need_weights=True,average_attn_weights=False)
Q,K,V=F.linear(X,mha.in_proj_weight,mha.in_proj_bias).chunk(3,dim=-1)
def split(tensor):
    return tensor.reshape(B,T,H,D//H).transpose(1,2)
heads,A=attention(split(Q),split(K),split(V))
joined=heads.transpose(1,2).contiguous().reshape(B,T,D)
manual=F.linear(joined,mha.out_proj.weight,mha.out_proj.bias)
torch.testing.assert_close(manual,api,atol=1e-14,rtol=1e-13)
torch.testing.assert_close(A,weights,atol=1e-14,rtol=1e-13)
print(api.shape,weights.shape)  # (2,3,4), (2,2,3,3)
```

`transpose` thay ý nghĩa trục; `contiguous` ở đây giúp cách ghép rõ ràng trước reshape. Không lấy `reshape(B, T, D)` thẳng từ B×h×T×d để ghép: nó có thể trộn head và vị trí sai. Mỗi hàng input ban đầu phải nhận đủ các head của chính vị trí ấy.

## E. Một ví dụ multi-head cực nhỏ tính tay

Để thấy ý nghĩa concat, giả sử một query, hai memory row, h=2, d_k=d_v=1 và W_O là identity2×2:

- Head1:q=ln2, K=[0,1]ᵀ, V=[1,3]ᵀ. Weight[1/3,2/3], output7/3.
- Head2:q=ln2, K=[1,0]ᵀ, V=[10,0]ᵀ. Weight[2/3,1/3], output20/3.
- Concat=[7/3,20/3], không phải mean ra một scalar9/2.

Nếu W_O khác identity, nó trộn các tọa độ concat theo matrix học được. Đây là projected QKV cho trước, không dùng nó để đếm parameter của một module thiếu định nghĩa input D. Bài chính có phép đếm80 parameter cho module D4, h2 đầy đủ.

## F. Suy permutation equivariance bằng ma trận

P là permutation matrix hoán vị token của self-attention không position/mask cố định. Q′=PQ, K′=PK, V′=PV:

\[
S'=PSP^\top, \qquad A'=PAP^\top, \qquad
Z'=PAP^\top PV=PZ.
\]

Softmax theo hàng tôn trọng hoán vị hàng/cột tương ứng. Với cross-attention, giữ Q và chỉ hoán vị memory bằng P thì A′=APᵀ, V′=PV nên Z′=Z. Nếu mask/position không được biến đổi tương ứng, các bước trên không còn áp dụng nguyên xi.

Một minh họa và phản ví dụ về weight cùng nằm trong mode `invariance`. Đây là chứng minh về cấu trúc phép toán, không là cam kết attention hiểu trật tự câu hoặc giải thích nhân quả.
