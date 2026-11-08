# Lời giải bài 36

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

Mười snippet Python dưới đây chạy độc lập từ thư mục bài này với requirements đã cài. Chúng không yêu cầu checkpoint của một lượt trước. Câu hỏi mở có nhiều lời giải hợp lý; đánh giá theo lập luận và bối cảnh.

## 1. Viết một cặp ưu tiên có thể đánh giá

A: “Chắc lỗi database, tăng pool ngay.” B: “Kiểm tra lỗi theo endpoint/dependency và đối chiếu thay đổi deploy. Nếu xác định regression, thực hiện rollback theo quy trình đã duyệt.” Ưu tiên B theo tiêu chí bằng chứng và hành động có điều kiện. Cần abstain khi thiếu chuyên môn đánh giá một khuyến nghị kỹ thuật cụ thể; nếu cả hai sai thì ghi both bad. Không chọn chỉ vì B dài hơn; tạo cặp đối chứng về độ dài.

## 2. SFT và dữ liệu ưu tiên khác gì?

Mẫu SFT là prompt cùng B. Thông tin rằng B thắng A và lỗi riêng của A không còn trong loss SFT. Chosen là người thắng cục bộ, có thể vẫn cần chỉnh sửa. Rejected có thể đúng nhưng ít phù hợp với mục tiêu của prompt. Rubric và đủ ngữ cảnh quyết định ý nghĩa nhãn.

## 3. Reward model tính tay

Xác suất .75, loss .287682. Đạo hàm theo r+ là −.25; theo r− là +.25. Cộng cùng100 không đổi hiệu và xác suất.

```python
import math
import torch
r = torch.tensor([math.log(3), 0.], dtype=torch.float64, requires_grad=True)
loss = -torch.nn.functional.logsigmoid(r[0]-r[1])
loss.backward()
assert torch.allclose(r.grad, torch.tensor([-.25,.25], dtype=torch.float64))
assert abs(loss.item() + math.log(.75)) < 1e-12
print(loss.item(), r.grad.tolist())
```

## 4. Bất đồng ba thuận một ngược

`L=−[3lnσ(w)+lnσ(−w)]/4`, đạo hàm `σ(w)−.75`. Tối ưu `w=ln3`, loss .562335. Với tất cả phiếu thuận, loss tiến về0 khi w→+∞; không đạt cực tiểu tại w hữu hạn trong mô hình không ràng buộc này.

```python
import math
w = math.log(3)
p = 1/(1+math.exp(-w))
mean_loss = -(3*math.log(p)+math.log(1-p))/4
assert abs(mean_loss-.5623351446188083)<1e-12
assert abs(p-.75)<1e-12
print(w, mean_loss)
```

## 5. Tính KL và mục tiêu

Expected reward `.8239592165`, KL `.1308120359`, J `.6931471806`. Reference J `.5493061443`. Log-ratio của hành động thứ hai là ln(.25/.5)=ln.5<0; KL cộng có trọng số cả hai hành động và không âm.

```python
import numpy as np
p=np.array([.75,.25]); q=np.array([.5,.5]); r=np.array([np.log(3),0.])
kl=np.sum(p*np.log(p/q)); j=p@r-kl
assert abs(j-np.log(2))<1e-12
assert np.log(p[1]/q[1])<0 and kl>0
print(p@r, kl, j, q@r)
```

## 6. Suy ra nghiệm đóng

Điều kiện dừng `r−β(log(p/q)+1)+λ=0` cho `p=q exp(r/β)×constant`. Chuẩn hóa tổngp=1 tạo Z. Cần β>0, reference dương trên support xét và Z hữu hạn. Khi β→∞ với reward hữu hạn, exp(r/β)→1, nên p*→q. Cộng offset reward không đổi nghiệm; nhân reward mà giữ β cố định thường làm đổi nghiệm.

```python
import math
import torch
from bandit import optimal
q=torch.tensor([.5,.5],dtype=torch.float64)
r=torch.tensor([math.log(3),0.],dtype=torch.float64)
p=optimal(q,r,1.)
assert torch.allclose(p,torch.tensor([.75,.25],dtype=torch.float64))
assert torch.allclose(p,optimal(q,r+100,1.))
assert torch.allclose(p,optimal(q,3*r,3.))
print(p.tolist())
```

## 7. Tính một cặp DPO

Raw gap `ln(.7/.2)=1.2527629685`; reference gap `ln(.6/.3)=.6931471806`; m=.5596157879. z=.2798078940, sigmoid=.5694991260, loss=.5629980309. Raw gap bỏ qua mốc ban đầu: có cặp reference đã rất thiên chosen; mức thay đổi ưu tiên tương đối với mốc mới là đại lượng của DPO trong bài.

```python
import math
m=math.log(.7/.2)-math.log(.6/.3)
z=.5*m
loss=math.log1p(math.exp(-z))
assert abs(m-math.log(1.75))<1e-12
assert abs(loss-.5629980308512323)<1e-12
print(m,z,1/(1+math.exp(-z)),loss)
```

## 8. Loss ban đầu ln2 nhưng có học không?

m=0, loss=ln2; đạo hàm theo m=−β/2, với β=.5 là−.25. Gradient qua policy còn tồn tại vì reference được giữ cố định. `reference=policy` chỉ tạo alias: freeze reference sẽ freeze policy; không freeze thì optimizer thay cả mốc tham chiếu. Cần deepcopy hoặc khôi phục hai đối tượng riêng. eval tác động layer chế độ train/eval; freeze tác động tham số; no_grad tác động graph của lượt tính.

```python
import torch
from objectives import dpo_loss
pc=torch.tensor([-.4],requires_grad=True)
pr=torch.tensor([-1.2],requires_grad=True)
loss,m=dpo_loss(pc,pr,pc.detach(),pr.detach(),.5)
loss.backward()
assert m.item()==0 and abs(pc.grad.item()+.25)<1e-7
assert abs(pr.grad.item()-.25)<1e-7
print(loss.item(),pc.grad.item(),pr.grad.item())
```

## 9. Kiểm gradient bằng sai phân số

m=.4, z=.28. Hai đạo hàm xấp xỉ −.301318 và+.301318. Từng đạo hàm là trung gian theo một logp, chưa là đạo hàm theo toàn logits/model parameters.

```python
import numpy as np
import torch
from objectives import dpo_loss
v=np.array([-.4,-1.2])
rc,rr=[torch.tensor([a],dtype=torch.float64) for a in [-.6,-1.]]
p=torch.tensor(v,dtype=torch.float64,requires_grad=True)
loss,_=dpo_loss(p[:1],p[1:],rc,rr,.7);loss.backward()
def oracle(a): return np.logaddexp(0,-.7*((a[0]+.6)-(a[1]+1.)))
fd=[]
for i in range(2):
    plus=v.copy();minus=v.copy();plus[i]+=1e-6;minus[i]-=1e-6
    fd.append((oracle(plus)-oracle(minus))/2e-6)
assert np.allclose(fd,p.grad.numpy(),atol=1e-8)
print(fd,p.grad.tolist())
```

## 10. Shift, mask và EOS

Input là `BOS USER route timeout ASSISTANT NET CHECK`; target là `IGN IGN IGN IGN NET CHECK EOS`. Các logit ở ASSISTANT/NET/CHECK lần lượt dự đoán NET/CHECK/EOS. V=29 cho score `−3ln29≈−10.10188749`. Thay logit tại target IGNORE không đổi phép sum trực tiếp, nhưng đổi prompt token thay hidden state/attention, nên có thể đổi logit completion. Mask target không cắt ảnh hưởng của context.

```python
import math
import torch
from data import Tokenizer,records,batch,IGNORE
from objectives import sequence_logps
t=Tokenizer(records()['train']);x,y=batch(t,[('route timeout','NET CHECK')])
a=torch.zeros(*x.shape,t.size,dtype=torch.float64)
score=sequence_logps(a,y)
assert t.size==29 and abs(score.item()+3*math.log(29))<1e-12
a[y==IGNORE]=123
assert torch.equal(score,sequence_logps(a,y))
print(score.item())
```

## 11. Sum và mean có thể đảo thứ tự không?

Sum A=−.2>B=−.24; mean A=−.1<B=−.08. Kiểm tất cả completion có ít nhất một target, EOS có vào score theo đúng quy ước, PAD không vào score. Thêm PAD hoặc gộp vào batch với completion dài hơn không được làm đổi score của cùng một ví dụ ngoài sai số float. Mean-token không phải loss DPO sigmoid sum của bài.

## 12. Preference tốt hơn nhưng chosen ít có khả năng hơn

Ban đầu m=0, loss=.693147. Sau m=ln1.5, sigmoid=.6, loss=.510826. Chosen giảm .4→.2, rejected giảm .3→.1 và C tăng .3→.7. Thông tin A>B chưa quyết định phải phân bố phần xác suất còn lại cho C thế nào.

```python
import math
q=[.4,.3,.3];p=[.2,.1,.7]
m=math.log(p[0]/q[0])-math.log(p[1]/q[1])
loss=math.log1p(math.exp(-m))
assert m>0 and p[0]<q[0]
assert abs(loss+math.log(.6))<1e-12
print(math.log(2),loss)
```

## 13. Chẩn đoán bandit reward hacking

RM chỉ biết ba phiếu thích feature1 hơn feature0 và một phiếu ngược. Fit cho w≈ln3, nhưng không có nhãn feature3; hàm tuyến tính cho điểm gấp ba feature1. Với β=.5, policy `[.3664495,.0399023,.5936482]`, utility=`.3664495−.5936482≈−.2271987`. Proxy reward tăng từ.582265 lên2.359154. β=5 giảm dịch chuyển trong ví dụ, chưa kiểm mọi hành vi hoặc bảo đảm an toàn; evaluator mới là một rule của toy world.

## 14. Chia dữ liệu và cache

Mỗi prompt có 4×3/2=6 cặp; 5 prompt có30cặp. Đây vẫn chỉ là một incident group. Split ở cấp incident, sau đó mới tạo cặp trong mỗi split. Cache cần đúng reference weights/revision, tokenizer/vocabulary, template, token IDs, EOS/mask, truncation/max length, completion, thứ tự/index record và reduction. Với lab còn ghi data hash và train hash. Tie/both bad cần trạng thái riêng hoặc quy trình xem lại, không đoán nhãn theo vị trí.

## 15. Đọc report TinyLM và thiết kế đánh giá

SFT bằng reference nên relative margin của mọi cặp là0. Nó vẫn sinh đúng route; chọn GUESS thay CHECK ở greedy làm exact quality=0 cho seed này. DPO đạt đủ chỉ tiêu nhỏ của toy task. Nó chưa chứng minh factuality, giữ năng lực cũ, chống injection, chất lượng tiếng Việt hoặc ưu thế hơn phương pháp khác.

Thêm baseline SFT chỉ học chosen, cùng dữ liệu/seed/budget phù hợp; đó là thí nghiệm đề xuất, chưa phải kết quả đã chạy. Các lát cắt hữu ích: incident/event mới, context thiếu/mâu thuẫn, câu hỏi tiếng Việt ngoài template. Rule lookup hiện đã giải được toàn bộ test này.

Bundle lưu/tải không đồng nghĩa resume optimizer. Nếu β/reference/tokenizer đổi, cần cấu hình thí nghiệm mới và tính lại cache tương ứng. Với strict loader của lab, metadata/reference/tokenizer mismatch bị từ chối. Ví dụ sau tạo bundle tối thiểu để thử validation, không giả làm file checkpoint huấn luyện:

```python
import copy
from data import digest
from model import state_hash
from training import validate_bundle
import torch
words=['toy'];reference={'w':torch.tensor([1.])}
m={'tokenizer_hash':digest(words),'reference_hash':state_hash(reference),'beta':.5}
b={'metadata':m,'words':words,'reference':reference}
validate_bundle(b,m)
for field in ['beta','words','reference']:
    bad=copy.deepcopy(b)
    if field=='beta':bad['metadata']['beta']=1.
    elif field=='words':bad['words']=['changed']
    else:bad['reference']['w'][0]=2.
    try:validate_bundle(bad,m)
    except ValueError:pass
    else:raise AssertionError('mismatch accepted')
print('3 mismatch cases rejected')
```

## 16. PPO, DPO và hệ thống thực thi

Hai clipped terms là2.4 và−1.6. πold dùng để thu rollout và làm mẫu số ratio PPO; πref làm mốc KL/reference của mục tiêu ưu tiên. Clip surrogate không cưỡng chế mọi xác suất nằm trong một khoảng cứng. Lab chỉ tối ưu enumerated bandit và DPO TinyLM, chưa chạy PPO.

Hệ thống vẫn cần xác thực, kiểm quyền trên tài nguyên/môi trường, review khi phù hợp, idempotency, budget/deadline, audit và kiểm kết quả theo hợp đồng công cụ. Trọng số model học từ preference không cấp quyền restart production. Xem lại bài26–33 nếu chưa phân biệt chất lượng đề xuất với thẩm quyền thực thi.
