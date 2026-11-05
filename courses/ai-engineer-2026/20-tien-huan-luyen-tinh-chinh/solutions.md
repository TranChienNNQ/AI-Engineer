# Bài 20 — Lời giải

[Bài học](README.md) · [Bài tập](exercises.md) · [Phụ lục](deep-dive.md)

## 1. Mục tiêu

Corpus log: causal LM dự đoán các token nội dung/EOS từ prefix. Prompt–nhãn: SFT tăng likelihood answer đúng tại context yêu cầu. Pair tốt/xấu: dữ liệu ưu tiên, cần objective thích hợp để dùng so sánh; không tự là cùng bài toán CE một answer. Model viết log được thưởng cho pattern log, không được dạy nhất thiết trả nhãn khi thấy prompt route.

## 2. Baseline theo mục tiêu

Giọng văn: thử prompt/ví dụ context và rubric giữ riêng trước khi train. Số dư hiện tại: đọc nguồn giao dịch đúng quyền và xác minh, không dựa weight nhớ con số cũ. Route 6 event cố định: rule mapping rồi đo accuracy/coverage trên trường hợp ngoài luật; lab rule 100%. SFT là một lựa chọn khi có hành vi cần học từ mẫu, không mặc định mọi tác vụ đều cần model cập nhật weight.

## 3. CPT/SFT

Tiếp tục LM trên manual từ checkpoint đã học là CPT; prompt–JSON là SFT khi có target hành vi. Lợi ích CPT phụ thuộc dữ liệu/model/ngân sách; phải đo. Lab không chạy một stage CPT riêng sau base: nó pretrain model nhỏ từ random rồi tạo các nhánh SFT. Không mô tả pipeline lab thành tiền huấn luyện lớn rồi domain-adapt đã được kiểm.

## 4. Shift

```text
full: [BOS,10,11,20,21,EOS]
x:    [BOS,10,11,20,21]
y:    [IGNORE,IGNORE,20,21,EOS]
```

Mask P=2 nhãn đầu. Nhãn tại indexP là answer token đầu, query ở prompt token cuối. Mask thêm một vị trí sẽ bỏ 20; không dùng cách đếm cả BOS vào P rồi mask thừa. Có 3 nhãn được chấm, gồm EOS.

## 5. Loss

Full CE≈0.292232567; answer CE=1.5 ln 2≈1.039720771. Full CE thấp vì 8 token prompt dễ dự đoán làm mẫu số lớn và loss trung bình thấp. Không so trực tiếp hai objective như cùng metric chất lượng answer.

## 6. Gradient

p=[2/3,1/3], L=ln 3≈1.098612; gradient=[2/3, −2/3]. Sau update z'=[0.626480514,0.066666667], p'nhãn 1≈0.363590533>1/3. Logits là biến trung gian trong Transformer; optimizer cập nhật weight tạo ra chúng. Nếu dùng mean qua nhiều target, cần hệ số chia tương ứng.

## 7. Gradient prompt

Loss mask làm direct gradient tại logits ignore bằng 0, nhưng prompt hidden là key/value được answer đọc. Giữ gradient output embedding `[B, T, D]` bằng hook/retain_grad rồi kiểm từng vị trí; xem snippet phụ lục. Chỉ xem embedding.weight.grad có thể nhầm do cùng ID xuất hiện ở nhiều vị trí. Detach prompt khỏi context thay đổi bài toán, không là điều completion-only bắt buộc.

## 8. Boundary

BPE có thể merge qua ranh giới chuỗi nếu cùng một pre-token run. Lab prompt kết thúc whitespace, answer bắt đầu non-whitespace; regex giữ hai run tách, nên code assert encoding ghép bằng nối encoding từng phần. Template khác phải kiểm span answer sau tokenize và truncation, còn target/EOS, role marker có đúng và có bị mask nhầm không. Không suy từ ranh giới ký tự sang token bằng số ký tự tùy ý.

## 9. Freezing

Head=DV+V=32×323+323=10659; backbone=33699−10659=23040. Eval chỉ đổi hành vi module như dropout, không freeze. Requires_grad=False cho parameter frozen; optimizer chỉ nhận parameter trainable. No_grad toàn backbone khi có adapter cần học sẽ chặn graph; freezing parameter vẫn khác việc không ghi graph activation.

## 10. LoRA

Wx=[−1,5]; Ax=5; BAx=[1, −0.5]; output=[0,4.5] với α/r=1. W 1024 ² có 1048576 parameter; A/B r(m+n)=4×2048=8192, tỷ lệ 0.78125%. Base weight vẫn cần cho forward, activation và các phần optimizer/dtype khác còn tồn tại; đây không là benchmark tiết kiệm tổng bộ nhớ.

## 11. Khởi tạo

B=0, A≠0:BA=0, forward bằng base, dA=0 ở bước đầu, dB=cg(Ax)ᵀ có thể khác 0. A=B=0:cảhai gradient 0 với cấu trúc này. Sau merge W'=W+cBA, không cộng cùng update lần nữa. Khi so forward cần cùng scale, dtype và dropout thích hợp; lab chưa triển khai merge adapter Transformer thật.

## 12. Contamination

Pretrain đã đọc incident test thì có overlap dữ liệu qua stage, dù chưa có nhãn SFT. Không nói độc lập nếu mục tiêu holdout incident. Lab chia 48 cặp service–operation thành 32/8/8 nhóm rồi mới tạo 6 event/nhóm; pretrain/SFT/tokenizer fit đều chỉ train. Nó vẫn chia sẻ từ vựng/event/template và không kiểm unseen event hoặc production ngoài miền.

## 13. Kết quả

Trong cấu hình đã chạy, full đạt 100%, scratch 58.3333%, head-only 12.5%, majority 33.3333%, rule 100%. Full làm task tốt hơn hai neural control ở đây nhưng không hơn rule và không bảo đảm áp dụng rộng. Head-only còn dưới majority nên không mặc định ít parameter là đủ. Output head đổi làm phân phối log cũ đổi dù backbone nguyên. Scratch chưa học base, nên LM CE của scratch cao không gọi là quên base; full/head có thể so trước/sau cùng base.

## 14. Compute

12 update/epoch×25=300/stage;1 pretrain+3 SFT=1200. Base tốt nhất ởepoch 18, tương ứng 216 update checkpoint, nhưng đã tốn 300 update để chạy và chọn trong 25 epoch. Full có thêm pretrain compute trước 300 SFTupdate; scratch chỉ 300 SFTupdate. Đây là cùng ngân sách SFT, không cùng tổng compute.

## 15. Kiểm artifact

Inference cần weight/config, tokenizer, template/task protocol, decoding policy và nguồn gốc. Resume còn cần optimizer, RNG, step/scheduler nếu dùng, trạng thái stage/dữ liệu. Model head-sft load về không tự có flags freeze; thiết lập lại nếu train.

Đoạn độc lập kiểm cấu trúc:

```python
import torch
from core import ByteBPE, LanguageModel, Config, BOS, PAD
from copy import deepcopy
from lab import model_hash

torch.manual_seed(20)
model = LanguageModel(ByteBPE().size, Config()).double().eval()
original_hash = model_hash(model)
branch = deepcopy(model)
with torch.no_grad():
    branch.head.bias.add_(1)
assert model_hash(model) == original_hash
assert model_hash(branch) != original_hash
x = torch.tensor([[BOS,100,101,102,103]])
future = x.clone(); future[:,3:] = torch.tensor([[104,105]])
with torch.no_grad():
    torch.testing.assert_close(model(x)[:,:3], model(future)[:,:3])
    padded = torch.cat([x,torch.full((1,3),PAD)],1)
    torch.testing.assert_close(model(x),model(padded)[:,:5])
print('PASS clone isolation, causal, padding')
```

So weight frozen trước/sau stage thực bằng tensor equality, không chỉ xem grad=None. Kiểm tokenizer mismatch bằng load checkpoint với tokenizer khác; không sửa artifact gốc. Report parent_hash full/head phải bằng hash base, scratch parent bằng initial_hash; phụ lục có đoạn kiểm run.

## 16. Mở rộng có kiểm soát

Một phương án là thêm một tỷ lệ LM train cũ vào objective khi SFT, giữ split/model/seed/budget rõ ràng. Xác định trước giới hạn tăng validation LM CE và tiêu chí routing validation; chọn trong những cấu hình đạt giới hạn ấy. Log số update lẫn số target token từng objective; trộn nhiều dữ liệu có thể thay đổi ngân sách, không chỉ một tham số.

Giữ baseline fullSFT không trộn, đo routing exact/format và LM CE trên cùng validation. Sau khi chốt mới đánh giá test phù hợp. Một seed chỉ cho quan sát cục bộ; muốn kết luận chắc hơn cần lặp seed và tập khác. Không chọn tỷ lệ trộn bằng test rồi coi test đó vẫn chưa dùng. Phần mở rộng này là đề xuất thí nghiệm, chưa có implementation đã chạy hoặc số liệu được tuyên bố trong bài.
