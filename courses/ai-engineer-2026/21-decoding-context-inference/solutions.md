# Bài 21 — Lời giải và cách kiểm

[Bài học](README.md) · [Bài tập](exercises.md) · [Phụ lục](deep-dive.md)

## 1. Vòng inference

Render template → tokenize → kiểm budget → prefill → chọn ID từ logits cuối → dừng hoặc nối ID/feed token mới → decode và validate output. Temperature chỉ biến đổi score khi inference, không cập nhật weight. Hidden tại token prompt cuối được dùng để dự đoán token kế tiếp nên đã có phân phối output đầu sau prefill.

## 2. Temperature

τ=0.5: [0.724638,0.181159,0.065217,0.028986]. τ=2: [0.370090,0.261693,0.202707,0.165509]. Tính bằng p^(1/τ) rồi chia tổng. Argmax vẫn A với τ>0 khi thứ hạng không hòa; greedy không đổi theo biến đổi này. τ=0 không hợp lệ trong phép chia; lab có nhánh greedy riêng.

## 3. Top-k

Giữ A, B với mass 0.75, q=[2/3,1/3,0,0]. Lab sort stable giảm dần và hòa ưu tiên ID nhỏ, giữ đúng k vị trí; các ID vốn bị mask vẫn có xác suất 0. Không áp quy tắc hòa này cho mọi thư viện vì có cách lọc theo ngưỡng thay vì exact-k.

## 4. Top-p

p 0.7 giữ A, B vì 0.5 chưa đủ,0.75 đã đủ. p 0.8 giữ A, B, C vì 0.75 chưa đủ,0.9 đã đủ; q=[5/9,5/18,1/6,0]. Giữ cả token làm vượt ngưỡng. Nucleus size phụ thuộc phân phối hiện tại, không cố định số candidate. Code đảm bảo ít nhất một candidate.

## 5. Filter order

Sau k 2, A có 2/3≥0.6 nên nucleus chỉ còn A. Nếu top-p 0.6 trực tiếp, A 0.5 chưa đủ nên giữ A, B. Model probability 0.5 của A không tự thành model certainty 1 dù policy lọc chỉ cho phép A. Khi ghi logprob cần ghi phân phối đang dùng.

## 6. Sampling

Tần suất hữu hạn có dao động ngẫu nhiên; seed chọn một chuỗi RNG cụ thể, không thay định luật phân phối. Cùng seed không bảo đảm cùng output sau đổi model, kernel hoặc dtype. Kiểm cùng generator/config cho cùng sequence trong cùng môi trường; kiểm candidate bị lọc không bao giờ xuất hiện và tần suất nhiều mẫu xấp xỉ q. Không dùng một sample để kiểm cả phân phối.

## 7. Beam

Greedy A, X 0, EOS: probability 0.06, logprob≈−2.813411. Beam 2 B, X 0, EOS: probability 0.36, logprob≈−1.021651. Duyệt 12 chuỗi của cây cho thấy 0.36 là lớn nhất ở đây. Beam hữu hạn vẫn có thể prune prefix dẫn tới optimum ở cây khác. Lab dùng logprob tổng thô, không length penalty; các chuỗi hoàn chỉnh trong ví dụ đều dài 3 token.

## 8. Kết thúc

Length chỉ nói hết budget; JSON có thể chưa parse được. Không coi auto-repair là output model nguyên bản hoặc bằng chứng tác vụ thành công. EOS là ID kết thúc dự đoán; stop string là điều kiện trên văn bản/byte cần matcher thích hợp; max_new_tokens giới hạn số ID mới. Stop string có thể nằm qua nhiều token/chunk, không kiểm độc lập từng chunk.

## 9. Budget

Input 70+output 30=100, vừa policy. Input 71+30=101 nên báo vượt hoặc thay chính sách rõ ràng. Prompt count gồm BOS trong lab. Forward dài nhất P+N−1 vì ID cuối vừa sinh chưa cần feed lại; policy reserve P+N≤C bảo thủ thêm một slot. Không bê chi tiết ấy sang mọi API.

## 10. Mask

Allowed mask là `[[T,T,T,T,F],[T,T,T,T,T]]`. Query cuối với score đều nhau đọc cả 4 value nên mean=2.5. Upper-left causal 1×4 chỉ cho đọc cột 0, output 1.0. SDPA không biết past length nếu chỉ đưa is_causal=True. SDPA boolTrue=cho đọc; MHA boolTrue=chặn. Chỉ một dấu đảo mask có thể làm toàn kết quả khác.

## 11. Shape

Q `[1,4,2,8]`; K, V sau nối `[1,4,7,8]`; score `[1,4,2,7]`; output mỗi head `[1,4,2,8]`; ghép `[1,2,32]`; head `[1,2,V]`. Position ID mới là 5,6. Không cần Q cũ để tính attention query mới; nhiều layer cần K, V riêng từng layer.

## 12. Memory

2×32×1×8×4096×128×2=536870912 byte=512 MiB. Batch 4→2147483648 byte=2 GiB. Chưa gồm model weight, workspace, allocator và activation khác. Cache không tăng giới hạn context, chỉ tái sử dụng kết quả prefix phù hợp. Với kiến trúc H_KV khác H_Q phải dùng số KV head thực, không tự lấy số query head.

## 13. Lượng vị trí

No-cache=16+17+…+23=156; cache=16+7=23. Đây là đếm projection/FFN positions, không là speedup đo thời gian; query mới vẫn đọc nhiều KV và cache concat còn copy. Nếu prefill chọn EOS ngay, dừng không cần decode forward; cache chỉ giữ prompt.

## 14. Streaming và telemetry

Incremental UTF-8 decoder nhận 195 trước trả chuỗi rỗng vì chưa đủ ký tự; nhận 169 trả `é`; flush cuối trả rỗng nếu không còn byte thiếu. Decode strict từng byte riêng sẽ lỗi.

TTFT=120 ms; TPOT=(250−120)/(4−1)≈43.333 ms; end-to-end=250 ms nếu request hoàn thành ở timestamp cuối. Chỉ 1 token thì TPOT theo công thức chưa xác định, nên báo None/N/A. Các timestamp là giả định, không phải benchmark lab. Throughput cần cửa sổ thời gian và concurrency riêng.

## 15. Kiểm cache

So logits cùng weight/input/dtype ở prefill; tách từng token hoặc nhiều cách chia chunk rồi nối logits để so full forward. Kiểm shape/length tăng đúng; prefix K/V cũ không đổi; sửa token tương lai không đổi logits prefix; position offset đúng. So argmax chưa đủ vì score sai vẫn có thể giữ token đứng đầu.

Đoạn độc lập từ thư mục bài 21:

```python
import torch
from core import LanguageModel, Config, BOS
from lab import cached_forward

torch.manual_seed(21)
model = LanguageModel(259,Config()).double().eval()
ids = torch.tensor([[BOS,100,101,102,103,104]])
with torch.no_grad():
    _, cache = cached_forward(model,ids[:,:3])
    saved_k, saved_v = cache['k'].clone(), cache['v'].clone()
    new, extended = cached_forward(model,ids[:,3:],cache)
    torch.testing.assert_close(extended['k'][:,:,:3],saved_k,rtol=0,atol=0)
    torch.testing.assert_close(extended['v'][:,:,:3],saved_v,rtol=0,atol=0)
    torch.testing.assert_close(new,model(ids)[:,3:],rtol=1e-10,atol=1e-10)
other = LanguageModel(259,Config()).double().eval()
try:
    cached_forward(other,ids[:,3:],cache)
except ValueError:
    pass
else:
    raise AssertionError('Cache owner mismatch was accepted')
print('PASS prefix cache, chunk logits, owner guard')
```

Không dùng cache cho model khác hoặc sau sửa prefix/weight mà không xử lý invalidation. Owner tag chỉ kiểm object identity, không tự chứng minh prefix còn đúng. Lab không hỗ trợ PAD/batch đa chiều dài hay cache chia sẻ.

## 16. Phép đo ứng dụng

Route: bắt đầu greedy, đo exact nhãn+EOS, valid format, accuracy theo lớp, trường hợp không đủ dữ kiện; so rule/classifier baseline thích hợp. Email mở: greedy là baseline, sampling có thể thử vài config; đo rubric, đúng dữ kiện và tính hữu ích, không ép một continuation tham chiếu duy nhất thành đáp án tuyệt đối.

Chọn decoder bằng validation; sau khi chốt mới dùng test phù hợp. Khi so cache/no-cache giữ model, tokenizer, prompt IDs, decoding policy và RNG; ưu tiên kiểm logits rồi output. Khi đo thời gian cần warmup, repetitions, thiết bị, length và concurrency rõ ràng. Demo NET của một prompt/seed không chứng minh bốn strategy giống nhau trên cả miền hoặc cache nhanh hơn một tỷ lệ cố định.
