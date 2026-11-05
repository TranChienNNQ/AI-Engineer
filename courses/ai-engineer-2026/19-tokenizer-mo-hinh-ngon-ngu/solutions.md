# Bài 19 — Lời giải

[Bài học](README.md) · [Bài tập](exercises.md) · [Phụ lục](deep-dive.md)

## 1. Đơn vị

`é` dựng sẵn có 1 code point và 2 byte UTF-8; token count phụ thuộc merge, có thể là các byte cơ sở hoặc một piece gộp. ID chỉ chọn hàng embedding; embedding là vector model học. Dạng `e` + dấu sắc có 2 code point. Emoji ghép có thể có nhiều code point. `len(str)` của Python không làm tokenization hoặc phân đoạn grapheme.

## 2. Dữ liệu

`'  a\n\tb'.split()` mất số space, newline và tab; nối lại thành `a b`. Indent code có thể đổi ý nghĩa hoặc gây lỗi. Lab giữ nguyên byte, không normalize và dùng regex giữ cả whitespace run. NFC có thể là chính sách riêng nếu được thống nhất, nhưng khi đó roundtrip phải hiểu theo chuỗi đã normalize, không hứa giữ nguyên input.

## 3. Merge

Cặp (100,101) và (101,100) đều có tần suất 2, chọn tuple nhỏ hơn (100,101) →259=`ab`. Sau đó (259,100) tần suất 2 →260=`aba`. Base 259 cộng 2 bằng 261. Byte cũ dùng cho chuỗi mới hoặc phần không merge; tạo token mới không thay nghĩa ID cũ.

## 4. Encode

`abac→[260,102]`, `ab→[259]`, `a→[100]`, `aba→[260]`. `[259]+[100]` khác `[260]` dù cùng decode `aba`. BPE áp ordered merge trên từng run, nên ranh giới ghép mới có thể tạo merge trước đó chưa có. Trong generation giữ prefix ID và nối ID mới, không tự retokenize prefix sau mỗi bước.

## 5. Chồng lấn

Có 3 cặp kề nhau, nhưng thay không chồng lấn từ trái sang phải cho `[aa,aa]`, tức 2 token. Test: `ByteBPE.merge([100]*4,(100,100),259)==[259,259]`. Sau một lượt không tự ghép `[259,259]` nếu chưa có rule tiếp theo cho cặp ấy.

## 6. Control và nhãn

PAD=0 là điền input; BOS=1 bắt đầu; EOS=2 kết thúc và được chấm; IGNORE=−100 bỏ vị trí loss, không phải token có embedding. `<BOS>` raw text không tạo ID 1. Lab thêm BOS/EOS bằng code; tokenizer decode bỏ special ID. Vì vậy roundtrip raw text đúng không đồng nghĩa decode giữ literal tên special cho các control ID được chèn riêng.

## 7. Byte

`é` UTF-8 [195,169], ID cơ sở [198,172]. Mỗi byte riêng là một phần của chuỗi đa byte nên có thể không decode strict được. Ghép raw byte đúng rồi decode; khi output model sai, báo invalid UTF-8 và dùng replacement để hiển thị nếu cần. Không lặng lẽ xóa byte và gọi kết quả là bản sinh nguyên gốc.

## 8. Likelihood

P=1/64=0.015625, NLL=ln64≈4.158883, CE=ln4≈1.386294, PPL=4. Không suy ra top-1 accuracy; probability nhãn và vị trí xếp hạng nhãn khác nhau. Trong ví dụ, chấm EOS, không chấm BOS. `lab.py --mode hand` kiểm số.

## 9. Gộp metric

CE=(4+6)/(2+6)=1.25; PPL=e^1.25≈3.490343. Mean PPL batch=(e²+e¹)/2≈5.053669 là phép đo khác và sai cho NLL/token tổng. Uniform V=323 có PPL=323 khi chấm mọi token trên full vocabulary; CE=ln323. Không đưa mean accuracy vào công thức PPL.

## 10. Batch

```text
x: [BOS,10,11]      y: [10,11,EOS]
   [BOS,12,PAD]       [12,EOS,IGNORE]
```

Có 5 nhãn. EOS không nằm tại vị trí input đang đoán chính EOS; shift một vị trí. EOS đã có trong full sequence để tạo y, không có nghĩa được đọc như feature tại query đó. PAD input mask và IGNORE label mask có chức năng khác.

## 11. Artifact

Không đủ cùng size: ID mapping có thể khác. Cần tokenizer đầy đủ (merge, base mapping, normalization, pre-tokenizer, special policy/version), model config và weight, cùng cách đóng khung/shift/decode. Lab lưu tokenizer.json và best.pt có config cùng hash merge; kiểm version/pre-tokenizer/normalization khi load. Hash không chứng minh artifact đáng tin hoặc chống mọi thay đổi code. Checkpoint lab không đủ resume.

## 12. Leakage

Fit tokenizer trên test làm test tham gia thiết kế representation. Chia window sau nối file có thể làm các window train/test trùng nhiều nội dung. Chia tài liệu/nhóm/thời gian phù hợp trước, fit merge train-only rồi freeze và tạo batch trong từng split. Với pretrained tokenizer dùng cố định, công bố nguồn/phiên bản và không nói nó được train-only trên corpus riêng nếu không có bằng chứng. Ngoài trùng exact còn cần xét trùng gần và dữ liệu theo nguồn.

## 13. Budget

Còn 40 token. Không biết 40 từ có vừa không cho tới khi encode bằng đúng tokenizer sau template. Stack trace có thể bị cắt mất exception gốc, quan hệ cause hoặc tên hàm; cần chọn phần liên quan và công bố truncation. Các số là giả định, không phải giới hạn dịch vụ thật.

## 14. Chất lượng

Không mâu thuẫn: nhiều token template rất dễ dự đoán dưới teacher forcing; khi tự sinh, prefix do model tạo có thể dẫn tới lặp/trộn. PPL đo likelihood trên test đã tokenize, không kiểm factuality hay cú pháp mọi câu sinh. Prompt mở có nhiều continuation hợp lệ nên một reference tùy chọn không tạo exact-match metric phù hợp. Add-one trên 323 class làm smoothing khá mạnh, nhất là context bigram ít mẫu; baseline này không đại diện tối ưu mọi n-gram.

## 15. Kiểm code

Đoạn độc lập sau chạy từ thư mục bài 19; kiểm artifact mismatch có ở phần kiểm run của tài liệu, cần dùng run của mình để thử thay tokenizer.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from lab import ByteBPE, LanguageModel, Config, BOS, PAD

torch.manual_seed(19)
tok = ByteBPE.fit(['aba', 'aba', 'ac'], steps=2)
for text in ['  mới\n\t🙂', 'e\u0301', '<BOS>', 'GET /v2?a=1', '']:
    assert tok.decode(tok.encode(text)) == text
assert ByteBPE.merge([100]*4, (100,100),259) == [259,259]
with TemporaryDirectory() as folder:
    path = Path(folder) / 'tokenizer.json'
    tok.save(path)
    restored = ByteBPE.load(path)
    assert restored.fingerprint() == tok.fingerprint()
    assert restored.encode('abac') == [260,102]
model = LanguageModel(tok.size, Config()).double().eval()
a = torch.tensor([[BOS,100,101,100]])
b = a.clone(); b[:,3] = 102
with torch.no_grad():
    torch.testing.assert_close(model(a)[:,:3], model(b)[:,:3])
    padded = torch.cat([a,torch.full((1,2),PAD)],1)
    torch.testing.assert_close(model(a),model(padded)[:,:4])
print('PASS roundtrip, save/load, causal, padding')
```

Để kiểm mismatch: sao chép run vào thư mục tạm, thay tokenizer.json bằng tokenizer khác có 64 merge/cùng size rồi gọi `load`; phải báo tokenizer/checkpoint mismatch. Không ghi đè tokenizer của run gốc. Roundtrip cần input ngoài train; model random cũng phải vượt kiểm nhân quả, vì đó là thuộc tính cấu trúc.

## 16. Thí nghiệm tokenizer

```python
from lab import ByteBPE, data
records = data()
for steps in [0,16,64]:
    tok = ByteBPE.fit(records['train'], steps=steps)
    tokens = sum(len(tok.encode(s)) for s in records['validation'])
    byte_count = sum(len(s.encode()) for s in records['validation'])
    print({'merges': steps, 'vocab': tok.size, 'validation_tokens': tokens,
           'bytes_per_token': byte_count/tokens})
```

0 merge: mỗi byte một token nên byte/token=1 trên nội dung; 16/64 merge tạo 275/323 mục nếu corpus đủ cặp. Đo được chuỗi ngắn hơn chỉ cho biết hiệu quả phân đoạn trên miền này. Muốn so model cần train/evaluate có thiết kế phù hợp, không so PPL/token khác tokenizer trực tiếp. Khi đổi mapping phải xây/load embedding và output head tương thích; nếu giữ D/context thì các block có thể cùng shape nhưng không tự bảo đảm weight đã học vẫn có nghĩa đúng. Bài tập không yêu cầu công bố một biến thể LM chưa được chạy.
