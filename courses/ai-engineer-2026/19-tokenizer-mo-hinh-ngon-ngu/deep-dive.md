# Bài 19 — Đào sâu và ứng dụng trong hệ thống backend

[Bài học](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Ba tình huống cần quyết định bằng sản phẩm

### Đọc ticket hỗ trợ

Input `Mã đơn DH-0012 bị lỗi, không phải DH-0013` có hai ID rất gần nhau. Nếu tiền xử lý bỏ chữ số hoặc cắt cuối, model mất chi tiết quyết định. Hãy giữ input nguyên gốc, lưu phiên bản tokenizer và chính sách truncation, sau đó mới tạo một bản biểu diễn phục vụ tìm kiếm nếu cần. Tokenizer không phải công cụ sửa nội dung ticket.

Một completion “đã hoàn tiền” không chứng minh đã có giao dịch. Khi xây API hành động phải kiểm trạng thái từ hệ thống giao dịch, không chỉ tin câu sinh. Đây là suy luận thiết kế từ sự khác biệt giữa dự đoán văn bản và xác minh dữ liệu, không phải đánh giá thực nghiệm của lab.

### Gợi ý code/JSON

Đổi indent hoặc dấu nháy có thể làm code/JSON không parse được. Decode thành chuỗi rồi chạy parser và schema phù hợp là phép kiểm riêng. Tokenizer roundtrip đúng chỉ bảo đảm đường chuyển đổi dữ liệu của nó, không bảo đảm model sinh JSON hợp lệ. Trong raw text, `<EOS>` chỉ là chữ ở tokenizer bài này; không dùng tìm chuỗi bằng substring để mô phỏng special ID.

### Trợ lý đọc log dài

Đếm token sau khi thêm ngữ cảnh và template. Giả sử budget minh họa 100, overhead 10, câu hỏi 20, output dự trù 30 thì còn 40 token cho log. Nên ưu tiên sự kiện liên quan và giữ nguồn/chỉ số dòng, thay vì cắt theo 40 từ. Đừng hard-code tỷ lệ “mỗi từ bằng N token” cho cả tiếng Việt, code và URL. Nếu dùng tokenizer khác lab, phải đo lại; bài này không cung cấp số token của dịch vụ thương mại.

## 2. Đếm token không chỉ là `len(text)`

Đo nhiều đơn vị trên cùng dữ liệu và cùng tokenizer:

```python
from lab import ByteBPE, data
tok = ByteBPE.fit(data()['train'])
for text in ['api timeout', 'API_TIMEOUT_504', 'Xin chào 🙂', '{"x": 1}', '  a\n\tb']:
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    print(repr(text), {'code_points': len(text), 'utf8_bytes': len(text.encode()), 'tokens': len(ids)})
```

Không có quy tắc rằng hai văn bản cùng số code point sẽ có số token bằng nhau. Byte vocabulary ngăn OOV cho UTF-8 input, không đảm bảo mọi piece là Unicode riêng hoàn chỉnh. Ví dụ `é` là byte C3 A9: hai byte ID đơn lẻ 198 và 172 chưa tự decode strict riêng lẻ, nhưng ghép theo đúng thứ tự decode được `é`.

## 3. Vì sao chỉ lưu vocabulary size là thiếu?

Tokenizer là mapping ID→bytes cùng quy tắc biến đổi input→ID. Checkpoint embedding hàng 260 được học với ý nghĩa của ID 260 lúc train. Nếu một tokenizer khác gán hàng đó cho piece khác, shape vẫn đúng nhưng ý nghĩa input bị thay đổi. Đây là lỗi hợp đồng dữ liệu khó thấy vì code vẫn chạy.

Lab lưu merge list có thứ tự, version, normalization và pre-tokenizer. Hash merge list được ghi trong checkpoint; các quy tắc base byte/special ID được cố định trong code version này. Hash không là chữ ký bảo mật hoặc xác nhận nguồn tin cậy; nếu thay code semantics mà giữ metadata cũ, kiểm hash không bảo vệ được. Hệ thống thật nên version toàn bộ preprocessing contract, không chỉ vocabulary.

Một phép thử sai khác cùng size:

```python
from lab import ByteBPE
a = ByteBPE.fit(['aba', 'aba', 'ac'], steps=2)
b = ByteBPE.fit(['xyx', 'xyx', 'xz'], steps=2)
assert a.size == b.size == 261
assert a.fingerprint() != b.fingerprint()
assert a.raw([260]) != b.raw([260])
```

## 4. Đánh giá LM theo token và theo byte

PPL/token phụ thuộc cách phân đoạn. Ví dụ một văn bản có NLL tổng giả định 12 nats. Tokenizer A tạo 6 token → CE=2, PPL≈7.389; tokenizer B tạo 12 token → CE=1, PPL≈2.718. Chỉ đổi mẫu số đã làm chỉ số khác, chưa có bằng chứng model B tốt hơn. Đây là ví dụ giả định để so đơn vị, không là kết quả hai model được train.

Có thể báo NLL chia số byte UTF-8, rồi đổi sang bits/byte bằng chia thêm ln2 khi cần một đơn vị độ dài khác. Phải thống nhất dữ liệu, normalization, BOS/EOS, reset context và likelihood đang chấm. Nếu NLL có EOS nhưng mẫu số chỉ có byte nội dung, cần công bố quy ước đó. Không biến phép đổi đơn vị thành thước đo ngữ nghĩa hay factuality.

Nâng cao: LM gán xác suất cho dãy token. Một chuỗi byte có thể được tạo từ nhiều dãy ID khác nhau (ví dụ `[a,b]` và `[ab]`). NLL trên **canonical encoding** của tokenizer không tự là tổng xác suất của mọi dãy token decode thành cùng chuỗi văn bản. Trong lab, metric được định nghĩa rõ là NLL/PPL trên canonical tokenization, không cộng mọi cách phân đoạn.

## 5. Context window và tính công bằng của phép đo

Trong lab mỗi câu ngắn vừa 128 token và độc lập nên không dùng sliding window. Với tài liệu dài, chia thành chunk rời làm token đầu chunk mất context trước đó. Sliding window có thể giữ thêm context, nhưng phần chồng lấn không nên bị chấm lặp ngoài chủ đích. Khi so số liệu cần báo giới hạn context, stride và cách tính target được chấm. [Hướng dẫn PPL của Hugging Face](https://huggingface.co/docs/transformers/main/en/perplexity).

Đừng ghép train và test rồi mới tạo window: hai window gần nhau có thể chia sẻ hầu hết nội dung qua split. Chia theo tài liệu/nhóm/thời gian phù hợp trước; fit tokenizer trên train khi nghiên cứu tokenizer mới. Nếu dùng pretrained tokenizer cố định, công bố nguồn/phạm vi đã biết thay vì giả rằng mình đã làm nó train-only trên corpus riêng.

## 6. Phân biệt ba loại “học được”

- Tokenizer học được merge `timeout` không đồng nghĩa hiểu timeout là gì; chỉ là pattern byte thường gặp.
- Embedding học được vector cho token không đồng nghĩa vector đó có một nghĩa từ điển cố định ở mọi context.
- Transformer LM học tiếp diễn nhiều context không đồng nghĩa có công cụ đọc production hoặc được huấn luyện trả lời mọi yêu cầu người dùng.

Dữ liệu tự giám sát không cần người gán từng nhãn next-token, nhưng vẫn cần chọn lọc nguồn, phân quyền sử dụng, chất lượng và mục tiêu đo. Một corpus đầy dòng template dễ đoán làm PPL thấp mà ít kiểm tra khả năng giải quyết vấn đề thật. Đó là lý do mẫu sinh lỗi trong bài được giữ nguyên cạnh metric.

## 7. Đọc artifact sau khi chạy

```python
# Sau lệnh train ở bài chính; đổi đường dẫn nếu run của bạn ở nơi khác.
from pathlib import Path
import json
import math
from lab import load, evaluate
run = Path('/tmp/lesson19-run-01')
model, tok, cfg, checkpoint = load(run)
records = json.loads((run / 'data.json').read_text())
report = json.loads((run / 'report.json').read_text())
measured = evaluate(model, records['test'], tok, cfg)
assert measured == report['test']
assert math.isclose(measured['perplexity'], math.exp(measured['ce']))
print(checkpoint['epoch'], tok.size, measured)
```

Đây là tái kiểm artifact đã chọn, không dùng test để điều chỉnh model. `best.pt` không đủ resume vì thiếu optimizer/RNG/trạng thái shuffle. Dùng pattern bài 12 nếu muốn resume thật. Không cố tái tạo số liệu bằng cách train lại rồi chọn run có test đẹp nhất.
