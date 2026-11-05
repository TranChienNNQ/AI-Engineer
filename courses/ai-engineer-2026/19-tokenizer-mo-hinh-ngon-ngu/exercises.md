# Bài 19 — Bài tập

[Bài học](README.md) · [Lời giải](solutions.md)

## 1. Các đơn vị của văn bản

Giải thích code point, byte, token ID và embedding qua `é`. Vì sao `len(text)` không phải token count? Một emoji nhìn thấy có luôn là một code point không?

## 2. Không làm mất dữ liệu

Vì sao `text.split()` rồi `' '.join(...)` không đủ cho tokenizer code/log? Cho một ví dụ bị thay đổi ý nghĩa. Nêu chính sách normalization của lab.

## 3. BPE tính tay

Với corpus `aba×2, ac×1`, tính hai merge đầu, ID mới và tổng size. Giải thích quy tắc hòa và vì sao byte cũ vẫn phải giữ.

## 4. Encode sau fit

Dùng tokenizer bài 3, encode `abac`, `ab`, `a`, và `aba`. Vì sao encode hai phần riêng rồi nối không luôn bằng encode chuỗi ghép?

## 5. Cặp chồng lấn

Trong `aaaa`, (a,a) xuất hiện bao nhiêu cặp kề nhau? Một lượt merge không chồng lấn tạo bao nhiêu token? Đề xuất test cho implementation.

## 6. Special token

Phân biệt PAD/BOS/EOS/IGNORE; vì sao IGNORE không là embedding ID? Raw text `<BOS>` có tạo special ID không? EOS có bị bỏ loss không?

## 7. UTF-8 và decode

Với `é`, byte cơ sở và ID tương ứng là gì? Vì sao token đơn lẻ có thể decode lỗi dù encode/decode cả input đúng? Khi generation UTF-8 sai, có nên lặng lẽ bỏ byte lỗi không?

## 8. Xác suất chuỗi

Các xác suất nhãn A,B,EOS là 1/2,1/4,1/8. Tính xác suất cả chuỗi, NLL tổng, CE và PPL. PPL=4 có suy ra accuracy=25% không?

## 9. Mean loss và PPL

Hai batch có tổng NLL 4 trên 2 token và 6 trên 6 token. Tính CE/PPL toàn dataset. Vì sao không mean hai PPL batch? Uniform model V=323 có PPL bao nhiêu?

## 10. Shift và padding

Token nội dung một tài liệu là [10,11], tài liệu thứ hai [12]. Viết x,y sau BOS/EOS và right padding. Có tổng bao nhiêu nhãn được chấm? Có nên cho model nhìn EOS cần dự đoán tại cùng vị trí không?

## 11. Hợp đồng tokenizer/model

Hai tokenizer cùng size 261 có đủ tương thích checkpoint không? Lập danh sách artifact tối thiểu cần lưu và giải thích hash của lab bảo vệ phần nào.

## 12. Split trước khi fit

Một người fit BPE trên cả train/validation/test rồi chia window ngẫu nhiên trên một file log đã nối. Chỉ ra hai nguồn sai lệch và cách sửa. Nếu tokenizer pretrained đã cố định thì cách mô tả khác gì?

## 13. Context thực tế

Budget giả định 100 token, overhead 10, câu hỏi 20, output 30. Còn bao nhiêu token tài liệu? Nếu log dài 40 từ có được mặc định vừa không? Nêu một nguy cơ cắt mất ngữ cảnh trong stack trace.

## 14. Đọc kết quả lab

PPL test≈1.5167 nhưng generation có `đọc đọc cấu vụ`. Có mâu thuẫn không? Vì sao không báo exact match với một câu bất kỳ? Baseline add-one có giới hạn gì?

## 15. Kiểm implementation

Viết test roundtrip cho whitespace/emoji/Unicode tổ hợp/literal special, save/load tokenizer, future-token invariance, PAD invariance và tokenizer/checkpoint mismatch. Dùng input mới không có trong train.

## 16. Thực nghiệm có kiểm soát

Chỉ trên train, fit BPE với 0/16/64 merge; đo size, số token trên validation và byte/token. Có được chọn “model tốt hơn” bằng token count hoặc so PPL khác tokenizer trực tiếp không? Nếu đổi merge, các module nào phải tương thích lại?
