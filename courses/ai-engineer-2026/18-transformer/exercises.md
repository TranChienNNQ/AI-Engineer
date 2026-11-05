# Bài 18 — Bài tập Transformer

[Bài học](README.md) · [Lời giải](solutions.md) · [Phụ lục](deep-dive.md)

Hãy tự vẽ và dự đoán số trước khi chạy. Bài 1–8 kiểm nền tảng; 9–13 kiểm implementation; 14–16 kiểm thực nghiệm. Không cần GPU hoặc mô hình tải ngoài.

## 1. Vẽ đường đi và suy shape

B=2, T=5, V=8, D=12, H=3, F=24, N=2. Ghi shape ID, embedding, Q/K/V sau tách head, attention score, output ghép head, FFN hidden và logits. Sau block số token có đổi không?

## 2. Residual và gradient

X=[2,−1], F(X)=[0.3,0.5]. Tính X+F(X). Nếu F(x)=2x ở bài toán scalar, L=Y²/2, Y=x+F(x), x=1 thì dL/dx bằng bao nhiêu? Vì sao không chỉ đi qua nhánh F?

## 3. LayerNorm tính tay

Với x=[1,2,3,4], γ=1, β=0, ε=0: tính μ, variance và output. Nếu x=[5,5,5,5] với ε=1e−5 thì output là gì? Vì sao không dùng variance chia D−1?

## 4. Gradient LayerNorm

Với ví dụ không hằng của bài 3 và L là feature đầu output, tính gradient input bằng công thức phụ lục. Gradient có chỉ nằm ở feature đầu không? Nó có lan sang token khác chỉ do `LayerNorm(D)` không?

## 5. FFN tính tay

Dùng W₁,W₂ và bias=0 của bài chính. Với u=[2,−1], tính uW₁, ReLU, FFN(u) và u+FFN(u). FFN có đọc hàng token bên cạnh không?

## 6. Đổi ví dụ tính tay sang causal

Giữ toàn bộ weight của block tính tay, nhưng token 0 chỉ được đọc token 0, token 1 đọc cả hai. Tính attention output, norm thứ nhất và output cuối. Nếu output cuối giống ví dụ không mask, có thể kết luận causal mask không có tác dụng không?

## 7. Pre-norm và post-norm

Nếu cả nhánh attention và FFN trả 0, output của một block pre-norm là gì? Post-norm là gì? Final norm của stack có làm hai kiến trúc tổng quát trở thành tương đương không?

## 8. Vị trí và hoán vị

Tính PE ở t=0 và t=1 khi D=4. Giải thích ba phép thử: hoán vị token không PE; hoán vị token nhưng giữ PE theo vị trí; hoán vị cả token và PE. Kết luận không PE có áp dụng nguyên vẹn khi causal mask giữ cố định không?

## 9. Shift và mask cho task copy

Source `[A,B,C]`: viết full sequence, input và labels có IGNORE. Query nào dự đoán A copy đầu tiên? Có bao nhiêu token được chấm? Tạo causal blocked mask cho 4 vị trí đầu, giải thích đường chéo.

## 10. Loss tính tay và tổng hợp batch

Hai token được chấm có xác suất nhãn đúng 1/2 và 1/4; ba vị trí khác là IGNORE. Tính CE mean. Nếu batch thứ nhất có 2 token, tổng CE=2; batch thứ hai có 6 token, tổng CE=12, CE toàn dataset là gì? `ignore_index` có loại class PAD khỏi softmax không?

## 11. Đếm tham số

D=12,H=3,F=24,N=2,V=8; MHA/Linear có bias, hai LN mỗi block, final LN, head không tying. Tính số tham số mỗi thành phần và toàn mô hình. Số tham số có phụ thuộc T và số vị trí PE cố định không?

## 12. Thiết kế kiểm nhân quả và padding

Viết code kiểm: (a) sửa input tương lai không đổi logits prefix của model hai tầng; (b) thêm PAD bên phải không đổi logits token thật; (c) gradient từ loss prefix tới future feature bằng 0. Vì sao không yêu cầu output PAD query bằng 0? Kiểm `LayerNorm([T,D])` có thể phá nhân quả dù attention đúng mask.

## 13. Encoder–decoder và ba mask

B=2,S=4,T=3,D=8,H=2. Ghi shape encoder memory, Q/K/V trong cross-attention sau split head, cross score, tgt mask và source padding mask. Tại sao vẫn cần `memory_key_padding_mask`? Đầu vào `nn.Transformer` có sẵn embedding không?

## 14. Đánh giá số liệu đã chạy

Model có token accuracy test=814/855 và greedy exact=94/128. Tính hai tỷ lệ và giải thích vì sao khác nhau. Vì sao baseline quy tắc 100% không làm lab vô nghĩa nhưng ngăn kết luận model là giải pháp tốt nhất cho chép chuỗi? Epoch cuối là 60, best là 58: dùng checkpoint nào?

## 15. Sửa một pipeline bị rò rỉ

Một người dùng full sequence làm cả input và labels cùng vị trí, chỉ đặt causal mask; tính CE gồm PAD; sau đó báo “generation accuracy” bằng argmax trên batch teacher-forced và chọn epoch test tốt nhất. Chỉ ra từng lỗi, thứ tự sửa và cách kiểm pipeline sau sửa.

## 16. Thực nghiệm có kiểm soát

Chạy cấu hình mặc định để có run riêng. Sau đó chỉ dùng train/validation để thử một thay đổi (N=1 hoặc FFN=32 chẳng hạn), giữ split, seed và cách đánh giá. Báo tham số, validation CE, token accuracy, exact match và số update. Có thể kết luận nhân quả chắc chắn từ một seed không? Muốn thử source dài 7–8 cần thay gì, và vì sao PE đủ dài chưa bảo đảm kết quả tốt?

## Mở rộng tùy chọn

- Thêm dropout đúng vị trí và kiểm train/eval; đừng kỳ vọng khớp số liệu lab gốc.
- Thay fixed PE bằng learned PE, thiết kế kiểm vị trí và giới hạn length.
- Viết phiên bản attention nhiều head bằng projection + functional SDPA bài 17, so forward/backward với Block cùng trọng số.
- Thêm report số chuỗi sai do EOS sớm, sai ký hiệu hoặc hết budget; giữ exact match gốc để tránh đổi metric sau khi nhìn lỗi.
