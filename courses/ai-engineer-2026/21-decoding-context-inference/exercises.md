# Bài 21 — Bài tập decoding, context và inference

[Bài học](README.md) · [Lời giải](solutions.md)

## 1. Vòng inference

Vẽ các bước từ prompt đến output cuối. Weight có thay đổi khi chỉnh temperature không? Vì sao logits cuối prefill đã đủ chọn token output đầu?

## 2. Temperature tính tay

Với p=[0.5,0.25,0.15,0.1], chọn logits ln(p). Tính phân phối khi τ=0.5 và τ=2. Greedy có đổi chỉ vì τ>0 đổi không? Vì sao không dùng phép chia τ=0?

## 3. Top-k

Tính q với k=2. Nếu logits hòa ở vị trí biên, lab chọn thế nào? Có thể suy mọi thư viện đều giữ đúng k token theo cùng quy tắc không?

## 4. Nucleus

Tính tập giữ và q với p-threshold=0.7 và 0.8. Candidate vượt ngưỡng có bị bỏ không? Top-p có cùng số candidate ở mọi prefix không?

## 5. Thứ tự filter

So top-k 2 rồi top-p 0.6 với top-p 0.6 trực tiếp trên phân phối gốc. Xác suất sample và model likelihood khác nhau ra sao?

## 6. Sampling và seed

Vì sao 10,000 mẫu không nhất thiết có đúng tỷ lệ lý thuyết? Cùng seed có bảo đảm output giống giữa CPU/GPU hay sau khi đổi model không? Nêu hai phép kiểm sampling hữu ích.

## 7. Greedy và beam tính tay

Dùng cây A/B trong bài. Tính probability và log probability chuỗi greedy, beam 2. Vì sao beam 2 tìm optimum ở cây này nhưng không bảo đảm optimum mọi cây? Lab có length penalty không?

## 8. EOS và length

Một JSON hết budget trước dấu đóng ngoặc. Có được báo thành công hoặc tự sửa rồi coi là output nguyên bản không? EOS, stop string và max_new_tokens khác nhau thế nào?

## 9. Context budget

Limit 100, template 10, câu hỏi 20, log 40, output 30. Có vừa policy lab không? Thêm một token log thì sao? Prompt token count có gồm BOS không? Vì sao số lượt forward có thể dùng ít hơn một slot so với policy reserve?

## 10. Mask cache

Past=3, new=2: viết allowed mask. Với một query cuối, Q=K=0, V=[1,2,3,4], output đúng là gì? Vì sao SDPA is_causal=True không tự biết offset? Polarity bool khác MHA ra sao?

## 11. Shape và vị trí

D=32, H=4, past=5, new=2, batch 1. Ghi shape Q, K, V và output trước/sau ghép head. Position ID của hai token mới là gì? Cần lưu Q cũ không?

## 12. Bộ nhớ KV

N_layer=32, B=1, H_KV=8, dₕ=128, T=4096, float16. Tính byte/MiB; B=4 thì sao? Có phải tổng VRAM không? Cache có làm tăng context limit không?

## 13. Đếm lượng tính lại

P=16, N=8. Tính số vị trí qua projection/FFN khi cache/no-cache. Có được gọi tỷ lệ đó là speedup không? Nếu token đầu là EOS, có cần thêm decode forward không?

## 14. Streaming và latency

Hai byte UTF-8 195,169 tới riêng. Incremental decoder nên trả gì mỗi lần? Với timestamp 0, token tới 0.120,0.160,0.205,0.250 s, tính TTFT/TPOT/end-to-end. Nếu chỉ một token thì TPOT theo công thức này là gì?

## 15. Kiểm implementation cache

Đề xuất test prefill, decode một token, chunk nhiều token và cache không đổi phần cũ. Vì sao chỉ so argmax chưa đủ? Cache có dùng sang model khác hoặc sau khi sửa prompt được không?

## 16. Thiết kế phép đo ứng dụng

Bạn cần route ticket ra nhãn cố định và soạn email có nhiều cách viết hợp lệ. Đề xuất baseline decoding, metric chất lượng/format/latency và cách chọn cấu hình bằng validation. Nêu điều phải giữ khi so cache/no-cache và giới hạn của demo một prompt bài 20.
