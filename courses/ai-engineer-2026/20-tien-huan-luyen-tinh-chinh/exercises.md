# Bài 20 — Bài tập tiền huấn luyện và tinh chỉnh

[Bài học](README.md) · [Lời giải](solutions.md)

## 1. Xác định mục tiêu học

Với corpus log chưa nhãn, bộ prompt–nhãn route và bộ so sánh hai câu trả lời tốt/xấu, cách tổ chức mục tiêu học khác nhau ra sao? Vì sao model viết log tốt chưa chắc trả nhãn đúng format?

## 2. Chọn cách giải bài toán

Phân tích ba yêu cầu: đổi giọng văn phản hồi; tra số dư hiện tại; route theo 6 event có quy tắc cố định. Với mỗi yêu cầu nêu baseline, phép đo và lý do có/không cần train ở bước đầu.

## 3. CPT và SFT

Một model tiếp tục next-token training trên manual nội bộ rồi được dạy prompt–JSON mẫu. Giai đoạn nào là CPT, giai đoạn nào là SFT? Có bắt buộc CPT luôn cải thiện SFT không? Lab đã thực sự chạy CPT chưa?

## 4. Shift và answer mask

Prompt=[10,11], answer=[20,21]. Viết full, x, y với BOS/EOS, completion-only. Nếu prompt có P token thì bao nhiêu nhãn đầu bị ignore? Tại sao không mask P+1?

## 5. Loss tính tay

8 prompt token có p=0.9,2 answer/EOS token có p=1/4,1/2. Tính full CE và answer CE. Có thể kết luận full CE thấp hơn là model trả lời tốt hơn không?

## 6. Một bước SGD

Logits=[ln 2,0], nhãn index 1. Tính softmax, loss, gradient và logits sau SGD lr 0.1. Xác suất nhãn đúng có tăng không? Trong model thật optimizer có cập nhật logits mỗi record như parameter không?

## 7. Prompt có gradient không?

Giải thích vì sao prompt-logit gradient trực tiếp bằng 0 nhưng prompt embedding có thể khác 0. Thiết kế một phép kiểm không nhầm gradient bảng embedding với gradient của từng vị trí.

## 8. Token boundary

Vì sao tokenize(prompt+answer) không luôn bằng tokenize(prompt)+tokenize(answer)? Trong lab điều gì bảo đảm ranh giới? Khi đổi chat template cần kiểm gì sau truncation?

## 9. Full, head-only và no_grad

Với D=32, V=323 tính head parameter, backbone frozen. `.eval()` có freeze không? Vì sao `no_grad()` toàn backbone có thể sai khi adapter nằm trong đó?

## 10. LoRA tính tay

Dùng W, A, B, x của bài chính để tính Wx, Ax, BAx và output. W 1024×1024, r 4 có bao nhiêu parameter A/B? Tỷ lệ so với W là bao nhiêu? Vì sao không gọi đó là tỷ lệ tiết kiệm tổng VRAM?

## 11. Khởi tạo LoRA

Dựa công thức gradient, phân tích A≠0, B=0 và A=B=0. Khi nào forward đầu bằng base nhưng vẫn có parameter học được? Sau merge W'=W+BA có nên tiếp tục cộng BA lần nữa không?

## 12. Split xuyên giai đoạn

Một incident test có transcript không nhãn xuất hiện trong pretraining. Vì SFT chưa thấy nhãn incident đó nên có được nói test độc lập không? Mô tả split lab và các giới hạn của nó.

## 13. Đọc bảng thực nghiệm

Full SFT exact 48/48, head-only 6/48, scratch 28/48. So với majority 16/48 và rule 48/48, kết luận nào được phép? Vì sao freeze backbone mà LM CE vẫn tăng? Vì sao LM CE scratch không gọi là forgetting từ base?

## 14. Ngân sách và checkpoint

192 mẫu, batch 16,25 epoch cho bao nhiêu update/stage? Tổng 4 stage? Best base epoch 18 có nghĩa chỉ tốn 216 update không? Full và scratch có cùng tổng compute không?

## 15. Artifact và kiểm chứng

Nêu artifact cần có để inference và phần còn thiếu để resume. Thiết kế test base không bị nhánh full sửa, frozen weight không đổi, tokenizer mismatch bị chặn, PAD/future không ảnh hưởng logits thật.

## 16. Thí nghiệm mở rộng

Đề xuất một thí nghiệm chỉ dùng train/validation để giảm regression trên LM cũ mà không bỏ metric routing. Nêu biến thay đổi, baseline, ngân sách, tiêu chí chọn best và giới hạn một seed. Có nên chọn tỷ lệ trộn bằng test rồi tiếp tục công bố test là holdout chưa dùng không?
