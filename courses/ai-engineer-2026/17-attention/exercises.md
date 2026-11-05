# Bài tập — Attention

[Về bài học](README.md) · [Đào sâu](deep-dive.md) · [Lời giải](solutions.md)

Bài có14 câu để bao phủ các phần quan trọng. Tính tay trước khi chạy code; không cần gửi bài cho agent để đọc đáp án.

## Bài 1. Vai trò Q, K, V

Một query dịch vụ đọc memory các trạng thái:đâu là query, key, value? Nếu giữ Q, K nhưng đổi V thì A hay Z đổi? Nếu thay Q thì có nhất thiết output giữ nguyên không? Phân biệt soft attention với argmax/dict lookup.

## Bài 2. Một query tính tay

q=ln2, K=[0,1,2]ᵀ, V=[[1,0], [0,2], [3,1]], d_k=1. Tính score, exp, weight và output. Với q=0 thì sao? Vì sao output có2 chiều trong khi query chỉ1 chiều?

## Bài 3. Ma trận2×3

Q=[[1,0], [0,1]], K=[[1,0], [0,1], [1,1]], V như bài 2. Tính QKᵀ, scaled scores, A, Z. Theo softmax chiều nào? A có cần đối xứng hoặc mỗi cột cộng1 không?

## Bài 4. Scaling và softmax

Giải thích giả định để dot có variance d_k. Với q=k=[1,1,1,1], dot/scaled score/cosine là gì? Cộng1000 vào cả hàng score có đổi weight không? Nhân2 thì sao? Vì sao implementation nên trừ max trước exp?

## Bài 5. Gradient theo Q, K, V

Dùng bài 2 và loss L=z_1. Tính G_V, G_A, G_S, G_Q, G_K. Vì sao tổng G_S bằng0? Nếu V thay đổi, có thể gradient theo score thay đổi dù score hiện tại giữ nguyên không?

## Bài 6. Mask tính tay

Chặn key3 ở bài 2. Tính A, Z mới. So với cách lấy A cũ rồi gán weight3=0. Vì sao nhân score bị cấm với0 không làm mask đúng? Hàng toàn key bị cấm được xử lý thế nào trong lab?

## Bài 7. Causal và padding

Viết causal allowed mask4×4. Padding key khác padding query ra sao? Vì sao diagonal vẫn được đọc trong next-token training? Hãy nêu một phép kiểm bằng cách đổi input tương lai mà không đụng prefix.

## Bài 8. API polarity

SDPA Bool True có nghĩa gì? MHA attn_mask và key_padding_mask thì sao? valid_keys có shape B×S, chuyển sang mask SDPA B×1×L×S và mask MHA key_padding thế nào? Eval có tự tắt SDPA dropout_p=.2 không?

## Bài 9. Self và cross-attention

X_q có5 vị trí, X_m có7 vị trí, d_model=12. Sau projection Q, K width4, V width6, ghi shape một sample của score, A, Z. Nếu Q, K, V cùng xuất phát từ X thì có luôn bằng nhau không? Self có đồng nghĩa causal không?

## Bài 10. Multi-head tính tay

Dùng ví dụ hai head ở phụ lụcE:tính hai weight, hai output và concat với W_O identity. Nếu thay concat bằng mean thì mất gì? Cùng tổng D, tăng số head có tự nhân số parameter projection thêm h lần không?

## Bài 11. Shape và parameter multi-head

B=3, T=5, D=12, h=3, d_k=d_v=4. Ghi shape sau projection, split, score, head output, concat và W_O. Tính parameter MHA đủ bias. Vì sao reshape trực tiếp B×h×T×d thành B×T×D có thể sai?

## Bài 12. Position, chi phí và diễn giải

Hoán vị đồng thời input self-attention không position/mask cho quan hệ output gì? Hoán vị K, V cùng nhau trong cross-attention thì sao? B=1, h=8, T=4096, float32, tensor score explicit tốn bao nhiêu MiB? Hai weight khác nhau có thể cho cùng output không? Cho phản ví dụ.

## Bài 13. Thực nghiệm và baseline

192 memory group, mỗi group2 query, split128/32/32 group cho bao nhiêu sample? Model dim8 có235 parameter được cộng thế nào? Batch32,40 epoch có bao nhiêu update? Query-key match metric khác accuracy value ra sao? Vì sao dict baseline100% không chứng minh neural model cần thiết?

## Bài 14. Đọc artifact và báo cáo

Chạy mode all, đọc metrics/report và probe attention. Viết8–12 câu nêu dữ liệu, splits, target, mask, model, loss, checkpoint, test, baseline và giới hạn. Value class0 có được mask như PAD key0 không? best.pt đã đủ resume chưa?

## Thực hành mở rộng

1. So forward explicit với NumPy và SDPA cho L≠S, d_v≠d_k, mask bool có batch/head broadcast.
2. Dùng sai phân trung tâm kiểm Q, K, V của bài 2; giữ mask và loss không đổi khi dịch từng phần tử.
3. So gradient parameter của MHA API với implementation split/concat thủ công.
4. Chạy Config(epochs=2, batch_size=30), dự đoán số update; thêm padding và hoán vị memory của model đã học, kiểm logits không đổi trong dung sai. Không dùng test để chọn cấu hình.
