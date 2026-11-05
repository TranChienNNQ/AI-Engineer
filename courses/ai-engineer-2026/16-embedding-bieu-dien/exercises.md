# Bài tập — Embedding và biểu diễn

[Về bài học](README.md) · [Lời giải](solutions.md)

Dự đoán/tính tay trước khi chạy. Ví dụ giả định không phải vector pretrained.

## Bài 1. ID không phải số đo

login=2,invoice=3,network=4 có chứng minh login gần nghĩa invoice hơn network không? Tính dot,cosine,L2 giữa hai one-hot khác nhau. Token có bắt buộc là một từ không? Tokenizer của bài xử lý `login,` giống `login` chưa?

## Bài 2. Lookup và shape

Dùng E5×2 ở bài, lookup ID[2,3,2,0] cho gì? Viết one-hot của ID3 và tích với E. Input B=7,T=6,V=100,D=8 có shape one-hot/embedding nào? E có bao nhiêu parameter? Linear100→8,bias=False trên one-hot có ít hay nhiều weight hơn bảng E?

## Bài 3. Pooling và PAD

Tính masked mean và naive mean qua4 vị trí của[2,3,2,0]. Tại sao padding_idx không tự sửa mean? Nếu thêm ba PAD nữa, hai cách thay đổi thế nào? UNK có bị loại khỏi denominator không? All-PAD được xử lý thế nào trong lab?

## Bài 4. Gradient hàng lặp

L=½‖p‖² với masked mean ở bài3. Tính gradient p, gradient mỗi occurrence, gradient hàng2 và3, rồi một bước SGD lr=.1. Loss sau update là bao nhiêu? Có tối ưu ID thành số thực không? Nếu một hàng vắng batch có grad0, mọi optimizer có bảo đảm hàng ấy không đổi không?

## Bài 5. Dot,cosine,L2

q=[1,0],a=[1,0],b=[10,10]. Tính ba phép đo; dot và cosine chọn hàng gần nào? Sau L2-normalization, kiểm d²=2−2cos cho b. Có áp công thức lên vector chưa normalize hoặc zero vector không?

## Bài 6. Static và contextual

Cùng token bank ở hai câu khác nghĩa có cùng lookup static không? Static có đồng nghĩa freeze không? Mean pooling có phân biệt “dog bites man” với “man bites dog” không? Token embedding có tự là sentence embedding tốt cho retrieval không?

## Bài 7. Vocabulary và OOV

Train “login error”,“invoice error”; validation “new login”. Theo quy tắc PAD0,UNK1,token train sort, ghi vocab và IDs validation. Nếu đổi mapping mà không đổi hàng E tương ứng, vì sao có thể sai dù shape vẫn đúng? Với input toàn OOV và UNK vector0, head có thể vẫn trả xác suất không?

## Bài 8. Hình học và diễn giải

Vì sao xoay E bằng Q trực giao và đổi head W→WQ có thể giữ logits? Một chiều embedding có bắt buộc tương ứng một khái niệm con người đặt tên không? `issue` gần `login` theo cosine có chứng minh đồng nghĩa không? Hình projection2D có đủ để kết luận nearest neighbor gốc không?

## Bài 9. Cấu hình và baseline

8 template, mỗi template3 topic×3 keyword×3 service, split4/2/2 template: số sample mỗi tập là gì? V=21,D=4,head3 có bao nhiêu parameter? Batch18,60 epoch có bao nhiêu update? Vì sao keyword rule100% hạn chế kết luận rằng model cần embedding để giải bài?

## Bài 10. Báo cáo thí nghiệm

Chạy mode all, đọc vocab,data,metrics,embeddings.tsv,report. Viết6–10 câu nêu split,tokenizer,OOV,model,cách chọn epoch,test,baseline và giới hạn geometry. Cần lưu gì để inference đúng? best.pt đủ resume chưa? Nếu PAD/UNK cùng vector0, có thể gộp hai ID làm một không?

## Thực hành mở rộng

1. Kiểm gradient hàng2,3 bằng sai phân trung tâm trên loss masked mean toy; giữ nguyên IDs và bảng còn lại.
2. Thêm PAD và đảo token thật trong classifier đã học, kiểm logits gần bằng nhau. Nhận biết đây vừa là kiểm mask vừa là giới hạn về thứ tự.
3. Hoán vị ID và hàng E tương ứng, kiểm giữ nguyên logits; nếu chỉ đổi ID thì so logits để phát hiện mismatch.
4. Chạy Config(epochs=2,batch_size=20) ở output mới: dự đoán số update, kiểm loss/count/confusion trên validation batch20 và54. Không dùng test để chọn cấu hình.
