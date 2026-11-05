# Bài tập — CNN và dữ liệu ảnh

[Về bài học](README.md) · [Lời giải](solutions.md)

Tính tay hoặc dự đoán trước khi chạy. Số liệu ví dụ là giả định nếu không ghi output lab.

## Bài 1. Pixel và trục

Ảnh RGB 20×30 được đọc thành NumPy HWC. Ghi shape trước/sau permute và sau thêm batch. Pixel [255,128,0] sau chia255 là gì? Vì sao reshape không thay thế permute? Grayscale có cần giữ channel không?

## Bài 2. Bốn ô convolution

Dùng X,K,b ở phần4, tính bốn ô Z. Có lật kernel trước khi tính Conv2d không? Bias cộng bao nhiêu lần mỗi ô? Nếu b=0 thì Z đổi thế nào?

## Bài 3. Output shape

Input H=W=6, kernel3, stride2, padding0, dilation1: output bao nhiêu? Đổi padding1 thì sao? Input batch8 RGB32×32 qua Conv2d(3,10,3,padding=1) cho shape và số parameter gì? H,W hoặc batch size tăng có làm số parameter lớp Conv đó tăng không?

## Bài 4. Pooling và thông tin

Cho ma trận A4×4 ở phần6, tính max pooling2,stride2. Tính average pooling trên các cửa sổ tương ứng. Pooling có weight học được không? Vì sao không nói max pooling bảo đảm invariant với mọi dịch chuyển?

## Bài 5. CNN nhỏ

Theo dõi shape với N=7 qua model bài này. Tính parameter từng lớp và tổng. Nếu input32×32 thay16×16 thì flatten có bao nhiêu feature/mẫu, head hiện tại có chạy được không? Flatten sai trục có thể làm gì với batch?

## Bài 6. Shared weight và gradient

Với L=sum(Z) của convolution phần4, tính dK và db. Vì sao phải cộng qua bốn patch? Nếu L=mean(Z) thì sao? Tính receptive field và jump sau Conv3→Pool2→Conv3→Pool2, các conv stride1,padding1, pool stride2. Có coi một logit chỉ nhìn vùng10×10 không?

## Bài 7. Cross-entropy

Hai logits[2,0], nhãn0. Tính p0, loss và gradient theo logits cho một mẫu. Tensor target của batch10 trong lab có shape/dtype gì? Có softmax trước CrossEntropyLoss không? Softmax khi nào cần dùng để diễn giải dự đoán?

## Bài 8. Preprocessing và augmentation

Pixel128, mean_train=.25,std_train=.5 trên thang0–1: giá trị chuẩn hóa bao nhiêu? Vì sao không tính mean/std từ cả test? Thanh dọc được flip ngang và rotate90: phép nào giữ nhãn0? Việc chuyển RGB sang L có giữ mọi thông tin không?

## Bài 9. Split theo nguồn

64 group, mỗi group2 view, split40/12/12 group cho bao nhiêu ảnh mỗi tập? Vì sao split ngẫu nhiên128 file có thể rò rỉ? Hash khác nhau có chứng minh ảnh khác nguồn không? Với test24 ảnh từ12 group có nên tính khoảng tin cậy như24 nguồn độc lập không?

## Bài 10. Chạy và báo cáo

Chạy mode all, xem manifest, preprocess, metrics.csv và report.json. Viết6–10 câu nêu dữ liệu/split, CNN, cách chọn epoch, confusion matrix, baseline và giới hạn. Vì sao accuracy có thể giữ100% nhưng CE còn giảm? Muốn inference và muốn resume cần giữ những gì khác nhau?

## Thực hành mở rộng

1. So conv_loop với F.conv2d trên mảng5×6, kernel2×3, stride1/2 và padding0/1; không dùng chính F.conv2d để tạo expected output.
2. Kiểm tổng hai convolution channel riêng bằng convolution hai channel tương ứng.
3. Kiểm group không trùng giữa split; train chuẩn hóa có mean gần0,std gần1; validation đọc hai lần giống nhau.
4. Chạy Config(epochs=2,batch_size=13): dự đoán kích thước các batch train và validation, kiểm evaluate gộp loss đúng khi batch cuối nhỏ.
