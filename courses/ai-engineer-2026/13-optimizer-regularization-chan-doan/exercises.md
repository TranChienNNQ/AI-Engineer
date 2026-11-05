# Bài tập — Optimizer, regularization và chẩn đoán

[Về bài học](README.md) · [Lời giải](solutions.md)

Dự đoán hoặc tính tay trước khi chạy. Ví dụ số là giả định nếu không ghi rõ output lab.

## Bài 1. Gradient và update

θ₀=1, gradient cho trước g₁=2,g₂=1, η=0.1. Tính hai bước SGD. Tại sao không thể dùng kết quả này để nói SGD kém hơn optimizer có θ₂ nhỏ hơn? Khi nào SGD API đang dùng gradient full batch thay vì mini-batch ngẫu nhiên?

## Bài 2. Momentum

Cùng chuỗi gradient, μ=0.9, không dampening/Nesterov/decay. Tính buffer và weight sau hai bước. Nếu bỏ buffer sau bước đầu, weight bước hai là gì? Nếu g₂=−1 thay vì 1, momentum bước hai cập nhật theo hướng nào?

## Bài 3. Adam

Với β₁=0.9, β₂=0.99, η=0.1, ε=10⁻⁸, m₀=v₀=0 và gradient 2,1, tính m,v,m̂,v̂ và weight hai bước. ε nằm ở đâu? Vì sao v không phải variance? Thay β₂=.999 có được giữ nguyên bảng số không?

## Bài 4. Learning rate và độ cong

Với J(w)=5w², w₀=1, suy ra điều kiện η để hội tụ bằng SGD thuần. Tính hai bước với η=.2 và .25. Loss phẳng có nhất thiết parameter không đổi không? Vì sao không dùng ngưỡng .2 cho mọi neural network?

## Bài 5. Hệ số regularization

J=½(w−3)²+λw²/2. Tìm nghiệm với λ=0,1,3. Tại λ=1 tính data loss, penalty và objective. Nếu thay penalty thành λw², nghiệm đổi thế nào? Lab có regularize bias không?

## Bài 6. L2 và AdamW

w=2, gradient dữ liệu bằng tensor 0, η=.1, decay=.1, optimizer mới. Tính weight sau một bước SGD decay, Adam coupled và AdamW. Nêu điều kiện để SGD decay tương đương L2 theo công thức của bài. Vì sao không bật decay và cộng L2 cùng lúc mà không ghi rõ?

## Bài 7. Dropout và chọn checkpoint

Activation bằng 2 đi qua Dropout p=.5 trong train có thể nhận những giá trị nào và kỳ vọng bao nhiêu? Có suy dự đoán eval của cả mạng bằng trung bình dự đoán train không? Run plain best val=.08976, dropout best val=.09182 nhưng dropout val cuối tốt hơn: theo giao thức bài chọn ai?

## Bài 8. Gradient clipping

g=(3,4), max_norm=1. Tính norm trước/sau clipping, vector gradient và update SGD η=.1. Khác clipping từng tọa độ ra sao? Gọi clipping trước backward có xử lý đúng gradient batch hiện tại không? Có suy update Adam luôn có norm≤.1 không?

## Bài 9. Cùng loss phẳng

Ba log có loss=1, update=0. Log A grad=−2, lr=.1; log B grad=−2, lr=0; log C grad=0 và input ReLU âm. Đặt giả thuyết và phép kiểm cho từng log. Nếu `.grad=None` thay vì 0, cần kiểm gì? Fit được tiny subset chứng minh điều gì và chưa chứng minh gì?

## Bài 10. Báo cáo thí nghiệm

Chạy mode study rồi đọc config, metadata, metrics.csv, summary và study.json. Viết 6–10 câu nêu dữ liệu/split, bốn cấu hình, metric chọn, best epoch và test của run được chọn. Vì sao objective_online không so trực tiếp với val_mse? Norm weight thấp nhất có phải model tốt nhất không? best.pt của bài đủ để resume chưa?

## Thực hành mở rộng

1. Kiểm L2 bằng autograd trên vector weight và so với λw; kiểm bias không nằm trong penalty.
2. Tạo hai model khởi tạo giống nhau, chạy 8 epoch Adam với l2=.001 và AdamW decay=.03. Đọc cột objective, penalty, gradient norm, update ratio; không chấm test để chọn cấu hình.
3. Chạy Config(epochs=8,clip_norm=1.) ở output mới, so định nghĩa preclip norm với norm gradient sau clipping. Không suy clipping luôn cải thiện validation.
4. Thay η=.15 trong quadratic_trace, dự đoán dấu weight và tốc độ giảm loss. Kiểm hai bước đầu, không coi đây là optimizer benchmark.
