# Bài tập — PyTorch, tensor, autograd và DataLoader

[Về bài học](README.md) · [Lời giải](solutions.md)

Dùng CPU; tính tay trước, chạy các mode lab sau.

## Bài 1. Tensor và phép nhân

Tính X*X và X@X với X=[[1, 2], [3, 4]]. Ghi shape. Dữ liệu [1, 2] mặc định có dtype gì, vì sao cần quan tâm trước Linear?

## Bài 2. Bộ nhớ dùng chung

Với snippet from_numpy trong bài, vì sao shared đổi thành[9, 2] còn copied vẫn[1, 2]? detach có tự tạo snapshot độc lập không?

## Bài 3. Autograd tính tay

Vẽ graph a=wx+b, L=a²+3a với x=2, w=0.5, b=1. Tính L, da, dw, db. Vì sao retain_grad chỉ cần cho quan sát a.grad?

## Bài 4. Tích lũy và graph mới

Hai forward/backward tại cùng parameter, không zero_grad, cho w.grad bao nhiêu? Vì sao không đơn giản gọi backward hai lần trên loss cũ? Sau zero_grad(set_to_none=True), .grad là gì?

## Bài 5. Mạng bài10

Ghi shape weight/bias Linear của mạng2→2→1. Chuyển W NumPy sang weight PyTorch. Với q=σ(−1.5), tính gradient hai weight và hai bias. Chạy mode hand kiểm tra.

## Bài 6. Một batch và một sample

TinyDataset trả sample shape gì? batch_size2 trên5 mẫu có bao nhiêu batch với drop_last False/True? Số sample xử lý tương ứng? n=1 thì squeeze() không chỉ định dim có rủi ro gì?

## Bài 7. Metric không đều batch

Predictor0 trên target1, 3, 5, 7, 9 cho batch MSE5, 37, 81. Tính mean trên batch và trên mẫu. Nếu bỏ batch cuối, metric thay thế là bao nhiêu?

## Bài 8. Training step và epoch

100 epoch, 5 mẫu, batch2, drop_lastFalse tạo bao nhiêu update? Sắp xếp zero_grad, forward, loss, backward, step. backward có tự sửa weight không?

## Bài 9. Mode và đánh giá

eval có tắt gradient không? no_grad có gọi eval không? Vì sao cần cả hai khi đo một model có Dropout? Bài này đã kiểm chứng accelerator hay Dropout chưa?

## Bài 10. Rà đề xuất và báo cáo

Một người shuffle toàn dữ liệu rồi nói đã có test độc lập; lấy mean các batch mean; sigmoid trước BCEWithLogitsLoss; báo train tốt là sẵn sàng production. Sửa bốn lỗi. Viết báo cáo5–8 câu cho thí nghiệm Linear của bài, có cấu hình và giới hạn.

## Thực hành mở rộng

Sửa batch_size thành3 rồi dự đoán số batch, shape và sample mean của predictor0. Đối chiếu bằng một DataLoader mới, giữ nguyên Dataset. Chạy lại code autograd và thêm một lần zero_grad để tự kiểm số14 thay vì28. Không dùng test để chọn cấu hình.
