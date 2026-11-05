# Lời giải — PyTorch và dữ liệu batch

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

X*X=[[1, 4], [9, 16]], X@X=[[7, 10], [15, 22]], cả hai(2, 2). torch.tensor([1, 2]) là int64; Linear cần kiểu thực tương thích tham số, không suy shape đúng là dtype đúng.

## Bài 2

from_numpy chia sẻ storage CPU tương thích; torch.tensor tạo bản sao. detach cắt graph nhưng vẫn chia sẻ storage. Snapshot độc lập dùng detach().clone(); không sửa dữ liệu mà graph đang cần.

## Bài 3

a=2, L=10. Hai nhánh góp4 và3 nên da=7, dw=14, db=7. w, b là leaf được lưu grad mặc định; a trung gian vẫn truyền gradient nhưng muốn đọc a.grad thì retain_grad. Không giữ a.grad không làm backprop ngừng ở a.

## Bài 4

Gradient w cộng thành28, b thành14 vì cùng parameter chưa cập nhật. Graph cũ thường đã giải phóng trung gian sau backward; forward mới tạo graph mới, còn grad buffer leaf vẫn tồn tại. set_to_none=True đặt grad vềNone, không tự tính một tensor gradient0 mới.

## Bài 5

Linear đầu weight(2, 2), bias(2, ); Linear cuối weight(1, 2), bias(1, ). Weight là transpose của W bài10. Gradient đầu[[-q, -2q], [q, q/2]], bias[-q, 0]; gradient cuối[[-q/2, q/2]], bias gần0. Tổng9 parameter. BCE0.201413278 và sau SGD0.1 là0.177846597.

## Bài 6

Sample X, y đều(1, ). Batch thông thường(2, 1), batch cuối(1, 1). Giữ cuối: 3 batch, 5 mẫu; drop: 2 batch, 4 mẫu. squeeze() trên(1, 1) thành scalar, mất cả batch; dùng squeeze(-1) nếu muốn vector(1, ), hoặc giữ cột.

## Bài 7

Mean batch mean=41; mean sample=(2*5+2*37+81)/5=33. Bỏ batch cuối cho21 trên4 mẫu, không phải metric trên5 mẫu. Khi model đổi giữa batch, training loss còn khác đánh giá cố định model cuối epoch.

## Bài 8

300 update. zero_grad→forward→loss→backward→step. Backward tính grad; step dùng grad sửa parameter. Epoch là một lượt DataLoader; không đồng nghĩa một update. Với drop_lastTrue chỉ200 update nhưng mỗi epoch mất một mẫu.

## Bài 9

eval không tắt graph; no_grad không gọi eval. Với module phụ thuộc mode cần eval để đổi hành vi và no_grad để khỏi ghi graph đo. Lab chỉ CPU Linear/ReLU, không kiểm chứng CUDA/MPS hoặc Dropout/BatchNorm. Không suy mọi module luôn cho cùng output ở train/eval.

## Bài 10

Shuffle không tạo split và không bảo đảm user/time độc lập. Mean batch mean sai khi batch khác kích thước. BCEWithLogitsLoss cần logit, không sigmoid trước. Train tốt chưa là đánh giá giữ riêng. Báo cáo mẫu: Đây là dữ liệu giảng dạy5 mẫu y=2x+1, không có validation/test. Model Linear(1, 1) float64 chạy CPU, khởi tạo w=b=0. Dataset/DataLoader dùng batch2, shuffle có seed, num_workers0, drop_lastFalse. Dùng SGD lr0.05, MSE, 100 epoch tương ứng300 update. Ban đầu MSE33; đọc MSE cuối từ mode train và đối chiếu nghiệm lý tưởng w2, b1. Metric cuối đo lại toàn5 mẫu với model cố định, không lấy mean tùy ý các training batch mean. Cần dữ liệu đại diện, baseline và train/validation/test đúng giao thức trước kết luận chất lượng thật.

## Mở rộng

Batch3 cho hai batch kích thước3, 2. Mean loss lần lượt35/3 và65; mean theo mẫu=(3*(35/3)+2*65)/5=33. Mean không trọng số hai batch mean=115/3≈38.333333, sai cho mục tiêu mean5 mẫu. Zero gradient giữa các graph mới trả gradient w14, b7 khi parameter chưa cập nhật.
