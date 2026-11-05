# Lời giải — Vòng train/eval và checkpoint

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

MSE=(1+9)/2=5. Gradient w bằng mean của `2*(ŷ−y)*x`: (0−6)/2=−3; gradient b=(−2−6)/2=−4. Sau SGD: w=0.3, b=0.4. Dự đoán mới 0.4,0.7; squared error 0.36,5.29 nên MSE=2.825. Model đã đổi nên hai số khác nhau. Đo lại trên train giúp kiểm cơ chế, không phải validation độc lập.

## Bài 2

Batch mean lần lượt 5,25, mean không trọng số là 15. MSE đúng=(1+9+25)/3=35/3≈11.666667. Cộng loss mean nhân số mẫu batch, chia tổng mẫu: (5×2+25×1)/3. Hoặc cộng toàn bộ squared error rồi chia số mẫu vì bài có một target scalar/mẫu. Với k target/mẫu và MSE mean trên mọi phần tử, denominator là số phần tử hợp lệ, không tự áp số mẫu nếu đang cộng lỗi trên tất cả target. Mask/weight lại cần denominator đúng theo định nghĩa riêng.

## Bài 3

Train cho 0 hoặc 2, do giữ lại với xác suất 0.5 và scale lên 2. Eval cho 1. no_grad ngăn ghi graph nhưng không đổi mode; Dropout vẫn bật trong train. evaluate để model ở eval, nên train_epoch phải gọi train để bật lại hành vi train. Không cần gọi backward trong eval; mode eval bản thân không tắt autograd.

## Bài 4

Theo thứ tự epoch 1–4, best_epoch là 1,2,2,2; bad_epochs là 0,0,1,2; dừng sau epoch 4 và dùng epoch 2. Epoch 3 giảm chỉ 0.005, chưa vượt ngưỡng 0.010. Epoch 5 chưa chạy. Với min_delta=0, epoch 3 trở thành best; epoch 4 chỉ bad=1; epoch 5 tiếp tục cải thiện và best=5, bad=0. Chuỗi kết thúc sau 5 giá trị mà chưa chạm patience. Dữ liệu giả định không đủ suy tương lai sẽ thế nào.

## Bài 5

Chọn A bằng test đã dùng test làm tín hiệu lựa chọn, phá vai trò giữ riêng. Theo quy tắc khai báo trước, chọn B vì best val 0.18 nhỏ hơn 0.20; sau đó chỉ chấm checkpoint B trên test. Nếu điểm test không tốt, báo giới hạn và xây giao thức/dữ liệu mới cho nghiên cứu tiếp; không đổi sang A rồi vẫn công bố số ấy như test độc lập. Một split hữu hạn có nhiễu, chưa chứng minh B thắng trên mọi phân bố hoặc seed.

## Bài 6

Train có 48/16=3 update/epoch; 60 epoch là 180 update. Validation 21 mẫu có hai batch 16,5; không có optimizer step. Online loss lấy qua các model trước update từng batch và Dropout bật; train_eval đo lại model cuối epoch với Dropout tắt. Hai cột không cần có thứ tự lớn nhỏ cố định. Ghi rõ định nghĩa giúp tránh kết luận một cột sai chỉ vì khác cột kia.

## Bài 7

Dict trả về giữ tham chiếu tensor, không phải snapshot độc lập. Dùng `copy.deepcopy(model.state_dict())` hoặc ghi file ngay ở thời điểm best. Dự đoán dùng best epoch 8; tiếp tục tiến trình bị ngắt dùng last epoch 11 với optimizer/RNG/bộ đếm cùng thời điểm. Ghép hai thời điểm không tái tạo trạng thái đã từng tồn tại trong run. Nếu muốn restart từ best, cần training checkpoint đồng bộ của best; đó là một lựa chọn khác và cần ghi rõ.

## Bài 8

Bước 1: v₁=2, w₁=1−0.1×2=0.8. Bước 2 đủ trạng thái: v₂=0.9×2+1=2.8, w₂=0.8−0.28=0.52. Nếu mất buffer, optimizer mới dùng v₂=1 nên w₂=0.70. Chênh 0.18 trong ví dụ tính tay. Đừng nhầm với lab MLP: phép kiểm thực tế sau nhiều update cho max chênh parameter≈0.151223918. Lab cũng xác nhận resume đủ trạng thái khớp chính xác với full trong môi trường đã ghi.

## Bài 9

Cần config/kiến trúc và dữ liệu/split giống, model, optimizer momentum, epoch đã hoàn thành, global_step, best snapshot/metric/epoch, bad_epochs và RNG CPU; history cần để nối log. Seed đưa RNG về đầu chuỗi, không về vị trí sau các lần Dropout. Lab dựng thứ tự bằng seed+epoch, ngắt ở cuối epoch, worker=0, không augmentation; vì vậy không có offset sampler giữa epoch cần khôi phục. Mở rộng NumPy phải lưu RNG NumPy; Python random, CUDA, scheduler, AMP cũng có trạng thái nếu dùng. Resume giữa epoch còn cần vị trí dữ liệu và gradient đang tích lũy nếu có. Không tuyên bố checkpoint của lab đủ cho các hệ thống đó.

## Bài 10

Báo cáo mẫu: Thí nghiệm dùng 90 mẫu tự tạo từ y=sin(3x)+0.3x+ε với noise std 0.2, chia 48/21/21 hàng train/validation/test bằng data_seed 17. Model MLP 1→16→1 dùng tanh và Dropout 0.15, CPU float64. Hai run có cùng split và seed 2026, dùng SGD momentum 0.9, batch 16, lr lần lượt 0.03 và 0.005. Mỗi run dùng validation MSE giảm nghiêm ngặt để giữ best, patience 15, tối đa 60 epoch. Run lr=0.03 có best epoch 58, val≈0.194300365; run lr=0.005 có best epoch 15, val≈0.230292046. Chọn lr=0.03 theo validation rồi mới chấm test một lần: MSE≈0.215743741 trên 21 mẫu. Run đầu đạt budget 60 epoch mà chưa dừng patience; run sau dừng patience ở epoch 30. Đây là dữ liệu nhỏ, một seed, không có baseline mạnh hoặc bằng chứng chất lượng vận hành.

Giữ code/phiên bản phụ thuộc, config, split và dữ liệu có thể dựng lại, metadata, log, checkpoint và quy tắc chọn; weight một mình không giải thích feature, môi trường và lựa chọn. Hash hỗ trợ phát hiện sai khác nhưng không thay thế dữ liệu/code. Run resume cần ghi parent checkpoint; tránh ghi đè artifact gốc.

## Thực hành mở rộng — Code đối chiếu

Từ thư mục bài 12, dùng môi trường của bài. Code tạo output trong thư mục tạm, không sửa các run đã lưu:

```python
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from lab import Config, make_data, make_model, make_loader, evaluate, fit, selection_trace

values = [.8,.6,.595,.61,.58]
trace = selection_trace(values, patience=2, min_delta=0)
assert len(trace) == 5 and trace[-1][3:5] == (5,0)

config = Config(batch_size=20, epochs=2, patience=10)
data, _, _ = make_data(config)
assert [len(y) for X,y in make_loader(data['train'],config)] == [20,20,8]
model = make_model(config)
before = {k:v.clone() for k,v in model.state_dict().items()}
a = evaluate(model, make_loader(data['validation'], config))
b = evaluate(model, make_loader(data['validation'], replace(config,batch_size=7)))
assert abs(a-b) < 1e-12
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())

with TemporaryDirectory() as directory:
    root = Path(directory)
    summary = fit(root/'run',config)
    assert summary['global_step'] == 6
    original = Config(epochs=4,patience=10)
    fit(root/'part',original,until=2)
    try:
        fit(root/'wrong',replace(original,lr=.02),resume=root/'part'/'last.pt')
    except ValueError as error:
        assert 'identical config' in str(error)
    else:
        raise AssertionError('Config mismatch must fail')
print('All extension checks passed')
```

Sai khác MSE rất nhỏ do thứ tự cộng số thực được chấp nhận qua dung sai; việc parameter không đổi được kiểm trực tiếp. Mã mở rộng này không đọc test và không chọn cấu hình bằng điểm test.
