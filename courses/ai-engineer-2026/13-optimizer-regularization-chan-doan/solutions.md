# Lời giải — Optimizer, regularization và chẩn đoán

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

SGD: θ₁=1−.1×2=.8; θ₂=.8−.1×1=.7. Weight nhỏ hơn không tự tốt hơn: cần objective và dữ liệu đánh giá, nghiệm cũng không luôn ở 0. Chuỗi gradient đặt trước để kiểm công thức, không mô phỏng các optimizer có đường đi khác nhau trên cùng hàm. Dùng toàn train cho một loss rồi backward là full batch, dù optimizer tên SGD; dùng DataLoader mini-batch ngẫu nhiên mới tạo gradient theo các batch đó.

## Bài 2

v₁=2, θ₁=.8; v₂=.9×2+1=2.8, θ₂=.52. Nếu mất buffer thì optimizer mới dùng v₂=1, θ₂=.7. Với g₂=−1, v₂=.8 nên θ₂=.8−.08=.72: vẫn giảm θ dù gradient tức thời âm, do lịch sử còn chi phối. Không khẳng định bước đó giảm objective hiện tại.

## Bài 3

Bước 1: m=.2, v=.04, m̂=2, v̂=4, update=.1×2/(2+10⁻⁸); θ≈.9000000005. Bước 2: m=.28, v=.0496; m̂=.28/.19≈1.473684211, v̂=.0496/.0199≈2.492462312; θ≈.806655200914. ε cộng sau căn v̂. v là trung bình động của g², chưa trừ bình phương mean nên không gọi variance. β₂ khác làm v, bias correction và bước sau thay đổi; bảng đặt β₂=.99, không được dùng như output mặc định .999. Code optimizer_trace kiểm cả hai bước với dung sai 10⁻¹³.

## Bài 4

w mới=(1−10η)w, nên cần |1−10η|<1 với w₀ khác 0: 0<η<.2. Với .2: w₁=−1,w₂=1, loss luôn5, parameter vẫn đổi. Với .25: w₁=−1.5,w₂=2.25, loss lần lượt11.25,25.3125. Độ cong 10 thuộc hàm này; mạng nhiều chiều, mini-batch, momentum hoặc adaptive update có động lực khác. Không dùng .2 như ngưỡng an toàn toàn cục.

## Bài 5

J'=(w−3)+λw=0 ⇒ w*=3/(1+λ). Nghiệm lần lượt3,1.5,.75. Tại λ=1, data loss=½(−1.5)²=1.125, penalty=½×1.5²=1.125, objective=2.25. Nếu penalty λw² thì gradient thêm2λw, nghiệm3/(1+2λ). Lab chỉ phạt weight Linear, không bias, và dùng data MSE mean khi huấn luyện MLP; hệ số ½ trong data loss ví dụ scalar là quy ước riêng đã ghi để dễ tính.

## Bài 6

SGD decay và AdamW cho1.98. Adam coupled đưa .1×2=.2 vào gradient; với moment ban đầu0, bias correction cho m̂=.2,v̂=.04; weight≈2−.1×.2/(.2+10⁻⁸)=1.900000005. SGD thuần không momentum, cùng parameter và λ, loss cộng λ/2 Σw² cho cùng công thức `(1−ηλ)w−ηg_data`. Các optimizer biến đổi gradient khác nhau nên không mở rộng kết luận tùy ý. Hai cơ chế cùng bật có thể tạo tác động kép ngoài ý định và làm khó so sánh; nếu chủ đích kết hợp phải ghi rõ objective, nhóm parameter và hệ số. Ví dụ dùng gradient 0, không phải grad=None.

## Bài 7

Trong train, activation nhận0 hoặc4 với xác suất .5; kỳ vọng2. Phi tuyến ở các tầng sau khiến không suy kỳ vọng dự đoán của toàn mạng bằng forward eval. Theo quy tắc best validation chọn plain, không đổi sang criterion val cuối sau khi xem kết quả. Nếu muốn đánh giá model cuối ngân sách thì phải khai báo giao thức đó riêng. Chênh lệch nhỏ trên một seed chưa là chứng cứ plain tốt nhất nói chung.

## Bài 8

Norm trước5, hệ số co=.2, gradient≈(.6,.8), norm≈1; SGD update≈(−.06,−.08), norm≈.1. Clipping từng tọa độ về [−1,1] cho(1,1), đổi hướng và norm√2. Gọi trước backward không clip gradient batch chưa được tính; đúng thứ tự backward→clip→step. Adam dùng moment và phép chia theo tọa độ; momentum/decay cũng ảnh hưởng update, nên giới hạn gradient không tự giới hạn update theo ηc. PyTorch dùng hệ số ổn định số nên các giá trị gần, không bằng tuyệt đối số thập phân tính tay.

## Bài 9

A gợi ý quên step, optimizer trỏ parameter cũ hoặc parameter bị ghi đè sau update; kiểm đường gọi step, identity và snapshot trước/sau. B có lr=0 đủ giải thích update0; kiểm config và param_groups của optimizer. C gợi ý ReLU chết, kiểm pre-activation âm và đạo hàm trước khi đổi optimizer. Các log chưa chứng minh duy nhất một nguyên nhân ngoài tình huống được dựng.

Grad=None cần kiểm đã backward chưa, requires_grad, graph có bị detach/item/no_grad không, parameter có dùng trong loss không và gradient có vừa được reset về None không. Tiny fit thành công chứng minh pipeline đơn giản có thể giảm loss trên tập có nghiệm đã biết; không chứng minh dữ liệu thật đúng, mọi gradient của MLP đúng hoặc model tổng quát hóa tốt. Tiny fit thất bại cũng cần kiểm khả năng biểu diễn và nhãn mâu thuẫn.

## Bài 10

Báo cáo mẫu: Dữ liệu synthetic gồm120 hàng với y=sin(3x)+.3x+noise std .3, split24/48/48 từ data_seed17. MLP1→32→32→1, tanh, seed2026, CPU float64 dùng full batch400 update/run. Bốn run là Adam plain, Adam với L2=.001, AdamW decay=.03 chỉ trên weight và Adam Dropout=.2. Learning rate chung .01 là cấu hình giảng dạy, không phải tuning tối ưu cho mỗi phương pháp. Chọn minimum validation data MSE qua epoch rồi qua run; baseline mean train có val≈.868515026 để tham chiếu. Plain best epoch63 cho val≈.089757914, thấp nhất trong bốn ứng viên. Chỉ sau lựa chọn mới chấm test của checkpoint đó, MSE≈.155154862 trên48 mẫu. L2 giảm norm và Dropout cải thiện val cuối nhưng không thắng theo best validation; một split/seed chưa đủ kết luận tổng quát.

Objective có L2 penalty còn val_mse không có và còn đo ở mode/thời điểm khác; so trực tiếp rồi suy train/val gap là sai. Norm thấp nhất không phải tiêu chí chất lượng đã đặt. best.pt chỉ có model/config/metric/data hash, thiếu optimizer/RNG/step và trạng thái resume; quay lại schema last.pt bài12 khi muốn tiếp tục. Adam cần lưu moment và step count, không chỉ weight.

## Thực hành mở rộng — Code đối chiếu

Từ thư mục bài13, dùng môi trường đã cài. Chạy trong thư mục tạm mới, không đọc test:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import torch
from lab import Config, make_model, weight_penalty, fit, quadratic_trace, clip_demo

model = make_model(Config())
penalty = .1/2 * weight_penalty(model)
penalty.backward()
for name, parameter in model.named_parameters():
    if name.endswith('weight'):
        torch.testing.assert_close(parameter.grad, .1*parameter)
    else:
        assert parameter.grad is None  # Bias không nằm trên đường penalty này.

with TemporaryDirectory() as directory:
    root = Path(directory)
    configs = [('l2',Config(epochs=8,l2=.001)),
               ('adamw',Config(epochs=8,optimizer='adamw',weight_decay=.03)),
               ('clipped',Config(epochs=8,clip_norm=1.))]
    for name,config in configs:
        fit(root/name,config)
        with (root/name/'metrics.csv').open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 8
        for row in rows:
            expected = float(row['data_loss_online'])+float(row['l2_penalty'])
            assert abs(float(row['objective_online'])-expected) < 1e-12
        if name == 'adamw':
            assert all(float(row['l2_penalty']) == 0 for row in rows)

clipped = clip_demo()
assert clipped['before'] == 5 and clipped['after'] <= 1
trace = quadratic_trace(.15)
assert abs(trace[1]['w']+.5) < 1e-12
assert abs(trace[2]['w']-.25) < 1e-12
print('Extension checks passed')
```

Với η=.15, hệ số là−.5: weight đổi dấu nhưng độ lớn giảm một nửa mỗi bước, loss giảm một phần tư. Clipping trong fit được thực hiện sau log preclip norm và trước step; clip_demo đo trực tiếp norm sau clipping. Nếu cần điều tra theo layer, bổ sung log riêng, không suy từ global norm rằng mọi layer ổn.
