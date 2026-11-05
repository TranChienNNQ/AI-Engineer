# Lời giải — Dữ liệu chuỗi và ý tưởng RNN

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

Batch(4,6,3), sample(6,3), x_t của toàn batch(4,3); một bước của một sample là vector(3,). Shuffle sample độc lập đổi thứ tự lấy mẫu, không tự đảo thời gian trong sample. Đảo các bước có thể đổi ý nghĩa/nhãn. Hai dãy ví dụ có tổng2 và mean2/3 nên mean không lưu thứ tự. Nếu dữ liệu là stream stateful, còn phải kiểm thứ tự nối sample.

## Bài 2

a1=.5,h1≈.462117157; a2≈−.384470711,h2≈−.366583715; a3≈.908354071,h3≈.720341322. Đảo[2,−1,1] cho final≈.401118659. Cùng weight dùng ở mọi bước; chỉ state thay đổi. h0=0 không xóa x1: x1 vẫn đi qua w_x vào h1, rồi có thể ảnh hưởng h2,h3.

## Bài 3

Output(4,6,5), h_n(1,4,5), weight_ih(5,3), weight_hh(5,5), mỗi bias(5,). Parameter=15+25+5+5=50, head5→2 thêm12 nên tổng62. batch_first không đổi hidden layout. Hai bias cộng thành bias hiệu dụng; mỗi bias có gradient bằng gradient theo tổng. SGD update cả hai với cùng lr làm tổng đổi gấp hai lần update một bias khái niệm với gradient đó. Không gán cả hai bằng b nếu muốn bias tổng=b.

## Bài 4

L≈.024275149188; δ3≈.106008056507,δ2≈.022940577530,δ1≈.004510391298. d_wx=δ1×1+δ2×(−1)+δ3×2≈.193585926782. d_wh=δ1×0+δ2×h1+δ3×h2≈−.028259592689. d_b=δ1+δ2+δ3≈.133459025336. Một parameter được dùng lặp lại nên tất cả nhánh cùng góp gradient về nó. Có loss trực tiếp tại h2 thì upstream về h2 gồm cả nhánh trực tiếp và nhánh từ h3; không chỉ còn δ3×w_h.

## Bài 5

Đạo hàm a^20 lần lượt≈9.536743×10⁻⁷,1,3325.256730. Đây là recurrence tuyến tính không tanh; tanh RNN còn có các đạo hàm activation và vector RNN có tích Jacobian. Clipping giới hạn gradient lớn hữu hạn, không tái tạo thông tin gradient đã mất. Detach giữ giá trị hidden nhưng cắt graph về trước; reset thay giá trị bằng0 hoặc initial state mới. Cần ghi rõ ranh giới stream khi carry state.

## Bài 6

A đọc tensor padded:1.5,2,2.5; state thật cuối1.5. B:2.5,6,10.5; state thật cuối10.5. Naive h_n/output bước cuối cho[2.5,10.5], gather theo lengths hoặc packed cho[1.5,10.5]. Packed bỏ vị trí ngoài lengths nên padding99 không đổi kết quả. Zero có thể là feature thật, và normalize có thể biến raw value khác0 thành normalized0; length phải từ metadata/cách tạo chuỗi, không đoán bằng pixel/feature khác0.

## Bài 7

Mean6 ô=14/6≈2.333333; mean3 token thật=14/3≈4.666667; mean per-sequence=((1+4)/2+9)/2=5.75. Mục tiêu khác: theo token làm chuỗi dài có nhiều đóng góp, theo chuỗi cho mỗi sample trọng số ngang nhau. Lab có một class target/chuỗi nên CE sum chia số chuỗi, không chia tổng lengths hoặc số ô padded.

## Bài 8

Input[10,12,11] ở index0–2→target13,index3; input[12,11,13] ở index1–3→target15,index4. Chia forecasting theo thời điểm target và thông tin sẵn có, kiểm nguồn/group/horizon thay vì random những window tương quan; scaler chỉ fit train. Dùng lịch sử đã quan sát từ train có thể hợp lệ cho validation, không phải mọi overlap đều leakage. Nếu feature dùng dữ liệu chưa biết ở lúc dự báo hoặc nhãn bao phủ đoạn tương lai chưa có thì vi phạm giao thức. Bài chỉ tạo hai window, không đánh giá một forecaster.

## Bài 9

128/32/32 sample. Cell có8×1+8×8+8+8=88 parameter, head18, tổng106. Train128/batch16=8 update/epoch,30 epoch là240 update. Endpoint rule đơn giản cùng đạt validation100%, nên kết quả chỉ cho thấy RNN giải được bài synthetic, không chứng minh phải dùng RNN hoặc RNN vượt baseline. Mean-only về toán học không phân biệt hai phần tử trong cặp đảo; lab không fit một mean-only model để tạo benchmark.

## Bài 10

Báo cáo mẫu: Dữ liệu synthetic có96 group, mỗi group chứa chuỗi gần tăng dài4–9 và bản đảo có nhãn đối nhau. Split theo group64/16/16 tạo128/32/32 sample và giữ cân bằng lớp. Mean≈.0246224512,std≈.7806527695 chỉ tính trên bước thật của train, trước padding. Model một lớp tanh RNN1→8 và head8→2 có106 parameter, dùng packed sequence và hidden cuối theo lengths. Adam lr=.01,batch16,seed2026,30 epoch cho240 update, global grad norm clip1. Chọn epoch30 theo validation CE thấp nhất≈.002202466, validation accuracy100%. Test sau lựa chọn có CE≈.002150564, accuracy100%, confusion[[16,0],[0,16]]. Majority validation50%, endpoint rule validation100%. Dữ liệu dễ, một seed và16 group test không đủ kết luận về chuỗi thực tế hoặc dài hơn.

Carry hidden giữa các batch độc lập có thể đưa thông tin mẫu khác vào dự đoán; vị trí index trong batch không xác định cùng stream. Inference cần weight/code, config, preprocessing và class mapping. best.pt thiếu optimizer/RNG,step/bộ đếm để resume; cần schema đầy đủ như bài12 và trạng thái stream/sampler nếu mở rộng stateful training.

## Thực hành mở rộng — Code đối chiếu

Từ thư mục bài15, dùng môi trường đã cài. Mã tạo output tạm; không dùng test để chọn cấu hình:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from lab import (SequenceClassifier, scalar_reference, make_records, fit_stats,
                 SequenceDataset, loader, evaluate, train, Config, setup)

setup()
sequences = [torch.tensor([[1.],[2.]]), torch.tensor([[3.]]),
             torch.tensor([[.1],[.2],[.3],[.4]])]  # lengths chưa sắp
lengths = torch.tensor([2,1,4],dtype=torch.long)
x = pad_sequence(sequences,batch_first=True)
model = SequenceClassifier();model.eval()
with torch.no_grad():
    packed = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
    _, batched_h = model.rnn(packed)
    for i,seq in enumerate(sequences):
        _, single_h = model.rnn(seq.unsqueeze(0))
        torch.testing.assert_close(batched_h[:,i:i+1],single_h,atol=1e-6,rtol=1e-6)
    reference = model(x,lengths)
    extra = torch.nn.functional.pad(x,(0,0,0,3),value=99.)
    for i,n in enumerate(lengths.tolist()):extra[i,n:]=99.
    torch.testing.assert_close(model(extra,lengths),reference,atol=1e-6,rtol=1e-6)

answer = scalar_reference()
for name,key,value in [('wx','dwx',.5),('wh','dwh',.25),('bias','db',0.)]:
    eps = 1e-6
    high = scalar_reference(**{name:value+eps})['loss']
    low = scalar_reference(**{name:value-eps})['loss']
    assert abs((high-low)/(2*eps)-answer[key]) < 1e-9

records = make_records();stats = fit_stats(records)
train_ds = SequenceDataset(records,'train',stats)
val_ds = SequenceDataset(records,'validation',stats)
assert [len(y) for x,l,y in loader(train_ds,19)] == [19]*6+[14]
a = evaluate(model,loader(val_ds,19));b = evaluate(model,loader(val_ds,32))
assert abs(a['loss']-b['loss']) < 1e-6 and a['confusion'] == b['confusion']
with TemporaryDirectory() as directory:
    report = train(Path(directory)/'run',Config(epochs=2,batch_size=19))
    assert report['global_step'] == 14 and report['prediction_roundtrip']
print('Extension checks passed')
```

Batch19 trên128 sample cho7 update/epoch, cuối14 mẫu; hai epoch là14 update. Validation32 sample chia19+13. So loss có dung sai float32 vì thứ tự cộng/tính batch có thể khác nhỏ. So h_n từng sample riêng là phép kiểm thứ tự, không chỉ kiểm shape.
