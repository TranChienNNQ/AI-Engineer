# Lời giải — CNN và dữ liệu ảnh

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

HWC=(20,30,3), CHW=(3,20,30), thêm batch thành(1,3,20,30). Pixel sau chia255 là(1,128/255≈.5019608,0). Permute đổi trục theo ý nghĩa; reshape không tự di chuyển channel đúng vị trí. Grayscale vẫn giữ C=1 để batch có dạng NCHW; không biến ảnh2D thành batch tùy ý.

## Bài 2

Z=[[.5,−.5],[−.5,1.5]]. Ví dụ trên phải: 2−3+.5=−.5. Conv2d dùng cross-correlation nên không lật kernel. Bias cộng một lần mỗi ô, sau tổng qua patch và channel. b=0 cho[[0,−1],[−1,1]]. Nếu lật kernel sẽ là một quy ước khác và cho kết quả khác với kernel này.

## Bài 3

p=0: floor((6−3)/2)+1=2, nên2×2. p=1: floor((6+2−3)/2)+1=3, nên3×3. Batch RGB qua Conv3→10,k3,p1,s1 cho(8,10,32,32); parameter=10×(3×9+1)=280. Số parameter lớp này không tăng khi H,W hoặc N tăng, nhưng activation và compute có thể tăng. Bias mặc định có, không quên10 bias.

## Bài 4

Max=[[6,2],[3,7]]. Average=[[3.5,1.25],[1.5,4]] vì các tổng lần lượt14,5,6,16 rồi chia4. Pool không có parameter học được. Chọn max giảm thông tin vị trí trong cửa sổ; đổi vị trí tương đối với ranh giới cửa sổ hoặc padding có thể làm output khác. Invariance là tính chất cần kiểm theo phép biến đổi/model cụ thể, không suy tuyệt đối từ một layer.

## Bài 5

Các shape chính: (7,1,16,16)→(7,4,16,16)→(7,4,8,8)→(7,8,8,8)→(7,8,4,4)→(7,128)→(7,2). ReLU giữ shape. Conv1 có40, Conv2 có296, Linear có258; tổng594. Input32×32 sau hai pool thành8×8, flatten8×8×8=512 feature/mẫu; Linear128→2 không nhận được512. Flatten toàn bộ có thể trộn N vào feature, làm mất một dự đoán/mẫu. Không ép reshape để che sai shape.

## Bài 6

dK=[[4,6],[4,5]], db=4. Mỗi ô output dùng cùng K nên mọi đóng góp quay về chung một parameter. Với mean bốn ô thì dK=[[1,1.5],[1,1.25]], db=1. Đây là loss sum/mean trước ReLU/pooling, không phải CE của mạng phân loại.

Bắt đầu r=j=1. Conv3: r3,j1; pool2: r4,j2; Conv3: r8,j2; pool2: r10,j4. Head flatten nối mọi ô nên logit phụ thuộc hợp nhiều vùng; r10 chỉ nói một vị trí ở feature map cuối, kể cả padding khi gần biên.

## Bài 7

p0=1/(1+e⁻²)≈.880797078, p1≈.119202922. CE=log(1+e⁻²)≈.126928011; gradient logits=[p0−1,p1]≈[−.119202922,.119202922]. Target class ID batch10 shape(10,), dtype torch.long; logits(10,2). Không softmax trước CE. Khi cần xác suất để hiển thị, dùng softmax ở inference; argmax logits cũng cho cùng class với argmax softmax. Xác suất lớn chưa chứng minh model đã calibration hoặc ảnh nằm trong phân bố train.

## Bài 8

(128/255−.25)/.5≈.503921569. Mean/std fit trên test đưa thông tin tập giữ riêng vào tiền xử lý; dùng stats train đã lưu cho validation/test/inference. Flip ngang giữ hướng dọc nên giữ nhãn0 trong bài này; rotate90 biến dọc thành ngang, nếu giữ nhãn0 sẽ sai. Chuyển RGB sang L làm mất màu; một bài toán phân biệt đỏ/xanh có thể không còn đủ thông tin. Lab chỉ dùng grayscale để đơn giản hóa hợp đồng input.

## Bài 9

Số ảnh80/24/24. Hai view cùng gốc có vị trí/độ sáng giống và chỉ khác nhiễu, nên split từng file có thể đưa gần trùng nguồn qua hai tập. Hash khác chỉ chứng minh byte khác, không chứng minh khác sản phẩm/patient/video. Cần group/source metadata và thiết kế split phù hợp. Test có12 group; không coi24 view là24 nguồn độc lập để áp một công thức confidence interval giả định độc lập. Bài không ước lượng khoảng tin cậy từ corpus nhỏ này.

## Bài 10

Báo cáo mẫu: Corpus tự tạo128 PNG grayscale16×16 của hai lớp dọc/ngang, có64 group và hai view/group. Split giữ group nguyên vẹn, cân bằng lớp, cho80/24/24 ảnh train/validation/test. Mean≈.122097693 và std≈.246240273 chỉ fit từ pixel train sau chia255. CNN hai Conv/ReLU/Pool rồi Linear có594 parameter, Adam lr=.01, batch16, seed2026,30 epoch tương ứng150 update. Train có random flip ngang; validation/test deterministic. Chọn epoch30 theo validation CE thấp nhất≈.0000252064, validation accuracy100%. Test sau lựa chọn có CE≈.0001328481, accuracy100%, confusion hàng=true,cột=pred là[[12,0],[0,12]]. Baseline majority train cho validation accuracy50%. Bài toán synthetic đơn giản, test từ12 group và một seed nên chưa là bằng chứng trên ảnh thực tế.

Accuracy chỉ xét argmax đúng/sai; tăng xác suất lớp đúng có thể giảm CE mà không đổi argmax. Inference cần kiến trúc/code, weight, preprocessing và class mapping cùng phiên bản. Resume còn cần optimizer/RNG, epoch/step và trạng thái lựa chọn/log như bài12. best.pt ở bài14 không đủ resume chính xác. Giữ manifest, dữ liệu tái tạo, config và môi trường để đối chiếu kết quả.

## Thực hành mở rộng — Code đối chiếu

Chạy từ thư mục bài14, output tạm và không dùng test để chọn cấu hình:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch.nn import functional as F
from lab import (conv_loop, make_corpus, fit_stats, PNGDataset, loader,
                 make_model, evaluate, train, Config)

image = np.arange(30,dtype=np.float64).reshape(5,6)
kernel = np.array([[1.,-1.,2.],[0.,2.,-2.]])
x = torch.tensor(image)[None,None]
k = torch.tensor(kernel)[None,None]
for stride in (1,2):
    for padding in (0,1):
        expected = conv_loop(image,kernel,stride,padding)
        actual = F.conv2d(x,k,stride=stride,padding=padding)
        np.testing.assert_allclose(actual[0,0].numpy(),expected,rtol=0,atol=0)

multi_x = torch.cat([x,2*x],dim=1)
multi_k = torch.cat([k,-k],dim=1)
actual = F.conv2d(multi_x,multi_k)
expected = conv_loop(image,kernel)+conv_loop(2*image,-kernel)
np.testing.assert_array_equal(actual[0,0].numpy(),expected)

with TemporaryDirectory() as directory:
    root = Path(directory)
    records = make_corpus(root/'corpus')
    groups = {s:{r['group'] for r in records if r['split']==s}
              for s in ['train','validation','test']}
    assert groups['train'].isdisjoint(groups['validation'])
    assert groups['train'].isdisjoint(groups['test'])
    assert groups['validation'].isdisjoint(groups['test'])
    stats = fit_stats(root/'corpus',records)
    ds = PNGDataset(root/'corpus',records,'train',stats)
    pixels = torch.stack([ds[i][0] for i in range(len(ds))])
    assert abs(pixels.mean().item()) < 1e-6
    assert abs(pixels.std(correction=0).item()-1) < 1e-6
    val = PNGDataset(root/'corpus',records,'validation',stats)
    assert torch.equal(val[0][0],val[0][0])
    assert [len(y) for x,y in loader(ds,13)] == [13]*6+[2]
    assert [len(y) for x,y in loader(val,13)] == [13,11]
    model = make_model()
    a = evaluate(model,loader(val,13))
    b = evaluate(model,loader(val,24))
    assert abs(a['loss']-b['loss']) < 1e-6
    assert a['confusion'] == b['confusion']
    report = train(root/'run',Config(epochs=2,batch_size=13))
    assert report['train_n'] == 80 and report['prediction_roundtrip']
print('Extension checks passed')
```

Train batch13 có7 update/epoch,2 epoch cho14 update. Khi đo bằng float32, đổi cách gom batch có thể có sai khác cộng số thực nhỏ; dùng dung sai và vẫn kiểm confusion/count. Không yêu cầu thí nghiệm2 epoch đạt số của bảng30 epoch.
