# Bài 14 — CNN và dữ liệu ảnh

[Mục lục](../README.md) · [Bài 13](../13-optimizer-regularization-chan-doan/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Bài dùng ảnh PNG tự tạo và PyTorch thật trên CPU; không tải dataset, model pretrained hoặc dùng GPU. Các ảnh thanh dọc/ngang là bài toán giảng dạy có chủ đích để kiểm cả dữ liệu, convolution và vòng học, chưa đại diện ảnh thực tế phức tạp.

## 1. Mục tiêu và kiến thức cần trước

Sau bài này, bạn có thể chuyển ảnh thành tensor đúng trục, tính một convolution nhỏ bằng tay, suy shape và số parameter của CNN, giải thích pooling và receptive field, huấn luyện bộ phân loại ảnh có split theo nhóm và tải model kèm preprocessing để dự đoán.

Cần bài 3 về shape, bài 7 về confusion matrix, bài 10–11 về autograd/DataLoader, bài 12–13 về validation và optimizer. Đọc phần 2–8, chạy mode `hand`, `shapes`, `images`; sau đó đọc pipeline ảnh và chạy `train`. Chưa cần biết xử lý ảnh cổ điển hoặc mua GPU.

## 2. Ảnh là những con số có cấu trúc

Ảnh grayscale có một giá trị mỗi pixel, thường biểu diễn độ sáng. Ảnh RGB có ba channel đỏ, lục, lam. Với ảnh 8-bit, mỗi channel thường lưu số nguyên 0–255; không phải mọi định dạng ảnh đều là 8-bit hoặc có ba channel. Ảnh RGBA có thêm alpha; ảnh palette cần chuyển mode trước khi hiểu các số như màu.

Trong ví dụ 8-bit này, pixel RGB `[255,128,0]` sau chia 255 là `[1,0.5019608,0]`. Chia 255 chuyển thang giá trị, chưa phải standardization theo mean/std.

| Biểu diễn | Shape ví dụ | Ý nghĩa trục |
|---|---|---|
| Mảng NumPy RGB | `(16,16,3)` | H,W,C |
| Một tensor ảnh RGB | `(3,16,16)` | C,H,W |
| Một tensor grayscale | `(1,16,16)` | C,H,W, vẫn giữ channel |
| Batch grayscale 8 ảnh | `(8,1,16,16)` | N,C,H,W |

Bài dùng batch theo NCHW với Conv2d. Channel của feature map sau convolution không còn có nghĩa bắt buộc là màu RGB: đó là những giá trị tính từ filter được học.

```python
import numpy as np
import torch
rgb = np.zeros((2,3,3), dtype=np.uint8)  # H=2, W=3, C=3
rgb[0,1] = [255,128,0]
chw = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
batch = chw.unsqueeze(0)
print(chw.shape, batch.shape, chw[:,0,1])
```

`permute` đổi thứ tự trục; `reshape(3,2,3)` chỉ sắp cách đọc bộ nhớ, không thay thế phép đổi HWC→CHW. `unsqueeze(0)` thêm trục batch, không biến channel thành batch. Mode `images` kiểm một pixel cụ thể để tránh chỉ kiểm shape rồi bỏ qua ý nghĩa.

![Các ảnh PNG và trục tensor](figures/01-images.png)

*Các ảnh minh họa lấy từ corpus tự tạo của lab; 0=dọc, 1=ngang. Mỗi sample vào model có shape (1,16,16).*

## 3. Tại sao dùng convolution thay vì flatten ngay?

Flatten ảnh rồi đưa vào Linear vẫn có thể học, nhưng không tự thể hiện rằng pixel gần nhau có quan hệ không gian. Convolution đưa vào hai giả định: mỗi lần tính nhìn một vùng cục bộ, và dùng cùng filter ở nhiều vị trí. Một mẫu biên hữu ích ở trái ảnh có thể được tìm bằng cùng filter ở phải ảnh.

Một filter không phải quy tắc “biết sẵn vật thể”: weight khởi tạo rồi được học qua loss/backprop. Filter đặt tay trong phần sau chỉ để thấy phép tính. Feature map là kết quả filter trượt qua ảnh; sau nhiều lớp, feature có thể phức tạp hơn nhưng không được mặc định mỗi channel tương ứng một khái niệm con người đặt tên.

Ví dụ ảnh grayscale 16×16, Conv2d 1→4, kernel 3×3, có bias: số parameter=4×(1×3×3+1)=**40**. Một Linear 256→4 chỉ cho 4 số và có 1.028 parameter; output của Conv là 4×16×16 khi padding=1, nên hai lớp này **không có cùng output**. Nếu Linear tạo cùng 1.024 số từ 256 input thì có 256×1.024+1.024=263.168 parameter. Ví dụ làm rõ sharing, không phải benchmark CNN luôn nhanh hoặc tốt hơn MLP. [CS231n: convolutional networks](https://cs231n.github.io/convolutional-networks/)

## 4. Convolution tính thế nào? Ví dụ từng ô

Trong học sâu, phép thường được gọi convolution là **cross-correlation**: kernel không bị lật trước khi nhân. Bài dùng đúng quy ước Conv2d của PyTorch. Với một channel, stride 1, không padding:

\[
Z_{i,j}=b+\sum_{u,v}K_{u,v}X_{i+u,j+v}.
\]

Lấy ảnh và kernel đặt tay:

```text
X = [[1,2,0],      K = [[1, 0],      b = 0.5
     [0,1,3],           [0,-1]]
     [2,1,0]]
```

Kernel 2×2 có bốn vị trí đặt vừa ảnh 3×3:

- Trên trái: 1×1+2×0+0×0+1×(−1)+0.5=**0.5**.
- Trên phải: 2×1+0×0+1×0+3×(−1)+0.5=**−0.5**.
- Dưới trái: 0×1+1×0+2×0+1×(−1)+0.5=**−0.5**.
- Dưới phải: 1×1+3×0+1×0+0×(−1)+0.5=**1.5**.

Kết quả Z=`[[0.5,−0.5],[−0.5,1.5]]`. Bias được cộng **một lần vào mỗi ô output**, không cộng cho mỗi phần tử patch.

```python
import torch
from torch.nn import functional as F
x = torch.tensor([[[[1.,2.,0.],[0.,1.,3.],[2.,1.,0.]]]])
k = torch.tensor([[[[1.,0.],[0.,-1.]]]])
z = F.conv2d(x,k,bias=torch.tensor([.5]))
print(z)
```

![Một convolution tính tay](figures/02-convolution.png)

*Kernel dùng lại ở bốn patch. Code conv_loop dùng vòng lặp NumPy độc lập để đối chiếu F.conv2d.*

### Nhiều channel và nhiều filter

Nếu có C_in channel, một filter bình thường có C_in×kH×kW weight. Nhân patch của từng channel với phần kernel tương ứng, **cộng qua cả channel**, rồi cộng một bias để ra một ô của một output channel. Không áp một kernel 2D riêng rẽ rồi mặc định ba màu tự trở thành ba class.

Với groups=1, shape weight là `(C_out,C_in,kH,kW)`, bias `(C_out,)`. Ví dụ Conv2d(3,8,3) có 8×(3×9+1)=224 parameter. Batch size và H,W không xuất hiện trong số parameter, nhưng vẫn ảnh hưởng số phép tính và bộ nhớ activation. [Conv2d API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html)

## 5. Stride, padding và kích thước output

Stride s là khoảng dịch giữa hai vị trí kernel. Padding p thêm biên quanh input; zero padding thêm số 0 ở biên. Với dilation d, kích thước kernel hiệu dụng là d(k−1)+1. Cho từng trục:

\[
H_{out}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}\right\rfloor+1.
\]

W dùng công thức tương tự. Bài thực nghiệm dilation=1, groups=1; dilation khác chỉ được giới thiệu qua công thức.

Với H=5,k=3,d=1:

| Padding | Stride | Tính tay | H_out |
|---:|---:|---|---:|
| 0 | 1 | floor((5−3)/1)+1 | 3 |
| 1 | 1 | floor((5+2−3)/1)+1 | 5 |
| 0 | 2 | floor((5−3)/2)+1 | 2 |
| 1 | 2 | floor((5+2−3)/2)+1 | 3 |

Đừng bỏ floor khi chia không hết: H=6,k=3,p=0,s=2 cho floor(3/2)+1=2. Với kernel/padding/stride không hợp lệ, output có thể không tồn tại và framework báo lỗi. `padding=1` không luôn giữ kích thước: điều đó đúng trong trường hợp k=3,s=1,d=1 của lab. Với các biên hữu hạn, padding còn ảnh hưởng nội dung feature ở rìa.

## 6. ReLU và pooling làm gì?

ReLU áp từng phần tử: max(0,z). Từ Z ở phần 4, A=`[[0.5,0],[0,1.5]]`. MaxPool2d kernel2,stride2 trên A cho một số **1.5**. Pooling không có weight học được; nó chọn max trong mỗi cửa sổ và giảm kích thước theo cấu hình.

Với A=`[[1,3,2,0],[4,6,1,2],[0,1,5,4],[2,3,0,7]]`, max pool2,stride2 cho `[[6,2],[3,7]]`. Nếu dùng average pooling trên cùng các cửa sổ thì giá trị khác; không gọi hai phép là một. Lab chỉ huấn luyện với max pooling mặc định ceil_mode=False. [MaxPool2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MaxPool2d.html)

Pool làm mất thông tin vị trí chi tiết trong cửa sổ. Nó có thể giúp giảm độ nhạy với một số dịch chuyển nhỏ, nhưng không bảo đảm toàn mạng bất biến với mọi phép dịch. Convolution stride1 thường có tính equivariance theo dịch chuyển ở vùng thích hợp: input dịch thì feature dịch tương ứng. Boundary, stride, pooling và head phụ thuộc vị trí có thể làm quan hệ đó không còn chính xác. Lab không chứng minh tính bất biến của CNN.

## 7. CNN nhỏ: theo shape đến logits

Model trong [lab.py](lab.py) nhận grayscale 16×16. Hãy đọc shape trước khi chạy:

| Lớp | Output với batch N | Parameter |
|---|---|---:|
| Input | N×1×16×16 | 0 |
| Conv2d 1→4, k3,p1 → ReLU | N×4×16×16 | 40 |
| MaxPool k2,s2 | N×4×8×8 | 0 |
| Conv2d 4→8,k3,p1 → ReLU | N×8×8×8 | 8×(4×9+1)=296 |
| MaxPool k2,s2 | N×8×4×4 | 0 |
| Flatten(start_dim=1) | N×128 | 0 |
| Linear 128→2 | N×2 | 2×128+2=258 |
| Tổng | 2 logits/mẫu | **594** |

```python
from lab import shape_demo
print(shape_demo())
```

Flatten bắt đầu ở trục 1 để giữ batch. `flatten()` không chỉ định trục có thể trộn cả batch; input sai 32×32 cũng làm head cố định 128 feature không khớp. Không sửa bằng reshape ép số phần tử để che lỗi kiến trúc. Nếu muốn nhiều kích thước ảnh, cần thiết kế resize/crop hoặc head khác; bài chưa huấn luyện adaptive pooling.

### Receptive field: một feature nhìn vùng nào?

Gọi r là độ rộng vùng input có thể ảnh hưởng một vị trí, j là khoảng cách trên ảnh gốc giữa hai vị trí feature kề nhau. Bắt đầu r=1,j=1. Với layer kernel k,dilation1,stride s: r_mới=r+(k−1)j; j_mới=js.

Conv3→Pool2→Conv3→Pool2 cho `(r,j)` lần lượt `(3,1),(4,2),(8,2),(10,4)`. Một ô feature cuối có receptive field lý thuyết 10×10, bao gồm phần padding nếu gần biên. Head Linear nối tất cả ô, nên một logit có thể phụ thuộc nhiều vùng hợp lại của ảnh, không chỉ một ô 10×10. Đây là phạm vi phụ thuộc tiềm năng, không có nghĩa mọi pixel đóng góp ngang nhau.

![Shape và receptive field](figures/03-architecture.png)

## 8. CNN cũng học bằng backprop

Weight convolution dùng chung qua nhiều patch nên gradient weight nhận tổng đóng góp từ các vị trí đã dùng nó. Không tạo một bộ weight riêng cho từng ô output.

Để tính tay dễ kiểm, chọn **L=sum(Z)** của convolution phần 4, chưa qua ReLU/pooling. Upstream gradient mỗi ô là 1, nên gradient kernel là tổng bốn patch:

```text
dK = [[1+2+0+1, 2+0+1+3], = [[4,6],
      [0+1+2+1, 1+3+1+0]]    [4,5]]
db = 1+1+1+1 = 4
```

Mode `hand` kiểm forward, dK và db bằng autograd float64. Nếu L=mean(Z) thì các gradient này chia 4. Nếu L là classification loss qua ReLU, pooling, head, upstream gradient khác; không lấy dK trên làm gradient của CNN đang phân loại. Tại max pool có một max duy nhất, gradient truyền về vị trí được chọn; bài không khảo sát quy tắc hòa max.

Model phân loại cuối nhận **logits N×2** và target class ID shape `(N,)`, dtype long: 0=dọc,1=ngang. CrossEntropyLoss đã kết hợp log-softmax với loss theo nhãn; không softmax logits trước khi đưa vào loss ở đây. Ví dụ logits `[2,0]`, nhãn0: p₀=e²/(e²+1)≈0.880797; loss=−log(p₀)≈**0.126928011**. Với một mẫu, gradient logit là `[p₀−1,p₁]≈[−0.119203,0.119203]`. [CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)

## 9. Đường đi từ file ảnh đến batch

Lab thực sự ghi PNG rồi đọc lại bằng Pillow; không chỉ giữ tensor đã tạo trong RAM. `read_gray` thực hiện:

1. `with Image.open(path)` và chuyển mode `'L'` rõ ràng.
2. Copy pixel thành mảng uint8, kiểm shape 16×16.
3. Chuyển float32, thêm channel và chia255 → shape `(1,16,16)`, giá trị [0,1].
4. Dataset có thể flip ngang nếu thuộc train; sau đó áp `(x−mean_train)/std_train`.
5. DataLoader gộp thành `(N,1,16,16)` và nhãn `(N,)` long.

Dùng context manager để đóng file, và mảng copy độc lập với đối tượng ảnh. Bài không tự resize ảnh sai kích thước: nó báo lỗi, giúp thấy hợp đồng input. Ảnh RGB thực tế được chuyển về grayscale bởi loader này, nên thông tin màu mất đi; muốn học RGB phải đổi cả preprocessing và in_channels. [Pillow: Image](https://pillow.readthedocs.io/en/stable/reference/Image.html)

**Tính tay preprocessing:** pixel128 → 128/255≈0.5019608. Nếu mean_train=.25,std_train=.5 thì giá trị chuẩn hóa≈0.5039216. Sau standardization, giá trị có thể âm hoặc lớn hơn1. Mean/std phải dùng cùng thang đã chia255, không trộn mean0–255 với input0–1.

Trong run thực, mean≈**0.122097693**, std≈**0.246240273**, tính chỉ từ 80 ảnh train trước random augmentation, std dùng correction=0 trên tất cả pixel. Ảnh một channel nên có một mean/std; RGB thường cần xét theo channel. Validation/test dùng nguyên stats train. Không fit lại stats trên ảnh cần dự đoán; checkpoint lưu stats và class mapping để tránh train/serving mismatch.

Với ảnh thực tế còn phải xử lý EXIF orientation, RGB/BGR theo thư viện đọc, alpha, ảnh hỏng, aspect ratio khi resize/crop, metadata và nguồn nhãn. Lab chỉ kiểm PNG grayscale tự tạo, không triển khai các xử lý đó. Resize có thể méo hình; crop có thể cắt mất đối tượng. Đó là quyết định dữ liệu cần kiểm theo bài toán, không chỉ đổi shape cho code chạy.

## 10. Split theo ảnh gốc và augmentation giữ nghĩa nhãn

Corpus có 64 group ảnh gốc, 32 group mỗi lớp. Mỗi group có cùng vị trí/độ sáng thanh và hai view nhiễu pixel độc lập. Split theo group và cân bằng lớp: 40 group train,12 validation,12 test; sau tạo hai view mỗi group có **80/24/24 ảnh**, tổng128 PNG. Group ID luôn chỉ nằm trong một tập; manifest lưu đường dẫn, nhãn, group, split và SHA-256 file.

Nếu chia ngẫu nhiên từng view, ảnh gần giống cùng nguồn có thể xuất hiện ở train và test. Với dữ liệu thật, group có thể là sản phẩm, bệnh nhân, video hoặc phiên chụp; hash file chỉ tìm trùng byte, không đủ phát hiện mọi ảnh gần trùng. Lab dùng group biết từ lúc tạo, không coi chỉ cần hash là hết leakage.

Augmentation không tự tạo thêm các quan sát độc lập. Lab chỉ random horizontal flip trên train: thanh dọc vẫn dọc, thanh ngang vẫn ngang. Rotate90 đổi dọc thành ngang, nên xoay mà giữ nguyên nhãn sẽ tạo mẫu sai. Một biến đổi hợp lệ ở bài này có thể không hợp lệ với chữ viết, hướng giao thông hoặc ảnh y khoa. Validation/test không random transform; `PNGDataset(..., augment=True)` với tập khác train bị chặn.

![Group split và ý nghĩa augmentation](figures/04-data-pipeline.png)

*Split theo nguồn trước, augmentation online sau. Hai view của cùng group không được đi sang hai tập.*

## 11. Thực nghiệm: huấn luyện, chọn epoch và dự đoán

Một kiến trúc/cấu hình được định trước: model 594 parameter ở phần 7, Adam lr=.01, batch16, seed2026, tối đa30 epoch, CPU float32. Train có random flip, shuffle; worker0. Mỗi epoch5 optimizer update, tổng150 update. Đo lại train với transform deterministic riêng; validation không augmentation. Giữ checkpoint có validation CE thấp nhất; hòa giữ epoch sớm hơn. Không quét hyperparameter trên test.

Kết quả thực chạy:

| Đại lượng | Kết quả |
|---|---|
| Best epoch theo validation loss | 30 |
| Validation CE / accuracy | ≈0.0000252064 / 24/24=100% |
| Test CE / accuracy sau lựa chọn | ≈0.0001328481 / 24/24=100% |
| Baseline majority train: validation accuracy | 50% |
| Confusion test, hàng=true,cột=predicted, thứ tự dọc/ngang | `[[12,0],[0,12]]` |

Chọn bằng **loss**, không bằng accuracy cuối: accuracy có thể đã100% trước khi CE tiếp tục giảm. Test chỉ được chấm cho checkpoint đã chọn sau vòng train. Sau đó code dùng một ảnh test để kiểm hai đường inference cùng cho dự đoán bằng nhau, không dùng ảnh đó để chọn lại model. `prediction_roundtrip=True` xác nhận load checkpoint, preprocessing và class mapping nhất quán.

Đây là mẫu hình synthetic rất dễ, không chứng minh CNN đạt100% trên ảnh thật hoặc chịu được domain shift. Test24 ảnh chỉ đến từ12 group nên không coi là24 quan sát nguồn độc lập để báo khoảng tin cậy ngây thơ. Không so CNN với MLP hoặc model pretrained trong lab này.

![Đường học và confusion matrix từ run thực](figures/05-training.png)

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import Config, train, predict_file
with TemporaryDirectory() as directory:
    out = Path(directory)/'run'
    report = train(out, Config(epochs=2))  # Thử pipeline ngắn; không kỳ vọng số bảng 30 epoch
    image_path = next((out/'data'/'images').glob('*.png'))
    print(predict_file(out/'best.pt', image_path))
    print(report['parameters'])  # 594
```

Output gồm data/images, data/manifest.json, config.json, preprocess.json, metadata.json, metrics.csv, best.pt và report.json. Best.pt chứa model state, config, preprocessing, class mapping, epoch/val loss và data hash; dùng tải inference, chưa chứa optimizer/RNG để resume. Nếu muốn resume, mở rộng checkpoint theo bài 12. Không đóng gói các output runtime vào bộ tài liệu; ảnh sẽ được tạo lại bằng code.

Bài này là **classification**: một nhãn cho toàn ảnh. Detection còn cần vị trí/box; segmentation cần nhãn theo pixel. Chúng có target, head, loss và metric khác; không chỉ đổi tên lớp của bộ phân loại này. Các kiến trúc ResNet, transfer learning và Vision Transformer là hướng đọc tiếp, chưa được huấn luyện hoặc benchmark ở đây.

## 12. Chạy thực hành và tự đánh giá

Đã kiểm Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Pillow 13.0.0, macOS arm64. CNN dùng CPU float32; phần convolution tính tay dùng float64. Phụ thuộc ở [requirements.txt](requirements.txt). Từ thư mục bài 14:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode shapes
.venv/bin/python lab.py --mode images
.venv/bin/python lab.py --mode all --out /tmp/lesson14-run-01
```

Windows dùng `.venv\Scripts\python` và output phù hợp. Chọn thư mục output mới, không ghi đè run trước; bỏ --out để lab tạo output tạm và in đường dẫn. Chép output ra nơi lưu lâu dài nếu cần giữ. `--mode train` chỉ chạy pipeline ảnh. Không cần torchvision hoặc mạng để tải dữ liệu sau khi cài phụ thuộc. Với nền tảng không có wheel bản ghim, xem [hướng cài PyTorch](https://pytorch.org/get-started/locally/) và ghi môi trường bạn thực chạy; seed không bảo đảm giống từng bit giữa phiên bản/thiết bị.

Năm hình đã có PNG/SVG; [make_figures.py](make_figures.py) tái tạo bằng Matplotlib 3.11.2 và run tạm:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Làm [10 bài tập](exercises.md), rồi xem [lời giải](solutions.md). Cần tự suy shape,40/296/258 parameter, bốn ô convolution, gradient shared kernel và phát hiện augmentation làm sai nhãn. **Học tiếp:** [Bài 15 — Dữ liệu chuỗi và ý tưởng RNN](../15-du-lieu-chuoi-rnn/README.md). [Nguồn và phạm vi đã đọc](../SOURCES.md).
