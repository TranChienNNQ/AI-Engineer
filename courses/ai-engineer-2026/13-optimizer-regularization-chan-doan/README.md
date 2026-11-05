# Bài 13 — Optimizer, regularization và chẩn đoán quá trình học

[Mục lục](../README.md) · [Bài 12](../12-train-eval-validation-checkpoint/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Dữ liệu tự tạo, PyTorch thật trên CPU. Bài phân biệt ba việc: **tìm tham số tốt cho mục tiêu đang tối ưu**, **hạn chế việc mô hình học quá sát train**, và **tìm nguyên nhân khi quá trình học không như mong đợi**. Một optimizer khác không tự sửa leakage hoặc nhãn sai.

## 1. Mục tiêu và cách học

Sau bài này, bạn có thể tính hai bước SGD/momentum/Adam; viết L2 đúng hệ số; giải thích AdamW khác Adam với L2; đọc gradient norm và mức thay đổi parameter; đặt phép kiểm để phân biệt một số nguyên nhân loss đứng yên. Thực hành cuối bài so bốn cấu hình regularization trên cùng dữ liệu, chọn checkpoint bằng validation và giữ test đến sau lựa chọn.

Cần bài 2 về gradient descent, bài 10–11 về backprop/autograd và bài 12 về vòng train/eval, validation/checkpoint. Bắt đầu bằng tính tay ở phần 2–6, chạy mode `optimizers`, `decay`, `clip`; sau đó đọc cách chẩn đoán rồi chạy `diagnose`, `study`.

## 2. Optimizer nhận gì và quyết định gì?

Backprop cho gradient gₜ=∇J(θₜ₋₁). Optimizer dùng gradient, learning rate và có thể cả lịch sử để tạo bước cập nhật Δθ. Nó không quyết định dữ liệu đúng, mục tiêu đúng hoặc test độc lập.

Với n mẫu và một target scalar/mẫu, bài dùng:

\[
L_{data}(\theta)=\frac{1}{n}\sum_i(\hat y_i-y_i)^2.
\]

Khi có L2, mục tiêu huấn luyện J còn gồm penalty. Validation/test vẫn đo **data MSE**, không cộng penalty vào metric rồi so với run không penalty. Bài 12 đã tách mode train/eval; ở đây còn phải tách rõ data loss và objective.

### SGD: stochastic ở đâu?

SGD không momentum cập nhật `θ ← θ − ηg`. Khi g tính từ một mẫu hoặc mini-batch ngẫu nhiên, đó là ước lượng gradient toàn train và có nhiễu do lấy mẫu. Khi dùng toàn train, công thức cập nhật vẫn vậy nhưng không có nhiễu mini-batch. `torch.optim.SGD` có thể dùng cho cả hai; tên API không tự quyết định cách lấy batch.

**Ví dụ giả định:** θ₀=1, η=0.1, hai gradient được cho trước g₁=2, g₂=1. SGD cho θ₁=0.8, θ₂=0.7. Đây là chuỗi gradient đặt tay để kiểm công thức, không phải ba optimizer cùng đi trên một hàm rồi có cùng gradient sau bước đầu.

## 3. Momentum: gradient mới cộng với lịch sử

Trong phiên bản không dampening, không weight decay, không Nesterov:

\[
v_1=g_1,\qquad v_t=\mu v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta v_t.
\]

v là buffer momentum, μ kiểm mức giữ lịch sử. Không dùng nhầm công thức có `(1−μ)g` mà vẫn giữ nguyên η rồi coi kết quả phải giống. PyTorch khởi tạo buffer bằng gradient bước đầu. [SGD API](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html)

Vẫn θ₀=1, η=0.1, g₁=2, g₂=1 và μ=0.9:

- Bước 1: v₁=2, θ₁=1−0.1×2=0.8.
- Bước 2: v₂=0.9×2+1=2.8, θ₂=0.8−0.1×2.8=**0.52**.

Lịch sử làm bước thứ hai khác SGD. Nó có thể giúp đi qua nhiễu hoặc hướng nhất quán, nhưng cũng có thể gây vượt quá vùng tốt. Momentum không bảo đảm mỗi batch loss đều giảm. Buffer là lý do phải lưu optimizer state khi resume, như bài 12.

## 4. Adam: hai trung bình động và hiệu chỉnh độ lệch

Adam dùng từng tọa độ parameter để giữ trung bình gradient m và trung bình bình phương gradient v. Khi chưa có weight decay, không AMSGrad:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

Các phép bình phương/chia/căn ở đây theo phần tử. m₀=v₀=0; hiệu chỉnh bias bù việc bắt đầu các trung bình từ 0. ε nằm **ngoài căn** trong công thức này. v không phải phương sai đã trừ mean². Adam vẫn cần chọn η; không nên hiểu “adaptive” là tự chọn mọi siêu tham số. [Adam API](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)

**Tính tay:** θ₀=1, η=0.1, β₁=0.9, β₂=0.99, ε=10⁻⁸; vẫn g₁=2, g₂=1. β₂=0.99 được chọn cho dễ tính, khác mặc định PyTorch 0.999.

| Đại lượng | Bước 1 | Bước 2 |
|---|---:|---:|
| m | 0.2 | 0.28 |
| v | 0.04 | 0.0496 |
| m̂ | 2 | 0.28/0.19≈1.473684211 |
| v̂ | 4 | 0.0496/0.0199≈2.492462312 |
| θ sau update | 0.9000000005 | 0.8066552009 |

Bước đầu giảm gần 0.1, dù g₁=2, vì chia theo căn moment bậc hai. Đừng suy rằng mọi bước Adam đều giảm đúng η; bước hai đã khác. Mode `optimizers` đối chiếu cả hai bước của ba optimizer với công thức viết bằng số Python.

```python
from lab import optimizer_trace
trace = optimizer_trace()
for name, rows in trace.items():
    print(name, [row['weight'] for row in rows])
```

![So sánh các bước đặt gradient trước](figures/01-optimizers.png)

*Đây là kiểm toán công thức, không phải benchmark xem optimizer nào hội tụ tốt hơn.*

## 5. Learning rate và độ cong: tại sao có thể dao động?

Xét hàm một chiều J(w)=5w², gradient 10w, w₀=1. SGD cho:

\[
w_{t+1}=(1-10\eta)w_t.
\]

Muốn w tiến về 0 từ khởi tạo khác 0 cần |1−10η|<1, tức **0<η<0.2**. Đây là kết luận cho hàm bậc hai cụ thể, không phải khoảng η chung cho neural network.

| η | Hệ số | Hai weight tiếp theo | Hành vi |
|---|---:|---|---|
| 0.01 | 0.9 | 0.9, 0.81 | Giảm dần |
| 0.1 | 0 | 0, 0 | Tới nghiệm một bước trong ví dụ này |
| 0.2 | −1 | −1, 1 | Đổi dấu, loss đứng ở 5 |
| 0.25 | −1.5 | −1.5, 2.25 | Biên độ và loss tăng |

![Learning rate trên cùng hàm bậc hai](figures/02-learning-rate.png)

*Loss phẳng còn có thể do parameter đang dao động, không chỉ do parameter không đổi.*

Khi thử optimizer mới, chọn learning rate trên validation với ngân sách được ghi rõ. Cùng một η số học không có nghĩa bước cập nhật hoặc mức độ công bằng giống nhau giữa SGD và Adam. Bài không benchmark các optimizer trên MLP; chỉ kiểm công thức và thực hành regularization với cấu hình khai báo trước. Schedule, warmup, RMSProp, Nesterov và optimizer phân tán nằm ngoài code bài này.

## 6. Regularization: thay ưu tiên của việc học

Regularization đưa thêm ràng buộc hoặc cơ chế làm mô hình bớt dựa vào các chi tiết của train. Nó có thể tăng train error nhưng cải thiện ngoài mẫu; cũng có thể làm cả hai tệ hơn nếu quá mạnh hoặc không phù hợp.

### L2 với hệ số được định nghĩa rõ

Bài chọn:

\[
J=L_{data}+\frac{\lambda}{2}\sum_{W}\|W\|_F^2,
\qquad \nabla_WJ=\nabla_WL_{data}+\lambda W.
\]

Tổng chỉ qua **weight matrices của Linear**, không qua bias. Đây là chính sách của lab, không phải mọi mô hình đều bắt buộc loại bias. Nếu viết λΣW² không có 1/2 thì gradient penalty là 2λW. Mức λ còn phụ thuộc quy ước loss mean/sum và cách cộng penalty; không chép con số giữa hai code mà bỏ qua quy ước. [L2 regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization)

**Ví dụ tính tay:** J(w)=½(w−3)²+λw²/2. Đạo hàm là (w−3)+λw; nghiệm w*=3/(1+λ). Không regularization cho 3; λ=1 cho 1.5; λ=3 cho 0.75. Nghiệm co về 0 vì bài đã đổi mục tiêu, không phải optimizer bỗng tính sai. Data loss tại λ=1 là 1.125, penalty là 1.125, objective là 2.25. Không gọi 2.25 là data loss.

```python
import torch
w = torch.tensor(2., dtype=torch.float64, requires_grad=True)
lambda_ = 0.1
penalty = lambda_/2 * w.square()
penalty.backward()
print(penalty.item(), w.grad.item())  # 0.2, 0.2
```

### Weight decay có luôn giống L2 không?

SGD thuần, không momentum, dùng cùng tập parameter và hệ số λ:

`w_new = w − η(g_data + λw) = (1−ηλ)w − ηg_data`.

Trong điều kiện đó, cộng L2 và co weight mỗi bước cho cùng update. Khi optimizer có lịch sử hoặc chia gradient theo tọa độ, không được suy tương đương này tự động còn đúng.

Adam với L2 kiểu coupled đưa λw vào gradient trước khi tính m, v. AdamW tách phép co weight khỏi m, v của gradient dữ liệu. Trong PyTorch 2.15, `Adam(..., decoupled_weight_decay=True)` cũng hỗ trợ hành vi tương đương AdamW; lab dùng `torch.optim.AdamW` để chỉ rõ lựa chọn. [AdamW API](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)

**Ví dụ một bước:** w=2, g_data=0, η=0.1, weight_decay=0.1; optimizer mới, mặc định betas/eps của Adam:

- SGD decay: w mới=(1−0.01)×2=**1.98**.
- Adam coupled: g=0.2 đi vào m,v; sau hiệu chỉnh, update≈0.1×0.2/(0.2+10⁻⁸), w mới≈**1.900000005**.
- AdamW: co weight thành **1.98**, moment gradient dữ liệu bằng 0 nên không thêm bước theo gradient.

Mode `decay` kiểm các số này. Code gán **gradient tensor bằng 0**, không để `.grad=None`. Không dùng ví dụ này để suy mọi parameter có grad=None vẫn được cập nhật.

![Mục tiêu L2 và cách decay](figures/03-regularization.png)

*Hệ số penalty và cách optimizer dùng nó đều ảnh hưởng kết quả. Giá trị λ/weight_decay giữa Adam và AdamW không tự có cùng ý nghĩa tác động.*

Không vô tình vừa cộng L2 vào loss vừa bật decay trên cùng weight nếu mục đích chỉ là một cơ chế. Lab cấm kết hợp hai cách trong ablation để dễ diễn giải.

## 7. Dropout, early stopping và dữ liệu

Dropout ngẫu nhiên bỏ activation trong train, scale phần giữ lại bằng 1/(1−p); eval là identity cho layer Dropout. Với p=0.5, activation 2 được biến thành 0 hoặc 4, kỳ vọng bằng 2. Đây là kỳ vọng của activation sau Dropout; không suy dự đoán của toàn mạng phi tuyến trong eval bằng trung bình mọi dự đoán train. [Dropout API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html)

Early stopping ở bài 12 giới hạn mức fit bằng validation. Nó không cộng penalty tường minh vào loss. Lab bài 13 chạy đủ ngân sách 400 epoch rồi chọn best epoch qua validation, giúp quan sát cả đoạn sau best; không gọi đó là đã tiết kiệm compute nhờ dừng sớm.

L1 dùng tổng trị tuyệt đối, có điểm gãy ở 0 và khuyến khích sparsity trong các bài toán phù hợp; bài chỉ giới thiệu, chưa thực nghiệm L1/proximal optimizer. Data augmentation chỉ hữu ích khi phép biến đổi giữ ý nghĩa nhãn và phù hợp dữ liệu; không áp ngẫu nhiên một biến đổi vào mọi bài toán. Chuẩn hóa feature có thể giúp tối ưu nhưng không tự thay thế regularization; scaler phải fit từ train như bài 5.

## 8. Thực hành bốn cấu hình regularization

Thí nghiệm tạo 120 hàng: x đều trong [−1,1), y=sin(3x)+0.3x+ε, noise Gaussian std=0.3. Generator riêng data_seed=17 tạo dữ liệu và split ID không trùng: 24 train, 48 validation, 48 test. MLP **1→32→32→1** có tanh, hai vị trí Dropout; tổng 1.153 parameter. Cùng seed khởi tạo 2026, CPU float64, full batch train mỗi epoch, 400 epoch=400 update/run. Không có random mini-batch; nhiễu Dropout chỉ xuất hiện ở run bật Dropout.

Bốn ứng viên định trước, lr=0.01:

| Run | Optimizer và regularization |
|---|---|
| plain | Adam, không L2/decay, p=0 |
| l2 | Adam, thêm 0.001/2 × tổng bình phương weight, không decay, p=0 |
| adamw | AdamW, decay=0.03 chỉ trên weight, bias decay=0, p=0 |
| dropout | Adam, không L2/decay, p=0.2 ở hai layer Dropout |

Mỗi run giữ epoch có **validation data MSE** nhỏ nhất, hòa giữ epoch sớm hơn; chọn run best validation thấp nhất, hòa ưu tiên ứng viên đầu đã khai báo. Đây là so bốn cấu hình cụ thể, chưa tách mọi tương tác hoặc tìm hyperparameter tốt nhất cho từng phương pháp. So sánh p=0 và p=0.2 có cùng khởi tạo weight, nhưng gradient sau đó khác vì Dropout thay đổi forward.

Kết quả thực chạy:

| Run | Best epoch | Best val MSE | Train MSE cuối | Val MSE cuối | Norm weight cuối |
|---|---:|---:|---:|---:|---:|
| plain | 63 | 0.089757914 | 0.081752044 | 0.149218425 | 12.362015 |
| l2 | 70 | 0.090190619 | 0.093813536 | 0.137423154 | 5.631940 |
| adamw | 70 | 0.089841328 | 0.082174901 | 0.146898029 | 11.965555 |
| dropout | 82 | 0.091817363 | 0.112807123 | 0.107021615 | 9.354766 |

Mean predictor fit từ train có validation MSE≈0.868515026; đây là baseline ngữ cảnh, không thêm một ứng viên vào phép chọn bốn run. Chọn **plain, epoch 63**, test MSE≈**0.155154862 trên 48 mẫu**. Mode `study` chỉ chấm test cho model được chọn sau khi hoàn tất lựa chọn.

L2 ở cấu hình này giảm norm weight, Dropout cho val cuối tốt hơn plain cuối, nhưng **best validation của plain vẫn thấp nhất**. Không viết lại quy tắc để bắt regularization phải thắng; cũng không dùng norm thấp nhất làm tiêu chí thay validation. Chênh best val nhỏ và một split/seed chưa chứng minh phương pháp nào hơn nói chung. Bảng cuối epoch và best epoch là hai thời điểm khác nhau.

![Đường học và norm từ các run thực](figures/04-study.png)

*Đường train/validation được đo lại ở eval; không cộng L2 penalty vào validation. Norm weight không phải thước đo chất lượng tổng quát.*

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import study
with TemporaryDirectory() as directory:
    report = study(Path(directory)/'study')
    print(report['selected'], report['test_mse'])
```

Các artifact gồm config.json, metadata.json (môi trường, split ID, data/code hash), metrics.csv, summary.json và best.pt. File best chỉ dùng tải model đã chọn; **không phải checkpoint resume**: không chứa optimizer/RNG/bộ đếm như last.pt bài 12. Nếu mở rộng resume với Adam, còn phải khôi phục m, v và step count. Không sửa mã bài 12 rồi mặc định optimizer state SGD dùng được cho Adam.

## 9. Log gì để hiểu một update?

| Cột trong metrics.csv | Thời điểm / định nghĩa |
|---|---|
| data_loss_online | MSE forward train trước update, Dropout bật nếu p>0 |
| l2_penalty | λ/2 × tổng bình phương weight trước update; bằng 0 với AdamW |
| objective_online | Tổng hai cột trên; AdamW không được biểu diễn thành penalty trong cột này |
| grad_norm_preclip | L2 norm nối tất cả gradient của objective sau backward, trước clipping |
| update_ratio | ‖θ_sau−θ_trước‖₂/(‖θ_trước‖₂+10⁻¹²), gồm weight và bias |
| train_mse, val_mse | Sau update, model cố định, mode eval, chỉ data MSE |
| weight_norm | Căn tổng bình phương weight sau update, không gồm bias |

Gradient norm khác update norm. Adam, momentum và decay có thể biến cùng gradient thành update khác. Ratio gần parameter bằng 0 có thể rất lớn; số toàn mạng còn che khuất layer bất thường. Cần kiểm từng layer khi điều tra sâu; không áp một “ngưỡng tốt” chung cho mọi model. Việc theo dõi actual update, thử fit tập nhỏ và kiểm gradient tham khảo [CS231n: Learning](https://cs231n.github.io/neural-networks-3/); các tình huống và ngưỡng thực hành ở đây tự thiết kế.

## 10. Gradient clipping: giới hạn gradient trước step

Với norm clipping, vector gradient g được co gần theo `g × min(1, c/‖g‖₂)`. Ví dụ g=(3,4), norm=5, c=1 thì gradient mới≈(0.6,0.8), giữ hướng. Nếu SGD thuần lr=0.1, update≈(−0.06,−0.08). Đây không phải cắt mỗi tọa độ xuống [−1,1]; cách đó sẽ cho (1,1).

```python
import torch
w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
w.grad = torch.tensor([3.,4.], dtype=torch.float64)
before = torch.nn.utils.clip_grad_norm_([w], max_norm=1., error_if_nonfinite=True)
print(before.item(), w.grad.tolist())  # norm cũ 5; gradient mới gần [0.6,0.8]
```

Gọi sau `backward()`, trước `optimizer.step()`. Hàm sửa `.grad` in-place và trả norm trước clipping. Code PyTorch dùng hệ số ổn định số nên không nhất thiết ra đúng 0.6,0.8 tuyệt đối. Với momentum/Adam hoặc decay, clipping gradient **không bảo đảm norm update parameter ≤ ηc**. [clip_grad_norm_ API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)

Clipping có thể hạn chế gradient lớn hữu hạn; không chữa dữ liệu NaN, loss sai, leakage hoặc learning rate không phù hợp. Bốn run chính mặc định không clipping để không thêm biến can thiệp; `Config(clip_norm=1.0)` là nhánh thực hành riêng. AMP cần xử lý scale trước clipping, chưa được thực nghiệm trong bài.

## 11. Chẩn đoán: triệu chứng → giả thuyết → phép kiểm

Không chẩn đoán chỉ từ một hình. Bắt đầu bằng một batch nhỏ: kiểm shape/nhãn/phân bố input, data loss hữu hạn, gradient có tồn tại không, parameter có thực sự thay đổi không. Giữ một snapshot để đo update; đừng chỉ thấy backward chạy mà kết luận model đã học.

| Triệu chứng | Một số giả thuyết | Phép kiểm giúp phân biệt |
|---|---|---|
| Loss phẳng, grad khác 0, update=0 | Quên step, lr=0, optimizer giữ parameter khác | In lr, kiểm identity parameter trong optimizer, đo trước/sau step |
| Loss phẳng, grad bằng 0 | ReLU chết, saturation, nhánh hằng; cũng có thể đã ở nghiệm | Kiểm activation/gradient từng layer, thử input khác và mục tiêu đơn giản |
| `.grad=None` | Chưa backward, graph bị cắt, parameter không được dùng hoặc đã reset | Kiểm requires_grad/grad_fn, đúng tensor loss và vị trí zero_grad |
| Loss/grad bùng lên hoặc NaN | Scale input/target, phép chia/log sai miền, lr lớn, lỗi số | Kiểm finite từ input đến loss, giảm lr có kiểm soát, khoanh batch gây lỗi |
| Train giảm, validation xấu dần | Overfit, lệch phân bố hoặc đánh giá không đồng nhất | So cùng mode/metric, kiểm split, giữ best; thử regularization theo validation |
| Train và val đều cao | Chưa tối ưu tốt, model/feature thiếu, nhãn nhiễu, regularization mạnh | Tắt regularization trên tiny subset, kiểm baseline và tăng độ khó từng bước |
| Val tốt bất thường | Bài dễ hoặc leakage, mẫu lặp, preprocessing nhìn val | Kiểm theo user/time, fit tiền xử lý từ train, rà target-derived feature |

### Ba lỗi có cùng loss=1

Mode `diagnose` dựng ba tình huống scalar trong bốn bước:

1. Quên `step`: prediction=w, w=0, loss=(w−1)², grad=−2, update=0.
2. Có `step` nhưng lr=0: cùng loss=1, grad=−2, update=0.
3. ReLU chết: prediction=ReLU(w), w=−1, loss=(prediction−1)²=1, grad=0, update=0.

Hai lỗi đầu cần kiểm cả lr và đường gọi step; gradient đơn thuần vẫn chưa phân biệt được. ReLU có input âm nên đạo hàm 0, không phải lỗi của autograd. Kết hợp thêm ví dụ η=0.2 ở phần 5: loss cũng phẳng nhưng update **khác 0** do weight đổi dấu.

![Cùng triệu chứng, nhiều nguyên nhân](figures/05-diagnosis.png)

### Fit một tập rất nhỏ

`tiny_overfit()` dùng Linear(1,1), bốn mẫu trên đường y=2x+1, không regularization, SGD lr=0.1, 200 update. MSE sau học≈1.948×10⁻²⁰. Đây là kiểm cơ chế forward/loss/backward/step trên bài toán có nghiệm đã biết, không phải bằng chứng MLP thực tế tổng quát hóa tốt. Một pipeline sai vẫn có thể nhớ ít mẫu; một model thiếu năng lực cũng không thể fit mọi tiny subset tùy ý. Kết hợp kiểm dữ liệu, gradient và baseline thay vì coi một phép thử là chứng nhận toàn hệ thống.

## 12. Chạy thực hành và tự kiểm

Đã kiểm Python 3.14.6, torch 2.15.0, NumPy 2.5.4, macOS arm64, CPU float64, một thread; [requirements.txt](requirements.txt) ghi phụ thuộc. Từ thư mục bài 13:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode optimizers
.venv/bin/python lab.py --mode decay
.venv/bin/python lab.py --mode clip
.venv/bin/python lab.py --mode diagnose
.venv/bin/python lab.py --mode all --out /tmp/lesson13-run-01
```

Windows dùng `.venv\Scripts\python` và đường dẫn output tương ứng. Bỏ `--out` để tạo output tạm mới; chọn đường dẫn mới nếu đã tồn tại. `--mode study` chỉ chạy bốn cấu hình. Output tạm cần chép ra nơi lưu lâu dài nếu muốn giữ. Không yêu cầu GPU hay dịch vụ tracking. Với nền tảng không có wheel phù hợp, xem [cài PyTorch](https://pytorch.org/get-started/locally/) và ghi lại môi trường thực chạy; seed không bảo đảm giống từng bit giữa phiên bản/backend.

Code ở [lab.py](lab.py). Năm hình PNG/SVG có sẵn; tái tạo bằng [make_figures.py](make_figures.py), Matplotlib 3.11.2, run tạo trong thư mục tạm:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py
```

Làm [10 bài tập](exercises.md), đối chiếu [lời giải](solutions.md). Cần tự tính được Adam bước hai, giải thích L2/AdamW, đọc đúng metric và chọn phép kiểm cho loss phẳng. **Học tiếp:** [Bài 14 — CNN và dữ liệu ảnh](../14-cnn-du-lieu-anh/README.md). [Nguồn và phạm vi đã đọc](../SOURCES.md).
