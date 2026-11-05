# Lời giải — Neuron, activation và mạng nhiều lớp

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các số liệu đã đối chiếu với code. Có thể xem từng lời giải độc lập; đọc xong chưa thay thế việc tự tính và giải thích.

## Bài 1

z=0.5×2+(−2)×(−1)−0.5=2.5. Identity/ReLU cho2.5; sigmoid≈0.924142; tanh≈0.986614.

Nếu là head binary, z=2.5 là logit, p=sigmoid(z) là xác suất mô hình gán cho lớp 1, nhãn ở 0.5 là1. Nếu là hidden ReLU, a=2.5 chỉ là activation; nó không phải xác suất250%.

Tại input (0, 0), z=b=−0.5: ReLU=0, sigmoid≈0.377541. Nếu b=0, z=0, ReLU=0, sigmoid=0.5. Bias thay vị trí của phép biến đổi dù không làm đổi input.

## Bài 2

ReLU=`[0,0,0,1,2]`; Leaky ReLU α=0.1=`[-0.2,-0.1,0,1,2]`.

Sigmoid(0)=0.5, đạo hàm=0.25; tanh(0)=0, đạo hàm=1. Sigmoid'(10)≈0.0000453958, rất nhỏ hơn 0.25, thể hiện bão hòa ở phía dương lớn.

ReLU có độ dốc 0 phía âm, 1 phía dương nên không có một độ dốc duy nhất cho toàn trục. Hai đạo hàm một phía tại 0 khác nhau, nên không có đạo hàm cổ điển tại đó. Một quy ước gradient 0 là lựa chọn của thuật toán/implementation, không sửa được định nghĩa đạo hàm cổ điển.

Một input âm cho ReLU=0 chưa chứng minh neuron luôn ra0. Ví dụ z=x−1: tại x=0 ra0, nhưng tại x=2 ra1. “Chết” liên quan đến toàn bộ vùng train/đường cập nhật đang xét.

GELU(−1)=−1×0.158655≈−0.158655; GELU(1)≈0.841345. Vì tích zΦ(z) có thể âm hoặc lớn hơn1, nó không phải xác suất đầu ra mặc định.

## Bài 3

| Lớp | W | b | Z, A |
|---|---|---|---|
| 3→4 | (3, 4) | (4, ) | (5, 4) |
| 4→2 | (4, 2) | (2, ) | (5, 2) |
| 2→1 | (2, 1) | (1, ) | (5, 1) |

Số tham số=(12+4)+(8+2)+(2+1)=29. Batch50 vẫn 29, nhưng số activation trung gian tăng. Bỏ bias còn 12+8+2=22.

Một mẫu có 3 feature phải đưa thành(1, 3). Weight theo quy ước `torch.nn.Linear` có shape lần lượt(4, 3), (2, 4), (1, 2), là transpose của W trong lab; bias vẫn theo số đầu ra. Đây chỉ là đối chiếu quy ước, chưa chạy PyTorch.

## Bài 4

Z¹=`[[1,1],[-1.5,2]]`; H=`[[1,1],[0,2]]`; Z²=`[[1.5],[-1.5]]`.

Sigmoid≈`[0.817574,0.182426]`, nhãn [1, 0]. Với y=[1, 0], cả hai loss đều $\ln(1+e^{-1.5})$, mean≈0.201413 nat. Parameter count=6+3=9. Mỗi bias gắn với một neuron và dùng cùng giá trị cho mọi hàng.

Chạy riêng hàng thứ hai vẫn cho Z¹=[−1.5, 2], H=[0, 2], logit−1.5. Dense và activation từng phần tử trong lab không kết hợp các hàng batch. Không suy tính độc lập giữa hàng này sang mọi module, ví dụ những phép toán dùng thống kê batch sẽ cần xét riêng.

## Bài 5

W_eq=W¹W²=`[[3],[0]]`; b_eq=b¹W²+b²=[−1.5]. Mạng identity cho output `[1.5,−4.5]`. Có hidden ReLU cho`[1.5,−1.5]`.

Hàng đầu Z¹=[1, 1] nằm phía dương nên ReLU giữ nguyên. Hàng sau Z¹=[−1.5, 2] có phần âm bị thay bằng 0; nếu giữ −1.5, nó đóng góp thêm−3 vào logit.

Mười lớp affine vẫn gộp thành một affine; nếu cuối là sigmoid thì vẫn logistic trên x gốc, không phân tách hoàn hảo XOR bốn góc theo ngưỡng0.5.

Mở rộng 2→1→2: mọi output biến thiên theo cùng một tổ hợp scalar của hai input. Hai cột của W_eq đều là bội của vector W¹, nên chúng không thể là hai vector độc lập(1, 0) và(0, 1) của ma trận đơn vị. Đây là giới hạn bottleneck, không phải phi tuyến mới.

## Bài 6

| X | H | z | p | Nhãn |
|---|---|---: |---: |---: |
| (0, 0) | (0, 0) | −2 | 0.119203 | 0 |
| (1, 0) | (1, 0) | 2 | 0.880797 | 1 |
| (0, 1) | (0, 1) | 2 | 0.880797 | 1 |
| (1, 1) | (0, 0) | −2 | 0.119203 | 0 |

W¹=`[[1,-1],[-1,1]]`, b¹=`[0,0]`; W²=`[[4],[4]]`, b²=`[-2]`. Số phần tử tham số=4+2+2+1=9, kể cả tham số có giá trị0.

(0, 0) và(1, 1) cùng đến hidden(0, 0). Cả hai cùng nhãn 0 nên việc gộp này không làm mất thông tin cần để phân loại đúng bảng chân trị, nhưng biểu diễn không còn tái tạo được input gốc duy nhất.

Biên hidden là h₁+h₂=0.5. Trong X: |x₁−x₂|=0.5, tức x₁−x₂=0.5 hoặc x₁−x₂=−0.5. Tại x=(0.6, 0.4), H=(0.2, 0), z=−1.2, p≈0.231475, nhãn 0. Bảng XOR chỉ định bốn góc, chưa cho nhãn tham chiếu cho điểm liên tục này; nó là dự đoán mở rộng của bộ trọng số đã chọn.

`xor_demo` không fit trọng số nào. Nó chứng minh một cách biểu diễn bằng weight đặt tay, không là kết quả training hoặc đánh giá giữ riêng.

## Bài 7

- Latency: output số thực có đơn vị, identity với MSE/MAE phù hợp; kiểm tra miền không âm/đuôi theo bài toán.
- Đúng một nhóm routing: 3 logits, softmax theo lớp, categorical cross-entropy.
- Nhiều chủ đề đồng thời: 3 hoặc C logits, mỗi nhãn một sigmoid, BCE theo nhãn với quy tắc giảm loss rõ ràng.
- Ưu tiên có/không: 1 logit, sigmoid để diễn giải, BCE từ logit; ngưỡng chọn bằng development.

Softmax buộc tổng xác suất lớp bằng 1, phù hợp một lớp loại trừ nhau; không đúng với các biến cố nhiều nhãn có thể cùng xảy ra.

z=0 cho p=0.5; sigmoid(p)≈0.622459. Sigmoid lần hai coi p như một logit mới và làm đổi ý nghĩa. Không truyền output đã sigmoid vào loss vốn nhận logit rồi lại kỳ vọng nó tính cùng BCE ban đầu.

Softmax[0, ln2, ln3]=[1/6, 1/3, 1/2]. Cộng1000 không đổi xác suất toán học; code trừ max trước exp để hạn chế overflow. Phép trừ dự đoán(n, 1) với nhãn(n, ) có thể broadcast thành(n, n). Lab dùng`logits[:,0]` để lấy vector(n, ) rõ ràng, kể cả n=1 vẫn giữ trục batch.

## Bài 8

`fit_readout` chỉ thay w, b của head binary. Với raw dùng H=X; với fixed_relu dùng H=[ReLU(x₁−x₂), ReLU(x₂−x₁)]. W¹, b¹ của phép biến đổi thứ hai không được cập nhật.

Tại w=b=0, p=0.5 cả bốn. BCE=objective=ln2≈0.693147 vì penalty=0. H có hai hàng nhãn 0 bằng(0, 0), mỗi hàng nhãn 1 kích hoạt riêng một cột. Vì vậy dw=[−1/8, −1/8], db=0. Sau bước lr=0.2, w=[0.025, 0.025], b=0. Cập nhật này chưa dùng gradient hidden.

Trên X gốc, gradient từng weight tại 0 là 0 vì các đóng góp positive/negative đối xứng triệt tiêu; gradient bias cũng 0. Với objective logistic lồi có L2, điểm gradient 0 là nghiệm tối ưu. Không có nghĩa mọi logistic phải random init: với feature phù hợp, khởi tạo 0 là bình thường và ví dụ head fixed_relu đã di chuyển ngay.

Sau3000 bước, raw BCE=objective≈0.693147. Fixed_relu có BCE≈0.154492, penalty≈0.128074, objective≈0.282566, w≈[3.578739, 3.578739], b≈−1.789369. Objective là tổng BCE và penalty, nên không được báo0.282566 như data BCE.

Accuracy train1 trên bốn hàng không chứng minh W¹, b¹ đã học, end-to-end hội tụ, calibration hoặc chất lượng ngoài mẫu. Cần code cập nhật hidden thật sự và giao thức dữ liệu đánh giá phù hợp trước các kết luận ấy.

## Bài 9

Thứ tự: forward tính dự đoán → tính loss → tính gradient → cập nhật tham số. Sau cập nhật phải forward lại để có dự đoán với tham số mới. Không đọc cache cũ như thể đã được tính lại.

Chỉ cập nhật W², b² thì chưa có căn cứ nói W¹, b¹ được học. Trace Z, A giúp kiểm tra tính toán nhưng tự lưu các mảng không tự tạo công cụ vi phân; autograd cần ghi đúng phép toán/quy tắc gradient và theo dõi phụ thuộc.

Với hai hidden unit có vai trò/kết nối đối xứng hoàn toàn, tín hiệu và gradient có thể bằng nhau qua các bước, nên chúng tiếp tục trùng chức năng. Khởi tạo head bằng 0 trên H cố định khác vì mỗi cột H đã có ý nghĩa/giá trị riêng; ở XOR fixed_relu, hai weight bằng nhau vẫn tạo nghiệm phù hợp, nhiệm vụ không cần chúng khác nhau.

Hidden width, activation và số bước được lựa theo test qua nhiều lần thử làm test tham gia lựa chọn. Dùng development để chọn, đánh giá độc lập sau chốt theo bài 5–8.

## Bài 10

Mạng2→2→1 có 9 tham số dùng chung cho mọi mẫu; batch 100 chỉ tạo thêm activation/trung gian. Năm lớp affine vẫn gộp được thành affine, không tự thêm phi tuyến. Hidden sigmoid chỉ là activation trong(0, 1), không có nhãn/xác suất riêng mặc định. XOR 100% trên bốn điểm không là bằng chứng hơn cây, trên dữ liệu nhiễu hoặc sẵn sàng production.

Lab từ chối bias shape(2, 1), input1-D hoặc NaN trong snippet. Các shape ấy có thể tạo broadcasting ngoài ý định hoặc không đúng hợp đồng; giữ batch/feature rõ ràng giúp tìm lỗi tại nơi phát sinh thay vì ở metric cuối.

Một báo cáo mẫu 7 câu:

> Đây là thí nghiệm trên bốn hàng chân trị XOR, không có validation/test. So sánh head logistic trên X gốc với head logistic trên hai feature ReLU do tác giả đặt tay. Chỉ w, b đầu ra được fit; W¹, b¹ hidden giữ cố định. Dùng mean BCE+0.01/2×||w||², không phạt bias, khởi tạo head bằng 0 và chạy 3000 bước full-batch với learning rate 0.2. Raw X cho train BCE≈0.693147, accuracy 0.5; fixed_relu cho train BCE≈0.154492, accuracy 1. Objective fixed_relu≈0.282566 gồm cả penalty, nên khác BCE và không chứng minh hidden đã được học. Trước kết luận tổng quát hóa cần dữ liệu/giao thức phù hợp, baseline và tập đánh giá nằm ngoài lựa chọn; một thí nghiệm end-to-end còn phải thật sự cập nhật các tham số hidden.

## Nếu còn vướng

- Viết một hàng input rồi tính từng neuron bằng tổng scalar trước khi vector hóa.
- Ghi shape bên cạnh mỗi phép nhân, đếm cột W theo neuron đầu ra.
- Tách bốn cột z → activation → xác suất theo head → nhãn; không bước nào tự mang nghĩa của bước khác.
- Đọc code xem mảng tham số nào được gán giá trị mới, không suy việc học từ tên gọi “neural network”.
