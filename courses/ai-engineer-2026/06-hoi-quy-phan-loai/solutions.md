# Lời giải — Hồi quy và phân loại

[Về bài học](README.md) · [Về đề bài](exercises.md)

Kết quả được đối chiếu với code. Mọi dữ liệu là ví dụ giảng dạy; một điểm số hoặc vài nhãn đúng chưa chứng minh khả năng làm việc trên dữ liệu triển khai.

## Bài 1

1. Thời gian job: hồi quy; đầu ra số, có thể bắt đầu từ mô hình tuyến tính và MSE/MAE phù hợp. Kiểm tra miền không âm và lỗi ở đuôi.
2. Một nhóm routing trong ba nhóm: phân loại đa lớp, softmax và categorical cross-entropy là một thiết kế cơ bản.
3. Nhiều chủ đề đồng thời: nhiều nhãn, có thể dùng sigmoid theo từng nhãn với BCE phù hợp. Không buộc tổng xác suất nhãn bằng 1.
4. Số request: đích đếm vẫn là số; có thể đặt hồi quy nhưng cần cân nhắc mô hình miền không âm, kỳ vọng số đếm và cách đánh giá. Không mặc định mọi số nguyên là class ID.
5. Mã 0,1,2 có thể chỉ là tên lớp hoặc thứ bậc. Dùng MSE sẽ áp đặt khoảng cách; cần chứng minh ý nghĩa khoảng cách ấy, hoặc chọn cách phân loại/ordinal phù hợp.

Đổi từ latency sang vượt 100 ms đã biến đích số thành biến cố nhị phân. Có thể dùng mô hình xác suất phân loại hoặc phương pháp khác phù hợp; điểm accuracy không cùng ý nghĩa với MSE latency. Một dự đoán mean latency dưới ngưỡng cũng chưa tự cho biết xác suất vượt ngưỡng.

## Bài 2

Với w=1,b=1, dự đoán `(2,3)`, sai số `(-1,-2)`. MSE=(1+4)/2=2.5 ms²; MAE=(1+2)/2=1.5 ms.

$$
g_w=\frac22[(-1)\times1+(-2)\times2]=-5,\qquad g_b=\frac22(-1-2)=-3
$$

Cập nhật lr=0.1 cho w=1.5,b=1.3. Dự đoán `(2.8,4.3)`, sai số `(-0.2,-0.7)`. MSE=(0.04+0.49)/2=0.265 ms², RMSE=$\sqrt{0.265}\approx0.514782$ ms. Nếu cần, MAE mới=0.45 ms. Không đọc 0.265 thành sai lệch trung bình 0.265 ms.

## Bài 3

| x | z | Odds | p |
|---:|---:|---:|---:|
| −1 | −ln3 | 1/3 | 1/4 |
| 0 | 0 | 1 | 1/2 |
| 1 | ln3 | 3 | 3/4 |

Ngưỡng 0.5 với hòa gán 1 cho nhãn `[0,1,1]`. Ngưỡng 0.8 cho `[0,0,0]`.

Mỗi lần tăng x thêm 1 làm odds nhân 3. x=2 cho odds=9, p=0.9. Từ x=0→1, p tăng 0.25; từ 1→2 chỉ tăng 0.15. Hệ số tuyến tính trên log-odds không phải mức tăng cố định trên p.

Một dự đoán p=0.75 là xác suất mô hình gán. Calibration cần nhiều mẫu phù hợp để đối chiếu tỷ lệ quan sát trong các nhóm xác suất; không được suy từ một số nằm đúng miền hoặc một dự đoán đúng.

## Bài 4

Ban đầu hai p đều 0.5, mean BCE=ln2≈0.693147. Các sai số p−y là `(0.5,-0.5)`:

$$
g_w=\frac12[0\times0.5+2\times(-0.5)]=-0.5,\qquad g_b=0
$$

Cập nhật cho w=0.5,b=0. Logits mới `(0,1)`, xác suất `(0.5,0.731059)`.

Mean BCE mới:

$$
\frac{\ln2+\ln(1+e^{-1})}{2}\approx0.503204
$$

Với p≥0.5 gán 1, cả trước và sau đều dự đoán `[1,1]`; accuracy vẫn 1/2. Xác suất cho mẫu có y=1 đã tốt hơn dù nhãn sau ngưỡng chưa đổi. BCE là mục tiêu liên tục cho xác suất, không bảo đảm mỗi bước làm accuracy nhảy lên. Bước kế tiếp còn có thể thay bias; ví dụ chỉ xét một bước đầu.

## Bài 5

$\|w\|^2=2^2+(-1)^2=5$. Penalty=(0.2/2)×5=0.5, objective J=0.7+0.5=1.2.

Gradient trọng số cộng λw=`(0.4,-0.2)`; gradient bias cộng 0 vì quy ước không phạt bias. Nếu thư viện dùng tổng loss hoặc hệ số khác, cùng trị số λ có thể tạo bài toán tối ưu khác; cũng phải xét thang đo đặc trưng và trọng số mẫu.

Lab chọn bằng **validation data loss**, không bằng train objective. Test ở ngoài quá trình chọn. Kể cả muốn chọn theo một mục tiêu nghiệp vụ khác, cần định nghĩa trước và dùng development thích hợp, không chọn sau theo test đẹp nhất.

## Bài 6

Theo thứ tự z=`[-1000,0,1000]`, y=`[1,0,0]`, BCE từng mẫu gần `[1000,ln2,1000]`, mean $=(2000+\ln2)/3\approx666.897716$.

Xác suất toán học sigmoid(−1000) cực nhỏ nhưng dương; biểu diễn float có thể làm tròn về 0. Tính loss từ logit giữ được âm log hữu hạn. Nếu đầu vào thật sự được định nghĩa là xác suất 0 cho kết quả xảy ra thì log loss vô hạn, như bài 4; đó là hai tình huống khác nhau.

Với hằng số c:

$$
\frac{e^{z_j+c}}{\sum_ke^{z_k+c}}=\frac{e^ce^{z_j}}{e^c\sum_ke^{z_k}}=\frac{e^{z_j}}{\sum_ke^{z_k}}
$$

Vì thế thêm 1000 không đổi softmax `[1/6,1/3,1/2]` về mặt toán học. Thuật toán trừ max giúp tránh mũ hóa giá trị quá lớn. Trong batch phân loại riêng từng mẫu, mỗi mẫu cần tổng xác suất trên lớp bằng 1; gộp cả batch vào một tổng duy nhất sẽ làm dự đoán một mẫu phụ thuộc các mẫu khác theo một ý nghĩa sai nhiệm vụ.

## Bài 7

| τ | Nhãn dự đoán | FP | FN | Chi phí | Accuracy |
|---:|---|---:|---:|---:|---:|
| 0.3 | [0,1,1,1] | 1 | 0 | 1 | 3/4 |
| 0.5 | [0,0,1,1] | 1 | 1 | 4 | 2/4 |
| 0.7 | [0,0,0,1] | 1 | 2 | 7 | 1/4 |

Chọn 0.3 trong các ứng viên bằng validation. Không dùng test để chọn lại nếu muốn nó còn là đánh giá ngoài quá trình phát triển.

So chi phí kỳ vọng: hành động 1 tốn $C_{FP}(1-p)$, hành động 0 tốn $C_{FN}p$. Hành động 1 không đắt hơn khi $p\ge C_{FP}/(C_{FP}+C_{FN})$. Với chi phí 1/3, ngưỡng 0.25.

Ở p=0.4, hành động 1 tốn 0.6, hành động 0 tốn 1.2, nên chọn 1 theo giả định này. Nếu xác suất chưa đáng tin cậy, chi phí phụ thuộc trường hợp, hoặc chỉ xử lý được một số cảnh báo, phải thiết kế quyết định theo bối cảnh. Ngưỡng 0.25 không tự là tối ưu thực nghiệm của mọi validation nhỏ.

## Bài 8

Softmax=`[1/6,1/3,1/2]`, nhãn argmax là account. Với lớp thật technical, categorical loss=−ln(1/3)=ln3≈1.098612.

Các sigmoid riêng cho `[1/2,2/3,3/4]`. Nhiều nhãn có thể cùng đúng nên không có ràng buộc tổng bằng 1; khác giả định “đúng một lớp”. Điều đó chưa mô hình hóa mọi phụ thuộc nhãn.

Mở rộng: với y=`[1,0,1]`, BCE ba nhãn lần lượt là ln2, ln3, ln(4/3). Tổng bằng ln8, mean bằng ln2≈0.693147. Đây là trung bình ba bài nhị phân theo nhãn trong ví dụ, không phải categorical loss cho một class ID technical. Cách lấy trung bình qua nhãn/mẫu cũng cần được ghi khi so với thư viện.

## Bài 9

Mean/SD chỉ fit từ train của ứng viên. Validation data loss chọn λ; sau chốt mới truy cập test trong mode huấn luyện. Lab không refit trên train+validation.

Binary `model_scores` trả logit, có thể âm hoặc lớn hơn 1; cần sigmoid rồi áp ngưỡng nếu muốn nhãn. Dùng trực tiếp `z>=0` cũng tương đương p≥0.5, nhưng không tương đương mọi ngưỡng khác.

Nếu $x'=(x-\mu)/s$, mô hình trên feature chuẩn hóa là $w^Tx'+b$. Đổi về tọa độ gốc cho $w_{raw}=w/s$ và $b_{raw}=b-w_{raw}^T\mu$. Vì thế trị số hệ số khác nhưng dự đoán giữ nguyên. Lab lưu cả mean/scale thay vì fit lại lúc serving.

Chọn λ hoặc threshold theo test để lại thông tin test trong quyết định. Một đánh giá độc lập sau thay đổi cần dữ liệu/giao thức nằm ngoài quá trình chọn ấy. Baseline cũng phải được fit đúng phía: tỷ lệ lớp 1 lấy từ train, không từ nhãn test.

## Bài 10

100% trên 4 mẫu giả định không chứng minh accuracy tương lai bằng 100%, không chứng minh calibration, không bao phủ mọi nhóm tải/người dùng hay dữ liệu ngoài phân bố. Cần dữ liệu đánh giá đại diện và đủ độc lập theo thiết kế bài 5.

Mở rộng XOR: để `(0,0)` có nhãn 0 cần b<0. Hai điểm nhãn 1 đòi hỏi w₁+b≥0 và w₂+b≥0, nên w₁,w₂≥−b. Khi ấy logit tại `(1,1)` thỏa $w_1+w_2+b\ge-b>0$, lại phải bị gán 1, mâu thuẫn với nhãn 0. Một ranh giới tuyến tính trên hai feature gốc không thể phân tách hoàn hảo bộ này.

Với feature tương tác, $z=x_1+x_2-2x_1x_2-0.5$ cho logits `[-0.5,0.5,0.5,-0.5]`, phân loại đúng XOR ở ngưỡng 0.5. Nó tuyến tính theo vector feature mới $(x_1,x_2,x_1x_2)$ nhưng phi tuyến khi nhìn lại hai tọa độ ban đầu. Đây là ví dụ về thay biểu diễn, không phải một mạng neural đã được huấn luyện.

## Nếu còn vướng

- Nhầm logit/xác suất/nhãn: viết ba cột riêng, chỉ áp ngưỡng sau sigmoid.
- Nhầm gradient: so định nghĩa loss và hệ số 1/n, 2/n trước khi sửa code.
- Nhầm regularization với metric: tách data loss, penalty và điểm validation.
- Nhầm nhiều lớp/nhiều nhãn: hỏi một mẫu có được mang đồng thời nhiều nhãn đúng không.
- Nhầm mô hình tuyến tính và biểu diễn: viết rõ feature mà mô hình thật sự nhận.

Việc tài liệu/code hoàn thiện không thay thế bằng chứng từ bài làm hoặc giải thích của người học.
