# Lời giải — Xác suất, thống kê và bất định

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các ví dụ là giả định. Kết quả xác định được đối chiếu bằng phép tính trực tiếp; kết quả mô phỏng ghi rõ seed, số lần lặp và môi trường trong bài học. Không thay số ngẫu nhiên bằng một giá trị lý thuyết cho đẹp.

## Bài 1

$P(E\cap A)=90/10000=0.009$. $P(A)=(90+495)/10000=0.0585$. $P(E\cup A)=0.01+0.0585-0.009=0.0595$.

Nhóm không cảnh báo có $10+9405=9415$ request, trong đó 10 lỗi, nên $P(E\mid A^c)=10/9415\approx0.001062=0.1062\%$.

Hai biến cố không độc lập: $P(E)P(A)=0.01\times0.0585=0.000585$, khác 0.009. Cũng có thể so $P(A\mid E)=0.9$ với $P(A)=0.0585$.

Mở rộng: mật độ 2 giây⁻¹ trên đoạn dài 0.5 giây có tổng diện tích 1 nên là một mật độ hợp lệ. Xác suất trong [0.1,0.2] bằng $2\times(0.2-0.1)=0.2$. Theo mô hình liên tục, xác suất tại đúng điểm 0.15 bằng 0; đó không phải xác suất của một khoảng quanh 0.15 hay xác suất máy đo làm tròn ra 0.15.

## Bài 2

$\mathbb E[K]=np=2$, $\operatorname{Var}(K)=np(1-p)=1.6$.

- $P(K=0)=0.8^{10}=0.1073741824$.
- $P(K=2)=\binom{10}2\,0.2^2\,0.8^8=0.301989888$.
- $P(K\ge1)=1-P(K=0)=0.8926258176$.

Kỳ vọng là trung bình theo phân bố, không phải kết quả bắt buộc của một lượt thử. Khi có sự cố chung, các request có thể phụ thuộc nhau, nên không tự động dùng phân bố nhị thức với n=10 chỉ vì mỗi request có hai trạng thái. Cũng phải kiểm tra giả định p không đổi giữa các phép thử.

## Bài 3

Với `[2,4,4,6]`, tổng 16, mean=4 ms, median=(4+4)/2=4 ms. Các độ lệch là `[-2,0,0,2]`, tổng bình phương bằng 8 ms².

| Đại lượng | Giá trị |
|---|---:|
| Phương sai chia n | 2 ms² |
| Phương sai chia n−1 | $8/3\approx2.666667$ ms² |
| SD chia n | $\sqrt2\approx1.414214$ ms |
| SD chia n−1 | $\sqrt{8/3}\approx1.632993$ ms |

Với c=4, MSE bằng 2 ms². Với c=3, sai số dự đoán–đích là `[1,-1,-1,-3]`, tổng bình phương 12, MSE bằng 3 ms².

Đổi 6 thành 60 làm mean=(2+4+4+60)/4=17.5 ms, median vẫn 4 ms. Mean nhạy với độ lớn của giá trị ở đuôi hơn median trong ví dụ này. Điều đó chưa đủ để xóa 60: trước hết cần xác định đó là lỗi dữ liệu hay một request thật cần quan tâm.

## Bài 4

| | Lỗi | Không lỗi | Tổng |
|---|---:|---:|---:|
| Cảnh báo | 90 | 45 | 135 |
| Không cảnh báo | 10 | 855 | 865 |
| Tổng | 100 | 900 | 1000 |

$P(E\mid A)=90/135=2/3\approx66.67\%$. Bằng công thức: $0.9\times0.1/(0.9\times0.1+0.05\times0.9)=2/3$.

Tỷ lệ nền cao hơn làm nhóm lỗi đóng góp nhiều hơn vào cảnh báo. Đây là so sánh giữ nguyên hai xác suất cảnh báo có điều kiện; trên hệ thống thật phải kiểm tra điều đó có còn đúng không. Nếu $P(A)=0$, nhóm điều kiện có xác suất 0 nên định nghĩa phép chia đang dùng không xác định; hàm lab báo lỗi thay vì trả tùy tiện 0 hay 1. Không thấy cảnh báo trong một mẫu hữu hạn chưa chứng minh $P(A)=0$ trong mô hình tổng thể.

## Bài 5

SD của một X là $\sqrt{0.2\times0.8}=0.4$. SE của tỷ lệ:

- n=100: $\sqrt{0.16/100}=0.04$.
- n=400: $\sqrt{0.16/400}=0.02$.

Giữ p cố định và các quan sát iid, muốn SE giảm một nửa thì cần n tăng 4 lần. SD của từng X không đổi. Thu thêm log chỉ giúp ước lượng tỷ lệ rõ hơn trong mô hình ấy, không tự chữa lỗi hệ thống hoặc làm p giảm.

Đầu ra mô phỏng không cần đúng tuyệt đối 0.2 và 0.02. Với seed và phiên bản được ghi, có thể tái tạo; đổi seed sẽ có dao động. Tính lý thuyết và lấy mẫu là hai đường kiểm tra khác nhau.

## Bài 6

Khoảng Wilson tương ứng xấp xỉ `[0.133366,0.288831]` và `[0.176377,0.225919]`. Cùng tỷ lệ nhưng mẫu lớn hơn có ít dao động lấy mẫu hơn dưới giả định iid, nên khoảng hẹp hơn.

0 lỗi trên 100 cho khoảng `[0,0.036995]`, không chứng minh p=0. Khi không có quan sát lỗi, vẫn còn các giá trị p dương tương thích với dữ liệu ở mức suy luận này.

Cách sửa phát biểu: “Ta dùng một phương pháp khoảng tin cậy danh nghĩa 95%; khi lặp lấy mẫu trong mô hình phù hợp, phương pháp hướng tới tỷ lệ phủ gần mức ấy.” Trong diễn giải frequentist, không gán trực tiếp xác suất posterior 95% cho tham số cố định sau khi đã có một khoảng cụ thể.

Hai lý do tỷ lệ phủ không bằng 95%:

1. Wilson có tỷ lệ phủ xấp xỉ, phụ thuộc n,p và tính rời rạc của số đếm. Với n=100,p=0.2, phép cộng xác suất nhị thức cho khoảng **0.940520**.
2. Mô phỏng chỉ lặp hữu hạn nên còn dao động Monte Carlo. 5000 lần, seed=2026 trong môi trường đã ghi cho **0.943800**.

Khoảng này nói về tham số tỷ lệ lỗi p, không nói phạm vi 95% thời gian xử lý hoặc kết quả một request mới. Lấy mẫu tương quan, chọn lọc hoặc môi trường đổi p còn có thể làm suy luận mất hiệu lực; tăng số lần mô phỏng không sửa được các giả định sai ấy.

Mở rộng: với k=20,n=100,z=1.96, D=1.038416, tâm khoảng khoảng 0.211098 và nửa độ rộng khoảng 0.077733. Vì vậy tâm Wilson không nhất thiết bằng đúng $\hat p=0.2$.

## Bài 7

Với ngưỡng 0.5, cả hai mô hình dự đoán `[1,0]`, accuracy=1.

| Mô hình | Brier | Log loss dùng ln |
|---|---:|---:|
| A: [0.8,0.2] | 0.04 | $-\ln0.8\approx0.223144$ |
| B: [0.6,0.4] | 0.16 | $-\ln0.6\approx0.510826$ |

Ví dụ A: hai bình phương sai số đều 0.04; xác suất gán cho kết quả thực sự xảy ra của cả hai mẫu đều 0.8. Brier/log loss thưởng cho việc gán nhiều xác suất hơn cho kết quả quan sát trong hai mẫu này.

Accuracy chỉ dùng nhãn sau ngưỡng nên bỏ thông tin về độ lớn xác suất. Hai mẫu quá ít để kết luận khả năng calibration trên phân bố triển khai. Điểm số thấp hơn trên một tập cụ thể cũng không tự chứng minh hiệu chuẩn tốt ở mọi nhóm hoặc môi trường tương lai.

## Bài 8

$H(0.5)=1$ bit. $H(0.2)=H(0.8)\approx0.721928$ bit. Sự đối xứng do đổi tên hai kết quả 0 và 1 không đổi độ bất định của phân bố Bernoulli.

Khi p=1 nhưng y=0, entropy đầu ra bằng 0, Brier=$(1-0)^2=1$, log loss=$-\ln(1-1)=+\infty$. Phân bố rất chắc chắn không có nghĩa dự đoán đúng.

Một p hoặc entropy không xác định nguyên nhân của bất định. Chẳng hạn p=0.5 có thể đến từ dữ liệu khó phân biệt với đầu vào hiện có, từ mô hình thiếu dữ liệu, hoặc cả hai. Cần phương pháp, giả định và bằng chứng bổ sung để phân tích các nguồn, không chỉ đọc một con số.

## Bài 9

- **Thiếu trạng thái cache:** mô hình thấy cùng đầu vào nhưng nhãn biến động. Đây có thể được bàn như dao động aleatoric đối với đầu vào hiện tại; ghi thêm trạng thái cache/tải có thể giải thích một phần. Không kết luận mọi dao động vĩnh viễn không giảm được.
- **Request 1000 lô:** dữ liệu huấn luyện chưa phủ vùng ấy. Cần đánh giá ngoại suy, thu dữ liệu đại diện và cân nhắc xử lý khi ngoài phạm vi. Một xác suất tự tin không chứng minh mô hình biết vùng mới.
- **Tỷ lệ nền đổi:** cần kiểm tra thay đổi phân bố, xác suất điều kiện và calibration theo thời gian. Có thể phải cập nhật mô hình hoặc quy trình vận hành; không sửa chỉ bằng tăng số chữ số thập phân của ước lượng cũ.

Các cách gọi giúp đặt câu hỏi, chưa phải phép phân rã đo lường đầy đủ. Thu thêm cùng loại dữ liệu ở vùng cũ, hoặc hỏi mô hình có tự tin không, chưa đủ giải quyết mọi tình huống.

## Bài 10

Tỷ lệ vẫn bằng 20%. Nhưng 100 bản sao của mỗi quan sát không tạo thêm thông tin độc lập về các request chưa quan sát. Dùng n=10 000 như các phép thử iid mới sẽ tạo khoảng hẹp giả tạo; lượng quan sát độc lập ban đầu vẫn là 100 trong giả định đề bài.

Một mẫu chỉ lấy lúc tải thấp có thể ước lượng rất chính xác **nhóm tải thấp**, nhưng không đại diện cho nhóm tải cao. Sai số do chọn mẫu khác sai số ngẫu nhiên; một khoảng hẹp không bảo đảm không thiên lệch.

Nên ghi thời gian, cơ chế lấy mẫu, điều kiện tải, nhóm request, phiên bản hệ thống và đơn vị độc lập thực sự. Nếu dữ liệu có cụm hoặc chuỗi thời gian, cần cách đánh giá phản ánh cấu trúc đó. Bài 5 sẽ phát triển việc tách dữ liệu theo mục tiêu triển khai.

## Nếu còn vướng

- Nhầm Bayes: bỏ công thức, đếm lại đúng nhóm làm mẫu số.
- Nhầm SD/SE: viết rõ đang mô tả từng request hay một tỷ lệ tính trên nhiều request.
- Nhầm khoảng tin cậy: quan sát hình nhiều lần lấy mẫu, giữ p thật cố định và để khoảng thay đổi.
- Nhầm confidence với đúng/sai: thử p=1 khi y=0 và đối chiếu entropy/log loss.

Tiến bộ cần dựa trên bài làm và giải thích của người học; việc tài liệu hoặc code chạy xong không thay thế bằng chứng đó.
