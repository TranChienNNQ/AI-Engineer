# Lời giải — Mô hình học từ dữ liệu

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các kết quả dưới đây được đối chiếu bằng `lab.py` và tính toán độc lập. Sai khác ở các chữ số rất nhỏ có thể do số thực dấu phẩy động.

## Bài 1

| $x$ | $y$ | $1.5x+0.5$ | Sai số | Bình phương sai số |
|---:|---:|---:|---:|---:|
| 1 | 3 | 2.0 | −1.0 | 1.00 |
| 2 | 5 | 3.5 | −1.5 | 2.25 |
| 3 | 7 | 5.0 | −2.0 | 4.00 |

MSE $=(1+2.25+4)/3=7.25/3\approx2.4167$ ms². Nhỏ hơn $9.6667$ ms² của $w=1,b=0$, nên tốt hơn **về MSE trên tập huấn luyện này**. Chưa đủ kết luận hiệu quả trên dữ liệu thực chưa thấy hoặc theo những mục tiêu khác như chi phí và độ trễ.

## Bài 2

Với lưới của chương trình: $w=2.4,b=0$, dự đoán lần lượt 2.4, 4.8, 7.2. MSE $=(0.36+0.04+0.04)/3\approx0.1467$ ms².

Để khớp từng điểm khi $b=0$, cần $w=3$, $w=2.5$ và $w=7/3$ tương ứng. Một tham số không thể đồng thời mang ba giá trị khác nhau. Vì vậy không thể đạt MSE bằng 0.

Phần mở rộng: với $b=0$, MSE là:

$$
L(w)=\frac{(w-3)^2+(2w-5)^2+(3w-7)^2}{3}
=\frac{14w^2-68w+83}{3}
$$

Hoàn thành bình phương, biểu thức nhỏ nhất tại $w=17/7\approx2.428571$, cho MSE $1/7\approx0.142857$ ms². Giá trị này không nằm trên lưới bước 0.1. Do đó kết quả grid search chưa phải nghiệm tối ưu trên toàn bộ số thực. Không bắt buộc tự suy ra công thức này để hoàn thành bài cơ bản.

## Bài 3

Trên lưới 0.1, MSE tốt nhất xấp xỉ **0.0467 ms²**. Hai bộ $(w,b)=(2.0,1.0)$ và $(1.9,1.2)$ đều cho giá trị đó về mặt toán học. Số thực và thứ tự duyệt có thể khiến code trả một trong hai; không nên chỉ so một bộ tham số để kết luận chương trình sai.

Hiệu hai giá trị đích liên tiếp là 1.5 rồi 2.4, trong khi $x$ đều tăng 1. Một đường thẳng có mức tăng không đổi, nên không thể đi qua cả ba điểm. Loss dương không tự chứng minh lỗi code.

Với lưới mịn hơn có chứa $w=1.95,b=1.10$, các dự đoán là 3.05, 5.00, 6.95. MSE $=(0.0225+0.09+0.0225)/3=0.045$ ms², thấp hơn kết quả lưới 0.1. Lưới mịn hơn cũng cần thử nhiều ứng viên hơn. Chất lượng trên dữ liệu mới vẫn phải đánh giá riêng, không được kết luận từ train loss.

## Bài 4

`x` là đầu vào; `w,b` là tham số; `train` là dữ liệu dùng để chọn tham số. Bước lưới 0.1 là cấu hình của thuật toán tìm kiếm, khác tham số mô hình. Khi suy luận với phiên bản mô hình cố định, `w,b` giữ nguyên còn `x` đến từ request. `train` không phải đầu vào của mỗi lần dự đoán.

Chương trình minh họa giữ tham số trong bộ nhớ. Một sản phẩm thực tế cần cơ chế lưu/tải phiên bản tham số và xử lý dữ liệu đầu vào nhất quán.

## Bài 5

Các lý do có thể nêu:

- Một mẫu test không đại diện cho mọi kiểu request hoặc điều kiện tải.
- Dữ liệu được tạo theo đường thẳng; hệ thống thực chưa được đo để xác nhận giả định ấy.
- 100 lô nằm xa khoảng 1–3 lô của dữ liệu huấn luyện: ngoại suy xa.
- Latency có thể phụ thuộc cache, tải CPU, kiểu truy vấn và yếu tố khác chưa có trong đầu vào.

Nên đo nhiều kích thước request, gồm phạm vi vận hành dự kiến, nhiều lần dưới các điều kiện tải phù hợp. Giữ dữ liệu đánh giá độc lập, kiểm tra sai số theo từng nhóm và điều tra các lỗi lớn. Đó là đề xuất thí nghiệm; chưa có kết quả thực nghiệm nào được tạo trong bài này.

## Khi nào có thể học tiếp?

Nếu giải thích được tham số được chọn như thế nào và vì sao train loss chưa đủ, có thể sang đạo hàm/gradient descent. Nếu nhầm giữa $y$ và $\hat y$, quay lại bảng tính tay. Nếu chương trình chạy nhưng chưa hiểu vòng tìm kiếm, thêm `print(w, b, loss)` cho vài ứng viên, không cần in toàn bộ lưới.
