# Bài tập — Mô hình học từ dữ liệu

[Về bài học](README.md) · [Lời giải để tự đối chiếu](solutions.md)

Mọi số liệu dưới đây là giả định. Làm bằng tay hoặc viết dự đoán trước, sau đó mới chạy code. Không cần gửi câu trả lời trong chat để học tiếp.

## Bài 1. Tính dự đoán và loss

Với dữ liệu `(1, 3), (2, 5), (3, 7)` và $\hat y=1.5x+0.5$:

1. Tính từng dự đoán, sai số và bình phương sai số.
2. Tính MSE và nêu đơn vị.
3. So với $w=1,b=0$, ứng viên mới tốt hơn theo thước đo nào? Đã đủ để kết luận tốt hơn trong thực tế chưa?

## Bài 2. Vai trò của bias

Dự đoán điều gì xảy ra khi buộc $b=0$. Từ thư mục bài 1, chạy:

```bash
python3 lab.py --variant no-bias
```

1. Ghi tham số và MSE tìm được.
2. Vì sao không thể chọn một $w$ duy nhất khớp cả ba mẫu?
3. Phần mở rộng: nếu không giới hạn lưới 0.1, $w$ tốt nhất có nhất thiết là số code tìm được không?

## Bài 3. Dữ liệu có nhiễu

Biến thể sau dùng `(1, 3.2), (2, 4.7), (3, 7.1)`:

```bash
python3 lab.py --variant noisy
```

1. Ghi MSE và một bộ tham số tìm được.
2. Vì sao không có đường thẳng nào đi qua chính xác cả ba điểm?
3. Loss lớn hơn 0 có tự động chứng minh thuật toán sai không?
4. Nếu làm bước lưới nhỏ hơn, loss có thể giảm không? Giảm loss huấn luyện có đủ chứng minh tốt hơn trên dữ liệu mới không?

## Bài 4. Phân biệt tham số và đầu vào

Giải thích vai trò của `x`, `w`, `b`, `train` và bước lưới `0.1`. Khi phục vụ một request mới bằng mô hình đã huấn luyện, những giá trị nào thường giữ nguyên? Những giá trị nào đến từ request?

## Bài 5. Phát hiện kết luận sai

Một đồng nghiệp nói: “MSE test bằng 0 ở mẫu `(4, 9)`, vậy hệ thống chắc chắn dự đoán chính xác latency khi xử lý 100 lô.”

Nêu ít nhất hai lý do kết luận này chưa có cơ sở. Đề xuất thêm dữ liệu hoặc phép kiểm tra để đánh giá tốt hơn.

## Tiêu chí tự đánh giá

- Tính được bài 1, có đơn vị và không nhầm MSE với sai lệch trung bình tính bằng ms.
- Giải thích được bài 2 bằng các phương trình, không chỉ thuật lại đầu ra code.
- Nhận ra ba điểm có nhiễu không nhất thiết cùng nằm trên một đường thẳng.
- Phân biệt được tham số học từ dữ liệu và cấu hình do người thiết kế chọn.
- Không suy độ tin cậy thực tế từ một mẫu test giả định.

Nếu vướng một ý, đối chiếu đúng phần tương ứng trong [lời giải](solutions.md), rồi thử tự giải thích lại với một ví dụ khác.
