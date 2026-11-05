# Quản lý hồ sơ học và nghiên cứu

## Đầu vào

Yêu cầu lưu/tiếp tục/cập nhật, vị trí hồ sơ nếu đã có, nội dung phiên học và nguồn đã truy cập. Không mặc định có bộ nhớ bền giữa các chat hoặc quyền truy cập tài liệu ở dịch vụ khác.

## Quy trình

1. Khi tiếp tục, tìm hồ sơ trong vị trí người dùng chỉ định hoặc thư mục dự án được phép. Đọc trạng thái hiện tại trước khi sửa. Nếu không có, nói chưa tìm thấy và yêu cầu bối cảnh tối thiểu; không bịa lịch sử học.
2. Khi lưu lần đầu, dùng một tệp Markdown nhỏ theo [mẫu hồ sơ](../topic-note.md). Ưu tiên vị trí người dùng chọn; nếu chưa chọn, dùng `outputs/learning-notes/` trong workspace được phép. Tên tệp là slug chủ đề dễ đọc; không chèn thông tin riêng tư vào tên.
3. Tách thông tin người dùng đã xác nhận, giả định còn mở, nội dung đã học và bằng chứng về khả năng vận dụng. “Đã trình bày” khác “đã kiểm tra hiểu”.
4. Lưu nguồn bằng liên kết và trích yếu đủ kiểm tra; không sao chép toàn văn tài liệu chỉ để giữ lịch sử. Giữ ngày/phiên bản cho dữ kiện thay đổi theo thời gian; không bịa ngày truy cập nếu chưa truy cập.
5. Khi cập nhật kết luận, ghi điều thay đổi và lý do, giữ lịch sử ngắn cho thay đổi có ý nghĩa. Không xóa ghi chú của người dùng hoặc ghi đè hồ sơ không liên quan.
6. Khi tiếp tục học, tóm tắt điểm dừng và phần kiến thức cần tiếp theo; kiểm tra lại nhận định phụ thuộc thời gian trước khi dùng. Không tái kiểm tra toàn bộ kiến thức ổn định nếu không cần.
7. Chỉ ghi đã lưu khi thao tác ghi thành công. Nếu môi trường không cho ghi, cung cấp nội dung để người dùng lưu và nêu rõ chưa lưu được. Không cam kết tự đồng bộ hoặc tự theo dõi nếu chưa có công cụ và cấu hình tương ứng.

## Đầu ra

Liên kết tới hồ sơ đã tạo/cập nhật và tóm tắt thay đổi; hoặc phần tiếp nối bài học dựa trên hồ sơ đã đọc. Khi lượng bằng chứng lớn, có thể tách sổ bằng chứng ra tệp riêng và liên kết từ hồ sơ, nhưng không tạo nhiều tệp nhỏ không cần thiết.

## Tiêu chí chất lượng

- Một phiên mới có thể tiếp tục từ hồ sơ mà phân biệt được điều đã biết, đã giả định và chưa xác minh.
- Nguồn và kết luận truy lại được; thay đổi quan trọng có lý do.
- Mức thành thạo dựa trên sản phẩm/câu trả lời, không dựa trên việc agent từng giải thích.
- Chỉ lưu thông tin liên quan mục tiêu học; không tự đưa dữ liệu ra dịch vụ ngoài.
