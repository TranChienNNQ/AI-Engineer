# Lời giải — Train/validation/test và rò rỉ dữ liệu

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các phép tính đã được đối chiếu với mã đi kèm. Dữ liệu nhỏ, dựng sẵn để thấy luồng thông tin; không suy ra hiệu quả triển khai thực tế từ các điểm số này.

## Bài 1

1. Gradient dùng train của lần fit tương ứng.
2. Mean điền thiếu cũng học từ train; trong CV phải là train-fold, không toàn development.
3. Checkpoint được chọn bằng validation; train vẫn dùng cho gradient.
4. Ngưỡng được chọn trong development/validation theo tiêu chí định trước, không chọn bằng test cuối.
5. Chọn seed tốt nhất bằng test đã để phản hồi test tham gia lựa chọn. Điểm tốt nhất không còn là đánh giá độc lập của quy trình đã chốt trước.
6. Chuyển đơn vị bằng hệ số cố định không học gì từ test nên bản thân phép biến đổi đó không gây leakage.
7. Refit được định trước có thể hợp lệ, miễn fit lại các thành phần đúng trên tập được phép và test chưa tham gia chọn. Một số bước như calibration/early stopping có thể cần quy trình development riêng, không gộp mù quáng.

Validation tác động thông qua lựa chọn cấu hình/checkpoint/ngưỡng. Không trực tiếp tính gradient trên nó không làm ảnh hưởng đó biến mất.

## Bài 2

Với k=5, các dự đoán validation là `(12,12,13)`:

- x=0.5 dùng train x=0,1,2,3,4, tổng y=60 nên trung bình 12.
- x=2.5 dùng x=2,3,1,4,0, tổng y=60 nên trung bình 12.
- x=4.5 dùng x=4,5,3,2,1, tổng y=65 nên trung bình 13.

MSE $=(1.5^2+(-0.5)^2+(-1.5)^2)/3=19/12\approx1.583333$ ms². k=3 có validation MSE $19/108\approx0.175926$, nhỏ nhất trong các ứng viên nên được chọn. k=1 có train MSE 0 vì tự lấy chính mẫu train làm láng giềng gần nhất, không phải bằng chứng dự đoán mới tốt.

Test x=1.5 dùng x=1,2,0, cho $(12+11+10)/3=11$. Test x=3.5 dùng x=3,4,2, cho $(14+13+11)/3=38/3$. MSE $=(1/4+25/36)/2=17/36\approx0.472222$ ms².

Đổi k theo điểm test là một quyết định mới dựa trên test; muốn đánh giá độc lập sau thay đổi cần một giao thức/dữ liệu ngoài quá trình chọn ấy. Validation/test khác điểm là bình thường trên hai mẫu hữu hạn khác nhau.

## Bài 3

Train mean=2, SD chia n=1, nên test z=$(8-2)/1=6$.

Fit cả `[1,3,8]`: mean=4, tổng bình phương độ lệch $9+1+16=26$, variance $26/3$, SD $\sqrt{26/3}\approx2.943920$. Test z=$4/\sqrt{26/3}\approx1.358732$.

Những giá trị giữ riêng đã thay đổi tham số tiền xử lý dù chưa có nhãn. Với mục tiêu đo khả năng trên dữ liệu chưa được dùng trong phát triển, đó vẫn là thông tin đi sai ranh giới. Một mô hình bất biến với biến đổi ấy có thể cho cùng điểm, nhưng không làm đổi nguồn gốc của các tham số đã fit.

## Bài 4

Không trùng row ID, nhưng train và test cùng có user B,C. Phép đo gần với trường hợp user đã biết, không đủ để kết luận cho user mới.

Một phép chia nhỏ không trùng user là train=`A1,A2`, validation=`B1,B2`, test=`C1,C2`. Nhưng mỗi phía chỉ một nhóm: không đủ đại diện, dễ lệch nhãn và không đo được nhiều kiểu khách hàng. Đây chỉ là ví dụ grouping, không phải khuyến nghị triển khai.

Trong lab 72 dòng, cách theo dòng để cả 12 user được nhớ ở train, nên test có nhãn đã nhớ và đạt 100%. Cách theo user giữ user 10,11 mới hoàn toàn; fallback train là 0, đúng một trong hai nhãn cân bằng nên đạt 50%. Hai phép chia cùng số dòng nhưng khác số user đã thấy và khác câu hỏi đánh giá.

## Bài 5

F1 hợp lệ theo dữ kiện: sự kiện và thời điểm có sẵn đều trước t=5. F2 không hợp lệ vì dù sự kiện ở t=4, hệ thống chỉ nhận giá trị ở t=6. F3 là tương lai. F4 là giá trị sửa sau t=5, không được dùng nó để tái hiện quyết định cũ.

Với F4, có thể dùng phiên bản đã thực sự có trước t=5 nếu truy được lịch sử, hoặc áp dụng cơ chế thiếu dữ liệu đã định; không nhất thiết loại vĩnh viễn cả loại feature. Event time cũ không chứng minh giá trị đã có tại thời điểm dự đoán.

Latency sau hoàn tất là đích hoặc hậu quả sau xử lý; đưa nó vào đầu vào của nhiệm vụ dự đoán latency ngay khi nhận request không chứng minh khả năng dự đoán mong muốn. Nếu nhiệm vụ khác là phân tích hồi cứu thì phải định nghĩa lại rõ, không báo hai nhiệm vụ như một.

## Bài 6

Nhãn của t=6 có ở 8, thỏa `<9`; t=7 có ở 9 và t=8 có ở 10, không thỏa. Train của lần fit trước validation chỉ giữ t=1…6 trong lịch này.

Validation t=9,10 có nhãn ở 11,12, nên trước t=11 chưa biết đủ cả hai. Phải đợi theo lịch rồi chọn cấu hình trước lần đánh giá tiếp theo. Khi refit trước t=13 với các mẫu t=1…10, toàn bộ nhãn có ở t=3…12, đều thỏa `<13`; các mẫu từng bị loại ở lần fit cũ không bị cấm vĩnh viễn.

Nhãn test t=13,14 chỉ có ở 15,16. Dùng chúng **sau đó để chấm các dự đoán đã tạo** là cần thiết cho đánh giá offline; dùng chúng để fit hoặc chọn trước thời điểm dự đoán mới là sai luồng.

Gap đếm dòng không tự biết các dòng cách nhau bao lâu, cửa sổ đích dài ngắn khác nhau, ingest chậm hay backfill. Cần kiểm tra thời điểm có sẵn thực và mục tiêu triển khai, không chỉ con số gap.

## Bài 7

Mean/scale toàn development đã nhận giá trị của validation-fold, nên mỗi mô hình trong fold không còn được xây chỉ từ train-fold. Test ngoài cùng vẫn khóa không xóa được rò rỉ bên trong quy trình chọn cấu hình.

Thứ tự đúng: chia fold → fit scaler trên train-fold → transform train/validation-fold bằng scaler ấy → fit mô hình train-fold → chấm validation-fold. Lặp lại từ đầu cho fold tiếp theo.

Với seed=2026, fold validation đầu chứa chỉ số 0,1; train-fold gồm 3,5,2,4 theo thứ tự ghép trong code. Mean train-fold bằng 3.5, scale $\sqrt{1.25}\approx1.118034$. Toàn development có mean 2.5, scale $\sqrt{35/12}\approx1.707825$.

Trong lab, CV ba fold chọn k=1 theo mean fold MSE≈2.833333 so với k=3≈3.666667. Validation cố định chọn k=3. Các điểm giữ riêng, cỡ tập fit và cách tổng hợp đã đổi; dữ liệu lại rất nhỏ nên khác lựa chọn là điều có thể xảy ra. Không được lấy giao thức có kết quả đẹp hơn sau khi xem số rồi mô tả là đã định từ đầu.

## Bài 8

Đã định trước refit và giữ k=3:

- x=1.5: ba láng giềng là x=1,2,0.5, có y=12,11,10.5; dự đoán $67/6\approx11.166667$.
- x=3.5: láng giềng x=3,4,2.5, có y=14,13,12.5; dự đoán $79/6\approx13.166667$.

Cả hai sai số đều −1/3, nên MSE bằng **1/9≈0.111111 ms²**. Cải thiện so với không refit ở đúng ví dụ này không chứng minh refit luôn cải thiện. Nếu dùng điểm test để quyết định có refit hay không, test đã ảnh hưởng thiết kế quy trình và không còn độc lập với lựa chọn ấy.

Khi refit, fit lại scaler/imputer và mô hình trên đúng tập train+validation được phép, giữ nguyên thiết kế đã chốt. Nếu có bước cần tập calibration hoặc early-stopping riêng thì xử lý theo quy trình tương ứng; không dùng test để thay thế.

## Bài 9

Thử rất nhiều lựa chọn có thể tối ưu theo nhiễu của một validation nhỏ. Điểm tốt nhất trên validation không tự là ước lượng độc lập cho cấu hình được chọn.

Test ngoài cùng chỉ chấm sau khi quy trình đã chốt; nested CV đặt việc chọn cấu hình trong vòng trong rồi dùng phần ngoài để chấm toàn quy trình. Cả hai vẫn cần tôn trọng nhóm, thời gian và ranh giới fit. Đổi tên một tập đã dùng để tuning không làm mất lịch sử phản hồi.

Chạy lại nguyên quy trình cố định để tái lập số không tự tạo một quyết định thích nghi mới. Thử một biến thể sau khi xem test, chọn biến thể có lợi rồi báo lại như chưa nhìn test thì khác. Nếu có sửa lỗi hoặc thay đổi thiết kế sau đánh giá, cần mô tả trung thực và đánh giá độc lập phù hợp khi kết luận chất lượng cuối.

## Bài 10

Một phương án có thể chấp nhận:

- **Đơn vị dự đoán:** ticket tại thời điểm mở. Gộp các tin nhắn thuộc cùng ticket, không để tin nhắn sau mở lọt vào đặc trưng thời điểm mở.
- **Nhóm:** theo dõi ticket ID và customer ID. Vì cần khách hàng mới, nhóm test không xuất hiện trong train/validation; nhóm validation cũng mới so với phần dùng fit để chọn.
- **Thời gian:** train là dữ liệu đủ nhãn trước cutoff phát triển; validation ở giai đoạn sau, chờ nhãn đủ rồi chọn; test ở giai đoạn muộn hơn trên khách hàng chưa thấy. Theo dõi cả thời điểm nhãn hoàn tất, không chỉ ngày mở ticket.
- **Feature hợp lệ:** nội dung lúc mở và mức dịch vụ đã biết lúc ấy. **Nguy cơ tương lai:** ghi chú đóng ticket hoặc tổng thời gian/nhãn xử lý sau hoàn tất; thống kê lịch sử cũng phải đúng as-of time.
- **Fit/tuning:** từ vựng, điền thiếu, biến đổi và mô hình chỉ fit trên train của lần đánh giá; chọn cấu hình/ngưỡng ở development. Sau khi chốt mới chấm test, báo theo các nhóm và mức độ hiếm của nhãn phù hợp.
- **Tái lập:** lưu phiên bản dữ liệu, mã chia, row/group ID, cutoff, availability của nhãn/feature, seed, pipeline, tiêu chí chọn và lịch sử thử nghiệm.

Cần đủ khách hàng độc lập ở mỗi phần để phép đo có giá trị; một khách hàng nhiều ticket không thay thế nhiều khách hàng mới. Cũng cần ghi mục tiêu, hạn chế và biến động môi trường. Đây là thiết kế đề xuất, chưa có thí nghiệm thật trên ticket nào được chạy.

## Nếu còn vướng

- Nhầm vai trò: hỏi “con số này dùng để cập nhật trọng số, chọn thiết kế hay báo cuối?”.
- Nhầm leakage với điểm xấu: vẽ mũi tên thông tin từ mỗi tập vào các bước fit/chọn.
- Nhầm đơn vị độc lập: truy về người dùng/ticket/tài liệu gốc, không chỉ row ID.
- Nhầm thời gian: theo dõi một mẫu qua lúc dự đoán, lúc có feature và lúc có nhãn.

Hoàn thành code không tự chứng minh người học đã hiểu; dùng lời giải thích và thiết kế phép đánh giá làm bằng chứng.
