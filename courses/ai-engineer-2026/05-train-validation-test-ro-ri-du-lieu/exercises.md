# Bài tập — Train/validation/test và rò rỉ dữ liệu

[Về bài học](README.md) · [Lời giải](solutions.md)

Các ví dụ là giả định. Ghi lựa chọn và lý do trước khi chạy code. Lệnh bên dưới chạy từ thư mục bài 5 với Python có NumPy.

## Bài 1. Gán đúng vai trò dữ liệu

Mỗi thao tác nên dùng tập nào hoặc cần sửa ra sao?

1. Tính gradient của trọng số.
2. Học trung bình để điền giá trị thiếu.
3. Chọn checkpoint theo loss mỗi epoch.
4. Chọn ngưỡng phân loại theo chi phí bỏ sót/báo nhầm.
5. Thử 50 seed rồi báo seed có điểm test cao nhất.
6. Đổi đơn vị ms sang giây bằng phép chia cố định 1000 trước khi chia dữ liệu.
7. Sau khi chốt cấu hình từ trước, refit trên train+validation rồi chấm test chưa dùng chọn.

Giải thích vì sao “không tính gradient trên validation” chưa có nghĩa validation hoàn toàn không ảnh hưởng mô hình cuối.

## Bài 2. Tính lại ví dụ chọn k

Dùng dữ liệu train/validation của bài học và quy tắc hòa khoảng cách ưu tiên dòng train trước.

1. Tính ba dự đoán validation khi k=5 và MSE của chúng.
2. Dùng bảng train/validation MSE để chọn k. Vì sao không chọn bằng train MSE nhỏ nhất?
3. Với k đã chốt, tự tính hai dự đoán test và MSE cuối.
4. Nếu test MSE cao hơn validation MSE, có được chọn lại k bằng test rồi gọi điểm mới là đánh giá độc lập không?

Kiểm tra bằng `python3 lab.py --mode select`.

## Bài 3. Fit scaler trên dữ liệu nào?

Train có `[1,3]`, test có `[8]`. Dùng chuẩn hóa $(x-\mu)/s$, SD chia n.

1. Tính mean/SD trên train và z của test khi dùng đúng tham số train.
2. Tính lại nếu fit trên cả `[1,3,8]`.
3. Chưa dùng nhãn test, vì sao vẫn có thể sai giao thức đánh giá trên dữ liệu mới?
4. Nếu điểm mô hình không thay đổi sau hai cách, điều đó có chứng minh cách fit trên cả test hợp lệ không?

```python
import numpy as np
from lab import fit_scaler, transform
train = np.array([1.,3.])
test = np.array([8.])
mean, scale = fit_scaler(train)
print(mean, scale, transform(test, mean, scale))
leaked_mean, leaked_scale = fit_scaler(np.r_[train,test])
print(leaked_mean, leaked_scale, transform(test, leaked_mean, leaked_scale))
```

## Bài 4. Không trùng dòng nhưng trùng người dùng

Có các quan sát `A1,A2,B1,B2,C1,C2`, ký tự đầu là user ID. Một phép chia cho train=`A1,B1,C1`, validation=`A2`, test=`B2,C2`.

1. Có trùng row ID không? Có trùng user giữa train/test không?
2. Phép đo phù hợp hơn với user đã biết hay user mới?
3. Đề xuất phép chia nhỏ minh họa không trùng user giữa ba tập; nêu giới hạn vì số nhóm quá ít.
4. Chạy mode `groups`: vì sao mô hình ghi nhớ đạt 100% ở phép chia theo dòng và 50% ở phép chia theo user, dù cùng số dòng 48/12/12?

Không kết luận mọi dự án phải giữ user hoàn toàn riêng nếu chưa xác định mục tiêu triển khai.

## Bài 5. Kiểm tra thông tin có sẵn

Mô hình phải dự đoán tại t=5. Giả sử mọi timestamp là thứ tự thời gian rõ ràng; thông tin phải có sẵn trước t=5.

| Feature | Sự kiện xảy ra | Giá trị có sẵn / được tạo | Mô tả |
|---|---:|---:|---|
| F1 | 4 | 4 | Trạng thái đã đo và gửi tới hệ thống |
| F2 | 4 | 6 | Số liệu quá khứ nhưng ingest chậm |
| F3 | 6 | 6 | Thông tin sau khi xử lý xong |
| F4 | 3 | 8 | Giá trị sửa/backfill sau này, gắn event time cũ |

1. Feature nào có thể dùng theo dữ kiện này?
2. Vì sao chỉ kiểm tra `event_time < 5` chưa đủ?
3. Với F4, cần loại hoàn toàn loại feature ấy hay tìm được phiên bản lịch sử phù hợp?
4. Nếu target chính là latency, đưa latency sau hoàn tất vào đầu vào có giúp chứng minh khả năng dự đoán ngay lúc nhận request không?

## Bài 6. Nhãn tương lai và refit muộn hơn

Dùng lịch trong bài: decision t=1…14; nhãn có tại t+2; lần fit trước validation dùng cutoff nghiêm ngặt `<9`.

1. Mẫu t=6,7,8 có dùng được để fit trước t=9 không?
2. Vì sao validation t=9,10 chưa thể dùng đủ để chọn cấu hình trước t=11?
3. Khi refit trước test t=13, nếu đã định trước dùng các mẫu quyết định t=1…10, những nhãn nào đã có theo điều kiện `<13`?
4. Test t=13,14 chỉ có nhãn khi nào? Chờ nhãn để chấm offline có phải leakage không?
5. Vì sao số mẫu gap cố định chưa tự bảo đảm mọi feature/nhãn đều hợp lệ?

Kiểm tra `python3 lab.py --mode time`.

## Bài 7. Tiền xử lý trong cross-validation

Một người fit scaler trên toàn development rồi mới chia CV. Họ nói “test vẫn khóa nên không sao”.

1. Chỉ ra thông tin validation-fold đã đi vào đâu.
2. Viết lại thứ tự fit/transform đúng ở mỗi fold.
3. Kiểm tra mean scaler ở fold đầu bằng snippet; so với mean toàn development.
4. Vì sao k tối ưu qua CV nhỏ này có thể khác k tối ưu qua một validation cố định?

```python
import numpy as np
from lab import training_data, cv_folds, fit_scaler
x, y = training_data()
folds = cv_folds(seed=2026)
val_idx = folds[0]
train_idx = np.concatenate(folds[1:])
print('train fold:', train_idx, 'validation fold:', val_idx)
print('fold scaler:', fit_scaler(x[train_idx]))
print('whole-development scaler:', fit_scaler(x))
```

## Bài 8. Refit là một bước được định trước

**Mở rộng tính tay.** Giả sử ngay từ đầu quy trình đã quy định: chọn k từ validation, sau đó refit k=3 trên train+validation. Thứ tự dữ liệu sau ghép là toàn bộ train cũ trước, rồi validation theo thứ tự cũ. Quy tắc hòa giữ nguyên.

1. Tính hai dự đoán tại test x=1.5 và 3.5 khi tập nhớ đã mở rộng.
2. Tính MSE. Vì sao không được dùng kết quả để lựa giữa “refit” và “không refit” sau khi đã nhìn test rồi báo như một đánh giá độc lập?
3. Nếu còn scaler/imputer học từ dữ liệu, khi refit phải làm gì với chúng?

```python
import numpy as np
from lab import training_data, validation_data, test_data, knn_predict, mse
x, y = training_data()
vx, vy = validation_data()
final_x, final_y = np.r_[x,vx], np.r_[y,vy]
tx, ty = test_data()
pred = knn_predict(final_x, final_y, tx, k=3)
print(pred, mse(ty, pred))
```

## Bài 9. Validation cũng có thể bị tối ưu quá mức

Nhóm đã thử hàng nghìn kiến trúc/prompt trên một validation rất nhỏ rồi báo điểm validation tốt nhất như hiệu quả với dữ liệu chưa thấy.

1. Điểm nào trong lập luận còn thiếu?
2. Test ngoài cùng hoặc nested CV giúp tách vai trò ra sao?
3. Nếu test đã liên tục được dùng để chọn prompt, đổi tên nó thành `final_test_v2` có sửa được không?
4. Chạy lại nguyên chương trình đã chốt để tái lập số có tự động làm test “bẩn” không? Phân biệt chạy lại với thay lựa chọn theo kết quả.

## Bài 10. Thiết kế một phép đánh giá cho ticket

Nhiệm vụ giả định: ngay khi ticket được mở, dự đoán có cần ưu tiên xử lý hay không. Sản phẩm nhắm tới **khách hàng mới ở giai đoạn tương lai**. Dữ liệu gồm nhiều tin nhắn trong cùng ticket, nhiều ticket cùng khách hàng; nhãn chỉ được xác nhận khi xử lý xong.

Đề xuất ngắn gọn:

- Đơn vị dự đoán và grouping cần theo dõi.
- Cách kết hợp thời gian với nhóm train/validation/test.
- Hai feature hợp lệ và hai feature có nguy cơ lộ tương lai.
- Ranh giới fit tiền xử lý, tuning và chấm cuối.
- Những thông tin phải ghi để người khác kiểm tra lại đánh giá.

Không cần mua dữ liệu, gọi API hoặc xây mô hình thật. Mục tiêu là thiết kế phép đo đúng câu hỏi trước khi tối ưu điểm.

## Tiêu chí tự đánh giá

Giải được bài 2–3; tìm được nhóm trùng ở bài 4; dùng đúng availability time ở bài 5–6; giải thích vì sao phải fit lại pipeline trong từng fold. Bài 8 là mở rộng, giúp phân biệt refit hợp lệ với chọn quy trình dựa trên test. Xem [lời giải](solutions.md) để tự đối chiếu.
