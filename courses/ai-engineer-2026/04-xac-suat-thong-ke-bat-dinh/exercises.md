# Bài tập — Xác suất, thống kê và bất định

[Về bài học](README.md) · [Lời giải](solutions.md)

Số liệu đều là giả định. Tính tay trước, sau đó dùng code đối chiếu. Lệnh chạy từ thư mục bài 4 với Python đã cài NumPy. Có thể tự đọc toàn bộ lời giải, không cần phản hồi trong chat.

## Bài 1. Biến cố, điều kiện và mật độ

Dùng bảng 10 000 request trong bài: có lỗi 100; vừa lỗi vừa cảnh báo 90; cảnh báo nhầm 495; không lỗi và không cảnh báo 9405.

1. Tính $P(E\cap A)$, $P(A)$, $P(E\cup A)$.
2. Tính $P(E\mid A^c)$: tỷ lệ lỗi trong nhóm không cảnh báo.
3. Hai biến cố E và A có độc lập không? Kiểm tra bằng một đẳng thức cụ thể.
4. **Mở rộng:** một thời gian T được mô hình hóa liên tục, mật độ bằng 2 giây⁻¹ trên [0,0.5] giây, bằng 0 ngoài khoảng đó. Mật độ lớn hơn 1 có sai không? Tính $P(0.1\le T\le0.2)$ và $P(T=0.15)$ theo mô hình liên tục.

## Bài 2. Kỳ vọng không phải kết quả bắt buộc

Giả sử 10 request độc lập, mỗi request lỗi với xác suất p=0.2. Gọi K là số lỗi.

1. Tính kỳ vọng và phương sai K.
2. Tính xác suất không có lỗi, đúng 2 lỗi, và có ít nhất 1 lỗi.
3. Vì sao quan sát đúng 2 lỗi không được đảm bảo dù kỳ vọng bằng 2?
4. Nếu một sự cố chung khiến nhiều request lỗi cùng lúc, công thức nhị thức còn tự động đúng không?

Kiểm tra bằng `python3 lab.py --mode probability` và hàm `binomial_pmf`.

## Bài 3. Tự tính thống kê mô tả

Dữ liệu latency là `[2,4,4,6]` ms.

1. Tính mean, median, tổng bình phương độ lệch.
2. Tính phương sai chia n, phương sai chia n−1 và hai độ lệch chuẩn, nêu đơn vị.
3. Nếu dự đoán hằng số c=4, MSE là bao nhiêu? Với c=3 thì sao?
4. Thay số 6 bằng 60: dự đoán mean hay median thay đổi nhiều hơn, rồi kiểm tra.

```python
from lab import descriptive_stats
print(descriptive_stats([2,4,4,6]))
print(descriptive_stats([2,4,4,60]))
```

## Bài 4. Bayes khi tỷ lệ nền thay đổi

Trong một hệ thống giả định khác, 10% request lỗi; hệ thống cảnh báo 90% số request lỗi và cảnh báo nhầm 5% số request không lỗi.

1. Lập bảng đếm trên 1000 request theo đúng các tỷ lệ đó.
2. Tính xác suất lỗi khi có cảnh báo bằng cả bảng đếm và công thức Bayes.
3. So với ví dụ tỷ lệ nền 1%, vì sao xác suất này thay đổi?
4. Trong một mô hình khác giả sử $P(A)=0$, vì sao không thể tính $P(E\mid A)$ bằng phép chia trong bài? Phân biệt giả định ấy với việc không thấy cảnh báo trong một mẫu hữu hạn.

Kiểm tra `python3 lab.py --mode bayes`.

## Bài 5. Độ lệch chuẩn và sai số chuẩn

Trong mô hình iid Bernoulli p=0.2:

1. Tính SD của một quan sát X.
2. Tính SE của tỷ lệ khi n=100 và n=400.
3. Muốn giảm SE một nửa thì cần tăng n bao nhiêu lần? Giả định nào đang được giữ nguyên?
4. Có thêm log có làm xác suất lỗi thật tự giảm không?

Có thể mô phỏng n=400 bằng:

```python
import numpy as np
from lab import sample_proportions
estimates = sample_proportions(p=0.2, n=400, repeats=5000, seed=2026)
print(np.mean(estimates), np.std(estimates, ddof=1))
```

## Bài 6. Khoảng tin cậy và giới hạn của nó

Chạy `python3 lab.py --mode intervals --repeats 5000 --seed 2026`.

1. Đọc khoảng cho 20/100 và 200/1000. Vì sao cùng tỷ lệ mẫu mà độ rộng khác nhau?
2. Khoảng cho 0/100 có chứng minh tỷ lệ lỗi thật bằng 0 không?
3. Sửa câu: “Sau khi quan sát dữ liệu, p có xác suất 95% thuộc khoảng Wilson này.”
4. Vì sao tỷ lệ phủ mô phỏng và mức danh nghĩa có thể không bằng nhau? Phân biệt dao động Monte Carlo với độ phủ xấp xỉ của chính phương pháp.
5. Khoảng này có chứa 95% thời gian xử lý của request mới không?

**Mở rộng:** tự thay k=20,n=100,z=1.96 vào công thức tâm và nửa độ rộng; đối chiếu hàm `wilson_interval`.

## Bài 7. Cùng nhãn, khác chất lượng xác suất

Hai mẫu có nhãn thật `y=[1,0]`. A dự đoán `[0.8,0.2]`, B dự đoán `[0.6,0.4]`. Quy tắc nhãn là p≥0.5 → 1.

1. Tính accuracy, Brier và log loss của từng mô hình.
2. Vì sao accuracy không phân biệt được hai cách dự đoán này?
3. Hai mẫu có đủ chứng minh A hiệu chuẩn tốt hơn trên dữ liệu tương lai không?

```python
from lab import probability_scores
print(probability_scores([1,0], [0.8,0.2]))
print(probability_scores([1,0], [0.6,0.4]))
```

## Bài 8. Entropy thấp có thể đi cùng sai lầm

1. Tính H(0.5), H(0.2) và H(0.8) bằng công thức entropy Bernoulli.
2. Khi mô hình dự đoán p=1 cho nhãn 1 nhưng nhãn thật là 0, entropy, Brier và log loss là gì?
3. Vì sao một giá trị entropy không đủ tách bất định aleatoric và epistemic?

Kiểm tra bằng `binary_entropy` và `probability_scores` trong lab. Không dùng log trực tiếp của 0 để ép ra một số hữu hạn.

## Bài 9. Thảo luận nguồn bất định

Xét ba tình huống giả định:

- Cùng số lô và số truy vấn, nhưng latency dao động vì trạng thái cache chưa có trong đầu vào.
- Mô hình chỉ thấy request 1–10 lô, nay nhận 1000 lô.
- Tỷ lệ lỗi trong log cũ là 1%, nhưng sau đổi hạ tầng lên 10%.

Với từng tình huống, nêu điều chưa biết, một cách thu thập bằng chứng hoặc cải thiện phù hợp, và một điều không thể kết luận chỉ từ xác suất đầu ra. Không cần ép mọi tình huống vào một phân loại duy nhất.

## Bài 10. Nhiều dòng không đồng nghĩa nhiều mẫu độc lập

Bạn có 100 request độc lập, trong đó 20 lỗi. Một đồng nghiệp lặp mỗi dòng 100 lần, coi đó là 10 000 request mới và tính khoảng Wilson hẹp hơn.

1. Tỷ lệ quan sát có đổi không?
2. Số quan sát độc lập có tăng không? Khoảng mới có phản ánh đúng lượng thông tin không?
3. Tương tự, tại sao một mẫu log chỉ lấy lúc tải thấp có thể cho ước lượng rất ổn định nhưng không đại diện cho lúc tải cao?
4. Đề xuất cách ghi nhận điều kiện thu thập để đánh giá được kết luận.

## Tiêu chí tự đánh giá

- Tính xác suất có điều kiện với đúng mẫu số, không đảo điều kiện.
- Nói rõ n hay n−1 và đơn vị phương sai/SD.
- Phân biệt SD của dữ liệu với SE của ước lượng.
- Giải thích đúng ý nghĩa khoảng tin cậy và điều kiện iid.
- Không xem confidence/entropy/accuracy như bằng chứng đầy đủ cho độ tin cậy của hệ thống.

Nếu chưa chắc, xem [lời giải](solutions.md) của đúng bài đang vướng, rồi thử lại với các con số khác.
