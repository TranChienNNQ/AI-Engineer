# Lời giải — Precision, recall và mất cân bằng

[Về bài học](README.md) · [Về đề bài](exercises.md)

Kết quả số được đối chiếu bằng code. Mọi ví dụ là dữ liệu giảng dạy. Việc biên soạn xong không có nghĩa người học đã thành thạo; hãy giải thích mẫu số và lựa chọn đánh giá bằng lời của mình.

## Bài 1

TP: A,F; FP: B,H; FN: C; TN: D,E,G. Ma trận là `[[3,2],[1,2]]`, support positive=3, số dự đoán positive=4, tổng=8.

- Accuracy=(3+2)/8=5/8=0.625.
- Precision=2/(2+2)=1/2: trong 4 ticket báo ưu tiên có 2 ticket thật sự cần ưu tiên.
- Recall=2/(2+1)=2/3≈0.666667: trong 3 ticket thật sự cần ưu tiên bắt được 2.
- Specificity=3/(3+2)=3/5=0.6; FPR=2/5=0.4.
- F1=4/(4+2+1)=4/7≈0.571429.

Đổi positive bằng đảo nhãn thật và dự đoán cho TP'=3,FP'=1,FN'=2,TN'=2. Precision'=3/4=0.75; recall'=3/5=0.6. Accuracy vẫn 5/8 vì tập mẫu dự đoán đúng không đổi. Điều này giải thích vì sao báo precision/recall phải nêu lớp quan tâm.

## Bài 2

| Quy tắc | Accuracy | Recall | F1 | Balanced accuracy | Precision |
|---|---:|---:|---:|---:|---|
| M | 0.965 | 0.75 | 6/13≈0.461538 | (0.75+950/980)/2≈0.859694 | 1/3 |
| B | 0.98 | 0 | 0 | 0.5 | Không xác định, 0/0 |

Với FP=1,FN=10: M cost=30+10×5=80,45 cảnh báo; B cost=10×20=200,0 cảnh báo. Nếu FP=FN=1: M cost=35, B cost=20, nên B có cost thấp hơn theo đúng giả định mới. Không có mâu thuẫn: hai cấu hình ưu tiên lỗi khác nhau. Một yêu cầu bắt buộc recall tối thiểu còn có thể loại B dù cost bình quân thấp hơn.

Sau thêm TN, M có TN=9950,N=10000; accuracy=(9950+15)/10000=0.9965. Precision=1/3,recall=3/4,F1=6/13 đều giữ nguyên. Không có positive nào mới được bắt thêm, nên không chứng minh khả năng phát hiện lớp hiếm tốt hơn.

## Bài 3

| Bảng | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| A | NaN | 0 | 0 | 0.5 |
| B | NaN | NaN | NaN | NaN |
| C | 1 | 3/5=0.6 | 6/8=0.75 | NaN |

A có FN=2, nên F1 dạng đếm là 0/(0+0+2)=0, không phải NaN. B chỉ có TN: không biết khả năng bắt positive. C có positive thật và dự đoán positive, nhưng không có negative thật để tính specificity/FPR.

Với `zero_division=0`, lab thay precision của A bằng 0; B có precision,recall,F1 thành 0. C không đổi ba tỷ lệ này, FPR/specificity từ NaN thành 0 theo quy ước. Balanced accuracy của B/C vẫn NaN vì lab yêu cầu đủ hai lớp thật. Không nên âm thầm lấy trung bình trên tập lớp khác nhau rồi so như cùng một câu hỏi.

## Bài 4

| τ | TP | FP | FN | TN | P | R | F1 | Accuracy | Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 2 | 2 | 2 | 6 | 0.5 | 0.5 | 0.5 | 8/12 | 12 |
| 0.5 | 3 | 3 | 1 | 5 | 0.5 | 0.75 | 0.6 | 8/12 | 8 |
| 0.4 | 4 | 4 | 0 | 4 | 0.5 | 1 | 2/3 | 8/12 | 4 |

Ở τ=0.7, score đúng 0.7 được chọn; ở τ=0.4, score đúng 0.4 cũng được chọn. Mỗi lần hạ qua hai mẫu ở ví dụ này thêm một TP và một FP: một FN được sửa, một TN thành FP, tổng số đúng giữ nguyên. FN đắt gấp 5 FP nên cost giảm.

Chọn τ=0.4 trong ba ứng viên bằng validation. Khi cost hòa, `choose_threshold` chọn ngưỡng lớn hơn bằng khóa `(cost,-threshold)`. Quy tắc định trước tránh việc dùng test để phá hòa theo hướng đẹp điểm.

Test sau chốt có ma trận `[[3,3],[1,1]]`, P=1/4,R=1/2,F1=1/3,accuracy=1/2,cost=8. Baseline luôn 0 có TN=6,FN=2,accuracy=6/8=0.75,cost=10. Nếu đổi ngưỡng vì test chưa đạt kỳ vọng, test đã tham gia phát triển. Để có đánh giá độc lập sau thay đổi cần tập hoặc giao thức nằm ngoài các quyết định phát triển đó; không chỉ chạy lại cùng test rồi gọi là độc lập.

## Bài 5

| τ | TP | FP | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.9 | 1 | 0 | 1 | 1/2 |
| 0.8 | 1 | 1 | 1/2 | 1/2 |
| 0.7 | 2 | 1 | 2/3 | 1 |

Precision giảm rồi tăng. Hạ ngưỡng thêm một FP làm tỷ lệ xấu đi; thêm TP tiếp theo cải thiện tỷ lệ. TP/FP không giảm khi hạ ngưỡng; FN/TN không tăng. Recall không giảm nếu có positive thật. Accuracy/F1 có thể tăng hoặc giảm tùy nhãn của nhóm vừa được thêm; ví dụ này accuracy 2/3→1/3→2/3 và F1 2/3→1/2→4/5.

## Bài 6

Tại .8 chọn A,B,C, không tách B/C. Các điểm sau ngưỡng có `(R,P,FPR)` là `(1/2,1,0)`, `(1,2/3,1/2)`, `(1,1/2,1)`; điểm đầu để vẽ là `(0,1,0)` nhưng precision đầu không đo được.

AP=(1/2)×1+(1/2)×(2/3)+0×(1/2)=5/6≈0.833333.

Hình thang PR: đoạn recall 0→1/2 có diện tích 1/2; đoạn 1/2→1 có diện tích `(1/2)×(1+2/3)/2=5/12`; đoạn recall không đổi có diện tích 0. Tổng=11/12≈0.916667. Phải ghi rõ cách tính khi báo “PR-AUC”.

Các cặp positive–negative:

- A–B: thắng; A–D: thắng.
- C–B: hòa, nửa điểm; C–D: thắng.

ROC-AUC=(1+1+0.5+1)/4=7/8=0.875. Phá hòa theo vị trí file tạo ra quyết định không đạt được từ score≥ngưỡng thật, khiến metric phụ thuộc thứ tự dữ liệu.

Hàm s³ tăng nghiêm ngặt trên miền score ví dụ, giữ thứ tự/hòa nên AP/AUC không đổi. Các trị số mới không tự thành xác suất đã hiệu chuẩn; tại cùng ngưỡng số, nhãn cũng có thể đổi. Xếp hạng và calibration là hai khía cạnh khác nhau.

## Bài 7

Với π=0.1: positive thật=1000,negative=9000; TP=800,FN=200,FP=180,TN=8820. Precision=800/980=40/49≈0.816327.

Với π=0.01: positive=100,negative=9900; TP=80,FN=20,FP=198,TN=9702. Precision=80/278=40/139≈0.287770.

Thay π vào $0.8\pi/[0.8\pi+0.02(1-\pi)]$ cho cùng kết quả. Với π=0.5, precision=0.4/(0.4+0.01)=40/41≈0.975610. Không được gán số ấy cho π=1%.

Giả định chuyển được TPR/FPR là then chốt. Nếu đặc trưng hoặc nhãn thay đổi khiến mô hình bỏ sót/báo nhầm khác trước, hai tỷ lệ thay đổi; công thức với tỷ lệ cũ không còn dự báo chính xác. Các số đếm trên là kỳ vọng, không phải cam kết một mẫu ngẫu nhiên sẽ có đúng các số đó.

## Bài 8

Support `[90,8,2]`, tổng cột `[92,6,2]`.

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| billing | 87/92≈0.945652 | 87/90≈0.966667 | 87/91≈0.956044 |
| technical | 3/6=0.5 | 3/8=0.375 | 3/7≈0.428571 |
| account | 0 | 0 | 0 |

Accuracy=90/100=0.9. Macro-F1=(87/91+3/7+0)/3=6/13≈0.461538. Weighted-F1=`0.90×87/91+0.08×3/7+0.02×0`≈0.894725. Macro-recall=(87/90+3/8+0)/3≈0.447222.

Khi một mẫu bị đoán sai lớp a→b, nó đóng góp một FN cho a và một FP cho b. Mỗi mẫu đúng đóng góp một TP. Vì có 90 đúng,10 sai nên micro TP=90,FP=FN=10, F1=180/200=0.9. Điều kiện một nhãn, đủ lớp là cần thiết cho lập luận này.

Account có recall/F1=0 trên đúng hai mẫu; weighted-F1 cao phần lớn nhờ billing. Bỏ account sẽ biến macro thành trung bình trên một tập lớp khác; chỉ hợp lệ nếu câu hỏi đó được định nghĩa rõ từ trước, không báo thay thế như thể vẫn đo đủ ba lớp. Hai mẫu cũng chưa đủ đánh giá chắc chắn account ngoài ví dụ.

## Bài 9

Không trọng số: $L'(q)=[-2/q+98/(1-q)]/100=0$ cho $2(1-q)=98q$, q=0.02. Đạo hàm bậc hai dương trên (0,1), nên đây là cực tiểu duy nhất.

Weight positive=49 cho đóng góp positive tổng 98, bằng đóng góp negative. Khi chia tổng weight=196, $L'_w(q)=[-98/q+98/(1-q)]/196=0$, q=0.5. Không có feature mới; mọi mẫu vẫn có cùng score, nên không cải thiện khả năng xếp hạng mẫu.

Loss tối ưu không trọng số≈0.098039 nat, loss tối ưu có weight≈0.693147 nat. Không so hai trị số như cùng mục tiêu trên cùng phân bố. Mục tiêu có weight khác đã đổi; cần đánh giá cùng protocol/metric trên development phù hợp.

Resampling toàn bộ trước split có thể để bản sao positive gốc xuất hiện ở cả train và test; nếu tạo/chọn mẫu dựa trên dữ liệu giữ riêng còn có đường rò rỉ thông tin khác. Nó cũng đổi tỷ lệ lớp test, khiến precision/accuracy không đại diện trực tiếp cho lưu lượng tự nhiên.

Trong mỗi CV fold: fit scaler chỉ trên fold-train, fit sampler chỉ trên dữ liệu train đã biến đổi, fit model trên train tương ứng; fold-validation chỉ được transform/dự đoán. Class weight suy từ số lượng train cũng không lấy từ toàn bộ nhãn giữ riêng. Stratification giữ tỷ lệ lớp tương đối, không đảm bảo group/user tách biệt hay feature có sẵn ở thời điểm dự đoán.

## Bài 10

Các vấn đề gồm: chưa định nghĩa positive và đơn vị mẫu; thiếu support/số cảnh báo; test đã bị thay tỷ lệ; nhãn chỉ có ở vùng mô hình báo; chọn ngưỡng bằng test; thiếu baseline/cost, giai đoạn đánh giá và phép chia theo nhóm/thời gian. Precision 90% có thể là 9/10 hoặc 9000/10000 với mức bằng chứng khác nhau.

Chỉ gán nhãn cảnh báo thì chưa biết bao nhiêu positive bị bỏ trong nhóm không cảnh báo, tức chưa biết FN để tính recall. Không tự đặt FN=0 chỉ vì chưa quan sát được.

Một báo cáo mẫu 7 câu cho thực hành:

> Đây là thí nghiệm giảng dạy trên score tự tạo, positive=1 nghĩa ticket cần ưu tiên; quyết định là score≥ngưỡng. Trên validation 12 mẫu gồm 4 positive, thử ba ngưỡng 0.7,0.5,0.4 bằng chi phí FP=1,FN=5. Chọn và chốt 0.4 vì validation cost thấp nhất trong ba ứng viên, bằng 4. Test giữ riêng có 8 mẫu gồm 2 positive; không dùng test để chọn lại. Kết quả test TN=3,FP=3,FN=1,TP=1, cho precision=0.25,recall=0.5,F1≈0.333333,accuracy=0.5,cost=8. Baseline luôn 0 có accuracy=0.75,cost=10 trên cùng test. Tập nhỏ tự tạo chưa chứng minh chất lượng, calibration hay chi phí khi vận hành; cần dữ liệu đại diện và nhãn đủ ở cả vùng báo/không báo để đánh giá tiếp.

## Nếu còn vướng

- Nhầm precision/recall: khoanh mẫu số trước, ghi câu hỏi bằng lời.
- Nhầm ma trận: viết trục và thứ tự nhãn, kiểm tra tổng hàng/cột.
- Nhầm FPR với tỷ lệ cảnh báo sai: negative thật và positive dự đoán là hai tập khác nhau.
- Nhầm macro-F1: tính F1 từng lớp rồi mới trung bình.
- Nhầm cải thiện mô hình: tách dữ liệu, mục tiêu huấn luyện, ngưỡng quyết định và metric báo cáo.
