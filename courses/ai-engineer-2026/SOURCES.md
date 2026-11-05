# Nguồn và phạm vi đối chiếu

[Về mục lục](README.md)

Ngày truy cập các nguồn dưới đây: **04/11/2026**. “Đã đọc” chỉ áp dụng cho phạm vi được ghi, không có nghĩa đã hoàn thành toàn bộ khóa. Các ví dụ số liệu trong bài 1–22 do bộ tài liệu tự tạo; kết quả số được kiểm tra bằng mã đi kèm.

## Nguồn dùng trong bài 1

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression) | Tài liệu giảng dạy, Google | Phương trình, feature, label, weight và bias; đối chiếu thuật ngữ bài 1 | Mô hình nhập môn; không chứng minh latency thực tế là tuyến tính |
| [Linear regression: Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss) | Tài liệu giảng dạy, Google | Loss, MSE/MAE và ảnh hưởng của sai số lớn | Chọn loss phải theo bài toán; không mặc định MSE phù hợp mọi ứng dụng |
| [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Tài liệu thư viện chính thức | Phần data leakage và việc giữ tập test độc lập | Không đủ để thiết kế mọi phép chia dữ liệu theo thời gian/nhóm |

## Nguồn dùng trong bài 2

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Linear regression: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent) | Tài liệu giảng dạy, Google | Vòng lặp tính loss/gradient/cập nhật, đạo hàm MSE theo weight và bias, đường loss | Bài 2 tự suy diễn lại bằng dữ liệu riêng; tính lồi không có nghĩa learning rate bất kỳ cũng hội tụ |
| [Linear regression: Hyperparameters](https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters) | Tài liệu giảng dạy, Google | Phân biệt parameter/hyperparameter; learning rate ảnh hưởng độ lớn bước và hội tụ | Không có một learning rate tối ưu chung; các ngưỡng và kết quả số trong bài 2 chỉ áp dụng ví dụ đã nêu |

Phép khai triển đạo hàm, ví dụ một biến, bảng số và ba hình bài 2 là nội dung tự biên soạn. Hình tính bằng Matplotlib từ công thức và kết quả của `lab.py`; không lấy lại hình từ nguồn. Sai phân trung tâm được dùng để kiểm tra số, không coi như chứng minh tổng quát. Thư viện vẽ chỉ cần khi tái tạo hình, không cần cho bài thực hành huấn luyện.

## Nguồn dùng trong bài 3

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [NumPy: Absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html) | Tài liệu chính thức, NumPy | Mảng nhiều chiều, `shape`, `ndim`, `size`, `dtype`, indexing; đối chiếu biểu diễn | Chỉ đọc phần liên quan, không coi đã học toàn bộ hướng dẫn; ý nghĩa trục do ứng dụng quy ước |
| [NumPy: matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) | Tham chiếu API chính thức | Quy tắc `@` cho mảng 1-D, 2-D, các trục batch; điều kiện kích thước | Không đồng nhất mọi phép `dot`/`matmul`/nhân theo phần tử |
| [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Hướng dẫn chính thức | So kích thước từ phải, trục 1, ví dụ scalar/mảng | Broadcasting hợp lệ về kỹ thuật chưa đảm bảo đúng ngữ nghĩa bài toán |
| [NumPy: reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) | Tham chiếu API chính thức | Giữ số phần tử, suy một trục `-1`, thứ tự đọc và view/copy | Ví dụ trong bài dùng thứ tự C mặc định; không khẳng định reshape luôn là view |
| [NumPy: transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) | Tham chiếu API chính thức | Hoán vị trục; mảng 1-D không thành vector cột chỉ bằng `.T` | Chưa đi sâu bố trí bộ nhớ hoặc chi phí mọi phép biến đổi |
| [PyTorch: Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | Hướng dẫn chính thức | Phần giới thiệu, thuộc tính shape/dtype/device và phân biệt phép nhân | Bài 3 không chạy PyTorch; đây là cầu nối khái niệm, chưa kiểm chứng code PyTorch |

Các phép tính ma trận, gradient nhiều đặc trưng, ví dụ broadcasting sai ý định và bốn hình bài 3 là nội dung tự biên soạn. Đã kiểm tra bằng NumPy 2.5.4; gradient còn được so với vòng lặp và sai phân số. Hình dùng cùng mảng số với code; không sao chép hình từ tài liệu nguồn. Phiên bản thư viện vẽ và hướng dẫn tái tạo ghi trong bài.

## Nguồn dùng trong bài 4

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Seeing Theory: Basic Probability](https://seeing-theory.brown.edu/basic-probability/index.html) | Tài liệu giảng dạy, Brown University | Văn bản về biến ngẫu nhiên, kỳ vọng và phương sai | Trang lưu trữ; chưa tương tác/chạy các widget; không dùng phát biểu hội tụ như bảo đảm sai số giảm sau từng mẫu |
| [Seeing Theory: Compound Probability](https://seeing-theory.brown.edu/compound-probability/index.html) | Tài liệu giảng dạy, Brown University | Phần biến cố/tập hợp và ý nghĩa thu hẹp không gian khi điều kiện hóa | Không xem ví dụ điều kiện như bằng chứng nhân quả; Bayes trong bài được suy ra từ định nghĩa và bảng riêng |
| [NIST: Measures of Scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm) | Sổ tay kỹ thuật thống kê, NIST | Phương sai mẫu, SD, đơn vị và ảnh hưởng đuôi dữ liệu | Bài phân biệt rõ phương sai mô tả chia n với ước lượng mẫu chia n−1 |
| [Seeing Theory: Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/index.html) | Tài liệu giảng dạy, Brown University | Ước lượng điểm và cách diễn giải confidence level theo quy trình lấy mẫu | Chỉ đọc phần văn bản liên quan; không triển khai bootstrap trong bài 4 |
| [NIST: Confidence intervals for proportions](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) | Sổ tay kỹ thuật thống kê, NIST | Công thức khoảng Wilson và so với khoảng xấp xỉ đơn giản | Không coi độ phủ Wilson đúng 95% cho mọi n,p; ví dụ tự cộng xác suất nhị thức để kiểm tra |
| [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Tài liệu chính thức | Calibration, reliability diagram; Brier/log loss không chỉ đo calibration | Không huấn luyện bộ hiệu chuẩn hoặc cài scikit-learn; biểu đồ dùng nhóm số liệu tự tạo |
| [NumPy: Generator.binomial](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.binomial.html) | Tham chiếu API chính thức | Sinh số đếm theo phân bố nhị thức, tham số n,p,size | Mô phỏng iid với p cố định, không phải log thật; kết quả tái tạo trong môi trường đã ghi |
| [SciPy: entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) | Tham chiếu API và giải thích khái niệm | Công thức entropy Shannon, cơ số log và đơn vị bit/nat | Chỉ dùng làm nguồn công thức; code không gọi SciPy, không triển khai KL divergence |
| [Kendall & Gal (2017)](https://arxiv.org/abs/1703.04977) | Nghiên cứu gốc, arXiv/NIPS 2017 | **Chỉ đọc abstract:** phân biệt aleatoric/epistemic | Không tuyên bố đã đọc toàn văn hoặc tái tạo kết quả Bayesian Deep Learning của bài báo |

Bảng đếm Bayes, ví dụ latency, ba nhóm calibration và năm hình bài 4 do bộ tài liệu tự tạo. Mô phỏng dùng NumPy 2.5.4, seed=2026; các giả định iid/p cố định được nêu rõ. Độ phủ Wilson được kiểm tra bằng cả mô phỏng và cộng các xác suất nhị thức. Bài không suy từ dữ liệu giả định ra độ tin cậy của bất kỳ hệ thống đang vận hành nào.

## Nguồn dùng trong bài 5

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Tài liệu chính thức | Định nghĩa data leakage; fit preprocessing chỉ trên train; feature selection và Pipeline | Không lấy điểm số cao làm bằng chứng duy nhất của leakage; lab không chạy ví dụ chọn đặc trưng của trang |
| [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Hướng dẫn chính thức | Train/validation/test, nguy cơ chọn hyperparameter theo test, K-fold và dữ liệu nhóm | Bài dùng CV nhỏ tự viết bằng NumPy; chưa triển khai nested CV hoặc bảo đảm thống kê cho dữ liệu sản xuất |
| [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) | Tham chiếu API chính thức | Quy tắc group không trùng giữa phần fit và phần giữ riêng | Không gọi API; thí nghiệm dùng hai phép chia dựng sẵn, không giả là GroupKFold hoặc random split |
| [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Tham chiếu API chính thức | Các tập train mở rộng theo thời gian; `gap` là số mẫu bị loại cuối train | Lab kiểm tra availability/cutoff riêng, không gọi API; gap không tự giải mọi nhãn/feature đến chậm |
| [scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) | Tham chiếu API chính thức | Đóng gói các transformer và predictor để cross-validate cùng nhau | Không cài scikit-learn hoặc kiểm chứng code API; Pipeline không tự sửa splitter hay feature tương lai |
| [Feast: Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) | Tài liệu chính thức | Tái hiện feature theo mốc lịch sử; phần event time và created time, gồm lưu ý backfill | Chỉ đọc văn bản/mã minh họa, không chạy Feast; không mặc định mọi offline store/cấu hình đều lọc created time |

Ví dụ kNN, scaler 1-D, mô hình ghi nhớ user và lịch nhãn t+2 là thí nghiệm tự biên soạn. Đã chạy bằng Python 3.14.6/NumPy 2.5.4. Các phép chia dựng sẵn được ghi đúng bản chất; seed=2026 chỉ dùng cho xáo row ID và CV. Hình lấy từ các mảng/điểm số của lab, không sao chép hình từ nguồn. Không xem điểm số ví dụ như mức giảm leakage hay hiệu quả kỳ vọng trong thực tế.

## Nguồn dùng trong bài 6

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Google: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression) | Tài liệu giảng dạy chính thức | Phương trình tuyến tính, feature, weight và bias | Không xem ví dụ latency tự tạo như quan hệ thực nghiệm đã được chứng minh |
| [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) | Tài liệu chính thức | Công thức MAE, MSE và RMSE; đơn vị RMSE theo biến đích | Lab tự tính bằng NumPy; không gọi hoặc kiểm chứng các API metric của scikit-learn |
| [Google: Sigmoid function](https://developers.google.com/machine-learning/crash-course/logistic-regression/sigmoid-function) | Tài liệu giảng dạy chính thức | Logit, sigmoid, xác suất và log-odds | Xác suất do mô hình gán không tự chứng minh calibration |
| [Google: Loss and regularization](https://developers.google.com/machine-learning/crash-course/logistic-regression/loss-regularization) | Tài liệu giảng dạy chính thức | Log loss, regularization và trọng số tăng trên dữ liệu phân tách | Không dùng phát biểu đơn giản hóa về loss để kết luận MSE bị cấm trong mọi bài phân loại; gradient trong bài được tự suy diễn |
| [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) | Tài liệu chính thức | Mục logistic regression và các quy ước loss/penalty; đối chiếu vai trò regularization | Lab dùng mean loss + λ/2 nhân bình phương chuẩn trọng số, không phạt bias; không đồng nhất λ với C hoặc mọi solver |
| [Google: Thresholds and the confusion matrix](https://developers.google.com/machine-learning/crash-course/classification/thresholding) | Tài liệu giảng dạy chính thức | Đổi xác suất thành nhãn bằng ngưỡng, FP/FN và quy tắc ở biên | Chỉ đọc văn bản; không chạy widget. Ngưỡng theo chi phí được suy ra riêng với giả định ghi rõ trong bài |
| [SciPy: softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html) | Tham chiếu API chính thức | Công thức, trục chuẩn hóa và cách dịch logit để ổn định số | Không gọi SciPy; ví dụ chỉ tính đầu ra/loss, chưa huấn luyện mô hình đa lớp |
| [NumPy: logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html) | Tham chiếu API chính thức | Tính log của tổng các hàm mũ; dùng trong BCE từ logit | Không bảo đảm mọi phép tính với float đều tránh làm tròn; lab kiểm tra các trường hợp hữu hạn cụ thể |

Ví dụ hồi quy, logistic, chọn λ, ngưỡng chi phí và năm hình là nội dung tự biên soạn. Bộ dữ liệu nhỏ được dựng có chủ đích, không phải benchmark sản xuất. Code dùng Python 3.14.6/NumPy 2.5.4, fit scaler chỉ từ train, chọn bằng validation data loss và chỉ chấm test cho mô hình đã chọn. Gradient được kiểm tra bằng sai phân trung tâm; hồi quy còn đối chiếu nghiệm bình phương tối thiểu. Mã vẽ dùng công thức và đầu ra của lab, không sao chép hình từ nguồn. Bài không suy điểm số hoặc calibration của hệ thống thật từ các ví dụ ấy.

## Nguồn dùng trong bài 7

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Google: Accuracy, recall, precision](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) | Tài liệu giảng dạy chính thức | Công thức binary, khác biệt mẫu số, FPR, F1 và tình huống chia 0 | Chỉ đọc văn bản, không chạy widget; ví dụ và phân tích chi phí của bài là tự tạo, không mặc định F1 phù hợp mọi mục tiêu |
| [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | Hướng dẫn chính thức | Các mục confusion matrix, chuẩn hóa hàng/cột, F-beta, macro/micro/weighted và ROC | Chỉ đọc phần liên quan; lab tự viết bằng NumPy, không chạy các API hoặc ví dụ sklearn; không khái quát micro-F1=accuracy sang multi-label |
| [scikit-learn: precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) | Tham chiếu API chính thức | `pos_label`, averaging và `zero_division`, khác biệt giữa NaN và giá trị thay theo quy ước | Không gọi API; mặc định binary của lab là NaN, không phải mặc định trả 0 kèm cảnh báo của API |
| [scikit-learn: Balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) | Tham chiếu API chính thức | Định nghĩa trung bình recall theo lớp, bản không điều chỉnh chance | Lab binary yêu cầu đủ hai lớp thật để báo balanced accuracy; không coi quy ước với lớp vắng của lab giống mọi thư viện |
| [scikit-learn: Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) | Tham chiếu API chính thức | AP theo tăng recall, phân biệt với tích phân hình thang PR; đầu vào score có thể chưa qua ngưỡng | Lab dùng ví dụ có hòa điểm theo nhóm; không gọi API, không triển khai báo cáo AP đa lớp hoặc object detection |
| [Google: Class-imbalanced datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets) | Tài liệu giảng dạy chính thức | Thiểu số/đa số, khó khăn khi học lớp hiếm và tác động của downsampling/upweighting | Không dùng tỷ lệ ví dụ của nguồn làm thống kê thực tế; không coi cân bằng 50/50 hoặc một trọng số là tối ưu chung |
| [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) | Tài liệu thư viện chính thức | Resampling toàn bộ trước split gây leakage và làm lệch phân bố đánh giá; đặt sampler trong phần train | Chỉ đọc giải thích và mã minh họa, không cài/chạy imbalanced-learn, benchmark Adult hoặc SMOTE |

Ma trận 1000 ticket, score validation/test, ví dụ PR có hòa, ma trận ba lớp và năm hình do bộ tài liệu tự tạo. Công thức precision theo prevalence được suy ra từ xác suất điều kiện; số đếm là kỳ vọng với TPR/FPR cố định, không phải dự báo triển khai. Code dùng Python 3.14.6/NumPy 2.5.4; kiểm tra bằng cách đếm từng mẫu, phân số và ROC-AUC theo cặp. Phần weight giải mô hình xác suất hằng bằng công thức rồi tính weighted BCE, chưa huấn luyện bộ phân loại có feature hoặc chạy resampling. Hình dùng cùng mảng/công thức với lab; không sao chép hình của nguồn.

## Nguồn dùng trong bài 8

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Google: Decision trees](https://developers.google.com/machine-learning/decision-forests/decision-trees) | Tài liệu giảng dạy chính thức | Root, điều kiện, đường đi và dự đoán ở lá cho phân loại/hồi quy | Chỉ đọc giải thích; không chạy YDF hoặc lấy lại hình của nguồn |
| [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html) | Hướng dẫn chính thức | Mục mathematical formulation: split có trọng số, Gini, lá xác suất/mean và tiêu chí MSE; các giới hạn greedy, overfit, ngoại suy | Mini-tree tự viết chỉ nhận feature số hữu hạn, không missing/categorical/weight/pruning; quy tắc hòa và gain dương là của lab, không coi giống mọi API |
| [scikit-learn: Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) | Hướng dẫn chính thức | Baseline mean/median, nhãn phổ biến và prior xác suất từ train | Không chạy DummyClassifier/Regressor; phân biệt tối ưu hằng theo loss với tối ưu trên mọi mô hình |
| [Google: Random forests](https://developers.google.com/machine-learning/decision-forests/random-forests) | Tài liệu giảng dạy chính thức | Bagging có hoàn lại và tập feature ứng viên ngẫu nhiên tại nút | Giới thiệu đối chiếu với boosting; lab không triển khai hoặc benchmark random forest |
| [Google: Gradient boosted decision trees](https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt) | Tài liệu giảng dạy chính thức | Fit phần lỗi theo vòng và shrinkage | Không chạy TF-DF; nguồn dùng error=F−y rồi trừ, bài dùng residual=y−F rồi cộng; không trộn dấu hoặc coi shrinkage bảo đảm test tốt hơn |
| [scikit-learn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html) | Hướng dẫn chính thức | Các mục gradient boosting formulation, classification và shrinkage: âm gradient, link sigmoid và tương tác learning rate/số vòng | Lab chỉ fit squared-error boosting bằng stump; chưa chạy boosting log loss, Newton, histogram, early stopping patience, AdaBoost hoặc thư viện ensemble |

Ví dụ Gini sáu nhãn, hồi quy bốn điểm, hai vòng residual, XOR và năm hình là nội dung tự biên soạn. Thực nghiệm so 10 ứng viên trên cùng train/validation synthetic; train seed=2026, validation seed=2027, các phần có thiết kế nhiễu khác nhau được ghi rõ. Mean và tuyến tính fit từ train; test chỉ chấm mô hình đã chọn và baseline cố định, không quét test để tìm người thắng. Code dùng Python 3.14.6/NumPy 2.5.4; đối chiếu split bằng tính tay/phân số, kiểm tra giá trị lá và cập nhật residual. Điểm số không là benchmark hoặc bảo đảm ngoài mẫu cho dữ liệu thực tế. Hình dùng chính mô hình/mảng của lab, không sao chép hình của nguồn.

## Nguồn dùng trong bài 9

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Google: Nodes and hidden layers](https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers) | Tài liệu giảng dạy chính thức | Mô hình affine quen thuộc, thêm hidden layer và vai trò weight/bias | Chỉ đọc văn bản, không chạy widget; không coi neuron nhân tạo mô phỏng đầy đủ sinh học |
| [Google: Activation functions](https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions) | Tài liệu giảng dạy chính thức | Vai trò phi tuyến, công thức sigmoid/tanh/ReLU và phân biệt đầu ra từng hàm | Không dùng gợi ý chọn ReLU làm bảo đảm nó tốt nhất mọi bài toán; không lấy lại hình nguồn |
| [Dive into Deep Learning: MLP](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) | Giáo trình do nhóm tác giả xuất bản | Mục hidden layers/from linear to nonlinear: shape và affine collapse; mục activation: công thức, đạo hàm sigmoid/tanh/ReLU | Không chạy notebook hoặc suy “mọi hàm đều học được” từ diễn giải universal approximation; ví dụ và lời giải số trong bài tự xây |
| [PyTorch 2.15: Linear](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html) | Tham chiếu API chính thức | Phương trình xAᵀ+b, shape weight/bias để so với quy ước hàng của lab | Chỉ đối chiếu tài liệu API; không cài hoặc kiểm chứng code PyTorch, không gọi phép nhân X@W của lab là cùng shape lưu weight |
| [PyTorch 2.15: GELU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GELU.html) | Tham chiếu API chính thức | GELU=zΦ(z), phân biệt dạng chuẩn với xấp xỉ tanh | Lab dùng math.erf trong công thức CDF; không gọi kernel PyTorch hoặc benchmark hiệu năng/độ chính xác trên mọi float |
| [PyTorch 2.15: BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html) | Tham chiếu API chính thức | Hợp đồng kết hợp sigmoid/BCE và tính ổn định khi nhận logits | Không chạy API; không sử dụng các phát biểu về pos_weight như bảo đảm đơn điệu precision/recall; phần này chỉ minh họa việc tránh sigmoid hai lần |

Neuron đơn, forward batch 2→2→1, phép gộp affine, XOR hai ReLU và năm hình là nội dung tự biên soạn. Code Python 3.14.6/NumPy 2.5.4 dùng weight hidden đặt tay; thí nghiệm chỉ fit head logistic với L2 trên feature cố định, chưa backprop hidden và chưa chạy autograd. Bảng XOR có bốn hàng train, không có điểm validation/test hoặc kết luận ngoài mẫu. Kiểm tra forward bằng tính scalar/vòng lặp, số tham số bằng shape và gradient head bằng sai phân số. Hình dùng cùng mảng/công thức và đầu ra của lab, không sao chép hình nguồn.

## Nguồn dùng trong bài 10

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) | Ghi chú khóa học do nhóm giảng dạy xuất bản | Chain rule, upstream/local gradient, cộng tại nhánh, cache và phép toán ma trận | Ví dụ scalar/mạng của bài tự tạo; không lấy lại hình hoặc mặc định quy ước transpose của nguồn giống batch hàng trong lab |
| [D2L: Forward/backward và computational graphs](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) | Giáo trình do nhóm tác giả xuất bản | Graph, backward MLP, nhánh regularization hai W và quan hệ với forward cache | Nguồn minh họa vector cột, không bias; bài tự chuyển công thức sang batch hàng có bias. Không chạy notebook hoặc coi backward tự thay thế optimizer |
| [CS231n: Gradient checks](https://cs231n.github.io/neural-networks-3/) | Ghi chú khóa học chính thức | Sai phân trung tâm, absolute/relative error, float64, epsilon và kink | Không áp một ngưỡng relative chung cho mọi gradient gần0; kiểm tra số ở vài điểm không phải chứng minh gradient đúng toàn miền |
| [PyTorch: Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Tutorial chính thức | Graph/DAG, chain rule, gradient accumulation và seed cho loss scalar | Chỉ đọc giải thích/mã minh họa; lab dùng NumPy viết tay, chưa chạy torch.autograd hoặc API zero_grad. Chi tiết thao tác sẽ học ở bài11 |
| [Google: Training using backpropagation](https://developers.google.com/machine-learning/crash-course/neural-networks/backpropagation) | Tài liệu giảng dạy chính thức | Giới thiệu vai trò backprop và các khó khăn vanishing/exploding/dead ReLU | Chỉ đọc văn bản, không xem video hay chạy widget; không dùng gợi ý ổn định training làm bảo đảm hội tụ cho mọi mạng |

Graph có nhánh, phép tính gradient batch và năm hình là nội dung tự biên soạn; mạng tính tay dùng lại parameter bài 9. Code Python 3.14.6/NumPy 2.5.4 tính tất cả parameter gradient của một hidden layer, có ReLU/tanh và L2 hai W; đối chiếu bằng sai phân trung tâm, công thức scalar và quan hệ mean/sum. Mạng XOR 2→4→1 được fit end-to-end từ khởi tạo ngẫu nhiên seed 2026; có ghi thay đổi W₁,b₁ để phân biệt với head cố định. Điểm số chỉ trên bốn hàng train, không có validation/test hoặc kết luận ngoài mẫu. Hình lấy từ công thức, trace và kết quả chạy lab, không sao chép hình nguồn.

## Nguồn dùng trong bài 11

Truy cập 04/11/2026. Chỉ đọc phần liên quan; các hình/ví dụ tự tạo.

| Nguồn | Phần đã đọc | Giới hạn |
|---|---|---|
| [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | Shape, dtype, device, phép toán, chia sẻ NumPy | Chỉ chạy CPU trong lab; không kiểm accelerator |
| [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Leaf, graph, backward, accumulation, no_grad | Lab đối chiếu graph nhỏ, không triển khai engine tổng quát |
| [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) | Map Dataset, __len__, __getitem__, collate và batch | Dùng dữ liệu tự tạo và num_workers=0; không chạy FashionMNIST/multiprocessing |
| [Linear](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html) | Weight layout (out,in), bias | Copy weight bài 10 có transpose, chạy kiểm chứng CPU |
| [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html) | Nhận logit và mean reduction | Không dùng pos_weight hoặc thử loss đa nhãn trong lab |
| [Cài đặt](https://pytorch.org/get-started/locally/), [torch trên PyPI](https://pypi.org/project/torch/) | Hướng cài và wheel Python3.14 macOS arm64 | Không lấy nhãn phiên bản trên trang cài làm phiên bản kiểm; bản thực chạy là torch2.15.0 |

Môi trường kiểm: Python3.14.6, torch2.15.0, NumPy2.5.4, CPU macOS arm64. Kiểm cả bốn gradient mạng bài10 và bước SGD, graph scalar, batch2/2/1, metric33, vòng fit Linear. Ví dụ train5 mẫu không có validation/test hoặc kết luận production. Các hình tái tạo từ mã đi kèm. Môi trường ảo đặt ngoài bộ tài liệu và không đóng gói.

## Nguồn dùng trong bài 12

Truy cập 04/11/2026. Đọc các phần được ghi; ví dụ, quy tắc chọn, hình và log thí nghiệm do bộ tài liệu tự xây.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Train/evaluation loop, zero_grad/backward/step và no_grad | Không chạy FashionMNIST của nguồn. Lab dùng sample-weighted MSE, gọi tập chọn theo epoch là validation; không sao chép tên test của nguồn thành vai trò giữ riêng |
| [Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html) | eval, state_dict có parameter/buffer và tham chiếu tensor | Lab chỉ thực nghiệm Linear/Tanh/Dropout, không thực nghiệm BatchNorm |
| [Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html) | Bernoulli zeroing, scale 1/(1−p), identity khi eval | Chạy minh họa p=0.5 và MLP p=0.15 CPU; không suy Dropout luôn cải thiện ngoài mẫu |
| [SGD](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html) | Công thức momentum, buffer khởi đầu và optimizer state | Ví dụ tính tay không dampening/weight decay/Nesterov. Chỉ chạy SGD momentum, chưa chạy Adam/scheduler |
| [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | State dict, deepcopy snapshot, model/optimizer checkpoint, eval khi inference | Lab không pickle module toàn phần; best/last schema và chính sách resume là thiết kế của bài |
| [Serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html) | Phần weights_only: các kiểu được đọc và giới hạn bảo vệ | Chỉ tải checkpoint tự tạo bằng tensor/kiểu cơ bản; không kiểm file không rõ nguồn |
| [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html) | RNG seed/state và giới hạn giữa phiên bản/thiết bị | Kiểm resume ở cuối epoch, CPU một thread, worker=0. Không kiểm CUDA/MPS, AMP, distributed, resume giữa epoch hoặc augmentation ngoài torch |

Môi trường chạy: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, macOS arm64, CPU float64. Kiểm một bước SGD tính tay, early stopping trace, Dropout train/eval, snapshot best, checkpoint roundtrip và resume đủ model/optimizer/RNG/bộ đếm. Hai learning rate dùng cùng split synthetic 48/21/21 và seed; chọn run bằng validation rồi mới chấm test một lần. Con số chỉ minh họa quy trình, chưa là benchmark hoặc bằng chứng chất lượng production. Các hình vẽ lại từ code riêng và run thực; môi trường ảo, checkpoint sinh khi chạy và cache không nằm trong ZIP chia sẻ.

## Nguồn dùng trong bài 13

Truy cập 04/11/2026. Ví dụ tính tay, các lỗi có chủ đích và hình do bộ tài liệu tự tạo.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [SGD 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html) | Update SGD/momentum, khởi tạo buffer đầu và state | Không chạy Nesterov/dampening; không biến cùng chuỗi gradient đặt trước thành benchmark hội tụ |
| [Adam 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html) | Moment, bias correction, eps, coupled/decoupled decay | Tính tay dùng beta2=.99 cho dễ đối chiếu, MLP dùng mặc định .999; không thực nghiệm AMSGrad/fused/CUDA |
| [AdamW 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html) | Co weight tách khỏi moment và param groups | Lab decay chỉ weight, không bias; không khẳng định decay=.03 tương đương L2=.001 |
| [Google: L2 regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization) | Phạt tổng bình phương weight và đánh đổi fit/regularization | Bài tự quy ước lambda/2, phân biệt data loss/objective; không dùng diễn giải nguồn như bảo đảm weight không bao giờ đúng bằng 0 hoặc generalization luôn tốt hơn |
| [Dropout 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html) | Zeroing trong train, scale và identity khi eval | Không suy kỳ vọng output toàn mạng phi tuyến bằng output eval; hai vị trí Dropout chỉ là kiến trúc lab |
| [clip_grad_norm_ 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html) | Global norm, sửa grad in-place, trả norm và báo non-finite | Thực nghiệm vector gradient và nhánh fit có clipping; chưa kiểm AMP/unscale hoặc clip distributed |
| [CS231n: Learning](https://cs231n.github.io/neural-networks-3/) | Sanity checks, tiny subset, actual update ratio và activation/gradient | Không lấy ngưỡng ratio khoảng 1e-3 làm chuẩn chung. Tình huống scalar tự xây; tiny Linear fit không chứng minh MLP hoặc dữ liệu thật đúng |

Đã chạy Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, macOS arm64, CPU float64 một thread. Kiểm hai bước optimizer với công thức riêng, coupled/decoupled decay, L2 gradient và bias exclusion, clipping, hàm bậc hai, ba lỗi scalar và tiny fit. Bốn cấu hình regularization dùng cùng split synthetic 24/48/48; chọn checkpoint và run bằng validation, chỉ chấm test của run được chọn. Bản best.pt trong bài là artifact inference, không có trạng thái resume. Các số là minh họa một seed, không phải benchmark phương pháp. Trang D2L weight-decay thử truy cập nhưng công cụ không lấy được nội dung, không dùng làm bằng chứng. Hình được tái tạo bằng code riêng; không đóng gói môi trường ảo, cache hoặc checkpoint sinh ra khi chạy.

## Nguồn dùng trong bài 14

Truy cập 04/11/2026. Bài và hình dùng ví dụ/PNG tự tạo, không sao chép hình nguồn.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [Conv2d 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html) | Cross-correlation, channel sum, NCHW, weight layout, stride/padding/dilation và shape formula | Thực nghiệm groups=1,dilation=1, CPU; dilation/depthwise chỉ giới thiệu hoặc ngoài phạm vi, không benchmark CUDA |
| [CS231n: Convolutional networks](https://cs231n.github.io/convolutional-networks/) | Local connectivity, parameter sharing, pooling và gradient cộng về weight dùng chung | Ví dụ số và shape tự xây; không lấy nhận định translation invariance như bảo đảm mọi CNN bất biến dịch chuyển, không chạy các kiến trúc lịch sử của nguồn |
| [MaxPool2d 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MaxPool2d.html) | Cửa sổ max, stride, shape và ceil_mode | Lab pooling2 stride2,padding0,ceil_mode=False; không khảo sát tie/max-unpool |
| [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) | Logits và target class ID long, reduction sum/mean | Lab phân loại hai lớp, không class weight/ignore_index/label smoothing hoặc soft target; mean theo số mẫu vì loss không trọng số |
| [Pillow 13: Image](https://pillow.readthedocs.io/en/stable/reference/Image.html) | Image.open/context, fromarray/save, convert L/RGB và array copy | Chỉ đọc/ghi PNG tự tạo, không triển khai xử lý EXIF, alpha, ảnh hỏng hoặc decoder các định dạng khác |

Môi trường: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Pillow 13.0.0, Matplotlib 3.11.2 trên macOS arm64. Convolution tính tay dùng float64; CNN chạy CPU float32 một thread. Kiểm cross-correlation bằng vòng lặp NumPy độc lập, kernel gradient/shape/parameter count, pooling và cross-entropy tính tay; dữ liệu PNG có group split 40/12/12 group tương ứng 80/24/24 ảnh. Thống kê chỉ fit train; augmentation flip ngang chỉ train, checkpoint lưu preprocessing và class mapping. Chọn best epoch bằng validation CE rồi mới chấm test; kiểm load/inference nhất quán. Không dùng torchvision, ảnh người dùng, dataset ngoài, model pretrained, CNN/MLP benchmark hoặc resume trong bài này. Test24 ảnh đến từ12 group, không coi các view là nguồn độc lập để suy khoảng tin cậy. Code và năm hình là artifact chia sẻ; corpus/checkpoint runtime và môi trường ảo không đóng gói.

## Nguồn dùng trong bài 15

Truy cập 04/11/2026. Graph, dữ liệu, số tính tay và hình là nội dung tự tạo.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [D2L: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) | Hidden state, recurrence, parameter dùng chung theo bước và head | Ví dụ nguồn về language modeling; lab dùng số thực và classification, chưa tokenizer/embedding hoặc sinh văn bản |
| [D2L: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) | Unroll, chain rule, tích đạo hàm, full/truncated computation | Bài tự suy scalar tanh với loss cuối chuỗi; không lấy gợi ý truncated/short-term của nguồn như quy tắc tốt nhất mọi bài toán. Lab full BPTT chuỗi ngắn, chưa stateful/TBPTT |
| [RNN 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNN.html) | Equation có hai bias, batch_first và hidden layout, output/h_n, parameter count, zero hx mặc định | Một lớp, một hướng CPU; không chạy stacked/bidirectional, dropout RNN hoặc CUDA |
| [pad_sequence 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.rnn.pad_sequence.html) | Collate chuỗi variable length thành batch, padding_side và value | Lab right-padding feature số, không suy zero là ký hiệu tự bỏ qua |
| [pack_padded_sequence 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.rnn.pack_padded_sequence.html) | Lengths CPU, unsorted batch và enforce_sorted=False | Kiểm h_n theo thứ tự sample, thay vùng padding và chạy từng chuỗi; chưa ONNX export hoặc left-padding |
| [LSTM 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html) | Cell/hidden state và gate để giới thiệu động cơ | Chỉ đọc phần định nghĩa; chưa code công thức gate hoặc huấn luyện/benchmark LSTM/GRU |

Đã chạy Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2 trên macOS arm64. Toy hand/padding/gradient dùng float64; classifier dùng CPU float32 một thread. Kiểm forward/BPTT manual, sai phân số, parameter/shape, packed-vs-single và padding invariance. Dữ liệu96 group mỗi group2 sample đảo nhau, split64/16/16 group tương ứng128/32/32 chuỗi; stats chỉ fit bước thật train. Chọn best epoch bằng validation CE, chấm test sau lựa chọn và kiểm inference roundtrip. Majority và endpoint rule là baseline validation; không tuyên bố RNN vượt rule đơn giản hoặc đã forecast thực tế. Window toy chỉ căn chỉ số input/target; không huấn luyện forecaster. Test32 sample từ16 group, không coi là32 nguồn độc lập. Artifact best.pt không chứa resume state; không đóng gói dữ liệu/checkpoint runtime, cache hoặc môi trường ảo.

## Nguồn dùng trong bài 16

Truy cập 04/11/2026. Bảng toy, phép tính, corpus ticket và hình do bộ tài liệu tự tạo.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [Embedding 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html) | Lookup table, shape, padding_idx, from_pretrained/freeze và gradient | Lab dense gradient, không max_norm/sparse/scale_grad_by_freq hoặc pretrained download. Phân biệt gradient PAD qua lookup với các tác động optimizer bên ngoài |
| [one_hot 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.one_hot.html) | ID long sang one-hot và chiều num_classes | Toy chỉ đối chiếu forward lookup với matmul; khi so backward phải đồng nhất mask/padding convention |
| [cosine_similarity 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cosine_similarity.html) | Công thức norm/eps, chiều tính cosine | Zero vector không có hướng; kết quả số hữu hạn nhờ eps không chứng nhận ý nghĩa retrieval |
| [Google: Embedding space](https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space) | Biểu diễn vector, phụ thuộc tác vụ và giới hạn diễn giải từng chiều | Không lấy hình trục ngữ nghĩa giả định của nguồn như mô tả mọi embedding. Không coi mọi embedding là vector bị ràng buộc trong khoảng0–1 hoặc−1–1 |
| [Google: Obtaining embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/obtaining-embeddings) | Học biểu diễn cùng task và phân biệt static/contextual | Lab chỉ static lookup+mean classifier; không chạy PCA, contextual encoder hoặc API embedding |
| [Google: Measuring similarity](https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity) | Vai trò norm trong dot/cosine và quan hệ khoảng cách sau normalize | Lab so vector toy và cosine hàng token đủ4D, không triển khai vector database hoặc retrieval benchmark |
| [D2L: Word embedding/word2vec](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html) | Skip-gram/CBOW và nhãn tạo từ context | Chỉ giới thiệu đối chiếu với supervised ticket task, chưa huấn luyện word2vec/negative sampling |

Môi trường: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2 trên macOS arm64. Toy dùng float64, classifier CPU float32 một thread. Kiểm lookup/one-hot, gradient hàng lặp và bước SGD, masked mean, freeze/head gradient, dot/cosine/L2 và phép xoay giữ logits. Corpus216 ticket chia template group4/2/2 cho108/54/54 sample; vocabulary21 chỉ fit train, PAD/UNK tách vai trò. Model99 parameter học embedding từ classification loss, chọn best theo validation CE rồi chấm test. Keyword rule cũng đạt validation100%, không coi model cần thiết hoặc geometry đã được chứng nhận semantic retrieval. Bảng cosine vẽ trực tiếp đủ4 chiều, không projection2D. Checkpoint inference lưu mapping/tokenizer cùng weight; chưa có trạng thái resume. Không đóng gói dữ liệu/checkpoint runtime, cache hoặc môi trường ảo.

## Nguồn dùng trong bài 17

Truy cập 04/11/2026. Ví dụ tính tay, dữ liệu và bảy hình là nội dung tự biên soạn; không lấy lại hình từ nguồn.

| Nguồn | Phần đã đọc / vai trò | Giới hạn |
|---|---|---|
| [Bahdanau, Cho, Bengio: Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | Abstract và mốc bản đầu2014: soft search/alignment để giảm bottleneck fixed vector | Chỉ đọc abstract, không tái hiện thuật toán dịch máy, BLEU hoặc additive-attention training |
| [Vaswani et al.: Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | Mục3.2–3.5 và bảng chi phí mục4: QKV/scaling, multi-head, nguồn self/cross, causal/position | Bài dùng cấu hình nhỏ và ví dụ riêng, không tái lập Transformer/dịch máy của paper. Phép suy gradient, permutation và phản ví dụ output do bài tự xây |
| [D2L: Attention scoring functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html) | Dot/scaled dot, giả định mean0/variance1 và softmax trên key | Không coi giả định độc lập/variance là sự thật của mọi learned QK; chưa thực nghiệm additive attention |
| [SDPA 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) | QKV shape, True=allowed, additive mask, scale, dropout_p và causal square | Chỉ kiểm CPU float64/float32, dropout0 khi đối chiếu; chưa non-square causal, GQA, CUDA/FlashAttention hoặc mixed precision |
| [MultiheadAttention 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html) | Projection, batch_first, bool mask True=blocked, key_padding/query distinction và per-head weights | Lab kiểm forward/parameter gradient với manual split/concat, không benchmark kernel/fastpath. Không train multi-head Transformer trong bài |

Môi trường: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2 trên macOS arm64. Hand/API/MHA dùng CPU float64; recall model dùng CPU float32 một thread. Kiểm one-query và matrix attention bằng NumPy reference, backward QKV bằng công thức và sai phân số, softmax/mask, causal không nhìn tương lai, padding không ảnh hưởng output hợp lệ, multi-head projection/split/concat, và permutation/weight interpretation counterexamples. Task có192 mapping memory độc nhất, mỗi group2 query; split128/32/32 group tạo256/64/64 sample. Classification loss học single-head cross-attention, chọn best validation CE rồi chấm test. Target nằm trong memory theo định nghĩa retrieval toy, dict rule giải được100%; không coi đó là dự báo hay chứng minh neural model cần thiết. Best.pt là inference artifact, chưa có resume state; không đóng gói data/checkpoint runtime, cache hoặc môi trường ảo.

## Nguồn dùng trong bài 18

| Nguồn | Loại / tổ chức | Phần đã đọc và vai trò | Giới hạn |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | Bài báo gốc, Vaswani và cộng sự | Mục 3.1, 3.3–3.5: encoder–decoder, post-norm, FFN, embedding/head, sinusoidal PE; đối chiếu mask và shift | Công trình gốc năm 2017; trang HTML v7 hiển thị 2023. Không tái hiện kết quả dịch máy hoặc phần cứng của bài báo |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | Bài báo gốc, Ba/Kiros/Hinton | Abstract: chuẩn hóa theo từng mẫu, affine học được và cùng cơ chế thống kê train/test | Chỉ đọc abstract; chi tiết API/variance trong lab đối chiếu nguồn PyTorch phía dưới |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | Bài báo nghiên cứu, Xiong và cộng sự | Abstract: vị trí LN, gradient khởi tạo và warm-up trong điều kiện nghiên cứu | Không coi pre-LN là bảo đảm mọi mô hình không cần warm-up; không tái hiện nghiên cứu |
| [D2L — Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) | Giáo trình của tác giả D2L | Mục 11.7.1–11.7.2, phần tổng kết: cấu trúc và positionwise FFN dùng chung theo token | Không chạy notebook/dataset dịch của sách; code bài 18 tự viết, không cần cài d2l |
| [LayerNorm 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html) | Tài liệu API chính thức | Công thức, normalized_shape, eps, affine, variance correction=0 và thống kê train/eval | Phụ thuộc phiên bản; số tính tay eps=0 chỉ là ví dụ có variance dương |
| [TransformerEncoderLayer 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html) | Tài liệu API chính thức | batch_first, norm_first, activation, cấu trúc reference và tham số | Dùng so block pre/post forward và gradient; không suy hiệu năng GPU/fast path từ CPU lab |
| [Transformer 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html) | Tài liệu API chính thức | src/tgt/memory shapes, bool mask True=chặn, ba padding mask, batch_first/norm_first | Module reference nhận vector; lab thêm embedding, PE và head bên ngoài; chỉ kiểm forward/backward seq2seq, không train seq2seq |
| [MultiheadAttention 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html) | Tài liệu API chính thức | QKV, head dimension, dropout weight, mask/padding polarity, need_weights và output shape | Không coi output padded query bằng 0; không trộn bool polarity với functional SDPA |
| [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) | Tài liệu API chính thức | Raw logits, target chỉ số lớp, ignore_index và mean trên nhãn không ignore | Lab không class weight/label smoothing; bỏ vị trí không loại class khỏi denominator softmax |

**Phạm vi kiểm bài 18:** ví dụ LayerNorm/FFN/residual và trọn block dùng số tự tạo; so custom pre/post block với TransformerEncoderLayer cùng weight ở forward, input gradient, parameter gradient; kiểm PE/hoán vị, causal prefix/future gradient, PAD và seq2seq mask/shift. Decoder-only 17,672 tham số train CPU thật 60 epoch, 960 update, split source identity 512/128/128 từ 5,440 chuỗi độ dài 3–6; validation CE chọn best epoch 58, sau đó mới đo test teacher-forced và greedy. Test token accuracy=814/855, exact=94/128; baseline quy tắc copy=100%, không chứng minh neural model cần thiết. Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2; không có pretrained LLM, KV cache, GPU benchmark hoặc kiểm ngoại suy length. Checkpoint chỉ đủ inference, không đủ resume; bản ZIP không chứa checkpoint/data runtime/venv/cache. Nguồn được đọc ngày 04/11/2026, số liệu có thể khác khi đổi backend/phiên bản.

## Nguồn dùng trong bài 19

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Hugging Face BPE](https://huggingface.co/learn/llm-course/chapter6/5) | Phần training/tokenization: đếm cặp, merge theo thứ tự, byte vocabulary, triển khai sư phạm | Code/ví dụ bài 19 tự viết; regex whitespace-run riêng, không tái hiện tokenizer pretrained cụ thể; không dùng model/API của OpenAI |
| [Python 3.14 unicodedata](https://docs.python.org/3.14/library/unicodedata.html) | normalize: NFC/NFD/NFKC/NFKD, code point và tương đương Unicode | Lab không normalize; không dùng API riêng Python 3.15; Unicode surrogate đơn lẻ ngoài phạm vi UTF-8 strict |
| [Hugging Face perplexity](https://huggingface.co/docs/transformers/main/en/perplexity) | Định nghĩa NLL/PPL, phụ thuộc tokenization, causal vs masked LM, context và sliding window | Trang main được đọc 04/11/2026; không chạy ví dụ GPT-2 hoặc tải dataset của nguồn; lab mỗi tài liệu vừa context, không sliding window |
| [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) | Logits, nhãn class và ignore_index; đối chiếu tính CE | Lab shift nhãn bên ngoài model; không suy quy tắc shift của mọi thư viện khác |

**Phạm vi kiểm bài 19:** corpus backend tiếng Việt tự tạo 192 câu, chia nguyên câu 128/32/32 trước fit BPE; 256 byte + 3 special + 64 merge =323 ID. Kiểm BPE tính tay, roundtrip Unicode/whitespace/literal special, save/load và tokenizer mismatch, causal/padding/gradient, split, checkpoint và các đoạn Python. Transformer LM 33,699 tham số chạy thật CPU 30 epoch, 240 update; best epoch 30 theo validation CE; test 896 nhãn gồm EOS, CE=0.4165676321, PPL=1.5167465787. Hai baseline add-one dùng cùng tokenizer/nhãn; smoothing mạnh, không coi là so sánh tối ưu mọi n-gram. Mẫu sinh có lỗi lặp/trộn được giữ nguyên. Môi trường Python 3.14.6, torch 2.15.0; không có corpus production, pretrained LLM, chat tuning, KV cache hoặc GPU benchmark. Checkpoint chỉ inference; ZIP không chứa dữ liệu/checkpoint runtime/venv/cache. Ví dụ phân đoạn tokenizer thương mại không được gán ID hoặc số token nếu chưa chạy tokenizer đó.

## Nguồn dùng trong bài 20

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) | Abstract: domain-adaptive/task-adaptive pretraining và thực nghiệm trong các miền, tác vụ phân loại được nghiên cứu | Không tái hiện nghiên cứu; không suy CPT luôn có lợi cho mọi LLM; lab không chạy CPT riêng |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | Abstract: supervised demonstrations trước giai đoạn ranking/RLHF; phân biệt mục tiêu học | Bài báo 2022; không phải tài liệu API/sản phẩm hiện hành, không tái hiện pipeline RLHF hoặc số liệu nguồn |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | Completion-only/assistant-only loss, prompt–completion, chat template tạo mask và phân biệt định dạng dữ liệu | Trang truy cập 04/11/2026; không cài/chạy TRL, không hứa config dùng cho mọi model/template |
| [LoRA](https://arxiv.org/abs/2106.09685) | Abstract: freeze base và cập nhật rank-decomposition; đối chiếu khái niệm với phép tính ma trận tự tạo | Chỉ demo số/gradient trong lab, chưa injection/train/merge adapter Transformer; không lặp lại tuyên bố tiết kiệm VRAM như benchmark riêng |
| [PyTorch autograd 2.15](https://docs.pytorch.org/docs/2.15/notes/autograd.html) | Freezing parameter, requires_grad và các grad mode/no_grad/eval | Chạy full/head-only trên CPU, không có benchmark phân tán/GPU |
| [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) | Raw logits, nhãn class, ignore_index và mean trên target thật | Lab tự shift labels, không có label smoothing/class weight; không suy mọi Trainer tự shift giống nhau |

**Phạm vi kiểm bài 20:** dữ liệu routing tự tạo 48 nhóm service–operation, split nhóm 32/8/8, mỗi nhóm 6 event →192/48/48 record; cùng split cho pretrain/SFT, BPE 64 merge fit LM train-only. Model 33699 parameter, base 25 epoch và 3 SFT branch 25 epoch, 300 update/stage (1200 tổng), một seed CPU. Base bestepoch 18; full/scratch/head bestepoch 25 theo validation objective riêng, sau đó mới đo test. Exact gồm EOS: base 0/48,full 48/48,scratch 28/48,head 6/48; rule 48/48,majority 16/48. Đo LM CE cũ cho thấy regression sau SFT, không suy mọi năng lực bị quên. Kiểm mask sau shift, prompt gradient, LoRA tay/gradient, frozen weight 23040 giữ nguyên, parent hash/clone isolation, metric/checkpoint và tokenizer compatibility. Python 3.14.6,torch 2.15.0,NumPy 2.5.4,Matplotlib 3.11.2. Chưa có CPT/RLHF/LoRA training trên Transformer, pretrained LLM tải ngoài, đánh giá production hoặc đa seed; full có thêm pretrain compute so với scratch. Checkpoint inference không đủ resume; ZIP loại runtime data/checkpoint/venv/cache.

## Nguồn dùng trong bài 21

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Hugging Face generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) | Greedy, sampling, beam và khác biệt tìm kiếm chuỗi | Trang main đọc 04/11/2026; không cài Transformers hoặc chạy model tải ngoài; beam lab chỉ cây hữu hạn |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Abstract: suy biến generation và khái niệm nucleus sampling | Không tái hiện nghiên cứu chất lượng; không suy top-p luôn tốt hơn cho mọi task |
| [How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation) | Prefill/decode, tái sử dụng KV cũ trong autoregressive inference | Lab tự ghép cache một block; không triển khai multi-layer/prefix sharing/serving batch |
| [Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache) | Phần dynamic/static và đánh đổi bộ nhớ/tính toán | Chỉ định hướng; không chạy cache class, offloading hoặc quantized cache của thư viện |
| [SDPA 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) | Bool True=allowed, scale/dropout, is_causal upper-left khi ma trận không vuông | Đã kiểm mask offset bằng số; không benchmark kernel GPU hoặc triển khai GQA |
| [PyTorch autograd 2.15](https://docs.pytorch.org/docs/2.15/notes/autograd.html) | Eval khác no_grad/inference mode và hạn chế tensor inference | Lab dùng eval+no_grad, không benchmark inference_mode; trang inference_mode riêng không truy cập được nên không ghi đã đọc |
| [torch.multinomial 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.multinomial.html) | API lấy mẫu, replacement và generator | Chạy CPU RNG riêng; không bảo đảm seed giống bit giữa mọi backend |
| [Python 3.14 codecs](https://docs.python.org/3.14/library/codecs.html) | IncrementalDecoder, state, final flush và xử lý chuỗi byte chưa đủ | Lab chỉ demo UTF-8 hai byte; chưa có SSE/HTTP/stop-string matcher |

**Phạm vi kiểm bài 21:** phân phối số tự tạo [0.5,0.25,0.15,0.1], temperature/top-k/top-p/order và sampling 10000 lượt seed 21; beam 1/2 so exhaustive cây 12 chuỗi. KV cache tự ghép cho core một causal pre-norm block, CPU batch 1 không PAD; đối chiếu prefill, token đơn và chunk với full logits float 64 (max error 4.44e-16), K/V [1,4,8,8] tổng 4096 byte; mask upper-left sai/offset đúng cho output 1/2.5. Nối checkpoint full-SFT từ run mặc định bài 20: prompt 24 token, NET+EOS 4 token, cả 4 strategy cùng output ở seed 21, cache/no-cache khớp IDs; đếm 102/27 vị trí, không gọi là speedup. Default không cần checkpoint, dùng random weight để kiểm cơ chế và không tuyên bố chất lượng văn bản. Python 3.14.6,torch 2.15.0,NumPy 2.5.4,Matplotlib 3.11.2; 7 hình và 16 bài tập. Timestamp TTFT/TPOT và ví dụ 512 MiB là giả định/công thức, không benchmark phần cứng; chưa có neural beam cache, padding batch, server, KV sharing, continuous batching hoặc đánh giá decoder toàn test. Không train mới trong bài; ZIP loại runtime report/checkpoint/venv/cache.

## Nguồn dùng trong bài 22

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [JSON Schema object](https://json-schema.org/understanding-json-schema/reference/object) | properties/required/null/additionalProperties; đối chiếu contract object đóng | Lab xuất Draft 2020-12; không suy provider nào hỗ trợ toàn bộ draft hoặc constrained decoding |
| [JSON Schema composition](https://json-schema.org/understanding-json-schema/reference/combining) | oneOf/anyOf và nhánh chồng lấn; route/abstain | Không tự kiểm semantic đúng tác vụ hoặc quyền ở hệ thống ngoài |
| [jsonschema validation](https://python-jsonschema.readthedocs.io/en/stable/validate/) | Draft 202012 Validator, check_schema, is_valid/iter_errors | Đã chạy jsonschema 4.26.0; schema là code tin cậy cục bộ, không nạp schema tùy ý qua mạng |
| [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/) | Type coercion, strict validation và giới hạn theo type/input | Đã chạy Pydantic 2.14.2; không coi strict là chứng minh semantic |
| [FastAPI response model](https://fastapi.tiangolo.com/tutorial/response-model/) | Response validation/filtering, OpenAPI; tách candidate validation với response envelope | Đã chạy FastAPI 0.162.0; chưa auth/tenant/production gateway |
| [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/) | TestClient và kiểm endpoint trong process | Đã chạy ASGI 200/422/502/503/504 và HTTP localhost Uvicorn; không benchmark mạng/TLS/proxy |
| [Python 3.14 json](https://docs.python.org/3.14/library/json.html) | json.loads, object_pairs_hook, parse_constant và giới hạn dữ liệu parse | Lab reject duplicate key/NaN/fence và raw trên 16000 byte; không xây transport body-size limiter |
| [RFC 9110](https://www.rfc-editor.org/rfc/rfc 9110.html) | Mục 9.2.2 idempotency và 15.6.3–15.6.5 về 502/503/504 | Mapping candidate/refusal sang 502 là lựa chọn API minh họa; không triển khai idempotency store/HTTP Retry-After parser |
| [Emerging Patterns in Building GenAI Products](https://martinfowler.com/articles/gen-ai-patterns/) | Phần Evals/Scoring/Running Evals và task-specific evaluation | Kinh nghiệm tác giả, không bằng chứng judge nào luôn đúng; lab không chạy LLM judge/DeepEval/model thật |

**Phạm vi kiểm bài 22:** dữ liệu/candidate/error script tự tạo, provider rule để demo và scripted fixture để kiểm branch; không là model eval thật hoặc sample production. FastAPI/Pydantic strict cùng schema oneOf, null/required/extra field; parser duplicate key/nonstandard number/raw limit; evidence chỉ kiểm ID tồn tại. 20 fixture: completion 16, JSON 13, schema 9, accepted 8, correct 5; conditional accepted accuracy 5/8, task success 5/20, route 4/7,23 attempt. FakeClock cho 850/150/70 ms ở ca retry, không benchmark latency/billing. Đã kiểm ASGI HTTP, Uvicorn localhost health/proposal/OpenAPI (server dừng sau kiểm), bounded retry/deadline và mọi snippet Python. Python 3.14.6, FastAPI 0.162.0, Pydantic 2.14.2, httpx 0.28.1, Starlette 1.10.0, jsonschema 4.26.0, Uvicorn 0.54.0, NumPy 2.5.4, Matplotlib 3.11.2. Token counter 100/25 là giả lập; không giá dịch vụ thật, không account/key, không external LLM call. Chưa constrained decoding, semantic judge, SDK provider, cancellation thật, jitter, circuit breaker, auth hoặc tác động ghi. ZIP gồm schema JSON nguồn được kiểm; loại runtime report/venv/cache.

## Nguồn dùng trong bài 23

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) | Phần TF-IDF: raw TF, smooth IDF, L2 normalization; đối chiếu công thức | Code tự viết NumPy, không chạy scikit-learn; analyzer tiếng Việt/identifier là lựa chọn riêng |
| [Lucene BM25Similarity 9.12.2](https://lucene.apache.org/core/9_12_2/core/org/apache/lucene/search/similarities/BM25Similarity.html) | Công thức IDF, k1/b, TF saturation và length normalization | Nguồn có phiên bản cố định, không tuyên bố mới nhất; không chạy Lucene, không tái hiện field/norm encoding |
| [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) | LSA trên TF-IDF/count, không centering, components và dấu SVD | Lab dùng dense NumPy SVD rồi cắt 6 chiều; không phải sparse scalable implementation hay neural encoder |
| [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) | full_matrices=False, shape/order U,s,Vh, reconstruction | Đã chạy NumPy 2.5.4 CPU; singular directions có thể đổi dấu/biến thiên theo backend |
| [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) | Ý tưởng query/document embedding, symmetric/asymmetric search và cosine | Chỉ dùng để giải thích neural retrieval; chưa cài/chạy model, không đo chất lượng pretrained embedding |
| [Elastic RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) | Fusion theo rank, rank bắt đầu từ 1, rank constant và candidate window | Tự triển khai RRF, không chạy Elasticsearch; không suy hybrid luôn thắng hoặc không cần chọn tham số |
| [Faiss](https://faiss.ai/) | Giới thiệu exact/approximate dense vector search, distance và đánh đổi | Không chạy Faiss/ANN hay benchmark; exact lab là dot NumPy trên toàn corpus |
| [scikit-learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) | Discount, ideal ranking, top-k và tie; đọc parameters/examples | Lab tự tính gain 2^grade−1; API dùng relevance trực tiếp, cần đổi gain để đối chiếu, chưa chạy thư viện |

**Phạm vi kiểm bài 23:** nguồn đọc ngày 04/11/2026. Corpus 20 runbook và 18 query tự tạo (6 dev, 10 test answerable, 2 test no-answer); 19 document public, một team-blue. Không là dữ liệu production, không có nhãn độc lập nhiều người; dev/test khác query nhưng có thể chung chủ đề. NumPy CPU fit vocabulary/IDF/SVD trên corpus, không dùng qrels để fit; LSA 6 chiều chưa là neural embedding. TF-IDF/BM25/cosine/RRF/graded metrics tính tay đối chiếu code, postings BM25, L2/zero/OOV, lọc trước top-k và scope rỗng; kiểm 72 query-method save/load round-trip, chặn SVD khi load. Test macro NDCG@3: TF-IDF .97904694, BM25 .98346388, LSA/hybrid .97733116; không kết luận khác biệt có ý nghĩa thống kê. Mỗi phương pháp trả ứng viên cho 1/2 query no-answer, báo riêng. Python 3.14.6, NumPy 2.5.4, Matplotlib 3.11.2; mọi snippet đã chạy, 7 hình PNG/SVG. Chưa ANN/reranker/neural model/API server/auth, chưa incremental index hay benchmark latency. Scope demo chỉ lọc ứng viên, global IDF/SVD vẫn chứa tài liệu riêng tư, không là tenant isolation. ZIP loại runtime JSON/NPZ, venv/cache; chỉ giữ code/tài liệu/hình và JSON Schema nguồn của bài 22.

Đã thử hai trang Stanford IR book về IDF/BM25 nhưng truy cập timeout; không dùng nội dung những trang đó làm bằng chứng đã đọc. Đã chuyển sang tài liệu chính thức truy cập được ở trên.

## Nguồn dùng trong bài 24

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Microsoft: chunk documents](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents) | Common chunking techniques, overlap/factors, heading và input budget | Chỉ dùng nguyên lý và đánh đổi; không gọi Azure, không lấy size/overlap gợi ý làm tối ưu chung hoặc dùng số token/provider trong bài |
| [Hugging Face tokenizer API, main](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer) | Truncation only_first/only_second/longest_first, max_length, overflow và stride=overlap | Không chạy Transformers; lab dùng tokenizer word-level custom với budget riêng, không khẳng định tương đương subword |
| [Sentence-BERT 2019](https://arxiv.org/abs/1908.10084) | Abstract/metadata: Siamese/triplet cho sentence embeddings và cosine | Chưa đọc toàn paper ở lượt này, không dùng số benchmark làm kết quả lab; bi-encoder lab không phải SBERT |
| [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) | Background và symmetric/asymmetric query/document search | Không cài/chạy Sentence Transformers, không chọn/tải model pretrained; khả năng ngữ nghĩa cần đo theo tác vụ |
| [DPR, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.550.pdf) | Mục 3.1–3.2: dual encoder, NLL, random/BM25/in-batch negatives; đoạn 4.1 về passage/title | Lab dùng cosine/temperature, shared word-mean encoder nhỏ, không tái hiện BERT/DPR/FAISS hay benchmark bài báo |
| [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | Hai tầng retrieval/cross-encoder, đầu vào query–document chung | Không suy reranker luôn cải thiện, không chạy ví dụ pretrained của nguồn |
| [PyTorch TransformerEncoder 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoder.html) | Constructor, forward, key padding mask, nested tensor và reference implementation | Đã chạy một layer bidirectional, batch_first, tắt nested tensor; không causal, không kiến trúc pretrained lớn |
| [PyTorch torch.load 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.load.html) | weights_only=True, map_location CPU, state load và giới hạn nguồn không tin cậy | Bundle tự tạo chỉ đủ inference, không đủ resume; hash không là xác thực chống giả mạo |
| [Late Chunking v3](https://arxiv.org/abs/2409.04701) | Abstract/metadata bản sửa 07/07/2025: contextualize token dài trước, chunk/pool sau | Chỉ phần phân biệt khái niệm; chưa đọc/kiểm toàn bộ thực nghiệm, chưa triển khai hoặc khẳng định cải thiện trên corpus lab |

**Phạm vi kiểm bài 24:** nguồn truy cập ngày 04/11/2026. Dữ liệu tự tạo có 10 source (9 public, 1 restricted); 18 section public sinh 25 chunk theo section cap 48; fixed24/overlap0 có 40 chunk, overlap8 có 54. Source NFC giữ char offset/quote; coverage theo một canonical span/query, không là nhãn relevance ngữ nghĩa đầy đủ. Có 36 train, 18 dev và 20 test (18 answerable, 2 no-answer), khác query text nhưng chung intent/source/span, chưa kiểm unseen intent/source/time. Vocabulary 397 từ public chunk và train query; S10 bị loại trước fit. Bi-encoder word-mean/projection 32 chiều, 13,760 tham số, contrastive CE trên 36×25, 80 epoch chọn dev epoch 40. Cross-encoder một Transformer layer, 24,929 tham số, 216 train pair (36 positive/180 negative), weighted BCE với pos_weight=5, 30 epoch chọn dev epoch 10. Miner hybrid đóng băng, chỉ query/nhãn train; không chèn gold khi eval. Dev MRR@3 của BM25/dense/hybrid/rerank xấp xỉ .9352/.6574/.7778/.4630; test xấp xỉ 1/.75/.8889/.3148. Reranker thua baseline, không đổi số đo để giả cải thiện. No-answer: OOV gate trả rỗng một ca, ca có từ cache vẫn có ứng viên; chưa calibration/abstention hoặc trả lời RAG. Đã chạy 12 snippet, kiểm cửa sổ/offset/budget, masked pooling/PAD, CE/gradient và BCE số tay; train/checkpoint/evidence metrics; rerank giữ tập ứng viên; 74 query qua 3 retriever và reranker khi save/load; loader phát hiện cache mismatch. Python 3.14.6, torch 2.15.0, NumPy 2.5.4, Matplotlib 3.11.2, CPU một thread, seed 24; 7 hình PNG/SVG có code tái tạo. Không dùng model pretrained, API, GPU, ANN, service/auth, semantic chunking, MMR hay late chunking; số thời gian cục bộ trong report không là benchmark. ZIP loại runtime JSON/PT/NPY, venv/cache; giữ JSON Schema nguồn bài 22.

Trang Sentence Transformers losses bị timeout khi mở; không dùng như nguồn đã đọc, đã chuyển sang mục training của DPR cho phần negative/loss.

## Nguồn dùng trong bài 25

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Lewis et al., RAG — NeurIPS 2020](https://papers.nips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) | Abstract và methods, đặc biệt 2.1–2.3: parametric/non-parametric memory, RAG-Sequence/Token, retriever/generator | Công thức và số tính tay riêng; không tái hiện train BART/DPR, không dùng benchmark paper làm số liệu lab; retrieve–prompt không mặc định là thuật toán gốc |
| [ALCE — EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398.pdf) | Abstract, §3.3 citation support/precision, NLI và cách phân biệt citation không liên quan | Lab dùng annotation đóng cho claim/support spans, không chạy NLI/ALCE; không so trực tiếp điểm với benchmark hoặc coi quote match là entailment |
| [KILT — NAACL 2021](https://aclanthology.org/2021.naacl-main.200/) | Abstract/metadata: kiến thức bên ngoài, downstream task và provenance | Chưa đọc/triển khai toàn benchmark; không gọi metric riêng của lab là KILT score |
| [Lost in the Middle — TACL 2024](https://aclanthology.org/2024.tacl-1.9/) | Abstract: vị trí thông tin ảnh hưởng các model/tác vụ được nghiên cứu | Chưa benchmark model 2026 hoặc lab LLM; không suy mọi context dài/model đều có cùng mức giảm |
| [jsonschema 4.26.0 validation](https://python-jsonschema.readthedocs.io/en/stable/validate/) | Validator protocol, Draft202012Validator, check_schema/is_valid và giới hạn format | Đã chạy schema nguồn cục bộ, không nạp schema remote; schema chỉ kiểm contract, không chứng minh semantic |

**Phạm vi kiểm bài 25:** đợt đối chiếu/biên soạn 04–05/11/2026, hoàn thiện 05/11/2026. Dữ liệu tự tạo gồm 11 snapshot/10 source ID, 25 fact có span, 16 query (4 dev/12 test); mốc as_of mặc định cố định 2026-11-04, public/approved còn 8 source/13 chunk. Query lịch sử dùng P01@1 (24 giờ), hiện hành dùng P01@2 (72 giờ); hai runbook upload mâu thuẫn được giữ để kiểm status. Không là dữ kiện production. BM25 lọc ngày/quyền trước thống kê; whole-chunk greedy packing cap 110, window 320/output reserve 48 theo tokenizer word-level. Built-in generator chọn tối đa 2 câu có ít nhất 2 term query chung, không là LLM; reserve không cưỡng chế độ dài output sinh. Giao diện generator không nhận nhãn required/support; scorer dùng catalog đóng, unknown không tự là false. Test main: mechanical 12/12, packed đủ facts 10/10 query có required, grounded claim 22/22 và citation 22/22 nhưng task 10/12; no-answer trả claim 1/2. Cap 40/110/180 có packed fact recall .45/1/1 và task 4/12,10/12,10/12. Fixture lập trình sẵn 22 case: parse 21, schema 19, mechanical 12, task 5; micro attachment 19/20, grounded claim 11/20, citation precision 13/22. Đây không là tần suất lỗi LLM. Đã chạy 12 snippet Python, kiểm source/version/offset/quote/hash, context không có quyền, parser/schema, joint support, metric/mẫu số, budget/duplicate/oversize và renderer/snapshot links; thử lại CLI từ bản ZIP giải nén. Python 3.14.6, jsonschema 4.26.0, CPU; Matplotlib 3.11.2/NumPy 2.5.4 để tái tạo 7 hình PNG/SVG. Chưa model sinh/NLI/LLM judge/API, auth service, dynamic permission/caching hoặc phòng vệ prompt injection toàn diện. ZIP chỉ giữ tài liệu/code/hình, loại runtime prompt/answer/JSON/source snapshots; JSON Schema nguồn bài 22 vẫn được giữ. Query dev/test chung nguồn/ý định, annotation tự tạo, chưa kiểm tổng quát hóa hoặc statistical significance.

## Nguồn dùng trong bài 26

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Anthropic: Effective context engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) | Khái niệm context/prompt, thành phần context, compaction và note-taking cho tác vụ dài | Bài 29/09/2025, hướng dẫn của nhà cung cấp; không là benchmark hay bằng chứng một policy tối ưu chung |
| [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | Direct/indirect injection, hậu quả, phân tách dữ liệu, least privilege và các biện pháp giảm thiểu | Không khẳng định có phòng vệ tuyệt đối; lab chưa đo model thật hoặc mọi đường exfiltration |
| [OWASP Authorization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html) | Least privilege, deny by default, kiểm mọi request; phần attribute/relationship | Policy nhỏ trong RAM, không triển khai một authorization service hay SSO |
| [OWASP Transaction Authorization](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html) | Dữ liệu quan trọng của action, server-generated state, thứ tự kiểm, final gate, TOCTOU, expiry và uniqueness | Mượn nguyên lý cho ticket giả định; không xây hệ thống giao dịch tài chính hoặc review UI thật |
| [NIST SP 800-162](https://csrc.nist.gov/pubs/sp/800/162/upd2/final) | Metadata/abstract: ABAC dựa thuộc tính subject, object, operation và môi trường | Chưa đọc toàn PDF; policy lab không được chứng nhận tuân thủ chuẩn |
| [Greshake et al., indirect prompt injection, 2023](https://arxiv.org/abs/2302.12173) | Abstract/metadata về nguy cơ instruction từ dữ liệu ngoài qua ứng dụng LLM | Chưa tái hiện thực nghiệm paper, không dùng số liệu paper làm kết quả lab |
| [AgentDojo, 2024](https://arxiv.org/abs/2406.13352) | Abstract/metadata bản v3 về đánh giá utility/security với công cụ và dữ liệu không tin cậy | Không chạy AgentDojo hoặc gọi fixture riêng là benchmark này |
| [Python 3.14 secrets](https://docs.python.org/3.14/library/secrets.html) | OS randomness và token_urlsafe(nbytes) cho token khó đoán | Chạy Python 3.14.6; không dùng random cho grant; không triển khai bearer credential production |
| [jsonschema validation](https://python-jsonschema.readthedocs.io/en/stable/validate/) | Draft202012Validator, check_schema/is_valid và validation contract | Chạy jsonschema 4.26.0; schema không chứng minh quyền hoặc ý định |

**Phạm vi kiểm bài 26:** nguồn đọc ngày 05/11/2026. Dữ liệu tổng hợp có 5 principal, 8 resource, 6 record; ma trận đọc có 26/40 cặp được phép. Authorized BM25 lọc trước df/avgdl/rank; context giữ authority=data, lineage/version/hash, cache theo subject/config/revision và kiểm lại từng dependency khi hit. Query Alice mặc định có 7 item, context 227/prompt 322 token word-level custom, reserve 64/window 480; không là tokenizer/provider thật. Gateway chỉ có get_ticket và close_ticket trong RAM, parse/schema/ACL/role/scope, grant bound subject/tenant/proposal/resource/ACL/expiry và consume+mutate dưới RLock. Grant do trusted controller cấp sau bước review giả định, ngoài prompt; chưa có auth service/review UI/sandbox hoặc distributed transaction. Bộ 22 proposal có 17 attack, 5 benign; schema-only cố ý yếu có 13/17 event ngoài quyền/ý định, gateway 0/17, benign 5/5 cả hai. Sáu confidentiality check có unsafe exposure 6/6, guarded 0/6; freshness và taint/authority báo riêng. Answer-only BANANA_OVERRIDE vẫn sai, không được coi là đã chặn. Đây là regression deterministic, không là ASR của LLM hay risk production. Đã chạy 10 snippet Python độc lập, mọi mode CLI và kiểm version/revoke/subject/expiry/replay/8 thread dùng một grant chỉ một effect; đối chiếu budget/knapsack, quyền, cache và các mẫu số; chạy lại từ ZIP giải nén. Python 3.14.6, jsonschema 4.26.0, CPU; NumPy 2.5.4/Matplotlib 3.11.2 để tái tạo 7 hình PNG/SVG. Không mạng, không tài khoản/LLM/API hoặc effect thật. ZIP loại runtime JSON/prompt/report, venv/cache; giữ JSON Schema nguồn của bài 22.

## Nguồn dùng trong bài 27

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Anthropic: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) | Client/server tools, vòng tool_use → application → tool_result, tool_use_id và ví dụ round trip | Đối chiếu nguyên lý; không chạy SDK, model hay API cụ thể; protocol lab tự thiết kế, không phải wire format provider |
| [Building effective agents](https://www.anthropic.com/research/building-effective-agents) | Phân biệt workflow/agent, chaining, routing, parallelization và gate | Bài 19/12/2024 có lưu ý tooling đã đổi; dùng pattern, không chạy framework hoặc suy một pattern luôn thắng |
| [AWS: Timeouts, retries, backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/) | Failure/timeout, side effect có thể đã xảy ra, retry làm tăng tải, backoff/jitter và phạm vi timeout | Lab dùng time model và deterministic backoff, không có network client, jitter hoặc benchmark latency dịch vụ |
| [AWS: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) | Retry/side effects, client request identifier, receipt/effect atomicity; cùng token khác tham số và parameter mismatch | Backend RAM/RLock chỉ minh họa dedup trong process; chưa durability, crash recovery hay transaction phân tán |
| [RFC 9110 §9.2.1–9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2) | Safe/idempotent semantics, intended effect, logging và retry non-idempotent methods | Áp dụng phân biệt khái niệm; tool nội bộ phải tự định nghĩa semantics, không triển khai HTTP server |
| [Python 3.14 asyncio tasks](https://docs.python.org/3.14/library/asyncio-task.html) | Coroutine scheduling, TaskGroup, failure propagation/cancellation, timeout và wait_for | Runtime Python 3.14.6; trang docs đang cập nhật dòng 3.14. Demo sleep giả I/O, không là durable workflow engine hoặc remote rollback |
| [jsonschema validation](https://python-jsonschema.readthedocs.io/en/stable/validate/) | Validator protocol, check_schema, Draft202012Validator và validate/is_valid | Chạy jsonschema 4.26.0 cho contract nội bộ; schema chưa chứng minh quyền, nguồn hoặc nghiệp vụ đúng |

**Phạm vi kiểm bài 27:** nguồn truy cập/đọc ngày 05/11/2026. Lab độc lập có 3 principal, 3 ticket, KB/health theo tenant, 4 tool; model được thay bằng template draft và lời gọi do controller cố định. Workflow chính tuần tự, states có WAITING_REVIEW/NEEDS_HUMAN/CANCELLED/FAILED/UNKNOWN/SUCCEEDED, tối đa 2 attempt/read, 1 write, 10 call attempt và 1.000 modeled ms/run. Runtime kiểm cả input/output, ID/resource correlation, batch hợp lệ trước dispatch; identity/review/key ngoài proposal. Effect/receipt/key conflict kiểm trong Backend lock; grant bound proposal/principal revision/evidence hash/version/TTL. Một read retry có 5 call attempt, 420 modeled ms; happy 350; tính tay fan-out critical path 270, tool work 300, không là số latency mạng. Demo riêng TaskGroup thực sự concurrent với sleep 8/12 ms, có optional failure và group timeout. 22 scenario đạt expected state/effect 22/22, không phải 100% task completion. Trước đối soát có succeeded 4/failed 10/needs_human 2/waiting_review 2/cancelled 1/unknown 3; sau đối soát succeeded 6 và unknown 1, các state khác giữ nguyên. Tổng 6 effect trong 22 Backend độc lập, không có note green; poisoned body không được diễn giải để chọn tool, chưa là benchmark prompt injection/LLM. Đã chạy 13 snippet độc lập, 6 mode CLI và round trip model mô phỏng 2 turn/1 read; kiểm parse/schema/correlation, wrong resource, permission/stale/review, budget/backoff, missing/invalid batch không dispatch, 8 thread gửi cùng write/key chỉ 1 effect, conflict/revoke/expiry, ambiguous write/receipt và async fan-out bằng barrier. Sáu hình PNG/SVG đã kiểm bằng mắt, có script tái tạo; bản ZIP đã giải nén chạy lại. Python 3.14.6, jsonschema 4.26.0, CPU; NumPy 2.5.4/Matplotlib 3.11.2 chỉ để vẽ. Chưa provider SDK/LLM, auth/review UI, I/O/network deadline, persisted checkpoint, distributed transaction, outbox/saga hay production observability. Snapshot không là resume state được xác thực; receipt không bền sau crash. ZIP loại runtime JSON/report, venv/cache; giữ JSON Schema nguồn bài 22.

## Nguồn dùng trong bài 28

| Nguồn | Phạm vi đã đọc và dùng | Giới hạn |
|---|---|---|
| [Building effective agents](https://www.anthropic.com/research/building-effective-agents) | Workflow/agent, feedback môi trường, điều kiện dừng và quyền host giữ | Bài 19/12/2024 có lưu ý tooling đã đổi; dùng khái niệm, chưa benchmark hoặc chạy agent/framework của nguồn |
| [ReAct — ICLR 2023](https://arxiv.org/abs/2210.03629) | Abstract/metadata bản v3: reasoning/actions đan xen và feedback từ môi trường | Chưa đọc toàn paper hoặc tái hiện benchmark; lab không phải ReAct model, không lưu suy nghĩ nội bộ LLM |
| [LangChain memory overview](https://docs.langchain.com/oss/python/concepts/memory) | Short/thread-scoped và long-term namespace, history/context, semantic/episodic/procedural; semantic memory khác semantic search | Dùng cách phân loại; không cài/chạy LangChain/LangGraph hoặc chứng minh một memory design tối ưu |
| [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Checkpointer/store, phạm vi thread/cross-thread và giới hạn RAM khi restart | Lab tự viết SQLite latest checkpoint + event metadata, không triển khai graph/checkpointer API hoặc time travel |
| [Effective context engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) | Quản lý context qua vòng lặp, phần long-horizon compaction và structured note-taking | Bài 29/09/2025, hướng dẫn nhà cung cấp; summary lab deterministic và vẫn giữ observations, không là compression/quality benchmark |
| [CoALA — Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) | Abstract/metadata v3 tháng 03/2024: memory modules, structured actions và decision process | Chưa đọc toàn paper/triển khai kiến trúc, không lấy kết quả nguồn làm số liệu lab |
| [Python 3.14 sqlite3](https://docs.python.org/3.14/library/sqlite3.html) | File DB, placeholders, autocommit, explicit transaction, commit/rollback và mở lại connection | Chạy Python 3.14.6; lab dùng autocommit=True + SQL BEGIN/COMMIT/ROLLBACK tường minh, chưa production HA hoặc secure DB transport |
| [SQLite Transaction](https://www.sqlite.org/lang_transaction.html) | BEGIN/COMMIT/ROLLBACK, một writer, BEGIN IMMEDIATE và khả năng busy | Dùng transaction cho checkpoint/event/CAS local; không chứng minh exactly-once remote call hay khả năng chịu mọi lỗi storage |

**Phạm vi kiểm bài 28:** nguồn truy cập/đọc ngày 05/11/2026. Lab độc lập có 3 principal, 7 resource (3 ticket/2 runbook/2 health), 3 read tool; policy Python chọn action từ observation, không gọi LLM/API/mạng. State có owner/task/version, observation refs/TTL, pending, counters và revision; host reserve budget trước model/tool và lưu latest checkpoint + event metadata trong SQLite transaction/CAS. Mặc định 8 turn, 6 tool reservation, 6.000 context units custom; cap 900/lượt gồm reserve 64. Happy 4 turn/3 read/1.378 units; retry intent 3/2; billing handoff sau 1 read. Restart không reset budget; crash model có 5 turn; crash trước/sau read có 4 reservation với 3/4 actual read attempts. Không có exactly-once, lease/fencing hoặc write tool. Memory note do trusted controller review giả định, namespace tenant/actor, lexical recall sau ACL/TTL/source hash/version, CAS update và tombstone forget; không đổi model weights, không tự sửa control, chưa đo memory usefulness. 25 scenario đạt expected status 25/25: DONE 9/HANDOFF 5/STALLED 1/LIMIT_REACHED 4/FAILED 5/CANCELLED 1; semantic_residual vẫn DONE với BANANA_OVERRIDE sai nội dung, không được tính là đã chặn. Đã chạy 13 snippet độc lập, 7 mode CLI, start/resume ở hai process; kiểm namespace/run owner, source update/revoke/missed version bump/health TTL, snapshot revalidation, model-view copy, persistent budget/recovery, cancel chống stale writer, rollback đồng thời checkpoint/event bằng trigger lỗi, CAS hai connection và 8 memory writer chỉ 1 update thắng. Memory demo kiểm expiry đúng biên 30, cross-session reopen, khác actor/tenant, version conflict/source update/revoke/forget. Report/hash tái tạo nhất quán; 7 hình PNG/SVG đã kiểm bằng mắt; bản ZIP giải nén chạy lại CLI/snippet và tái tạo hình. Python 3.14.6, SQLite runtime 3.53.3, jsonschema 4.26.0, CPU; NumPy 2.5.4/Matplotlib 3.11.2 chỉ để vẽ. Raw DB là host-trusted, có thể giữ plaintext lịch sử; snapshot không xóa DB/backup và tombstone chưa là secure deletion. Chưa provider SDK/model, LangGraph, embedding/vector DB, auth service, review UI, mạng/deadline thật, distributed workflow hoặc semantic verifier. ZIP loại DB/report/runtime JSON, venv/cache; giữ JSON Schema nguồn bài 22.

## Nguồn định hướng đề cương

| Nguồn | Phần đã đọc / mục đích | Tính thời điểm và giới hạn |
|---|---|---|
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/) | Mục lục và phạm vi về mô hình, dữ liệu, đánh giá, hệ thống | Dùng để tổ chức nền tảng; cần đọc bài cụ thể khi biên soạn |
| [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Giới thiệu quy trình dữ liệu, mô hình, tối ưu và lưu mô hình | Chưa kiểm tra toàn bộ notebook; phiên bản sẽ chốt theo bài thực hành |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Trang giới thiệu, yêu cầu đầu vào và phạm vi chương | Chưa đọc hết khóa; mô hình/thư viện cụ thể thay đổi theo thời gian |
| [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | Mục lục và mô tả phần dữ liệu, kiểm thử, triển khai, monitoring | Tài liệu 2022: dùng nguyên lý, không lấy phiên bản công cụ làm chuẩn 2026 |
| [Building effective agents](https://www.anthropic.com/research/building-effective-agents) | Phân biệt workflow/agent và đánh đổi độ phức tạp | Bài 19/12/2024; chính trang có lưu ý công cụ đã thay đổi; góc nhìn nhà cung cấp |
| [Effective context engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) | Khái niệm context và cách lựa chọn thông tin | Bài 29/09/2025; hướng dẫn thực hành, không phải bằng chứng giải pháp nào luôn tốt hơn |
| [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) | Cấu trúc task/trial/grader và kết hợp các cách đánh giá | Bài 09/01/2026; cần thiết kế đánh giá riêng theo ứng dụng |
| [PEFT: Quantization](https://huggingface.co/docs/peft/developer_guides/quantization) | Giải thích lượng tử hóa và kết hợp PEFT | Chưa chạy ví dụ GPU; không cam kết mức VRAM hay tăng tốc cụ thể |

## Cách bổ sung nguồn

Ghi nguồn tại đoạn cần kiểm chứng, rồi bổ sung bảng này với phần thực sự đã đọc. Nêu phiên bản cho API/thư viện khi dùng. Không biến danh sách đọc thêm thành tuyên bố đã xác minh; không suy kết quả tuyển dụng hoặc mức lương từ các tài liệu kỹ thuật này.
