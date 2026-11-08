# Phụ lục bài 43 — Forecasting qua lăng kính xác suất và hệ thống

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Information set quan trọng hơn tên kiến trúc

Gọi ℱ_T là tập thông tin thật sự có ở origin T: target đã chốt, lịch đã công bố, phiên bản dữ liệu và dự báo ngoại sinh đã phát hành. Forecast là hàm `f(ℱ_T,h)`. Target tương lai có thể xuất hiện trong file offline để chấm điểm, nhưng không thuộc ℱ_T.

Một feature có `event_time < T` vẫn có thể rò rỉ nếu `available_at > T`. Một feature có `event_time > T` vẫn có thể hợp lệ nếu là lịch đã công bố trước T. Với dữ liệu sửa đổi, cần lưu cả effective time và ingestion/publication time để reconstruct bản người dự báo đã thấy. Đây là lý do as-of join khác lọc đơn thuần bằng ngày sự kiện.

Nhiệt độ thực ngày mai và forecast nhiệt độ ngày mai là hai biến khác nhau, có sai số khác nhau. Backtest bằng actual thời tiết rồi production bằng forecast thời tiết có thể tạo khoảng cách chất lượng. Scenario forecasting giả sử “nếu nhiệt độ là 30” cũng hợp lệ như một câu hỏi có điều kiện, nhưng không phải ex-ante forecast vô điều kiện. Đối chiếu [FPP3 §7.6](https://otexts.com/fpp3/regression.html).

## 2. Từ loss tới đại lượng cần dự báo

Với X là thông tin đầu vào, chọn a để tối thiểu expected squared error:

`E[(Y−a)²|X] = Var(Y|X) + (E[Y|X]−a)²`.

Thành phần thứ nhất không phụ thuộc a, nên tối ưu là conditional mean. Với absolute error, median là một tối ưu; khi phân phối có đoạn phẳng có thể có nhiều median. Pinball tại τ nhắm tới conditional τ-quantile dưới điều kiện thông thường.

Nếu sai số thiếu capacity tốn c_under mỗi đơn vị, thừa capacity tốn c_over, loss một bước là `c_under*(y−a)_+ + c_over*(a−y)_+`. Chia tổng hai cost cho pinball tương ứng τ=`c_under/(c_under+c_over)`. Đây là mô hình chi phí đơn giản: queueing, threshold, stockouts, SLA và quyết định liên thời gian có thể làm bài toán phức tạp hơn. Không lấy MAE nhỏ nhất làm mặc định tối ưu mọi quyết định.

Lab train MLP bằng MSE trên từng residual đã chuẩn hóa, nhưng chọn family/epoch bằng raw MAE. Đây là quyết định thiết kế được khai báo, không phải hai objective giống nhau. Chuẩn hóa riêng Y theo h còn làm mỗi horizon có trọng số khác trong raw squared error.

## 3. Một bước cập nhật exponential smoothing và một AR model

Simple exponential smoothing cập nhật level `ℓ_t=α*y_t+(1−α)*ℓ_(t−1)`, 0<α≤1. Nếu ℓ trước=100, y=120, α=.2 ⇒ℓ mới=104. Không có trend/season, forecast tương lai từ trạng thái đó đều 104. Holt/Holt-Winters thêm các trạng thái khác; đây chưa phải code ETS đã fit/tune. [FPP3 exponential smoothing](https://otexts.com/fpp3/exponential-smoothing.html) mô tả họ phương pháp cho trọng số giảm dần lên quá khứ.

AR(1): `y_t=c+φ*y_(t−1)+ε_t`. Nếu innovation có conditional mean 0 và model đúng, forecast một bước là c+φ*y_T. Tính lặp với c 1, φ.8, y_T 10 cho Các giá trị là 9; 8.2; 7.56. Khi |φ|<1, mô hình AR(1) ổn định thông thường có mean dài hạn c/(1−φ)=5, không phải c=1. Điều này không biến mọi dữ liệu thực thành AR(1).

AR(p) dùng p lag của y. MA(q) trong ARIMA dùng q **innovation/error** quá khứ; chúng thường cần ước lượng, không phải tập quan sát y để lấy trung bình trượt. ARIMA(p,d,q) kết hợp AR/MA sau d lần sai phân; SARIMA thêm cấu trúc mùa vụ. [FPP3 ARIMA](https://otexts.com/fpp3/arima.html) là tài liệu học tiếp; lab ridge/MLP không phải implementation đầy đủ của ARIMA.

## 4. Window cutoff và khoảng cách giữa các tập

Với label kết thúc ở `T+H`, train prefix `[0,N)` cho điều kiện `T+H<N`. Input đủ L điểm cho `T≥L−1`. Do đó số hàng là N−L−H+1 khi N đủ lớn. N=224/L=28/H=7 ⇒190. One-step có 196 hàng; đừng dùng số 190 để mô tả recursive training.

Một hàng có origin 220, H 7 mang labels 221…227. Nó không được fit ở cutoff 224, dù phần input chỉ nhìn quá khứ. Loại những hàng có label vượt boundary là điều kiện bắt buộc. Một khoảng trống bổ sung giữa train và validation có thể cần khi label/feature publication trễ, hoặc khi muốn một thiết kế kiểm định cụ thể; không có một số “gap” thần kỳ áp cho mọi chuỗi.

Validation input có thể dùng y của các ngày train cuối. Origin 230 được dùng y 224…230 vì đã xảy ra lúc đó, dù chúng thuộc target của forecast trước. Không được kết luận rò rỉ chỉ vì input của một split chạm lịch sử split trước; cần xét ℱ_T. Ngược lại centered moving average dùng cả y_(T+1) để làm feature tại T là leak. Trailing rolling mean gồm y_T chỉ hợp lệ nếu y_T đã chốt tại origin.

## 5. Ridge chuẩn hóa và intercept

Cho X∈R^(n×d), Y∈R^(n×H) là residual target. Tính μ_X,σ_X,μ_Y,σ_Y trên training rows, với std floor 1. Gọi Z là X chuẩn hóa; U là Y chuẩn hóa; A=`[1,Z]`. Lab giải:

\[
\min_B\|AB-U\|_F^2+\lambda\sum_{j=1}^{d}\|B_j\|_2^2,
\quad B=(A^\top A+\lambda D)^{-1}A^\top U,
\quad D=\mathrm{diag}(0,1,\ldots,1).
\]

Code dùng `solve`, không nghịch đảo ma trận explicit. Công thức là **sum squared error + λ penalty**, không phải mean squared error + cùng λ; nếu đổi cách chuẩn hóa loss phải đổi λ tương ứng. Intercept hàng 0 không bị phạt. Reverse normalization và cộng anchor y_T để có dự báo mức.

Direct known H 7 có B 53×7=371 hệ số kể cả intercept. Recursive known H 1 có B 35×1. MLP có 52×32+32+32×7+7=1927 parameter. Số parameter không tự cho biết model tốt hơn hay thời gian chạy; dữ liệu, optimization, inductive bias và evaluation mới quyết định.

Anchoring giúp model học thay đổi quanh mức cuối, nhưng nếu mức cuối là outlier thì mọi horizon có thể bị ảnh hưởng. Một model dùng normalized lag vẫn có thể gặp shift ở pattern/variance. Thêm feature về sự cố chỉ có ích nếu feature được biết đúng lúc; đừng lấy nhãn “sự cố tương lai” từ generator.

## 6. Recursive và direct không có một bên luôn thắng

Recursive train một bước bằng lịch sử actual, nhưng khi h>1 inference có input sinh ra. Đây là một dạng train/inference mismatch. Nếu slope ước lượng sai, rollout có thể khuếch đại hoặc kéo về mean sai. Với H 7, mỗi forecast trong lab dùng lịch sử do chính nó sinh ra sau bước đầu, giữ as-of của plan ở T.

Direct multi-output không đưa output trở lại input; có thể học hiệu ứng riêng từng h. Đổi lại output rộng hơn, số cửa sổ train ít hơn cho cùng prefix. Bảy output của ridge có thể được hiểu là bảy bài toán tuyến tính với cùng design matrix và cùng λ; không phải phân phối joint learned. Một model autoregressive có stochastic rollout cũng cần giả định về innovation để tạo sample path, không chỉ lặp point prediction.

Kết quả lab recursive tốt hơn direct một chút ở validation, còn direct tốt hơn ở test ổn định. Điều này minh họa không nên nhìn test để quay lại chọn family. RMSE của direct trên shift cũng thấp hơn recursive dù MAE cao hơn; “best model” phải gắn với metric/quy trình đã định.

## 7. Backtest: cố định model, refit hay cập nhật online?

Ba thiết kế khác nhau:

1. Fixed-origin: tại T phát toàn bộ H bước, không đọc actual bên trong khoảng đó.
2. Rolling-origin, fixed weights: cứ đến origin mới lấy actual vừa tới, giữ scaler/weights. Lab thuộc nhóm này sau refit 280.
3. Rolling refit: mỗi origin fit lại theo expanding/sliding window từ dữ liệu có lúc đó; cần đo thêm chi phí và tính khả thi.

Cả ba có thể hợp lệ cho sản phẩm khác nhau. Không so model A được actual mới và retrain với model B bị giữ lịch sử cũ rồi kết luận chỉ vì kiến trúc. Lịch phát hành hàng tuần với H 7 cũng khác lịch phát hành hằng ngày H 7, nơi target lặp nhiều lần. [FPP3 §5.10](https://otexts.com/fpp3/toolbox.html) giúp hình dung rolling origin; các boundary cụ thể của lab là thiết kế tự biên soạn.

Đừng tune liên tục trên calibration: sau khi dùng nó để quyết định cả family/feature rồi hiệu chỉnh interval trên cùng lỗi, bước phân tách đã thay đổi. Có thể thiết kế nested backtest, nhiều window calibration hoặc online update, nhưng phải mô tả rõ dữ liệu/decision ở mỗi mốc.

## 8. Scale metric, aggregation và hierarchy

MASE dùng scale train, nên vẫn chấm được test target 0 nếu scale>0. s nhỏ bất thường có thể làm MASE rất lớn; chuỗi constant khiến s 0 và metric không xác định theo công thức này. Lab kiểm lỗi thay vì ép số hữu hạn. Validation scale 3.3870968 từ 224 ngày; test scale 3.3846154 từ 280 ngày; không suy ngược scale từ test.

Với nhiều series, pooled MAE theo toàn bộ điểm làm series lớn/nhiều điểm ảnh hưởng mạnh. Macro mean MASE mỗi series cho trọng số khác. WAPE gộp thường nghiêng về series nhiều volume; không tự đo độ công bằng giữa cửa hàng. Cần báo cáo cả population, missing policy, horizon và aggregation.

Nếu cửa hàng cộng thành vùng rồi toàn quốc, forecast riêng từng cấp có thể không cộng khớp. Hierarchical reconciliation là nhánh khác; chưa thực hiện trong lab. Tương tự forecast demand khi stockout hoặc demand thưa cần định nghĩa observation và loss riêng, không lấy metric của chuỗi daily sạch áp máy móc.

## 9. Calibration từng h và giới hạn thống kê

Ở mỗi h, lab có m=12 residual tuyệt đối. Sort và chọn order statistic thứ 11. Khoảng đối xứng quanh point không cần giả định residual Gaussian để tính, nhưng **coverage đáng tin cần thêm điều kiện**. Với exchangeable calibration/test scores và một predictor cố định độc lập với các calibration labels, công thức split conformal có ý nghĩa hữu hạn mẫu. Phụ thuộc thời gian/shift không tự thỏa exchangeability.

Ở đây chỉ kiểm coverage thực nghiệm trên hai chế độ. Mỗi h có 8 test forecast; aggregate 56 vẫn phụ thuộc. Không dùng binomial standard error với 56 IID trials để công bố độ chắc chắn. Coverage riêng từngh, coverage theo nhóm và coverage đồng thời một path là ba câu hỏi khác nhau.

Nếu mỗi ngày có marginal coverage .8 và **giả sử độc lập**, xác suất cả 7 ngày đều covered là .8 ⁷≈.2097. Không giả sử độc lập thì không được dùng số đó cho dữ liệu lab. Minh họa cho thấy “80% từng điểm” không đồng nghĩa “80% cả tuần”. Muốn path-level band có thể calibrate score cực đại chuẩn hóa của cả vector horizon; cần thiết kế riêng và đủ data, không chỉ đổi tên biểu đồ.

## 10. Độ rộng, interval score và shift

q thực nghiệm của model chọn lần lượt khoảng `[2.6154,1.8322,1.2453,1.9038,2.5570,2.0383,1.7943]`. Width là 2 q, mean 3.9961. Không có ràng buộc q tăng theo h; chỉ 12 scores/h và residual model khác theo h.

Nếu actual vượt hi, interval score tăng 10 đơn vị cho mỗi đơn vị vượt với α.2. Vì vậy nhánh shift có cùng mean width nhưng score 47.4291 so 4.9105 ở stable. Đây là lời nhắc kiểm cả calibration và sharpness; width hẹp không tự là tốt.

Một chiến lược thực tế có thể dùng window gần đây, group calibration, drift alarm hoặc adaptive update. Mỗi chiến lược có trade-off về ổn định, delay và bảo đảm riêng. [Adaptive Conformal Inference](https://arxiv.org/abs/2106.00170) là hướng đọc tiếp từ abstract, không phải thuật toán đã được lab kiểm tra. Nếu refit point model, residual cũ cũng không tự đúng cho model mới; cần tái thiết kế calibration.

## 11. Deep forecasting và dữ liệu nhiều chuỗi

RNN đọc thứ tự bằng hidden state; TCN dùng causal convolution theo thời gian; Transformer dùng attention với representation thời gian/position thích hợp. Forecasting token thường là giá trị/vector/patch thời gian, không tự là token văn bản của bài 19. Dữ liệu có khác biệt tần suất, scale, missingness, seasonality và covariate availability.

[PatchTST abstract](https://arxiv.org/abs/2211.14730) mô tả patching và channel-independent weight sharing. [LTSF-Linear abstract](https://arxiv.org/abs/2205.13504) nhấn mạnh so sánh với linear baseline trong nghiên cứu của họ. Hai ví dụ dẫn đến thói quen kiểm baseline, protocol, dataset và compute, không dẫn đến tuyên bố một kiến trúc thống trị mọi forecasting. Chưa đọc toàn văn/tái lập nên không đưa các số cải thiện trong paper vào bài.

Một global model có thể tận dụng nhiều chuỗi nhưng cũng cần split giữa entity và giữa thời gian, tránh leakage do chuẩn hóa toàn bộ series, duplicate series hoặc pretraining overlap. Với model tiền huấn luyện, kiểm thêm license, context/horizon, tần suất và memory/latency. Lab một series nhỏ chưa đủ để đánh giá lợi ích transfer.

## 12. Checklist lập luận trước khi giao một forecast

Có thể dùng các câu hỏi sau để review sản phẩm của chính mình:

- Tôi dự báo target gì, cho origin/horizon/frequency nào, dữ liệu chốt lúc nào?
- Baseline có dùng đúng lịch sử/covariate và cùng schedule với model không?
- Model/scaler/feature selection/refit/calibration có thể tái dựng chỉ từ quá khứ ở từng mốc không?
- Tôi đang đo lỗi trên forecast pair, ngày, entity hay quyết định kinh doanh? Case zero/missing xử lý thế nào?
- Interval là pointwise hay pathwise; mức nominal có được kiểm thực nghiệm; shift có thể làm nó sai không?
- Artifact lưu gì; dịch vụ còn cần state/history/version nào; khi actual đến trễ thì metric có bị join sai không?

Nếu chưa trả lời được một câu, đó là điểm cần làm rõ trước khi diễn giải số đẹp. Thực hành bắt đầu bằng [bài tập](exercises.md); phần kiểm tra mã không thay thế đánh giá dữ liệu thật.
