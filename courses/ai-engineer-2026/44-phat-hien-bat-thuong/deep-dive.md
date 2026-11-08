# Phụ lục bài 44 — Giả định, thống kê và trạng thái của detector

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Unusual, harmful và unknown là ba khái niệm khác nhau

Một score có thể phản ánh khoảng cách, density, likelihood hoặc lỗi dự đoán. Model thường không biết utility của một sự kiện. Một khách hàng mới có pattern hiếm nhưng hợp lệ; request lỗi lặp lại quá thường xuyên có thể nằm ngay trong vùng reference. “Unknown class”, “out-of-distribution” và “sự cố nghiệp vụ” không đồng nghĩa hoàn toàn.

Với classifier nhiều lớp, max softmax thấp là một heuristic cho uncertainty, không tự là detector OOD được calibrated. Với density model, likelihood cao cũng không tự nói observation có semantic đúng. Cần đặt câu hỏi cụ thể: khác population nào, theo feature nào, và false positive/negative dẫn tới quyết định gì? Lab giữ score như score và đánh giá với nhãn generator, không biến nó thành probability.

Nếu nhãn có đủ, classifier supervised có thể là baseline hợp lý. Normal-only không tự cao cấp hơn supervised; nó là giả định dữ liệu khác. Khi nói “không dùng nhãn” cần tách fit representation, chọn hyperparameter, threshold và model selection. Trong lab, labeled validation chọn kNN sau khi score/threshold mỗi family đã fit/calibrate.

## 2. Robustness không có nghĩa không thể bị lừa bởi contamination

Median ít bị một điểm rất lớn kéo lệch hơn mean. Nhưng contamination đủ lớn hoặc có cấu trúc vẫn làm median/MAD/reference sai. Một cụm anomaly đông có thể thành cụm dày; detector density cần phân biệt population normal thực sự với “cái thường xuất hiện trong log”.

Trong ví dụ reference `[10,11,12,13,14]`, thêm 100 khiến mean từ 12 lên 26.6667, median thành 12.5. MAD cũ 1, MAD mới 1.5. Median cũng thay đổi, chỉ thay đổi ít hơn trong ví dụ. Nếu nhiều giá trị lặp, MAD có thể 0 dù có một vài outlier. Dùng `.6745` gắn với scaling dưới normal, không phải lời hứa mọi distribution sẽ có cùng tail probability. [NIST](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) bàn về masking/swamping và phân biệt labeling với formal identification.

Một fallback cho cảm biến rời rạc có thể dùng ngưỡng tuyệt đối theo độ phân giải; một target tiền tệ có thể cần log transform; missing cần flag hoặc quarantine. Phải đánh giá lại metric theo population sau transform/fallback. Lab chọn từ chối zero/near-zero scale, chưa tự chọn một chính sách nghiệp vụ.

## 3. Mahalanobis: covariance làm xoay thước đo

Với covariance symmetric positive-definite Σ=VΛVᵀ, đặt a=Vᵀ(x−μ). Khi đó:

\[
(x-\mu)^\top\Sigma^{-1}(x-\mu)=\sum_j\frac{a_j^2}{\lambda_j}.
\]

Hướng có variance nhỏ phạt mạnh hơn hướng thường biến thiên. Đó là lý do `[1,−1]` lạ trong dữ liệu tương quan dương. Với covariance gần singular, nghịch đảo có thể khuếch đại noise; regularization thêm λI làm mẫu số bớt nhỏ. Lab λ=.001 trên standardized features, không phải cùng ý nghĩa khi áp trên raw units.

Nếu dữ liệu Gaussian, tham số biết đúng và điều kiện phù hợp, squared distance có quan hệ với χ². Mean/covariance ước lượng, contamination, finite sample và non-Gaussian làm diễn giải tail thay đổi. Lab dùng threshold empirical held-out, không áp critical value χ² rồi tuyên bố FPR chính xác. Empirical covariance của lab không phải `EllipticEnvelope` hoặc MCD; chỉ có ý tưởng covariance-distance chung.

## 4. PCA residual và latent distance đo hai điều khác

Gọi V_k là k principal vectors, có `V_kᵀV_k=I`. Projector P=V_kV_kᵀ có P²=P, Pᵀ=P. Reconstruction residual `(I−P)z` chỉ giữ thành phần ngoài subspace. Vì vậy mọi z nằm trong span(V_k) đều có residual 0, kể cả z rất xa trung tâm.

Có thể báo cả reconstruction Q-statistic và latent standardized distance như T², hoặc kết hợp score. Nhưng một hàm kết hợp cần calibration/selection riêng, không chỉ cộng hai số khác thang tùy ý sau khi xem test. Mahalanobis regularized trong lab dùng cả hai hướng, còn PCA residual chỉ dùng hướng bị loại. Đây là hai mục tiêu score khác nhau.

Autoencoder tuyến tính với squared reconstruction, các điều kiện phù hợp và nghiệm tối ưu có thể học cùng subspace PCA; network phi tuyến, constraint, optimizer và dữ liệu có thể thay đổi kết quả. Không gọi PCA NumPy là đã train deep autoencoder. Với AE, decoder capacity lớn có thể tái tạo cả anomaly; score cần đánh giá held-out như mọi phương pháp khác.

## 5. Khoảng cách, density và dimension

kNN dùng scale train rồi giữ reference. Lưu N×d reference và brute-force Q queries tốn khoảng O(QNd) phép tính và có thể cần O(QN) distance buffer trong cách vector hóa lab. Với 300 reference/180 test thì nhỏ; với hàng triệu vectors cần chunking/index/ANN và đo recall/latency. Embedding đẹp không tự có metric-space phù hợp anomaly.

Score train có self-neighbor khác score query mới. Nếu dùng train-distance để đặt threshold novelty mà không loại self/định nghĩa leave-one-out, distribution calibration đã sai. Lab dùng 200 query calibration ngoài reference. Exact duplicate trong reference vẫn là hàng hợp lệ theo hợp đồng hiện tại; không có cơ chế deduplicate entity hoặc tách leakage nhóm.

LOF so density tương đối cục bộ, Isolation Forest dựa vào isolation paths, One-Class SVM học frontier. Không chọn bằng tên hoặc lời hứa “không cần assumption”. Theo [tài liệu scikit-learn](https://scikit-learn.org/stable/modules/outlier_detection.html), cần phân biệt API outlier và novelty, nhất là LOF. Chưa chạy chúng trong bài này nên không có thứ hạng thực nghiệm so với kNN/PCA của lab.

## 6. Ngưỡng empirical không giải quyết mọi multiple testing

Với 200 calibration scores và α=.05, rank 191. Strict `>` loại mọi tie đúng q; không buộc đúng 10 điểm bị đánh dấu. Calibration có tối đa 9 scores vượt q ở sample không tie; nó không chứng minh tương lai đúng 5% FPR. Không đọc tỷ lệ calibration rồi dùng nó như unbiased performance estimate của rule đã chọn.

Order statistic dạng `ceil((n+1)*(1−α))` có liên hệ conformal rank nếu score model fixed và các scores calibration/query có tính exchangeable phù hợp. Lab không chứng minh điều kiện này cho hệ streaming; CUSUM tạo score phụ thuộc state rõ ràng. Cũng chưa có multiple-testing/FDR/ARL control cho nhiều series/thời điểm. Một threshold danh nghĩa không phải xác suất có ít nhất một false alarm trong cả ngày.

Nếu marginal FPR là p trên M điểm normal, expected false hits=Mp bằng tuyến tính expectation, không cần độc lập. Muốn xác suất “ít nhất một” bằng `1−(1−p)^M` thì cần giả định độc lập tương ứng. Nếu nhiều series đều tăng tải trong cùng deployment, alert có thể thành burst đồng thời; tổ chức routing/dedup/incident grouping là bài toán hệ thống riêng.

Calibration cũng cần population rõ. Gom tất cả giờ/người dùng/thiết bị vào một q có thể tạo coverage/FPR rất khác theo nhóm. Contextual normalization, group thresholds hoặc model riêng có thể giúp, nhưng mỗi nhóm cần đủ reference/calibration và cơ chế fallback.

## 7. Forecast residual khác residual biết sau tương lai

Ở thời điểm t, detector được đọc y_t vừa đến. Calendar expectation phải được xây từ model/feature có trước khi y_t đến. Lag 24 dùng y_(t−24) đã quan sát, nhưng nếu điểm đó là anomaly thì một spike có thể tạo echo 24 giờ sau. Đây là failure mode có thật của quy tắc score, không phải future leakage.

Centered filter, interpolation dùng cả tương lai và full-series decomposition có thể hữu ích cho điều tra offline, nhưng không thể nhận cùng latency với causal detector. Nếu cho phép lookahead 5 bước, phải ghi detection delay tối thiểu 5 bước và đánh giá đúng schedule. Không dùng timestamp tại giữa window làm timestamp phát hiện nếu window chỉ đủ dữ liệu ở cuối.

Calendar model lab không có lag y, chỉ intercept/sin/cos/plan. Nó fit trên 168 giờvà giữ nguyên. Plan xuất hiện trong normal calibration ở các pha lịch khác nhau nên seasonal-naive residual khá lớn ngay cả khi không có sự cố; threshold≈9.978 robust units phản ánh reference đó. Calendar residual có thể giúp, nhưng lỗi expectation, plan sai hoặc regime mới vẫn có thể tạo false hits.

## 8. Streaming state là một phần của model behavior

CUSUM G⁺/G⁻ phụ thuộc cả prefix trước chunk. Reset ở mỗi batch làm score đổi theo batch size. Policy counters/cooldown cũng vậy. Lag detector cần history; nếu thay history thật bằng 0 ở chunk boundary, forecast residual thay đổi. Một system có state cần version cho cả state schema và score/policy configuration.

Lab replay hai chunk 336…429 và 430…503: CUSUM nhận state sau 429, policy nhận active/counters/cooldown/last_tick đã serialize JSON. Các method khác nhận actual history từ toàn fixture. Đây là checkpoint demonstration, **không chứng minh crash recovery end-to-end**: chưa có transaction giữa input offset, state update và notification receipt; chưa xử lý notification gửi thành công rồi crash trước lưu state.

Trong production, event-time order, watermark/late-data policy và idempotency key ảnh hưởng semantics. Ví dụ cùng observation bị apply hai lần làm CUSUM tăng hai lần. Core policy từ chối `start` không nối đúng `last_tick+1`, nhưng không có broker/partition ownership hoặc snapshot rollback. Xem lại bài 27–33 nếu xây hệ phát cảnh báo thật.

## 9. Đánh giá event cần nói rõ match và deadline

Bài dùng exact interval overlap của **timestamp firing**, không match alert active. Với event 354…354, CUSUM policy firing 355 bị tính outside, dù score còn cao do spike trước. Một ứng dụng cho grace window sau event có thể chấm khác; phải chốt trước, tránh mở tolerance đến khi model đẹp.

Policy calendar tìm 3/4 events, delay [1,1,1], không có false firing trong 168 giờ test. Nói “mean delay 1 giờ” mà bỏ 1 missed event là thiếu thông tin. Raw hits calendar tìm cả 4 events, nhưng 31 positive points chỉ có 28 TP; một event được tìm không có nghĩa mọi point bên trong đúng. Raw score và policy cũng có population alarm khác nhau, nên không trộn precision point với event recall rồi tính F1 tùy ý.

[Kim et al. abstract](https://arxiv.org/abs/2109.05257) nêu nguy cơ PA thổi phồng đánh giá. Bài chỉ dùng cảnh báo đó và ví dụ tự tính, không triển khai PA&K trong paper. Nếu chọn range-based metric khác, đọc định nghĩa, tolerance, weighting và penalty cho duplicate/late/long alerts. Không gọi một chỉ số là chuẩn thống nhất cho mọi incident.

## 10. Hướng deep/representation và thí nghiệm tiếp theo

Có thể dùng reconstruction/forecasting, one-class objective, self-supervised representation hoặc embeddings pretrained rồi detector đơn giản. [Deep One-Class Classification](https://proceedings.mlr.press/v80/ruff18a.html) giới thiệu Deep SVDD; bài chỉ đọc abstract, chưa kiểm toàn bộ điều kiện kiến trúc/objective hay chạy benchmark của paper. Với ảnh/âm thanh, nối bài 41: feature encoder cần phù hợp domain, preprocessing và feature version phải khớp calibration.

Một nghiên cứu tiếp theo hợp lệ bắt đầu từ dataset/protocol và failure mode. Ví dụ tạo thêm một development set để chọn k/regularization/threshold level, giữ test hiện tại làm báo cáo cũ; đánh giá future holdout mới hoặc cross-entity cohort chưa dùng. Với detector online có adaptation, backtest **policy update thực tế** bằng dữ liệu available tại từng mốc, không đưa label tương lai vào quyết định reset/refit.

Phân tích effort cũng quan trọng: kNN lưu reference, covariance cần estimate ổn định, AE cần train/compute, còn rule đơn giản dễ giải thích nhưng có thể bỏ pattern. Đo cùng event population, point metric/AP, event/delay/false firing, memory/latency và công sức review. Đừng mặc định phương pháp phức tạp phải thắng hoặc dùng test để chọn lại cấu hình.
