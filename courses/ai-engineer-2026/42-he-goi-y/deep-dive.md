# Phụ lục bài 42 — Dữ liệu, objective và evaluation của hệ gợi ý

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Preference, exposure và action là ba biến khác nhau

User có thể thích item nhưng chưa được thấy; có thể thấy nhưng không click vì đang bận; có thể click rồi thất vọng. Dataset click trộn nhu cầu, policy hiển thị, position và nhiều yếu tố khác. Đặt nhãn “observed positive” giúp nhớ đó là kết quả logging, không phải biết đầy đủ sở thích.

Impression cũng phải có định nghĩa: gửi response, render lên màn hình hay thật sự vào viewport? Delayed outcome có thể đến sau cửa sổ gán nhãn. Missing outcome do chưa đợi đủ khác non-conversion sau đủ thời gian. Schema và thời gian join là một phần ML problem, không chỉ plumbing backend.

Lab không tạo exposure process hay đo counterfactual reward. Dữ liệu generator có primary/secondary topic, nhưng policy chỉ nhận history và catalog metadata. Không lấy preference ẩn đó làm feature để báo personalized inference.

## 2. Unobserved entry và weighted objectives

MSE trên rating observed, weighted implicit MF và pairwise ranking là các objective khác nhau. MF chỉ định cách tạo score; không quy định phải dùng MSE, BPR hay một loss duy nhất. Weighted implicit MF có thể gán confidence khác nhau cho observed/unobserved, còn BPR so một observed i với sampled unobserved j. Cả hai vẫn là mô hình hóa với giả định.

Không nên diễn giải mọi unobserved là true negative. Nhưng cũng không thể chỉ train mô hình trả1trên observed rồi không có cơ chế phân biệt các candidate. Negative sampling/weighting tạo một learning problem khả thi; kết luận phải khớp phân phối đã dùng và label noise có thể có.

## 3. ID embedding và feature encoder

ID-only MF lưu một hàng embedding cho user/item đã học. Nó có thể khai thác latent co-interaction rất tốt mà không đọc nội dung; đổi lại cold-start phải có chiến lược. Feature encoder có thể chấm item mới từ text/image/metadata, miễn feature và preprocessing có sẵn đúng lúc và ở trong miền model.

Hai vector cùng d không tự tương thích — bài41. Một hai-tower huấn luyện joint user/context–item có thể dùng nội dung để retrieval, nhưng cần kiểm objective/sampling/index và feature freshness. Không chỉ lấy bất kỳ text embedding từ model khác cắm vào bảng item MF rồi giữ nguyên dot product.

MF score của lab không normalize L2. Tự chuyển sang cosine lúc serving làm đổi hàm chấm đã train và có thể đổi ranking. Item bias cũng không được bỏ khi muốn tái tạo score exact. Nếu triển khai dot-based ANN, có thể cần thiết kế cách biểu diễn bias hoặc scoring stage; lab chỉ exact scan nên chưa giải quyết vấn đề đó.

## 4. BPR objective không đồng nghĩa optimize trực tiếp NDCG

BPR làm tăng score difference của sampled pairs. Không trực tiếp dùng vị trí top5 trong loss; một cải thiện pairwise có thể không cải thiện top-K, đặc biệt khi negative sampling khác candidate distribution. Đây là lý do chọn checkpoint bằng validationNDCG và vẫn báo train pair loss riêng.

Với `L=softplus(-Δ)`, đạo hàm theoΔ là `-sigmoid(-Δ)`. Δ âm lớn cho loss lớn và gradient gần−1; Δ dương lớn cho gradient nhỏ. `softplus` ổn định hơn tính `-log(sigmoid(Δ))` ngây thơ khi giá trị cực đoan. Regularization giữ score/embedding tránh tăng vô hạn chỉ để tách observed/unobserved.

Lab regularize mean tổng norm các embedding xuất hiện trong sampled triples, bias cũng có penalty. User/item phổ biến bị tính nhiều lần. Điều này không bằng lấy norm của toàn embedding table một lần mỗi update, và λ không so trực tiếp với λ của implementation khác có reduction khác. Adamfullbatch384triple/epoch cũng khác LearnBPR bootstrap-SGD gốc; không gọi đó là reproduction benchmark paper.

## 5. Snapshot negative sampling và time leakage

Training được fit tại một cutoff, nên được biết toàn bộ observed positives trong train window. Pool ởt9 loại tất cả positives t<10 và chỉ có item available lúc đó. Đây là batch snapshot training, không phải online sampler tại từng eventt0…7. Nếu muốn mô phỏng online updates thì cần pool/history theo event time và protocol khác.

Đếm future positives bị sample chỉ là audit sau khi đã lấy mẫu; không remove hoặc đổi gradient. Nếu dùng toàn log để exclude future positive, sampler đã nhận thông tin không tồn tại ở train cutoff. Ngược lại, biết một j sau này thành positive không chứng minh nó đã là preference đúng ở mọi thời điểm trước đó.

In-batch negatives có thể lặp positive hợp lệ; hard negatives có thể là true relevant chưa được quan sát. Nếu dùng propensity/popularity distribution, lưu rõ phân phối/weights. Đổi sampler có thể cần retune và evaluation mới; loss giữa hai sampler không cùng một population đơn giản để so.

## 6. Temporal backtest cần nhất quán toàn hệ thống

Requestt11 được phép dùng eventt10 như history vì đã xảy ra. Nhưng không có nghĩa tự động được refit tất cả model/stats bằngt10: điều đó tùy lịch refresh được mô phỏng. Lab giữ global statistics/weights từ train, chỉ request context/history cập nhật. Sự khác nhau này phải được ghi để người khác chạy lại.

Per-user leave-last-out giữ tương đối theo mỗi user, chưa chắc giữ timeline toàn cục. Nếu model dùng item popularity từ sự kiện sau thời điểm một query trong test, kết quả có thể nhìn trước tương lai. Một backtest có nhiều thời điểm cần snapshot/index/features version tương ứng mỗi thời điểm, không một global snapshot sau cùng cho tất cả.

Catalog availability cũng là feature. Không để item có release t12 được gợi ý tại t11, và không lấy metadata chưa tồn tại từ tương lai làm content feature. Lab lưu cả teaching catalog, nhưng eligibility bảo đảm item tương lai không được chấm/nominate trước release.

## 7. Candidate generation đặt trần cho re-ranking

Gọi R_u là relevant set, C_u là candidates và L_u là danh sách sau rerank, với `L_u⊆C_u`. Khi đó:

\[
|L_u\cap R_u|\leq|C_u\cap R_u|,
\quad Recall(L_u)\leq Recall(C_u).
\]

Đây là quan hệ tập hợp, không cần model phải tốt hoặc calibration. Với một target, nếu candidate miss thì downstream hit0. Nếu có cơ chế bổ sung candidate sau đó, cần tính lại C thực sự dùng; không nhận trần của một stage rồi bỏ qua source mới.

Lab popularity top10→content_fresh chỉ sắp lại10item nên trần5/54. Đây khác ranking trên target được chèn sẵn cộng random negatives: protocol sampled luôn làm target có mặt, nên không đo lỗi retrieval miss. ANN cũng có approximation error riêng so exact retrieval; giữ riêng ANN recall, semantic candidate recall và final ranker metric.

## 8. Ties, denominator và aggregation thay ý nghĩa metric

Lab sort score giảm, ID tăng để tie reproducible. Content one-hot tạo nhiều tie, nên ordering policy ảnh hưởng NDCG và cold-item placement. Không coi đó là chi tiết triển khai vô hại. ID tăng không được nhận là “tự nhiên tốt nhất”; đây là convention để tái lập.

Binary NDCG tính IDCG với `min(K,|R|)`, không với số hit đã thấy trong output. Nếu dùng số hit làm ideal denominator thì model bỏ relevant có thể được điểm lạc quan. MRR@K chỉ lấy relevant đầu trong topK; Recall đếm tất cả target positives. Với graded gain phải nêu formula riêng.

Macro theo query khác average theo impression/event volume; user nhiều session có thể nặng hơn nếu mỗi query là một dòng. Lab mỗi test user một request, cohorts36/12/6 và headline làmacro54query. Không bỏ cold miss hoặc query khó để tăng điểm tổng; relevance-empty cần protocol được công bố, helper hiện tại từ chối.

## 9. Sampled metrics và nhãn thiếu là hai vấn đề riêng

Full-catalog evaluation bỏ một nguồn nhiễu do sampling candidates, nhưng chưa làm relevance annotation đầy đủ. Unknown items có thể hữu ích dù không là observed target. Sampled metric thay thêm tập các item cạnh tranh; việc A tốt hơn B trên một sample chưa bảo đảm thứ tự giữ trên toàn eligible universe.

Ví dụ rank20/100→rank2/6 là một subset có thể chọn được, không ước lượng kỳ vọng hoặc chứng minh bias bằng mọi hướng. Tài liệu On Sampled Metrics được dẫn ở mức abstract, không triển khai correction estimator. Nếu cần sampled protocol do catalog quá lớn, giữ seed/distribution/size và so cách đó trên cùng protocol; đừng so số sampled với full như cùng đơn vị.

## 10. Hybrid source coverage và diversity khác fairness

RRF cộng reciprocal rank từ mỗi source. Item được hai source nominate có hai contribution; cold item chỉ ở content có một. Cơ chế này giải thích vì sao hybrid có source cold-start vẫn có thể không hiển thị item mới ởtop5. Không có một guarantee fairness hoặc optimal mixture từ việc dùng c60.

Greedy cap topic tối đa2 giúp số topic trong mỗi list tăng. Nhưng union item coverage qua nhiều user có thể giảm, và relevance target có thể bị đẩy khỏi list. Trong lượt đã chạy: RRF→cap nâng mean topic1.907→3.111 nhưng unique items68→64 và Recall27/54→20/54. Đây là ba metric khác, không mâu thuẫn.

Fairness cần định nghĩa nhóm/cơ hội/outcome và đánh giá phù hợp. “Mỗi list có3topic” chưa chứng minh công bằng giữa provider, user population hoặc các nhóm xã hội. Lab không có group fairness objective hoặc sensitive attributes; không suy từ diversity heuristic thành fairness guarantee.

## 11. Propensity weighting cần estimand và logging phù hợp

Trong observation weighting, p có thể là xác suất một entry được quan sát trong population quan tâm. Tổng contribution `r/p` cần denominator/target population đúng; chia cho số observed rows bất kỳ không tự cho CTR population. Ví dụ hai contribution trong bài chỉ minh họa arithmetic:1/.5=2,0/.25=0, mean1; nó không là xác suất đã calibration và có thể vượt1 ở sample khác.

Policy-value evaluation là bài toán khác: thường còn cần target policyπ và logging policyμ, action/context/reward join đúng, overlap/support và assumptions để diễn giải. Trọng số importance liên quanπ/μ cho action đã log, không tự dùng item popularity thayμ. Nhiều trường hợp có weight nhỏ/propensity gần0 gây variance lớn; clipping/self-normalization đổi bias-variance trade-off.

Bài không triển khai IPS/SNIPS/OPE hoặc ước lượng propensity từ clicks. Schnabel etal. được dẫn cho selection bias và propensity-scored evaluation, không dùng kết quả lý thuyết để chứng nhận metric lab unbiased. Để nghiên cứu tiếp cần đọc đúng estimator/assumptions và thiết kế data logging trước.

## 12. Serving và monitoring cần cùng định nghĩa dữ liệu

Artifact model phải đi với ID mapping, feature/preprocess và index phù hợp. Cache user-history cần timestamp/watermark; stale eligibility có thể trả item đã ngừng hoạt động. Theo dõi version mismatch, fallback/cold rate, empty-list rate, candidate source coverage và latency từngstage giúp tìm lỗi mà NDCG tổng không nói hết.

Inference bundle lab chỉ giữ MF/config/ID/data hash và weights digest. Policies còn cần catalog/history/statistics từ teaching dataset để phục vụ; không phải package production tự chứa mọi store. Không optimizer/RNG/sampler checkpoint nên chưa exact resume; digest không chứng minh artifact từ nguồn đáng tin.

Muốn thêm model hiện đại, làm từng giả thuyết: feature encoder cho cold item; sequence encoder cho recency/order; learned ranker cho context; exploration cho evidence mới. Mỗi lần cần baseline, split, metric và chi phí phục vụ tương ứng. [Nguồn và phạm vi đã đọc](../SOURCES.md).
