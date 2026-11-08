# Bài tập bài 42 — Hệ gợi ý

[← Bài chính](README.md) · [Lời giải](solutions.md) · [Phụ lục](deep-dive.md)

Dùng thư mục bài42 làm working directory cho Python. Phân biệt số tính tay, observed target và outcome của người dùng thật.

## 1. Chọn nhiệm vụ và mục tiêu

Với thư viện bài học, phân biệt search theo keyword, related-items và personalized feed. Nếu chỉ tối đa click thì có thể khác mục tiêu học như thế nào? Đề xuất hai metric gần và hai guardrail, không giả định đã đo uplift.

## 2. Rating, click và exposure

Một ô user–item trống có thể mang những nghĩa nào? Vì sao click=1/nonclick=0 không tự tạo dataset preference đầy đủ? Viết schema tối thiểu để join request, impression, position và delayed outcome. Lab có thông tin nào, thiếu thông tin nào để tính CTR/propensity?

## 3. Popularity và eligibility

ItemA có10click train/0test, B có4train/20test. Popularity tại train cutoff được xếp thế nào? Nếu dùng counts toàn log thì sai ở đâu? Nêu cách xử lý item chưa release, đã retire, blocked hoặc seen; khi nào loại seen không phù hợp sản phẩm?

## 4. Content và item cosine tính tay

History topic`[0,0,1]` cho profile nào? Score topic0/1 theo dot là bao nhiêu? Với binary matrix`[[1,0,1],[1,1,0],[0,1,1]]`, tính cosine giữa item0 và1. Phân biệt content similarity với co-click và causal preference.

## 5. MF, rating và số tham số

p=`[.8,.2]`, qA=`[1,0]`, qB=`[0,1]`, bias0: score/ranking? Rating với μ3.5,b_u.2,b_i−.1,dot.8 là bao nhiêu? Với48user,72item,d12 và item bias, tính số parameter. Vì sao dùng BPR thì user bias hằng theo item không cần thiết và dot score không là probability?

## 6. Một bước BPR

p=`[1,0]`, qi=`[1,0]`, qj=`[0,1]`, không bias/regularization. Tính Δ, loss, gradient p/qi/qj và SGD lr.1 cập nhật đồng thời. Dùng autograd đối chiếu. Nếu cập nhật p rồi dùng p mới cho gradient qi thì có còn cùng một bước không?

## 7. Negative sampling theo snapshot

Lab lấy j từ tập nào? Một item user click ởvalidation có được xuất hiện làm negative train không? Tại sao không lọc mọi future positive bằng knowledge test? Kiểm pool không đổi khi sửa events t≥10. Nêu đánh đổi uniform/popularity/hard-negative.

## 8. Timeline và leakage

Kiểm một validation click tại t10 chưa thuộc history requestt10 nhưng thuộc history testt11. Weights và global stats có được fit lại với test không? Nêu một rủi ro của random event split và một rủi ro global timeline khi per-user leave-last-out.

## 9. Cold-start và inference bundle

Phân biệt new user/new item. MF ID-only xử lý thế nào trong lab? Lưu/load random MF với snapshot hash và kiểm mismatch bị từ chối. Bundle đã chứa mọi thành phần serving/exact resume chưa? Có thể tạo ID mới với embedding random rồi gọi đã giải quyết cold-start không?

## 10. Top-K metrics

Ranking`[A,B,C,D,E]`, relevant`{B,E,F}`, K5: tính Precision/Recall/MRR/DCG/IDCG/NDCG. Một relevant duy nhất thì Recall có quan hệ gì với HitRate? Nếu chỉ trả2item thì precision chia2hay5 trong lab? Relevance rỗng được xử lý thế nào?

## 11. Candidate ceiling và sampled metrics

Target không vào top10candidate thì re-ranker có thể tạo hit ởtop5 bằng cách chỉ sắp lại10item không? Với54query có5candidate hit, trần Recall@5 là bao nhiêu? Giải thích vì sao target rank20/100 nhưng rank2trên một subset6item không là cùng metric evaluation.

## 12. RRF và diversity cap

Nguồn1`[1,2]`, nguồn2`[2,3]`, c60: tính RRF của từng item và thứ tự. Với item0…8, topic=`id//3`, ranking0…8,K5,cap2/topic, danh sách nào được chọn? Vì sao một item mới chỉ có một nguồn có thể bị thiệt trong RRF và cap topic chưa chứng minh fairness?

## 13. Đọc thí nghiệm

Chạyall vào output mới; đối chiếu baseline, best checkpoint, cold cohorts, coverage và candidate demo. BPR có hơn random không, có hơn content/itemcosine không? Content+fresh có thắng mọi metric không? Diversity có tăng mọi loại coverage không? Không đổi test để ép kết quả đẹp.

## 14. Exposure bias và IPS

Ví dụ hai reward`[1,0]`, propensity`[.5,.25]`: mean reward và mean`r/p` là bao nhiêu? Vì sao ví dụ này chưa đủ đánh giá uplift hoặc full policy value? Cần những giả định/logging nào trước khi dùng propensity weighting? Tại sao closed feedback loop đáng chú ý?

## 15. Thiết kế nhánh hiện đại

Mô tả two-tower+ANN+ranker+re-ranker và một sequential model. Phân biệt retrieval recall/approximation với ranker quality và context time. Item text/image/audio embedding bài41 có thể giúp gì, và chưa giải quyết điều gì? Nêu baseline/evaluation cần thêm trước khi dùng LLM làm ranker.

## 16. Kế hoạch sản phẩm

Thiết kế protocol offline và online cho trang gợi ý bài học: target/window, snapshot, item eligibility, cold cohort, baseline/metric, đa dạng và chất lượng dài hạn, logging/version/cache/latency. Chỉ rõ claim nào offline có thể hỗ trợ và claim nào cần người dùng/experiment thật.
