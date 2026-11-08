# Lời giải bài 42 — Hệ gợi ý

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

## 1. Chọn nhiệm vụ và mục tiêu

Search có truy vấn keyword; related-items lấy một item làm seed; personalized feed có user/context/history. Một trang chủ bài học nên hỗ trợ nội dung phù hợp mục tiêu, không chỉ tiêu đề dễ click. Có thể đo read completion/helpful feedback và retention theo mục tiêu; guardrails như complaint/irrelevant rate và latency/empty-list rate. Phải định nghĩa logging/window và xác nhận metric phản ánh việc học; không gọi những tên metric này là bằng chứng đã tăng outcome.

## 2. Rating, click và exposure

Ô trống có thể là chưa hiển thị, chưa nhìn thấy, chưa quan tâm lúc đó hoặc outcome chưa đến. Click cũng chưa là satisfaction. Dataset đầy đủ cho một bài toán cần request/candidate/item/model IDs, event/processing time, impression definition/position, action/reward, observation window và join key.

Lab có clicks tổng hợp và catalog/history time, không impressions/position/propensity. Không thể lấy số clicks chia catalog size làm CTR hoặc gán user-item chưa click là0với nhãn dislike chắc chắn. Negative sampling là giả định học, khác ghi nhận preference âm thực.

## 3. Popularity và eligibility

Tại train cutoff A10>B4. Toàn log cho A10<B24 đã dùng futuretest để thay ranking. Việc này có thể làm offline metric đẹp hơn nhưng không mô phỏng thông tin lúc quyết định.

Chỉ cho item đã released, chưa retired, khôngblocked/seen theo policy. Item seen là đúng để loại trong nhiệm vụ khám phá bài chưa đọc của lab; nghe nhạc, refill sản phẩm hoặc ôn bài có thể cần repeat. Trả rỗng/ít hơnK khi không đủ hợp lệ; không phá điều kiện để đủ số lượng.

## 4. Content và item cosine tính tay

Profile=`[2/3,1/3]`, dot với item topic0/1 tương ứng2/3,1/3. Matrix ví dụ có mỗi item2user, item0/1giao1user nên cosine1/2. Content dựa metadata/representation, collaborative dựa pattern tương tác; co-click có thể do exposure chứ không là causal preference.

```python
import numpy as np
features=np.eye(2);profile=features[[0,0,1]].mean(axis=0)
assert np.allclose(profile,[2/3,1/3])
x=np.array([[1,0,1],[1,1,0],[0,1,1]],dtype=float)
cosine=x[:,0]@x[:,1]/(np.linalg.norm(x[:,0])*np.linalg.norm(x[:,1]))
assert abs(cosine-.5)<1e-12
```

## 5. MF, rating và số tham số

ScoresA.8,B.2, xếp A trước. Ratingillustration3.5+.2−.1+.8=4.4. Parameter count48×12+72×12+72=1512. Lab không có explicitrating objective, global/user bias hoặc sigmoid probability output.

Nếu score có b_u giống nhau cho tất cả item, `s_ui−s_uj` triệt tiêu b_u nên pairwise data loss không học bias đó. Item bias khác nhau nên còn tác dụng. Dot score có thể âm hoặc lớn hơn1; ngay cả sigmoid score sau negative sampling cũng cần xem sampling/calibration trước diễn giải xác suất click.

## 6. Một bước BPR

Δ1, losslog(1+e⁻¹)≈.313261688, g=.268941421. Gradp=`[-g,g]`, gradqi=`[-g,0]`, gradqj=`[g,0]`. Với lr.1 có p'=`[1.026894142,−.026894142]`, qi'=`[1.026894142,0]`, qj'=`[−.026894142,1]`.

```python
import numpy as np
import torch
from model import MF,bpr_loss
from arithmetic import hand
m=MF(users=1,items=2,dim=2).double()
with torch.no_grad():
    m.user.weight.copy_(torch.tensor([[1.,0.]]))
    m.item.weight.copy_(torch.eye(2));m.bias.weight.zero_()
loss,_=bpr_loss(m,torch.tensor([0]),torch.tensor([0]),torch.tensor([1]),reg=0)
loss.backward();h=hand()['bpr']
assert abs(float(loss.detach())-h['loss'])<1e-12
assert np.allclose(m.user.weight.grad.numpy()[0],h['gradient_p'])
# This example is bias-free: update only p and q, all gradients already computed.
with torch.no_grad():
    m.user.weight-=.1*m.user.weight.grad
    m.item.weight-=.1*m.item.weight.grad
assert np.allclose(m.user.weight.detach().numpy()[0],h['after_p'])
assert np.allclose(m.item.weight.detach().numpy(),[h['after_i'],h['after_j']])
```

Lấy p mới để tính gradient qi sẽ đổi trạng thái dùng cho gradient, không còn cùng simultaneous gradient step. Trong training thật, bias và regularization được cập nhật theo objective đầy đủ; ví dụ hand cố ý bỏ chúng để kiểm cơ chế.

## 7. Negative sampling theo snapshot

Pool là item availablet9 trừ trainpositives của user; một negative/positive/epoch, uniform với replacement giữa các draw. Validation target warm có thể trong pool, vì nó chưa được observed ở train snapshot. Không dùng test/validation future labels để loại nó.

```python
import copy
import numpy as np
from data import dataset,negative_pools
s=dataset();p=negative_pools(s)
assert any(r['target'] in p[r['user']] for r in s['validation'])
changed=copy.deepcopy(s)
for e in changed['events']:
    if e['time']>=10:e['item']=0
assert all(np.array_equal(p[u],negative_pools(changed)[u]) for u in p)
```

Uniform dễ giải thích nhưng có thể nhiều negative quá dễ. Popularity tập trung vào item hay xuất hiện và thay objective weighting. Hard negatives giúp phân biệt candidate gần nhưng dễ có false negative. Cần log phân phối, seed và cách tránh positive đã biết; future audit không được feedback vào gradient/sampling. Không so trainloss khác sampler như cùng population.

## 8. Timeline và leakage

```python
from data import dataset,seen_at,train_events,eligible
s=dataset();r=s['validation'][0]
assert r['target'] not in seen_at(s,r['user'],10)
assert r['target'] in seen_at(s,r['user'],11)
assert all(e['time']<10 for e in train_events(s))
assert all(78 not in eligible(s,q) and 71 not in eligible(s,q) for q in s['test'])
```

Weights/global popularity/S frozentrain, nhưng requestcontext dùng events trước t, kể cả validationclick t10 tại testt11. Không dùng eventtest chính thời điểm11làmhistory. Randomsplit có thể tạo lịch sử nhìn tương lai; per-userleave-last-out có thể dùng training của user khác xảy ra sau thời điểm query nếu không kiểm globaltime. Chọn protocol đúng mục tiêu refresh/serve, không chỉ split theo tỷ lệ.

## 9. Cold-start và inference bundle

New user không có ID embedding đã học; lab fallbackpopularity trênwarmmappeditems choMF. New item không có hàng embedding; MF không nominate nó, trong khi content có thể chấm metadata nếu profile user có. Evaluation vẫn giữ targetcoldtrongpopulation, nên miss không bị bỏ.

```python
import tempfile
from pathlib import Path
from data import dataset,fingerprint
from model import MF
from training import save_bundle,load_bundle
s=dataset()
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'mf.pt';save_bundle(path,MF(),s)
    model,meta=load_bundle(path,fingerprint(s))
    assert len(meta['item_ids'])==72
    try:load_bundle(path,'other snapshot')
    except ValueError:pass
    else:raise AssertionError('mismatch accepted')
```

Bundle có weights/spec/ID mapping/hash, chưa có catalog/history store và global stats để tự phục vụ mọi request, chưa có optimizer/RNG/sampler cho exact resume. Digest không xác thực người gửi. Tạo embedding random cho ID mới chỉ làm tensor lookup chạy được, chưa có bằng chứng model hiểu item đó.

## 10. Top-K metrics

Precision2/5,Recall2/3,MRR1/2. DCG=`1/log2(3)+1/log2(6)`≈1.01778256,IDCG=`1+1/log2(3)+1/log2(4)`≈2.13092975,NDCG≈.477623704. IDCG dựa sốrelevant khảdụngtheoannotation/K, không dựa sốhitmodeltrả.

```python
import math
from metrics import ranking_metrics
m=ranking_metrics(['A','B','C','D','E'],['B','E','F'],5)
expected=(1/math.log2(3)+1/math.log2(6))/(1+1/math.log2(3)+1/math.log2(4))
assert m['precision']==.4 and m['recall']==2/3 and m['mrr']==.5
assert abs(m['ndcg']-expected)<1e-12
assert ranking_metrics(['A'],['A'],5)['precision']==.2
```

Một target/request làm Recall bằng HitRate. Danh sách ngắn vẫn chia precision cho K=5; ranking rỗng cho0, relevant set rỗng bị từ chối trong helper. Nếu dataset thật có query không positive, cần định nghĩa population/metric khác hoặc phân nhóm riêng, không âm thầm bỏ query để làm đẹp điểm.

## 11. Candidate ceiling và sampled metrics

Nếu L⊆C thì L∩R⊆C∩R, nên Recall cuối ≤ Recall candidate. Với 5 candidate hit trên 54 query, trần là 5/54≈.092593. Re-ranker chỉ sắp lại 10 item không thể khôi phục target không có trong đó. Nếu thêm source ở bước sau thì C phải là union ứng viên thực sự đã dùng.

Rank 20/100 và rank 2/6 thuộc hai tập đối thủ khác nhau. Một subset có thể bỏ nhiều item đứng trên target; Hit@5 đổi từ 0 sang 1 không là cùng phép đo. Candidate demo không ép target vào set, nên giữ được các trường hợp retrieval miss. Không so trực tiếp sampled metric với full-catalog metric như cùng đơn vị.

## 12. RRF và diversity cap

RRF item1=1/61; item2=1/62+1/61; item3=1/62. Thứ tự 2,1,3. Greedy cap2/topic trên0…8 giữ 0,1,3,4,6. Item mới không được một nguồn nominate chỉ có một contribution; thêm content source chưa bảo đảm cold item vào top-K.

```python
from metrics import reciprocal_fusion,diversify
rank,s=reciprocal_fusion([[1,2],[2,3]],60)
assert rank==[2,1,3] and abs(s[2]-(1/61+1/62))<1e-12
catalog=[{'id':i,'topic':i//3} for i in range(9)]
assert diversify(list(range(9)),catalog,5,2)==[0,1,3,4,6]
assert diversify([0,1,2],catalog,5,2)==[0,1]
```

Cap có thể trả danh sách ngắn khi không đủ topic. Nó kiểm soát đa dạng chủ đề trong list, khác catalog coverage qua nhiều user và khác fairness. Fairness cần định nghĩa nhóm, outcome, cơ hội và metric phù hợp; lab không kiểm những điều đó.

## 13. Đọc thí nghiệm

MF NDCG test từ≈.03818 random lên .26044 sau train; Recall từ3/54 lên20/54. Nó chưa vượt item cosine với NDCG .52341/Recall34/54 hoặc content với Recall38/54. Content+fresh có Recall43/54 cao hơn item cosine, nhưng NDCG .51231 thấp hơn. Không có một cấu hình thắng mọi chỉ số.

RRF→diverse: mean topics1.907→3.111, unique items68→64, Recall27/54→20/54. Số topic/list tăng không buộc mọi item coverage hoặc relevance đều tăng. Best MF epoch35 theo validation NDCG .318610; hyperparameter/protocol giữ nguyên, không chọn trên test.

Cohort new-item có12query cần tách riêng: MF/RRF0hit nhưng content+fresh12hit. Các số này mô tả generator/split hiện tại, không quyết định model tốt nhất cho mọi recommender.

## 14. Exposure bias và IPS

Mean reward=.5; contribution `r/p=[2,0]`, mean=1. Đây là số học trên hai dòng giả định, chưa là expected CTR của population, causal uplift hoặc full policy value. Denominator và target population quan trọng; không chia cho observed count bất kỳ rồi nhận là unbiased.

Cần biết probability một entry được quan sát hoặc logging policy đã chọn action, cùng estimand, support/overlap, reward window và context join đúng. Full policy value còn cần target policy; không tự thay item popularity cho logging propensity. Propensity rất nhỏ có thể tạo weight lớn và variance cao; clipping/self-normalization đổi đánh đổi bias–variance.

Feedback loop có thể xuất hiện khi item được hiển thị nhiều nhận thêm click, rồi model học nó là phổ biến và tiếp tục hiển thị nhiều. Chưa đủ bằng chứng để coi pattern đó là sở thích user thuần túy.

## 15. Thiết kế nhánh hiện đại

Two-tower nhận user/context và item features, học representation tương thích; ANN lấy candidate gần đúng; ranker chấm features/context phức tạp hơn; re-ranker áp điều kiện danh sách. Sequence encoder thêm thứ tự/position/time của history để dự đoán next action, với mask và temporal protocol tránh tương lai.

ANN recall so exact embedding retrieval khác semantic recall của relevant items và khác final ranking metric. Content embedding text/image/audio có thể giúp item cold-start, nhưng cần processor/version và training phù hợp; nó chưa tự hiểu sở thích user hoặc giải quyết exposure bias.

LLM ranker cần baseline rẻ, retrieval tốt, grounding, budget/latency và evaluation tái lập. Câu giải thích thuyết phục không thay cho bằng chứng ranking hoặc tác động sản phẩm.

## 16. Kế hoạch sản phẩm

Định nghĩa trước “gợi ý bài chưa đọc có ích”, target window và positive/outcome hợp lý. Chia snapshot toàn hệ thống theo thời gian, giữ catalog release/retire, history as-of và model/index version. So popularity, content, item cosine, MF/hybrid trên cùng population/K; tách cold user/item và thiếu history, đồng thời kiểm coverage/diversity/latency.

Online cần experiment với assignment phù hợp, outcome về học tập và guardrails feedback/empty-list/latency. Log candidate, display position, impression, request và model version để audit. Cache phải phản ánh eligibility/history đủ mới; delayed outcome join đúng window.

Offline có thể hỗ trợ claim “ranking observed target trên protocol này tốt hơn”. Nó chưa chứng minh “tăng CTR/learning retention do model mới” hoặc “công bằng hơn” nếu chưa có định nghĩa và evidence phù hợp.
