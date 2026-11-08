# Lời giải bài 44 — Phát hiện bất thường

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Phụ lục](deep-dive.md)

Các đoạn Python độc lập, chạy từ thư mục bài 44 sau khi cài requirements. Đáp án thiết kế khác vẫn hợp lệ nếu giữ đúng data/decision timing và nêu rõ giả định.

## 1. Định nghĩa anomaly trong hệ backend

Ví dụ normal theo context: chiến dịch đã lên lịch, reference cùng mức tải từng có latency quanh 180 ms, SLO vẫn đạt. Ví dụ cần điều tra: tải không tăng nhưng latency nhảy 180 ms ở một nhóm dependency. Không đủ thông tin để khẳng định root cause. Chọn entity=`service+region`, frequency 5 phút, target p95 latency/bucket, reference từ vận hành đã review và cùng feature version. Score cao chưa đủ tự rollback hoặc khóa tài khoản; cần bằng chứng và quy trình hành động tương ứng.

## 2. Z-score, IQR và MAD tính tay

z=(140−100)/10=4. IQR 4, fence [4,20]. Median 12/MAD 1, modified absolute z(20)=5.396. MAD 0 cần fallback theo domain hoặc từ chối; helper chọn từ chối. Fence không là probability anomaly 95%, vì thiếu model phân phối/prior/nhãn cùng các điều kiện kiểm định.

```python
import numpy as np
from detectors import robust
assert (140-100)/10==4
q1,q3=10,14;iqr=q3-q1
assert (q1-1.5*iqr,q3+1.5*iqr)==(4.,20.)
m,s=robust([10,11,12,13,14])
assert m==12 and abs((20-m)/s-5.396)<1e-12
try:
    robust([1,1,1])
except ValueError:
    pass
else:
    raise AssertionError('MAD zero cần policy riêng')
```

## 3. Novelty, outlier và supervision

Lần lượt là novelty normal-only, outlier detection với contamination, supervised classification. Lab fit score bằng normal reference, nhưng chọn family bảng bằng labeled validation F1. Vì vậy toàn pipeline có dùng nhãn ở bước selection. Train log chứa một incident lặp dài có thể làm mean/covariance hoặc neighborhood dịch sang vùng lỗi; lúc đó mẫu lỗi tương tự được score thấp. Median/MAD không sửa được mọi trường hợp hay xác định nhãn nghiệp vụ.

## 4. Mahalanobis tính tay

Σ⁻¹=`[[1,−.9],[−.9,1]]/.19`; `[1,−1]` cho 20, `[1,1]` cho 20/19≈1.05263. Hướng trái dấu có variance nhỏ khi correlation dương. Regularization giúp conditioning và giảm khuếch đại noise ở hướng variance gần 0; không biến covariance thành robust hoặc xác suất chính xác.

```python
import numpy as np
cov=np.array([[1.,.9],[.9,1.]])
a=np.array([1.,-1.]);b=np.array([1.,1.])
assert abs(a@np.linalg.solve(cov,a)-20)<1e-12
assert abs(b@np.linalg.solve(cov,b)-20/19)<1e-12
```

## 5. kNN novelty score

Distances 3/2/1/7, lấy 1 và 2⇒1.5. Một training point có distance 0 với chính nó; đưa nó vào distribution calibration novelty mà không xử lý self đổi bài toán. Raw units lệch nhau có thể làm một trục áp đảo; fit scaler chỉ trên reference. LOF dùng relative local density, không chỉ mean distance.

```python
import numpy as np
reference=np.array([0.,1.,2.,10.])
distance=np.sort(abs(reference-3.))
assert distance[:2].mean()==1.5
assert np.sort(abs(reference-2.))[0]==0
```

## 6. Bắt lỗi leakage

Scaler toàn bộ đọc distribution test; q tối ưu test F1 dùng nhãn test để chọn; thêm query vào reference trước khi score tạo self-neighbor và thay policy thành adaptation chưa đánh giá. Centered smoothing có lookahead 2 giờ; nếu dùng nó phải ghi timestamp phát hiện lúc đủ window, hoặc chuyển causal filter.

Mutation check fit: thay mọi x từ index 300 trở đi và mọi label, fit lại chỉ x[:300], rồi chấm cùng một fixed query; coefficients/preprocess/score phải như trước. Mutation check streaming: thay y từ 400 trở đi, giữ score/state 168…399 không đổi. Core checks đã chạy các tính chất này; chúng chưa chứng minh timestamp metadata đúng trong production.

## 7. PCA và reconstruction

Projection [1,1], residual [1,−1], sum squared 2. `[10,10]` projection bằng chính nó, score≈0. AE có capacity lớn hoặc train contamination cũng có thể tái tạo anomaly; error thấp chưa đủ. Có thể thêm latent distance hoặc kNN; chốt rule/calibration/development trước, đo cohort along/off và holdout mới.

```python
import numpy as np
v=np.array([1.,1.])/np.sqrt(2)
def residual(z):
    z=np.asarray(z)
    return float(np.sum((z-(z@v)*v)**2))
assert abs(residual([2.,0.])-2)<1e-12
assert residual([10.,10.])<1e-25
```

## 8. Protocol của track bảng

Train 0…299 fit/scaler/reference; calibration 300…499 lấy q từng family; validation 500…619 chọn family bằng F1; test 620…799 báo cáo. Không refit sau chọn, nên có thể calibrate score trước labeled validation. Nếu refit trên tập khác, cần đánh giá lại q bằng calibration độc lập phù hợp; không giữ số cũ mà mặc định cùng false-hit rate. Mahalanobis test F1/AP cao hơn là quan sát, không cho phép đổi selected family kNN mà vẫn gọi test untouched cho decision đó.

## 9. Ngưỡng, tie và nominal rate

rank=ceil(20 ×.9)=18, q 18; score 18 không flag,19 có. n 200/α=.05⇒rank 191, tối đa 9 scores vượt q trong calibration; ties có thể ít hơn. FPR tương lai phụ thuộc score distribution/context/state và selection; không được suy chính xác 5%. n 1/α=.05⇒rank=ceil(2 ×.95)=2>1, helper từ chối.

```python
from detectors import threshold
q,k=threshold(list(range(1,20)),.1)
assert (q,k)==(18.,18) and not 18>q and 19>q
try:
    threshold([1.],.05)
except ValueError:
    pass
else:
    raise AssertionError('rank vượt calibration size')
```

## 10. Phát hiện sau quan sát và context

Được dùng y_t vì đang chấm observation đã đến; forecast trước t thì không. Plan phát hành t−48 có thể dùng xây expectation trước t; y_(t+1) chưa đến không được dùng cho detector causal tại t. Nếu score dựa y_t−y_(t−24), spike của hôm nay trở thành lag ngày sau và tạo residual lớn dù hôm sau normal. High residual có thể do incident, data error, plan sai hoặc model expectation cũ; cần điều tra.

## 11. CUSUM tính tay và trạng thái

G⁺=.5/1/1.5/2; G⁻=0, score bằng G⁺. Chia 2+2 mà reset: chunk sau trả 0.5/1 thay vì 1.5/2. Một spike lớn đẩy G lên cao, rồi k nhỏ cần nhiều bước để kéo về 0 nếu không có residual đối nghịch. Reset/clipping/rolling/adaptation thay thuật toán; phải chốt thời điểm/điều kiện reset chỉ bằng thông tin có lúc đó và backtest cả policy mới. Không dùng nhãn event-end tương lai để reset đúng chỗ đẹp.

## 12. Confusion matrix, AP và base rate

TP=1, FP=1, FN=1, TN=2, precision/recall/F1 đều 0.5, FPR 1/3. Positive ranks 1/3⇒AP 5/6. Ties gộp group, tất cả score bằng nhau cho AP bằng tỷ lệ positive; không đổi theo row order. Bayes arithmetic cho precision≈.0826446, tức 8.26446%. FPR là điều kiện trên normal, precision là điều kiện trên predicted positive; denominator khác nhau.

```python
import numpy as np
from metrics import confusion,average_precision
y=[0,1,0,1,0];scores=np.array([.1,.9,.8,.7,.2])
r=confusion(y,scores>.75)
assert r['precision']==r['recall']==r['f1']==.5 and r['fpr']==1/3
assert abs(average_precision(y,scores)-5/6)<1e-12
assert average_precision([1,0],[.5,.5])==average_precision([0,1],[.5,.5])==.5
pi,tpr,fpr=.001,.9,.01
precision=pi*tpr/(pi*tpr+(1-pi)*fpr)
assert abs(precision-.08264462809917356)<1e-12
```

## 13. Point adjustment và event delay

Point recall 1/10=.1; event recall 1/1=1; delay 9 bước. PA tô cả đoạn thành positive làmpoint recall 1 và có thể che mô hình chỉ tìm ra cuối incident. Cần chốt match theo timestamp firing hay active-overlap; có grace/lookahead không; event boundaries exclusive thế nào; duplicate/late alert phạt gì; missed events có luôn được báo không. Lab không PA, không tolerance, new firing phải nằm trong `[start,end)`.

## 14. Alert policy tính tay

Firing 2; sau non-hits 4/5/6 thì đóng ở 6. Cooldown bỏ 7/8/9/10, mở sớm nhất 11; counter đã đủ nên firing 11. Spike đơn không mở vì persist 2. Active trước event mới không là một quyết định phát hiện mới trong metric lab; nếu muốn active-overlap metric phải định nghĩa/chấm riêng. Đây là trade-off, không phải cách tăng event recall bằng đổi tên output.

## 15. Bundle, checkpoint và replay

Ví dụ q 1 chỉ là fixture I/O, **không phải threshold calibrated**. Score/q roundtrip và JSON state của policy phải giống. Thư mục temporary tự dọn:

```python
import json,tempfile
from pathlib import Path
import numpy as np
from data import tabular,fingerprint
from detectors import fit,score
from experiment import save_bundle,load_bundle
from alerts import apply
d=tabular();m=fit(d['x'][:300],'knn')
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'detector.json'
    save_bundle(path,m,1.,fingerprint(d,['x','label']))
    restored,q=load_bundle(path,fingerprint(d,['x','label']))
    assert q==1 and np.array_equal(score(m,d['x'][620:]),score(restored,d['x'][620:]))
    hits=[0,1,1,1,0,0,0,1,1,1,1,1]
    whole=apply(hits);a=apply(hits[:8]);state=json.loads(json.dumps(a['state']))
    b=apply(hits[8:],start=8,state=state)
    assert a['firings']+b['firings']==whole['firings']==[2,11]
    assert b['state']==whole['state']
```

Temporal cần calendar/scaler/threshold/policy versions, CUSUM G⁺/G⁻, ít nhất 24 actual cho seasonal lag và offset/timestamp. Fixture cung cấp history này, checkpoint chưa chứa tất cả. Digest không khóa có thể bị người sửa tính lại. Checkpoint chưa transaction với input offset/notification receipt nên crash có thể mất hoặc gửi lặp alert; cần thiết kế durability/idempotency riêng.

## 16. Đọc kết quả và thiết kế mở rộng

PCA residual chỉ nhìn hướng ngoài subspace, nên extreme along có thể reconstruct tốt. Calendar raw có 9 FP nhưng policy suppress được false firing ở sample này; vẫn bỏ spike 354 và 3 events được tìm trễ 1 giờ. CUSUM nhiều hits sau incident, raw event recall 1 đi cùng 113 FP; policy latched nên miss new events. Đừng trộn point precision với firing/event recall để tạo một F1 không có định nghĩa.

Thiết kế mở rộng: chốt entity/frequency/availability, thu reference được review, giữ các event cùng incident/thiết bị trong split thích hợp. Dùng development mới chọn k/representation/IF/AE/threshold/persistence, calibrate theo đúng model đã fit; sau đó test future/cold-device holdout mới chưa dùng. Không lấy bài test cũ đã xem để tiếp tục chọn và báo unbiased final performance. Báo point/AP/cohort, false firings theo thời gian/entity, detected delay kèm missed, peak memory/inference latency/reference storage và chi phí review. Với adaptation, replay đúng lịch update/no-label-until-available; đo cả việc incident bị hấp thụ thành normal. Chưa có kết quả thực thi nào cho các model mở rộng này trong bài 44.
