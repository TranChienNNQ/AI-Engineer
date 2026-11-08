# Lời giải bài 43 — Dự báo chuỗi thời gian

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Phụ lục](deep-dive.md)

Mỗi đoạn Python dưới đây độc lập, chạy từ thư mục bài 43 sau khi cài requirements. Đáp án thiết kế có thể khác nếu nêu rõ giả định và hợp đồng thời gian hợp lệ.

## 1. Định nghĩa bài toán capacity

Ví dụ target là peak request/giây trong mỗi ngày theo UTC, L=28/H=7, origin cuối Chủ nhật sau khi log đã chốt. Nếu quyết định capacity theo ca thì dùng bucket theo ca thay cho ngày. Tổng request/ngày giống nhau vẫn có burst khác nhau; latency còn phụ thuộc service time, cache hit, queue, dependency và concurrency, nên forecast volume chưa bảo đảm p99. Cần `issued_at`, `data_as_of`, `target_time`, timezone/bucket definition và version. Với log đến trễ, actual đến T chưa chắc available đến T.

## 2. Baseline tính tay

Naive 16/16/16; mean 13/13/13; drift slope 2 ⇒18/20/22. Chu kỳ 7 cần một tuần đầy đủ nên bốn điểm không đủ. Seasonal naive 9 bước:11/21/31/41/51/61/71/11/21. Helper lab yêu cầu 28 điểm; code tính công thức nhỏ dùng trực tiếp NumPy:

```python
import numpy as np
history=np.array([10,12,14,16],float)
h=np.arange(1,4)
assert np.repeat(history[-1],3).tolist()==[16,16,16]
assert np.repeat(history.mean(),3).tolist()==[13,13,13]
assert (history[-1]+h*(history[-1]-history[0])/(len(history)-1)).tolist()==[18,20,22]
week=np.array([11,21,31,41,51,61,71])
assert week[np.arange(9)%7].tolist()==[11,21,31,41,51,61,71,11,21]
```

## 3. Phân biệt trend, seasonality, missing và zero

Tăng trưởng chậm là trend; lặp cuối tuần là seasonality. Ngày đóng cửa có thể có sales 0 thực nếu định nghĩa target là sales; mất log là unknown, không tự thành 0. Với target demand tiềm ẩn, ngày đóng cửa hoặc hết hàng không đo đầy đủ nhu cầu. Gắn cờ missing/closure/stockout, xác minh nghiệp vụ và thiết kế target/imputation trước. Không dùng một quy tắc điền 0 cho cả hai trường hợp chỉ vì file thiếu dòng.

## 4. Tạo window và tính số hàng

Origin 6: input 5/6/7, target 8/9. Initial direct: origin 27…216, có 190 hàng, target cuối 223. Origin 220 có label 221…227 nên vượt 224 exclusive, bị loại. Refit 280: direct 246 hàng, one-step 252 hàng. Initial one-step 196 hàng. Oracle vòng lặp dưới kiểm điều kiện label trước khi so helper:

```python
from data import dataset
from features import supervised
for end,h,expected in [(224,7,190),(224,1,196),(280,7,246),(280,1,252)]:
    legal=[t for t in range(end) if t-27>=0 and t+h<end]
    x,y,ticks=supervised(dataset(),end,h,True)
    assert ticks.tolist()==legal and len(legal)==expected
    assert legal[-1]+h==end-1
```

## 5. Bắt lỗi leakage

Ba thao tác đầu không hợp lệ cho protocol này: random split phá mô phỏng tương lai; scaler toàn bộ nhìn distribution chưa đến; centered mean nhìn y_(T+1). Origin 230 dùng actual đến 230 hợp lệ nếu chúng đã có, vì phát forecast mới 231…237. Chúng không được quay lại làm input cho forecast đã phát tại 223.

Future mutation phải giữ cả output lẫn preprocessing fit không đổi khi chỉ thay target chưa biết. Mô hình được fit ở 224 thì y 224 trở đi chưa được dùng:

```python
import copy
import numpy as np
from data import dataset
from models import fit_ridge,predict
a=dataset();b=copy.deepcopy(a);b['y'][224:]+=123456
ma=fit_ridge(a,224,True,7,.1);mb=fit_ridge(b,224,True,7,.1)
assert np.array_equal(ma['coef'],mb['coef'])
for key in ma['prep']:
    assert np.array_equal(ma['prep'][key],mb['prep'][key])
assert np.array_equal(predict(ma,a,223),predict(mb,b,223))
```

Đây là một tính chất cần kiểm, không chứng minh toàn bộ pipeline production sạch nếu metadata/as-of join sai.

## 6. Covariate biết trước

Calendar, kế hoạch 224 công bố 220 và forecast nhiệt độ phát hành 222 có thể dùng. Kế hoạch 225 công bố 224 và nhiệt độ thực 226 chưa có. Với feature bắt buộc thiếu, có thể từ chối forecast, dùng fallback được định trước hoặc imputation có flag đã đánh giá; không thay bằng actual sau này. Lab chọn fail fast cho plan không available:

```python
from data import dataset
from features import row
d=dataset();d['plan_available_at'][225]=224
try:
    row(d,223,7,True)
except ValueError as e:
    assert 'available' in str(e)
else:
    raise AssertionError('phải từ chối plan chưa có tại223')
assert row(d,223,7,False).shape==(28,)  # nhánh lag không dùng plan
```

## 7. Anchoring và scaler

Lag anchored là −3/−1/0; target residual 1/−2/3; cộng anchor 50 trở lại thành 51/48/53. Anchoring trừ cùng một mốc 50 khỏi mọi lag, còn sai phân từng cặp cho chuỗi ba điểm 47/49/50 là 2/1. Std floor 1 nghĩa cột std 0.3 được chia 1, không chia 0.3, nên variance sau biến đổi vẫn 0.09. Muốn phục hồi target phải dùng y_mean/y_std rồi cộng anchor, theo đúng thứ tự.

## 8. Ridge và MLP

Objective `||AB−U||²_F + λ Σ_(j≥1)||B_j||²`, cột đầu A gồm toàn số 1; không phạt B₀. Direct:52 feature+intercept,7 output ⇒371 hệ số. Recursive:34+intercept,1 output ⇒35. MLP 1664+32+224+7=1927 parameter. `mean-SSE + λ_mean penalty` tương đương `sum-SSE + n*λ_mean penalty` nếu mean chỉ theo sample; nếu mean theo tất cả n×H phần tử còn thêm H. Phải ghi quy ước thay vì giữ λ một cách máy móc.

MLP có thể thiếu dữ liệu độc lập, overfit, optimizer chưa hợp, hoặc cấu trúc gần tuyến tính đã đủ. Lab chưa thử hết search space; không được kết luận kiến trúc luôn thua.

## 9. Recursive tính tay và teacher forcing

Các giá trị là 9; 8.2; 7.56. Dùng actual 20 ở bước 2 cho 17 là một forecast mới có thêm thông tin, không còn forecast 3 bước tại T ban đầu. Trong code lab, buffer sinh ra được nối vào history, và plan vẫn kiểm với `asof=T` gốc; ví dụ nhỏ:

```python
import numpy as np
def rollout(last,h):
    generated=[]
    for _ in range(h):
        last=1+.8*last
        generated.append(last)
    return np.array(generated)
assert np.allclose(rollout(10,3),[9,8.2,7.56])
# Không có biến future_actual trong API này.
assert not np.allclose(rollout(10,3),[9,17,14.6])
```

Đoạn nhỏ kiểm trực giác AR, không thay cho guard window/covariate của model nhiều feature.

## 10. Metric và case không xác định

MAE 3; RMSE√11≈3.3166; bias 5/3; WAPE 0.3; MASE 1.5. MAPE có phần tử chia 0, không xác định theo công thức chuẩn. Train 1…14 có bảy chênh lệch mùa vụ đều 7⇒s 7. Constant train có s 0; MASE không xác định. MASE<1 so với scale train, không phải phép so trực tiếp với test seasonal-naive MAE. Cần chấm baseline trên cùng test để biết ai thắng ở đó.

## 11. Lịch đánh giá và chọn model

56/7=8 tuần validation, 84/7=12 tuần calibration, 56/7=8 tuần test. Origin là ngày ngay trước mỗi block 7 target; lần lượt 223/279/363 cho block đầu. Refit model/scaler trên prefix 280; giữ cố định sau đó. MLP chọn epoch 131 bằng first-min validation MAE trong 180 epoch fit prefix 224, sau đó khởi tạo lại seed 43 và fit 131 epoch trên 280 ngày.

Chọn lại family bằng stable-test là dùng test cho model selection. Muốn retrain mỗi tuần cần backtest chính policy refit ấy: cập nhật train chỉ bằng target đã chốt, fit lại scaler, tính lại calendar/covariate as-of và quyết định calibration phù hợp model mới. Không gán metric protocol cũ cho hệ mới.

## 12. Quantile và calibration tính tay

Pinball lần lượt 1.8/.2. m 9, rank=ceil(10×.8)=8, q 8, point 20⇒[12,28]. Lab m 12/rank 11 cho **mỗi h** ; 84 là tổng qua cả 7 horizon, không phải 84 observations cùng một phân phối sai số horizon. Pooling thay đổi câu hỏi; muốn dùng chung phải định nghĩa và kiểm riêng.

```python
import numpy as np
from metrics import residual_quantiles,interval
q,k=residual_quantiles(np.arange(1,10)[:,None],.2)
assert k==8 and q.item()==8
assert np.allclose([20-q.item(),20+q.item()],[12,28])
r=interval([5],[0],[4],.2)
assert r==dict(coverage=0.,mean_width=4.,interval_score=14.)
try:
    residual_quantiles(np.ones((1,7)),.2)
except ValueError:
    pass
else:
    raise AssertionError('rank vượt số calibration origin phải bị từ chối')
```

## 13. Interval score và coverage

[0,4] không chứa 5, coverage 0, width 4, score 4+10×1=14. [-100,100] chứa 5 nhưng width/score 200, cho thấy một ví dụ coverage tốt không làm interval hữu ích hơn. Mỗi ngày 80% không bảo đảm cả tuần 80%; nếu độc lập thì 0.8⁷≈20.97%, nhưng lab không giả định độc lập để tính xác suất cả path.

Lab dùng q cố định theo residual quá khứ. Dependence/shift phá giả định thường dùng để suy coverage; chỉ được nói coverage đo được 16/56 hoặc 45/56 ở hai nhánh. Chưa có coverage guarantee, online adaptation hay uncertainty do chọn model được lượng hóa đầy đủ.

## 14. Đọc ablation và kết quả

Validation MAE: seasonal 3.3750, ridge_lags 3.0327, ridge_known 1.1114, MLP 1.2088. Nhóm calendar+plan giúp trong cấu hình generator này; muốn cô lập plan cần thêm lag+calendar và lag+plan với search/split giống nhau, rồi đánh giá hợp lệ. Nhánh shift coverage 28.57%, stable 80.36%, cùng mean width 3.9961. Dữ kiện là các số đó dưới protocol/seed hiện tại. Giả thuyết “calendar sẽ giúp tải backend của tôi” cần log thực, hợp đồng timestamp và backtest khác; không suy trực tiếp từ generator.

Cũng không đọc test rồi tăng capacity theo một rule tự chọn và gọi đó là policy đã được đánh giá. Muốn đánh giá quyết định cần định nghĩa cost/SLA, chọn rule trên development rồi holdout mới.

## 15. Checkpoint và forecast record

Ví dụ standalone fit ridge, tạo q hợp lệ về shape, roundtrip và cố tình truyền hash sai. q toàn số 1 ở đây là fixture kiểm I/O, **không phải interval calibrated để dùng dự báo**:

```python
import tempfile
from pathlib import Path
import numpy as np
from data import dataset,fingerprint
from models import fit_ridge,predict
from experiment import save_bundle,load_bundle
d=dataset();m=fit_ridge(d,280,True,1,1.,True)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'forecast.pt'
    save_bundle(path,m,np.ones(7),fingerprint(d))
    restored,q,_=load_bundle(path,fingerprint(d))
    assert np.array_equal(predict(m,d,363),predict(restored,d,363))
    assert np.array_equal(q,np.ones(7))
    try:
        load_bundle(path,'wrong-snapshot')
    except ValueError:
        pass
    else:
        raise AssertionError('snapshot mismatch')
```

Record có thể chứa series_id, issued_at, data_as_of, target_time, h, frequency/timezone, point/lo/hi/nominal_level, model/preprocess/calendar_version. Training resume cần thêm optimizer/scheduler/RNG/data iterator và lịch train; bundle hiện tại chưa đủ. Digest không có khóa/chữ ký, ai thay payload cũng có thể tính lại digest, nên không chứng minh nguồn tin cậy.

## 16. Thử nghiệm mở rộng có kỷ luật

Ví dụ giữ test hiện tại làm báo cáo cũ; chọn cấu hình lookback/ETS/boosting qua nhiều rolling development folds ở phần lịch sử còn phù hợp, giữ boundary label và availability. Khi đã dùng quá nhiều development, thu thập/giữ một block tương lai chưa dùng làm test mới. Chốt metric/selection rule trước, refit trên development hợp lệ, dùng một calibration block sau refit, rồi giữ weights/q cho lịch phát hành đã công bố. Nếu đổi lookback, feature function đang hard-code 28 phải được parameter hóa đồng bộ; không chỉ sửa SPEC rồi tin model đã đổi.

So baseline naive/seasonal/drift/mean, MAE/RMSE/bias/MASE theo h, interval score/coverage nếu có; ghi train/inference latency, footprint và chi phí refit. Đo end-to-end nếu quyết định triển khai. Dùng cùng origin/actual/covariate snapshot, không ép model mới thắng bằng test tuning.

Nhiều cửa hàng: tách thời gian và cohort cửa hàng mới, fit scaler chỉ trên lịch sử được phép, so local/global, báo macro/micro để volume không che nhóm nhỏ. Nếu tổng vùng phải bằng tổng cửa hàng, cần kiểm/reconcile hierarchy riêng. Những việc này là kế hoạch mở rộng, không phải kết quả đã chạy trong bài 43.
