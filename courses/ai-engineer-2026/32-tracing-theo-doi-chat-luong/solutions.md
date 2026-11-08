# Bài 32 — Lời giải

[← Bài chính](README.md) · [Bài tập](exercises.md)

Các snippet chạy độc lập từ thư mục bài sau khi cài requirements.txt. Trace IDs/timestamps thay đổi giữa các lần chạy; so cấu trúc và count thay vì ép raw report giống từng byte.

## 1. Chọn tín hiệu theo câu hỏi

API chậm: latency distribution và queue wait để thấy population, rồi trace để xác định path cụ thể; log có correlation hỗ trợ đối chiếu. Nhiều503: counter theo boundary/error layer, traffic/admission và trace mẫu. Mobile sai: labeled cohort/slice metrics, coverage/version và data checks. Backend hết trace: kiểm sampler, exporter failure/eviction/collector health trước kết luận không có request.

HTTP200/span UNSET cho biết operation không báo lỗi ở boundary đã instrument, chưa so prediction với ground truth. Một hệ thống có thể phục vụ rất nhanh các prediction sai.

## 2. Parent graph

API SERVER là parent của queue.publish PRODUCER và queue.wait INTERNAL. job.process CONSUMER có parent producer; model.call CLIENT là child consumer; model SERVER nhận header từ client; model.forward INTERNAL là child server. Cả bảy span cùng trace trong normal case.

```python
import asyncio
from demos import trace_demo
async def main():
    r=await trace_demo();spans=r['spans'];by_id={s['span_id']:s for s in spans}
    consumers=[s for s in spans if s['name']=='job.process']
    assert len(consumers)==2
    assert all(by_id[s['parent_id']]['name']=='queue.publish' for s in consumers)
    assert all(by_id[s['parent_id']]['kind']=='PRODUCER' for s in consumers)
    assert all(s['end_ns']>=s['start_ns'] for s in spans)
asyncio.run(main())
```

Parent producer đã kết thúc trước consumer vẫn mô tả nhân quả. Không cộng tất cả duration vì có lồng nhau và song song; timestamp containment không là invariant chung cho asynchronous parent/child.

## 3. Các ID và W3C header

Version00, trace ID32hex, parent span ID16hex, flags2hex với sampled bit. ID toàn zero invalid; propagator của SDK không tạo valid parent từ header ấy. Outgoing header giữ trace ID, parent ID đổi thành span hiện tại.

Prediction ID dùng ghép actual outcome theo schema nghiệp vụ; trace ID dùng nối operations, span ID cho một operation. Không cái nào là credential, auth scope hoặc idempotency key mặc định. Header hợp lệ về format chưa xác nhận quyền người gửi.

## 4. Context qua queue

Normal và broken đều HTTP200/200, có14 span. Normal có2 trace; broken có4 trace vì hai job.process trở thành root, downstream model vẫn nối vào trace mới của mỗi worker operation. Worker được tạo ở startup trước request nên cần extract carrier message, không dựa context tại startup.

Global current_trace_id có thể bị request B ghi đè lúc request A đang await. Context per task giúp local execution; serialized carrier giúp queue/network boundary. Cần test isolation/parent edges, không chỉ kiểm một chuỗi ID xuất hiện trong log.

## 5. Critical path

Song song: max(40,60)+10=70 ms lý tưởng; work total40+60+10=110 ms. Tuần tự:110 ms nếu bỏ overhead. Parent duration còn bao child, nên cộng cả parent và child tiếp tục double count.

Consumer có thể bắt đầu sau khi publish operation kết thúc vì message được xử lý về sau. Với fan-in từ nhiều trace có thể dùng links; lab chưa triển khai case ấy.

## 6. Histogram

Mean120 ms; nearest-rank p95=500 ms. Buckets cumulative20/50/100/+Inf là2/4/4/5. Chỉ biết4/5≤100 không biết giá trị cụ thể của observation còn lại; histogram không cho exact p95 như danh sách năm giá trị gốc.

```python
from telemetry import Histogram
from demos import nearest_rank
values=[.01,.02,.03,.04,.5]
h=Histogram((.02,.05,.1))
for value in values:h.observe(value)
s=h.snapshot()
assert s['cumulative']==[2,4,4,5] and s['count']==5
assert abs(s['sum']-.6)<1e-12 and nearest_rank(values,.95)==.5
```

Replica percentiles đã mất thông tin distribution/population, trung bình chúng không khôi phục quantile toàn bộ. Cần gộp distribution phù hợp rồi tính percentile, cùng window/unit/bucket schema. Bài dùng teaching Histogram, chưa Prometheus query hoặc native histograms.

## 7. Mẫu số và cardinality

Một successful API prediction có2 HTTPcompletion: API và model service. Forward span không phải request thứ ba. Queue wait chỉ observe sau dequeue; queue_full bị chặn trước message nên không có queue wait observation.

3×2×3=18tổ hợp; thêm10.000ID có thể thành180.000. Dùng service/route template/status class và version policy bounded; không rawURL, user/email/prompt/trace ID. Counter API của overload có3attempt completion, model chỉ có2; không gọi tổng5 là5 người dùng.

## 8. Sampling/export failure

```python
import asyncio
from demos import trace_demo
async def main():
    r=await trace_demo(export_fail=True)
    assert r['statuses']==[200,200] and len(r['predictions'])==2
    assert r['span_count']==0 and r['exporter']['failed_spans']==14
    assert sum(x['count'] for x in r['metrics']['requests'])==4
asyncio.run(main())
```

ALWAYS_OFF cũng có0 span nhưng không phải exporter failure; metrics/ledger/logs vẫn được ghi theo code path riêng. Nếu chỉ giữ error traces, tỷ lệ error trong tập trace bị selection bias. Tail sampler không khôi phục dữ liệu head sampler đã drop; còn cần theo dõi chính sampling/export pipeline.

## 9. Minimization và giới hạn filter

SECRET32 là canary synthetic ở body, Authorization, baggage, query, path và exception. Middleware chọn route template, không serialize raw payload/header/query; span scope tắt auto record_exception và set error code cố định. Test xác nhận các đường cụ thể không xuất canary vào span/log/metric.

Key allowlist không biết secret nằm trong allowed value hoặc dynamic span name. Cần category do code kiểm soát, review auto instrumentation và retention/access phù hợp. Hash ID dễ đoán chưa chắc ẩn danh và vẫn giữ cardinality cao. Bài chưa có detector PII tổng quát hoặc production privacy proof.

## 10. Join theo as-of và version

Prediction key là `(prediction_id, model_version)`. Label có event105/available200 không được dùng ở as_of120 dù outcome đã xảy ra. Cohort theo prediction time, không theo thời điểm feedback ingestion.

```python
from dataclasses import replace
from data import dataset
from quality import joined_labels
p,f=dataset()
labels,info=joined_labels(p,f+[f[0]],320)
assert len(labels)==30 and info['duplicates']==1
conflict=replace(f[0],feedback_id='conflict',label=1-f[0].label)
labels,info=joined_labels(p,f+[conflict],320)
assert len(labels)==29 and info['quarantined']['label_conflict']==1
wrong=replace(f[0],feedback_id='wrong-version',model_version='toy-v2')
labels,info=joined_labels(p,[wrong],320)
assert not labels and info['quarantined']['unknown_prediction_or_version']==1
```

Duplicate không tăngn; conflict không lấy last-write-wins; wrong version không được nối vào model khác. Policy correction/revision chưa triển khai, cần thiết kế explicit nếu label được sửa hợp lệ về sau.

## 11. Nhãn chưa đủ

Early:6/6đúng trên6/10 nhãn, coverage60%. Mature:6/10đúng, coverage100%. Bốn prediction sai chưa về lúc đầu, nên observed accuracy100% không đại diện toàn10 prediction.

```python
from data import dataset
from quality import snapshot
p,f=dataset()
early=snapshot(p,f,100,110,120)
assert early['status']=='PROVISIONAL' and early['labeled']==6
partial=[x for x in f if x.prediction_id not in {'now-00','now-01','now-02','now-03'}]
later=snapshot(p,partial,100,110,220)
assert later['mature'] and later['status']=='INSUFFICIENT_LABELS'
assert later['observed_metrics']['accuracy']==1 and not later['alerts']
```

Không có quality alert ở đây vì eligibility gate, chưa có nghĩa chất lượng tốt. Production có thể cần data/feedback coverage alert riêng; lab chưa tự tạo rule cho mọi khoảng trống. Maturity/80%coverage chỉ là policy, không chứng minh missing labels random hoặc đã hết trễ.

## 12. Confusion matrix và slice

Accuracy=(1+5)/10=.6; precision=1/(1+4)=.2; recall=1/(1+0)=1; F1=2/(2+4)=1/3. Mobile1/5=.2, web5/5=1. Recall100% có thể song hành4 falsepositive trong5predicted positive. Brier current .3005 so reference .1205.

```python
from data import Prediction
from quality import score_metrics
p=Prediction('only','toy-v1','features-v1',0,'web',.1,.1)
m=score_metrics([(p,0)])
assert m['accuracy']==1 and m['precision'] is None and m['recall'] is None
assert score_metrics([]) is None
```

Contract bài dùng None khi denominator=0, không báo zero như có bằng chứng model recall kém. Khi vẽ dashboard cần hiển thị undefined/no labels rõ, không đổi None thành0hoặc100%.

## 13. Drift không quyết định quality

TV=.5×(|.5−.2|+|.5−.8|)=.3. Current TV0 nhưng accuracy.6<reference.8; shift TV.3 nhưng accuracy1. Không phải mọi shift đều có hại hoặc mọi quality regression đều đổi histogram đang đo.

```python
from quality import quality_demo
from math import isclose
q=quality_demo()
assert q['current_distribution']['tv']==0
assert q['mature']['observed_metrics']['accuracy']==.6
assert isclose(q['shift_distribution']['tv'],.3)
assert q['shift']['observed_metrics']['accuracy']==1
```

Feature drift P(X), concept drift P(Y|X), training-serving skew train/serve là các đối tượng khác. Histogram đơn biến/bins thô có thể bỏ sót đổi trong bin hoặc joint relation. Fixture là counterexample synthetic, chưa detector concept drift đầy đủ hay p-value/statistical claim.

## 14. Alert và runbook

Mature có overall/mobile accuracy<.75, slices đủ n/coverage theo policy. Trước rollback: kiểm nguồn label/join/version, thời điểm/coverage, schema/preprocess, data mix, mẫu sai và eval tái hiện; trace giúp tìm implementation path/correlation, không tự chứng minh model là nguyên nhân.

Rule thresholds chỉ là minh họa, chưa kiểm định/thống kê nhiều lần hoặc multiple-comparison control. Lab chỉ trả records, không gửi thông báo, tự retrain, chạy monitor nền hay rollback. Sau này cần gate/quyền như bài26–30và quy trình version/release bài33.

## 15. Tái lập đúng phần cần ổn định

48 checks kiểm implementation;6 tracing scenarios gồm lỗi có chủ đích;2 live prediction normal không là30 rowreplay quality;14 span mô tả hai prediction, không là14 prediction. ASGI header propagation không chứng minh TCP/service từ xa hoặc collector/OTLP đã chạy.

```python
import asyncio
from lab import report
from demos import stable_facts
async def main():
    a=await report();b=await report()
    assert stable_facts(a)==stable_facts(b)
    assert a['evaluation']['passed']==a['evaluation']['checks']==48
    assert a['tracing']['normal']['spans'][0]['trace_id']
asyncio.run(main())
```

Raw timestamps/span IDs/histogram durations có thể đổi; parent relationship, status/count, quality math và pipeline behavior phải giữ dưới cùng config/fixture. Runtime report chứa IDs/timing, không kèm trong ZIP nguồn; cũng loại venv/cache/credential/dữ liệu riêng. Artifact hash/data version là phần quan trọng nhưng chưa một mình bao mọi source/runtime change.

## 16. Tiêu chí dự án

Workflow có context carrier tường minh và kiểm normal/broken/isolation, metric không phụ thuộc trace sampling, exporter error/drop visible, raw data minimization có canary. Ledger prediction/feedback dùng join key/version/as-of, không suy ground truth từ trace success.

Có missing/duplicate/conflicting labels, maturity/coverage/slice guards, undefined denominator và drift counterexample. Report tách đúng operation success, telemetry completeness và labeled quality; nêu network/collector/DB/judge nào chưa làm. Không cần cài cloud app hoặc dùng dữ liệu thật để học các boundary này.
