# Bài 34 — Lời giải và hướng đối chiếu

[Bài chính](README.md) · [Bài tập](exercises.md)

Các quyết định thiết kế có thể khác nếu boundary, population và giả định được nêu rõ. Không biến fixture tổng hợp thành kết quả production.

## 1. “Nhanh hơn” nghĩa là gì?

Chưa đủ. Cần cùng release/model/config, input/output mix, cache/cold/warm, concurrency/arrival process, connection/network, hardware/runtime, sample count/window, lỗi/timeout, quality và estimator. Mean20 lúc rảnh và p95200 dưới tải trả lời hai câu hỏi khác nhau; cần paired experiment phù hợp trước khi chọn.

## 2. Bộ đếm không được mất mẫu

Success/sent=70/90≈77.78%, success/planned=70%, attainment=50%. Attempt rate90/2=45/s, successful throughput70/2=35/s, goodput50/2=25/s. HTTP503 là response từ server; generator_drop chưa gửi nên chưa có response/latency service và không được gán status503.

Tỷ lệ success có mẫu số sent có thể trông tốt hơn success/planned khi generator thiếu năng lực. Báo cả counts và denominator.

## 3. Mean và nearest-rank percentile

Mean120 ms, p5030, p95/p99500. n5 khiến p99=max, chưa ước lượng ổn định đuôi production. None có nghĩa chưa có observation thuộc population, không phải latency0.

```python
from measurements import distribution,quantile
d=distribution([10,20,30,40,500])
assert (d['mean'],d['p50'],d['p95'],d['p99'])==(120,30,500,500)
assert quantile([],.99) is None
for values,q in [([1],0),([float('nan')],.95)]:
    try:quantile(values,q)
    except ValueError:pass
    else:raise AssertionError('invalid observation accepted')
```

## 4. Đặt lại đồng hồ

Dispatch100 ms, client20, scheduled120; success nhưng không good. Thêm generator_drop làm planned2/sent1, good vẫn0 nên attainment0. Client latency đẹp không sửa được late dispatch và thiếu tải.

```python
from measurements import summarize
rows=[dict(id=0,planned_ms=0,sent_ms=100,done_ms=120,status=200),
      dict(id=1,planned_ms=10,sent_ms=None,done_ms=100,status='generator_drop')]
s=summarize(rows,200,40)
assert s['generator_dropped']==1 and s['success_fraction_of_planned']==.5
assert s['latency_success_ms']['mean']==20 and s['scheduled_latency_success_ms']['mean']==120
assert s['good']==0 and s['attainment_of_planned']==0
```

## 5. Queue, execution và Little's Law

L=5×.2=1 request trung bình trong boundary đã nêu. L có thể gồm queue; worker capacity còn phụ thuộc execution, burst, tail, occupancy/resource và SLO. Offered10 gồm rejected nhưng W chỉ accepted thì không cùng population; 10×.2 chưa có nghĩa đúng cho boundary ấy.

Critical path=max(40,60)+10=70 ms; total work40+60+10=110 ms. Các nhánh có overlap, parent span thường chứa child nên cộng tất cả tạo double count. Contention/overhead thực có thể làm latency khác mô hình.

## 6. Closed/open và omission

Closed timeline đơn giản cho latency đều100, mean/p95100. Open mỗi20 tạo latency100+80i:100/180/260/340/420/500/580/660/740/820; mean460 và p95820. Nhu cầu events độc lập cần open. Closed phù hợp khi user thực sự chờ terminal trước thao tác tiếp; chưa phải sai trong mọi ngữ cảnh.

```python
from simulation import scenarios
s=scenarios();a=s['closed_one']['summary'];b=s['open_same_demand']['summary']
assert a['latency_success_ms']['mean']==100
assert b['latency_success_ms']['mean']==460 and b['latency_success_ms']['p95']==820
assert [r['done_ms']-r['planned_ms'] for r in s['open_same_demand']['rows']]==[100+80*i for i in range(10)]
```

## 7. Burst và goodput

4 accepted:2 hoàn tất20,2 hoàn tất40;4 rejected ngay0. Success latency[20,20,40,40], mean30/p9540; error latency0 trong admission lý tưởng. Good2 vì SLO30. Window40 ms → successful100/s, goodput50/s. Attainment/planned2/8=.25. Rate cao của burst hữu hạn không chứng minh steady capacity cùng SLO ở mọi workload.

```python
from simulation import simulate
r=simulate([0]*8,workers=2,queue_cap=2,service_ms=20,slo_ms=30)
s=r['summary']
assert s['statuses']=={'200':4,'503':4} and s['good']==2
assert s['success_rps']==100 and s['goodput_rps']==50
assert [x['start_ms'] for x in r['rows'][:4]]==[0,0,20,20]
```

## 8. Sweep không chỉ chọn RPS lớn nhất

Modeled service fixed2/20 ms chỉ xử lý tối đa100/s khi có việc liên tục. Arrival cao làm queue đầy và503; accepted còn trễ40 ms. 140/s có32 success nhưng8 good; 200/s có24 success nhưng6 good. Successful throughput cả cửa sổ vẫn quanh98/s trong khi attainment giảm20%/15%.

503 latency0 làm mean gộp giảm dù trải nghiệm kém. Mô phỏng không có HTTP/network/GPU, không suy capacity của phép đo thật từ100/s ấy. Số liệu HTTP còn phụ thuộc event loop và máy.

## 9. Bounded generator

Một request bị giữ chiếm cap; ba arrival tiếp theo tạo drop. Sau release event, request ấy503: sent1/errors1/planned4, không phải errors4. Hook emitted trong run dùng để đồng bộ fixture, không dựa vào một sleep thời gian tùy ý rồi giả biết tất cả arrival đã phát.

```python
import asyncio,httpx
from loadgen import Config,run
async def example():
    hold=asyncio.Event();entered=asyncio.Event();emitted=asyncio.Event()
    async def blocked(request):
        entered.set();await hold.wait();return httpx.Response(503,json={'code':'test'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(blocked),base_url='http://lab') as c:
        task=asyncio.create_task(run(c,Config(rate=500,count=4,max_inflight=1,max_lag_ms=1000),emitted))
        await asyncio.wait_for(entered.wait(),1);await asyncio.wait_for(emitted.wait(),1)
        hold.set();r=await task
    assert r['summary']['sent']==1 and r['summary']['generator_dropped']==3
    assert r['summary']['errors']==1
asyncio.run(example())
```

Sleep sau response gắn arrival tiếp theo với completion, tức closed-like behavior. Absolute schedule i/r giữ nhu cầu độc lập, nhưng vẫn phải báo lag/drop nếu generator không theo kịp.

## 10. Đọc một phép đo TCP thật

Không có một bộ p95/RPS/status counts bắt buộc cho mọi máy. Cần kiểm `planned=sent+dropped`, `sent=success+errors`, `good≤success`, time order và bounded server active≤2/waiting≤4. 503 là failure nhưng vẫn có latency; generator_drop chưa có client request latency.

Giới hạn: một máy/process/event loop; synthetic cooperative delay; ít request và repeat; payload đồng nhất/cache/keepalive; không GPU/TLS/proxy/remote network/task quality. Nếu nhiều drops, workload thực chưa bằng offered250/s, không dùng để chứng nhận server chịu được250. Assertions cố định timing sẽ biến noise hoặc machine differences thành flaky test thay vì kiểm định nghĩa.

## 11. Streaming boundaries

TTFT120, ITL30/40/30, mean token gap100/3≈33.33 ms; TPOT dùng response end120/3=40 ms; E2E240. Chênh20 ms finish overhead được phân bổ vào TPOT thứ hai. n1 thì per-extra-token TPOT là N/A. Chunk có thể chứa nhiều token, metadata hoặc buffer, cần biết payload/events thật.

```python
from measurements import streaming_example
s=streaming_example()
assert s['ttft_ms']==120 and s['itl_ms']==[30,40,30]
assert abs(s['tpot_to_last_token_ms']-100/3)<1e-12
assert s['tpot_using_response_end_ms']==40 and s['finish_overhead_ms']==20
```

## 12. Warmup, lặp và aggregation

Warmup xong/drain/reset rồi mới đo. Open window từ origin đến max(count/rate,last client terminal); closed đến last client terminal. Server idle có thể được chờ riêng sau client timeout. Warmup có thể vẫn thuộc allocated/billed cost của toàn phiên; không loại tự động chỉ vì latency summary bỏ nó.

Mean hai p95 chưa là global percentile. Giữ raw samples/counts và báo per-run distribution; nếu pooling thì nêu weighting/population. So hai release cần cùng workload, môi trường, latency endpoints, error/quality; giữ cold/warm cache policy, xen kẽ/randomize order phù hợp và đủ repeat/uncertainty. Lab chưa randomized order, confidence interval hoặc performance acceptance gate.

## 13. Tính chi phí token và retry

A1=(1000×2+100×8)/10⁶=.0028. A2=(500×2+500×.5+200×8)/10⁶=.00285. B1=(1000×2+1000×.5+300×8)/10⁶=.0049. Tổng.01055 USD. Chia3 attempt≈.00351667; chia2 successful task=.005275; chia1 qualified=.01055.

```python
from decimal import Decimal
from costs import attempt_cost,ledger,account
assert [attempt_cost(x['usage']) for x in ledger()]==[Decimal('.0028'),Decimal('.00285'),Decimal('.0049')]
r=account(ledger())
assert r['total_cost_usd']=='0.01055' and r['qualified_tasks']==1
assert r['cost_per_successful_task_usd']=='0.005275'
```

Cached input đã thuộc total input. Tính cả total theo giáfull rồi cộng cache sẽ double count. Failed A1 có usage trong fixture nên phải tính theo rate card; không suy quy tắc này cho mọi nhà cung cấp/status không có usage.

## 14. Usage chưa biết và duplicate

Không biết costC; report known subtotal.01055, missing1, total và mọi per-unitcost tổng là None. Chưa đối soát thì đừng gọi subtotal là total/invoice. Under giả định nonnegative rates của bài, nó là phần đã biết; credit/refund thực có thể làm cách hiểu lower bound khác.

Duplicate cần receipt/provider attempt identity, nội dung usage/outcome, revision/as_of và precedence rule. Hai record conflict có thể là correction hoặc bug; account() từ chối để người gọi reconcile. Tự cộng cả hai hoặc bỏ một record tùy thứ tự đều có thể sai.

## 15. Self-host cost và quality tradeoff

3/60000=.00005 USD/request;3/54000≈.00005556 USD/success;3/45000≈.00006667 USD/qualified. Nếu đã cấp trọn giờ và charge theoallocated hour, busy30% không tự giảm invoice còn.9 USD. Spot/reservation/serverless/elastic có quy tắc khác cần kiểm riêng.

Còn CPU/RAM/storage/network, shared services, retained replicas/idle/warmup, telemetry và vận hành. So API/self-host cần cùng task/quality/SLO/failure/capacity scope và cách phân bổ; ví dụ này chưa đủ quyết định mua tài nguyên.

## 16. Protocol cho sản phẩm AI

Một protocol tốt pin release/model/prompt/index/data/config và hardware/runtime; workload giữ độ dài, cache hit và tool mix; định nghĩa planned/send/first-content/last-token/end; open arrival hoặc user/thinktime theo bài toán; warm/cold/window/drain, cap/deadline và đủ repeat. Chấm quality/security/freshness trên tập/outcome phù hợp rồi mới tối ưu cost/goodput.

Dừng hoặc chưa kết luận capacity nếu generator thiếu headroom/drops, error/latency vượt mục tiêu, memory/queue tăng không ổn định hoặc workload/quality không còn đúng. Sau tối ưu kiểm numerical/schema/quality, SLO theo slice và cost/qualified task; quan sát release đang phục vụ và rehearsal rollback có dependency tương thích.

```python
from simulation import scenarios
from costs import self_hosted_example
from lab import hand
assert scenarios()==scenarios()
assert self_hosted_example()==self_hosted_example()
assert hand()==hand()
print('Deterministic definitions/cost fixture; TCP timing is measured separately')
```

`stable_facts()` giữ metadata/source/config, simulation, cost, hand và regression outcomes. Nó không giữ TCP timing/status/drop counts hoặc coi chúng là hằng số. Cùng source không đảm bảo cùng background load hay thermal/cache state.
