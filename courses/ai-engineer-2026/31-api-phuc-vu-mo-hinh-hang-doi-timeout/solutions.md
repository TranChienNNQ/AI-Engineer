# Bài 31 — Lời giải

[← Bài chính](README.md) · [Bài tập](exercises.md)

Snippet chạy độc lập từ thư mục bài sau khi cài requirements.txt. Các số thời gian trong phần tính tay không phải latency benchmark của máy đang chạy.

## 1. Model và hệ thống phục vụ

Request → validation → admission → queue → worker → model dùng chung → terminal result → response hoặc job GET. Lifecycle, capacity, cancellation, deadline, errors và result retention không tự được giải quyết bởi hàm predict.

Với x=(0,5;0,5), z=2×0,5+3×0,5−2=0,5; sigmoid(z)≈0,6224593, label high. Trọng số được đặt, không có dataset gán nhãn hoặc calibration nên không suy xác suất ticket thật khẩn cấp.

## 2. Model được load một lần

```python
import asyncio
from demos import local_client,PAYLOAD
async def main():
    async with local_client() as (client,svc):
        a=await client.post('/v1/predict',json=PAYLOAD)
        b=await client.post('/v1/predict',json=PAYLOAD)
        assert a.status_code==b.status_code==200
        assert svc.backend.load_count==1
        assert a.json()['result']['artifact_hash']==b.json()['result']['artifact_hash']
asyncio.run(main())
```

Warm-up xác nhận một forward path chạy được, chưa accuracy/SLO hay mọi shape/device path. Artifact cần model version, weights, feature order, preprocessing/scaler/vocab, threshold/output semantics và runtime phù hợp. Đổi preprocessing có thể vẫn cho số hữu hạn/HTTP 200 nhưng làm meaning sai; cần contract/parity tests với input tham chiếu.

## 3. Sync response và job

Predict giữ HTTP request đợi kết quả, dù coroutine dùng async/await. Job submit trả 202 trước kết quả; GET Location đến terminal. GET 200 nghĩa đọc record thành công, job bên trong còn có thể EXPIRED/FAILED. Label SUCCEEDED mới nói model operation đã trả kết quả hợp lệ theo contract, vẫn chưa nói prediction đúng trên dữ liệu thực.

Trong lab 202 chỉ là admission vào RAM. Restart mất queue/registry, record terminal có thể bị evict; chưa durable delivery hoặc per-user ownership. BackgroundTasks cũng không tự thay broker/result store/idempotency.

## 4. FIFO tính tay

A start 0/finish100/wait0. B start100/finish200/wait90. C start200/finish300/wait180. D/E bị reject vì B/C đã chiếm hai chỗ chờ. Nếu Q=4, D start300/finish400/wait270; E start400/finish500/wait360. Công suất service giả định vẫn 10 job/s.

Nếu C chỉ có budget150 từ arrival20 thì deadline170. Tới start dự kiến200, nó đã hết hạn; service phải skip, không cấp thêm150 sau dequeue. Đó là biến thể có timeout, khác bảng burst ban đầu không timeout.

## 5. Công thức và giả định

μ=1/0,1=10 job/s. M/M/1 W=1/(μ−λ): tại λ5→0,2 s; λ8→0,5 s; λ9,5→2 s. Little: L=8×0,5=4, cùng boundary/population. Đây là ví dụ lý thuyết vô hạn riêng, không là dự báo số entry trong Q=2 lab.

M/M/1 cần arrival Poisson, exponential service, một server, infinite buffer, λ<μ và steady-state cho các công thức đang dùng. Little có phạm vi rộng hơn, không cần hai phân bố ấy nhưng cần trung bình và boundary/rate nhất quán. Khi reject request, dùng effective/admitted rate; không ghép W của accepted với attempted λ.

## 6. Admission và queue cap

```python
import asyncio
from demos import overload_demo
async def main():
    r=await overload_demo()
    assert r['statuses']==[200,202,503]
    assert r['at_full']['busy']==1 and r['at_full']['queued']==1
    assert r['rejected_code']=='queue_full' and len(r['model_calls'])==2
    assert r['ready_while_full']==200
asyncio.run(main())
```

Queue_size=0 trong asyncio.Queue có nghĩa unbounded, nên Service từ chối. Semaphore chỉ cap running, các request chờ semaphore vẫn có thể rất nhiều. Await put vô hạn có thể chuyển backlog sang các coroutine producer. Lab reject bằng put_nowait; HTTP body/connection/task ngoài admission vẫn cần giới hạn khác nếu vận hành thật.

## 7. Budget còn lại

```python
from demos import hand_demo
h=hand_demo();b=h['budget_example']
assert b['total_ms']-b['elapsed_ms']-b['reserve_response_ms']==350
assert b['remaining_for_work_ms']==350
```

Đây là phép phân bổ budget giả định của client/system. Lab budget_ms bắt đầu tại submit sau validation, chỉ bao queue+inference. Khi dequeue dùng request deadline cũ; reset 500 ms mỗi stage sẽ kéo dài tổng thời gian quá yêu cầu.

Loop.time là monotonic trong process, không phải timestamp chung giữa máy. Qua network cần remaining duration hoặc timestamp với quy ước/skew, và server vẫn clamp budget theo policy.

## 8. Queue timeout và tombstone

```python
import asyncio
from demos import queued_timeout_demo
async def main():
    total=await queued_timeout_demo()
    queue=await queued_timeout_demo(queue_timeout=True)
    assert total['expired']['code']=='request_deadline'
    assert queue['expired']['code']=='queue_timeout'
    for r in [total,queue]:
        assert r['model_calls']==['j0001']
        assert r['at_expiry']['queued']==1 and r['retry_before_drain']=='queue_full'
        assert r['after']['queued']==0
asyncio.run(main())
```

Timer làm future terminal ngay nhưng entry vật lý còn trong FIFO đến worker dequeue. Worker skip và task_done, không gọi model. Join nói tất cả entry đã được xử lý/skip và bookkeeping cân bằng, không nói mọi job thành công. Xóa qua private queue internals có thể phá accounting/race.

## 9. Running expiry và cooperative cancellation

```python
import asyncio
from demos import running_timeout_demo
async def main():
    r=await running_timeout_demo()
    assert r['status']==503 and r['body']['state']=='EXPIRED'
    assert r['started']==['j0001'] and r['cancelled']==['j0001']
    assert r['finished']==[] and r['stats']['busy']==0
asyncio.run(main())
```

Fake backend đang await gate/sleep, nên cancellation ném CancelledError và unwind. Worker giữ busy đến unwind. Blocking forward không nhường event loop có thể trì hoãn timer; post-check bỏ kết quả quá hạn nhưng không dừng computation ngược về deadline. Native/GPU/remote cancellation cần contract/implementation khác, chưa được lab này kiểm.

## 10. Hủy chờ không kill thread

```python
import asyncio
from demos import cancel_demo
async def main():
    r=await cancel_demo(caller=True)
    assert r['cancelled_job']['state']=='CANCELLED'
    assert r['cancelled_job']['code']=='caller_cancelled'
    assert r['cancelled_backend']==['j0001'] and r['stats']['busy']==0
asyncio.run(main())
```

Đây là chủ động cancel coroutine HTTP-ASGI, Service.wait bắt cancellation để cancel job. Thread demo là một ca riêng: khi caller timeout, task/thread chưa xong; shield giữ handle theo dõi, không giết thread và không bắt caller tiếp tục chờ vô hạn. Sau đó mở gate và await physical completion để cleanup.

Nếu inference đưa vào thread, chỉ trả worker credit khi computation thật xong, hoặc khi backend thật sự xác nhận hủy. Không trả credit chỉ vì wrapper await đã bị cancel; không cắm thread vào cooperative Service mà giữ nguyên assumptions.

## 11. HTTP timeout thật

Socket mode đã quan sát statuses ready200/predict200/submit202/get200, job async SUCCEEDED; request thứ ba làm client ReadTimeout nhưng job sau đó SUCCEEDED, ba model call tổng cộng. Endpoint không tự theo dõi disconnect nên không tự hủy job vì client đóng socket.

Connect: mở connection; read: chờ chunk; write: gửi chunk; pool: lấy connection từ pool. Cần timeout tổng nếu muốn giới hạn toàn operation. Keep-alive timeout đóng connection idle giữa các request, không phải inference deadline. ASGI task.cancel và TCP disconnect không tương đương; không dùng check của cái này để suy cái kia.

## 12. Status code và retry

200 predict là model operation xong; 202 là nhận; 422 là input invalid; 500 là inference error; 503 là unavailable/overload/local expiry theo contract lab. GET record FAILED vẫn HTTP 200 vì việc đọc record thành công. ReadTimeout là exception client, không phải một HTTP 504 do app gửi.

504 phù hợp khi gateway/proxy không nhận upstream response kịp; 408 không là tên chung cho model compute timeout. Status phải đọc cùng error code và layer. Retry-After=1 là hint, không là cam kết còn capacity sau một giây.

Ba lớp ×3 attempts/lớp có thể tạo 27 downstream attempts. Retry cần tổng budget/cap, backoff/jitter và operation semantics. Nếu là write chưa rõ effect, dùng stable business key/receipt/reconcile; đừng suy client timeout tức chưa làm.

## 13. Retention và terminal state

```python
import asyncio
from demos import local_client,PAYLOAD
from service import Service
async def main():
    async with local_client(Service(record_limit=2)) as (client,svc):
        for _ in range(3):assert (await client.post('/v1/predict',json=PAYLOAD)).status_code==200
        assert (await client.get('/v1/jobs/j0001')).status_code==404
        assert len(svc.jobs)==2
        before=svc.get('j0003');svc.cancel('j0003')
        assert svc.get('j0003')==before
        before['result']['label']='changed'
        assert svc.get('j0003')['result']['label']=='high'
asyncio.run(main())
```

Eviction chọn terminal+physical_done; nếu record còn theo dõi running work thì phải giữ đến cleanup. Cancel SUCCEEDED không đổi state/result. Deepcopy ngăn người nhận dict sửa state nội bộ bằng reference, không là sandbox chống code Python tùy ý; job GET/DELETE chưa có auth/ownership.

## 14. Lifecycle và scaling

```python
import asyncio
from demos import shutdown_demo
async def main():
    r=await shutdown_demo()
    assert not r['stats']['accepting']
    assert r['stats']['busy']==r['stats']['queued']==0
    assert all(j['state']=='CANCELLED' and j['code']=='shutdown' for j in r['jobs'])
    assert r['all_workers_stopped'] and r['model_released']
asyncio.run(main())
```

Livez có thể 200 trước model ready; queue full tạm thời không tự unready. Shutdown chặn admission trước, drain trong grace rồi hủy phần chưa xong, cleanup worker/timer/model. Contract này dùng backend cooperative, không bảo đảm RAM jobs qua crash hoặc mọi native inference đều cancel được.

Hai coroutine worker chia queue/model trong process. Hai Uvicorn process có model/queue/registry riêng: job tạo ở process A có thể 404 khi GET đi B. Shared store/broker/routing cần thiết kế riêng; GPU memory có thể nhân theo process. Bài chỉ kiểm hai coroutine worker, chưa multi-process scaling benchmark.

## 15. Phạm vi bằng chứng

45 check xác minh API/lifecycle/parser/retention và expected outcomes. 11 scenario gồm nhiều case có intentional error/expiry/cancel, không là 11 task thành công hoặc accuracy. Numeric forward khớp tính tay, chưa chất lượng learned model.

ASGI kiểm routing/lifespan trong process; TCP thêm Uvicorn/socket/network timeout loopback; load test cần workload/arrival/concurrency/model/shape/measurement boundaries và latency distribution. Không lấy sleep5ms làm model latency benchmark.

Năm negative case mở rộng: body quá lớn trước validation, tenant quota/fair queue, load fail một shape/device, native operation không cooperative, crash sau202 trướcpersist. Load fail cơ bản đã có; các phần body/quota/native/restart durability còn chưa triển khai. Ghi rõ khoảng trống, không sửa tên một test có sẵn để nói đã kiểm.

## 16. Tiêu chí dự án

Người khác chạy lại được API, kiểm preprocessing/model parity, version/hash và queue/deadline/cancel/error/lifecycle. Dùng model nhỏ có input tham chiếu; reset synthetic state, không phụ thuộc account hoặc API key.

Nếu model blocking, chọn/kiểm backend execution và physical completion thay vì chỉ thêm async def. Tách numerical correctness, regression counts, actual network behavior và benchmark. Với job bền hoặc nhiều process cần thêm persisted state/delivery/ownership phù hợp; nêu điều chưa làm cùng kết quả, không công bố production SLO từ vài demo.
