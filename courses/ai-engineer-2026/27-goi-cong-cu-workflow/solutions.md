# Bài 27 — Lời giải

[← Bài chính](README.md) · [Bài tập](exercises.md)

Các snippet chạy độc lập từ thư mục bài, dùng môi trường trong requirements.txt. Thay đổi chỉ nằm trong Backend mới ở RAM.

## 1. Vòng gọi công cụ

Model đề xuất tên/args; host parse và kiểm permission/intent/budget, dispatcher gọi handler đã đăng ký, adapter trả result/error gắn call ID. Sau đó host có thể gọi model tiếp hoặc kết thúc workflow. Việc model nói “đã gọi API” không phải receipt của backend.

Protocol của bài là cấu trúc nội bộ. Tích hợp thật cần adapter theo provider và giữ call/result message đúng quy định; không giả định mọi API đều nhận name/arguments/id như lab. Server tool và client tool có nơi thực thi khác nhau.

## 2. Contract

Mô tả cần nói append là thêm note nội bộ, không đóng ticket/gửi thông báo; ticket phải cùng tenant, version là snapshot đã đọc, note có giới hạn; ghi mới cần intent/review của controller. Schema chỉ cho ID/version/note, còn identity/grant/key đi theo trusted channel.

```python
import json
from contracts import call,parse_turn
base=call('w','append_note',ticket_id='T-BLUE-1',expected_version=1,note='Kiểm tra retry')
cases=[]
x=json.loads(json.dumps(base)); x['name']='shell'; cases.append(json.dumps([x]))
x=json.loads(json.dumps(base)); x['arguments']['tenant']='green'; cases.append(json.dumps([x]))
x=json.loads(json.dumps(base)); x['arguments']['expected_version']=True; cases.append(json.dumps([x]))
x=json.loads(json.dumps(base)); x['arguments']['note']='   '; cases.append(json.dumps([x]))
cases.append('[{"id":"a","id":"b","name":"get_ticket","arguments":{"ticket_id":"T-BLUE-1"}}]')
for raw in cases:
    try: parse_turn(raw)
    except ValueError: pass
    else: raise AssertionError('Invalid proposal accepted')
print('5 invalid proposals rejected')
```

Chặn extra field hạn chế mass assignment nhưng không là auth. Backend phải vẫn kiểm tenant/quyền/resource; scope API production cần chặt hơn mô hình role write/read đơn giản này.

## 3. Correlation

Đảo thứ tự completion không đổi call mà result trả lời. Ghép bằng call ID, kiểm tập ID đầy đủ/không trùng và tên tool; không zip theo vị trí đến trước.

```python
import json
from backend import Backend
from contracts import call,pair_results
from workflow import Runtime
calls=[call('a','lookup_runbook',service='checkout'),call('b','get_health',service='checkout')]
results=Runtime(Backend()).read_batch(json.dumps(calls))
assert [r['call_id'] for r in pair_results(calls,results[::-1])]==['a','b']
for changed in [results[:1],[results[0],results[0]],[dict(results[0],name='get_health'),results[1]]]:
    try:pair_results(calls,changed)
    except ValueError:pass
    else:raise AssertionError('Bad correlation accepted')
```

Call ID là transport/protocol identity. Một ý định write được gửi lại với ID mới vẫn cần cùng idempotency key; một key mới cho mọi response model sẽ không dedup được intent trùng.

## 4. Đồ thị

Ticket → service là chaining; category technical/billing là routing. Sau khi biết service, runbook/health độc lập mới fan-out, draft là fan-in. Required evidence, review và live write policy là các gate khác nhau.

Hai write cùng expected_version có cạnh phụ thuộc hoặc conflict; chạy đồng thời không làm nghiệp vụ tự giao hoán. Code chính tuần tự; demo fan-out riêng chỉ hai read. Retry tạo vòng runtime, không được nhầm với DAG phụ thuộc không chu trình.

## 5. Đường găng

Sequential 350, parallel critical path 270, tổng tool work 300 ms. Draft 50 là tính toán giả định, không được tính như một tool request trong tổng 300. Human review time bị loại khỏi ví dụ; đo end-to-end thực tế cần tính riêng thời gian chờ người.

Parallel demo dùng asyncio sleep 8/12 ms, không gọi mạng; wall time có scheduling overhead và phụ thuộc máy. Không lấy 350/270 làm speedup đã benchmark cho service thật. Song song có thể giữ nguyên cost/request count và tăng tải tức thời.

## 6. Xác suất

`.98×.97×.99=.941094`; `1−.2³=.992`. Phép nhân đầu cần độc lập hoặc thay bằng các xác suất có điều kiện phù hợp. Phép retry cần cùng xác suất và độc lập giữa các attempt. Outage chung thường làm những giả định đó sai.

4 tầng mỗi tầng 3 attempt có thể khuếch đại tối đa `3^4=81` attempt ở tầng cuối cho một request gốc nếu mọi tầng đều retry xuống dưới. Retry policy nên có ownership/budget rõ, không nhân vô hạn ở SDK, service và workflow. Đây là giới hạn tính tay, không phải số đo lab.

## 7. Budget

500−200−50−200=50 ms. Không bắt đầu attempt cần 200 ms với budget còn 50 như thể deadline chưa trôi. Có thể không retry, giảm timeout theo remaining budget nếu semantics phù hợp, hoặc dừng có lý do.

read_retry có 5 call attempt: get_ticket hai lần, KB, health, write. Thời gian `20+50+40+80+120+50+60=420` modeled ms. call_limit=2 dừng sau ticket+KB; time_budget=100 dừng trước KB vì 40+80 vượt. State count khác call count: transition không nhất thiết gọi tool và một tool có thể nhiều attempt.

```python
from fixtures import scenario
r=scenario('read_retry')['before_reconcile']
assert r['calls']==5 and r['modeled_ms']==420
for name in ['call_limit','time_budget']:
    r=scenario(name)
    assert r['before_reconcile']['state']=='FAILED' and r['write_effects']==0
```

Admission biết duration trong simulator không dùng được như dự báo latency production. Deadline thật cần monotonic clock và I/O timeout/cancellation; tổng retry còn phải tính backoff.

## 8. Partial result

Health down hai lần vẫn có ticket/runbook đủ cho note triage; ghi rõ unknown. Runbook rỗng không có evidence bắt buộc nên NEEDS_HUMAN. bad_read_output là lỗi contract, không được chuyển thành “không tìm thấy” rồi bỏ qua.

```python
from fixtures import scenario
r=scenario('optional_health_down')['before_reconcile']
assert r['state']=='SUCCEEDED'
assert r['context']['health']['state']=='unknown'
assert 'health=unknown' in r['proposal']['arguments']['note']
assert scenario('no_runbook')['before_reconcile']['state']=='NEEDS_HUMAN'
assert scenario('bad_read_output')['before_reconcile']['state']=='FAILED'
```

Không bắt mọi Exception rồi trả `{}`: việc đó che programming error, quyền và dữ liệu sai kiểu thành dữ liệu tưởng hợp lệ. Partial policy cần do nghiệp vụ đặt; không dùng một fallback cho mọi tool.

## 9. Review gate

Prepare chỉ tạo draft, chưa có effect. execute trước approved bị từ chối. reject chuyển CANCELLED; không phải một write lỗi. Source version/args đổi sau review làm grant không khớp evidence/proposal.

```python
from workflow import Workflow
from backend import ToolError
w=Workflow();w.prepare()
try:w.execute()
except ToolError as e:assert e.code=='invalid_state'
else:raise AssertionError('Write without approval')
w.reject()
assert w.state=='CANCELLED' and w.backend.effects==0
```

Helper approve là trusted controller API, chưa có UI/auth event thật và không được expose như model tool. Controller có thể dùng quyền/intent đã xác lập trong session/workflow; không bắt buộc hỏi lặp lại chỉ vì đang dùng AI.

## 10. Dedup và conflict

Backend kiểm caller/tenant/write rồi tra receipt theo key. Cùng action/key trả receipt cũ dù call ID khác và grant đã consumed; không phải cấp thêm quyền hay thực thi lại. Khác args/cùng key bị conflict.

```python
import copy
from workflow import Workflow
from backend import ToolError
w=Workflow(run_id='same-intent');w.prepare();w.approve();w.execute()
c=copy.deepcopy(w.write_call);c['id']='new-transport-id'
receipt,_=w.backend.execute(w.subject,c,key=w.key,grant=w.review)
assert receipt==w.result and w.backend.effects==1
c['arguments']['note']='Nội dung khác'
try:w.backend.execute(w.subject,c,key=w.key,grant=w.review)
except ToolError as e:assert e.code=='idempotency_conflict'
else:raise AssertionError('Conflicting key accepted')
assert w.backend.effects==1
```

Request hash so nội dung, key nhận diện ý định, grant cho phép effect. Key mới có thể được coi là ý định mới và tạo note thứ hai nếu cũng được review đúng; vì vậy recovery phải giữ key, không tự thay ID để bỏ qua conflict.

## 11. UNKNOWN và đối soát

lost_write_response và invalid_write_output đều có effect 1 trước khi caller nhận xác nhận hợp lệ; sau đối soát receipt chuyển SUCCEEDED, effect vẫn 1. unknown_no_receipt trong simulator chưa ghi, nhưng controller chỉ thấy timeout và thiếu receipt nên giữ UNKNOWN.

Không tìm thấy receipt ở dịch vụ phân tán chưa chứng minh request không chạy: có thể đang xử lý, replica chưa đồng bộ hoặc retention đã hết. Backend lab là local authoritative store, không mô phỏng những yếu tố ấy; policy vẫn giữ UNKNOWN bảo thủ thay vì bịa kết quả.

## 12. Quyền/version lúc thực thi

Grant bound proposal/evidence/principal revision; append kiểm quyền sống và ticket version. Fixture revoke_before_write/stale_ticket đều không có effect. Sau effect, revoke không làm note đã ghi biến mất; mất quyền đọc receipt không là bằng chứng operation thất bại.

```python
from fixtures import scenario
from backend import Backend
from workflow import Workflow
for name in ['stale_ticket','revoke_before_write']:
    r=scenario(name)
    assert r['before_reconcile']['state']=='FAILED' and r['write_effects']==0
b=Backend();b.set_faults('append_note','timeout_after')
w=Workflow(b);w.prepare();w.approve();w.execute()
b.people['ops_blue']['write']=False
w.reconcile()
assert w.state=='UNKNOWN' and w.reason=='not_permitted' and b.effects==1
```

RLock chỉ đồng bộ thread cùng Backend/process. Không có durable receipt hoặc exactly-once distributed delivery. Production cần xác định transaction/conditional update, retention, retry/reconcile semantics và quyền đọc trạng thái operation.

## 13. Fan-out và cancellation

Hai read độc lập được schedule bằng TaskGroup, result trả theo call ID. Optional health unavailable được chuyển thành error envelope; nguồn bắt buộc lỗi làm exception propagate và group hủy sibling còn chạy. Timeout cũng có thể cancel task local.

Không suy server rollback: request đã gửi đi có thể vẫn chạy hoặc đã commit, nhất là khi dùng thread/blocking client/remote service. Demo không thực thi network và chỉ có hai read; không có toàn bộ concurrency limit/rate control hay scheduler phân tán.

```python
import asyncio
from backend import Backend
from parallel import read_fanout
b=Backend();b.set_faults('get_health','unavailable')
r=asyncio.run(read_fanout(b,optional_health=True))
assert r['results'][0]['status']=='ok'
assert r['results'][1]['status']=='error'
```

## 14. Đánh giá

Trước đối soát: SUCCEEDED 4, FAILED 10, NEEDS_HUMAN 2, WAITING_REVIEW 2, CANCELLED 1, UNKNOWN 3. Sau đối soát: SUCCEEDED 6, FAILED 10, NEEDS_HUMAN 2, WAITING_REVIEW 2, CANCELLED 1, UNKNOWN 1. Tổng mỗi bên 22. Có 6 write effect trong 22 Backend độc lập; không là 6 effect trên cùng run.

22/22 là expected-outcome checks, bao gồm fail/cancel đúng. Cần thêm task-quality/grounding eval nếu dùng LLM; rate đúng tool, args đúng, nghiệp vụ đúng, permission violation và duplicate effects là các phép đo khác nhau. Lab chưa đo chi phí provider, token, độ trễ mạng, tool selection hay ngữ nghĩa note do model.

## 15. Compensation

Note đã tồn tại và có thể đã được đọc; xóa nó không xóa tri thức người đọc. Email gửi rồi không thể chắc chắn thu hồi. Compensation là hành động nghiệp vụ sửa/giảm tác động, không phải rollback nguyên tử xuyên mọi hệ thống.

Có thể lưu NOTE_COMMITTED rồi NOTIFICATION_PENDING/FAILED, retry thông báo bằng key riêng, báo partial success rõ ràng. Nếu phải atomically lưu dữ liệu và ý định gửi event, cân nhắc transaction/outbox ở datastore; worker gửi vẫn cần dedup/retry. Lab chưa triển khai email/outbox/saga, không gọi chúng đã được kiểm.

## 16. Tiêu chí dự án

Người khác phải đọc được graph, contract, quyền và state transitions; chạy lại fixture với expected effect; thấy call ID, attempt, unknown và receipt. Code không được dispatch văn bản tool output hoặc tin approval trong body. Key/intent phải ổn định qua recovery, trace không chứa bearer grant.

Nếu thêm LLM, giữ cùng gateway và thêm adapter/model task eval. Thử benign, lỗi tạm thời, unknown, sai tham số, attack và missing evidence; ghi cả task completion và việc từ chối nhầm. Không dùng workflow từ chối mọi write để đạt “an toàn tuyệt đối” rồi bỏ qua utility.
