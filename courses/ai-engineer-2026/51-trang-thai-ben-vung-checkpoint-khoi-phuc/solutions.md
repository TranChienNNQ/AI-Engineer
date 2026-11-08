# Lời giải bài 51 — Trạng thái bền vững, checkpoint và khôi phục

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Đào sâu](deep-dive.md)

Các snippet độc lập, chạy từ thư mục bài 51. Dữ liệu nằm trong TemporaryDirectory; worker crash chỉ tác động process con do thí nghiệm tạo. Không gọi provider hoặc tool ghi ticket.

## 1. Lưu ticket có phải đã lưu agent?

world.sqlite lưu business data, chưa có accepted plan, observations, pending action, counters, deadline, principal/task/release binding hoặc final delivery state. Bài 50 giữ các thông tin thực thi trong RAM. Một ticket status open không cho biết search nào đã trả rỗng hoặc health nào agent đã dùng.

Checkpoint giữ state tại boundary; context là phần đưa vào planner và có thể dựng lại; long-term memory phục vụ nhiều run; telemetry phục vụ quan sát. Một hệ có thể lưu tất cả bằng database, nhưng mục đích/invariant/retention khác nhau. Không dùng log tự do thay journal mà chưa thiết kế atomicity/replay.

## 2. Vẽ state machine

ready → planning khi reserve lượt; planning → pending khi proposal/tool action được guard và commit; pending → ready khi observation commit. ready → paused sau boundary chủ động; resume paused → ready. planning có thể tạo final sau validation. Budget/fault có thể kết thúc stopped/failed/blocked.

Pending phải chứa ý định có identity để lần sau biết operation nào chưa được giải quyết. Final phải có result cùng state, tránh answered nhưng không có nội dung đã lưu. Schema kiểm shape; runtime kiểm authorization/source/plan semantics.

## 3. Thiết kế ID

Giữ run support-504, step S2 và op support-504:S2. Attempt tăng từ 1 lên 2; checkpoint revision tăng theo mỗi commit; worker mới tăng epoch. Observation O2 được gán khi response được commit, không vì probe thấy một acknowledgement. MCP call_sequence có thể bắt đầu lại trong session mới.

```python
from contracts51 import operation_id,recovery_rule
assert operation_id('support-504','S2')=='support-504:S2'
assert recovery_rule('search_runbooks')=='bounded_read_retry'
assert recovery_rule('commit_ticket')=='reconcile_or_review'
```

Lab không gửi op_id làm server idempotency key. Chỉ đặt tên ổn định chưa khiến external service deduplicate. S4 query mới có op_id mới; không đổi args S2 để che lần tìm thất bại.

## 4. Commit trước/sau I/O

Reserve planning turn → guard proposal → commit plan/intent → reserve attempt → current permission check → external call → commit observation/xóa pending → validate/commit final → deliver. Journal append và head update phải chung transaction để readers không thấy head chưa có snapshot hoặc snapshot được công bố dở.

Không giữ write transaction qua I/O vì kéo dài lock và vẫn chưa atomically cover remote effect. Intent/reservation/result là các transaction ngắn; khoảng giữa chúng được biểu diễn rõ bằng phase pending, không được lờ đi.

## 5. Crash gap

Trước crash after_response: 2 reservations, 2 probe acknowledgements, 1 observation. Sau resume: 5 reservations, 5 acknowledgements, 4 observations. S2 trả hai lần nhưng chỉ lần retry được commit vào execution state. Probe là test instrument độc lập, agent không đọc nó để recover.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch,inspect
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'gap';setup(p)
    assert launch(p,crash_at='after_response')['exit']==83
    s=inspect(p)
    assert s['state']['pending']['op_id']=='support-504:S2'
    assert len(s['acknowledgements'])==2 and len(s['state']['observations'])==1
    launch(p,owner='worker2',takeover_epoch=s['header']['epoch'])
    a=inspect(p)
    assert sum(x['op_id']=='support-504:S2' for x in a['acknowledgements'])==2
    assert len(a['state']['observations'])==4
```

Với create-ticket, pending có thể che một ticket đã tạo. Cần status lookup hoặc API dedup với key đã persist, giữ unknown khi chưa xác minh; không retry bằng key mới. Write recovery như vậy chưa được code ở bài này.

## 6. Reserve mà chưa gửi

attempt_reserved có 5 reservations nhưng 4 acknowledgements; after_response có 5 và 5. Reservation được commit trước dispatch, nên crash có thể tiêu budget dù chưa gửi. Không thấy observation cũng phù hợp trường hợp đã chạy rồi mất kết quả, nên không đủ căn cứ để hoàn budget.

Bound này bảo thủ. Nó kiểm soát số cơ hội thử qua restart, không đếm chính xác SQL/network/backend operations. Việc cấp lại budget cần policy/controller và bằng chứng, không tự động suy từ một field rỗng.

## 7. Commit observation rồi chết

observation_committed lưu O2 search rỗng trước crash. Resume dựng guard từ O1/O2, biết S2 đã empty, tạo revision 2/S4 rồi đọc health. Không gọi S2 lại; S4 là operation khác có query lấy từ symptom.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch,inspect
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'committed';setup(p)
    launch(p,crash_at='observation_committed')
    b=inspect(p)
    assert len(b['state']['observations'])==2
    launch(p,owner='worker2',takeover_epoch=b['header']['epoch'])
    a=inspect(p)
    assert len(a['acknowledgements'])==4
    assert sum(x['op_id']=='support-504:S2' for x in a['acknowledgements'])==1
    assert len(a['state']['plans'])==2
```

Replay dùng dữ liệu đã commit để chọn bước; retry gây execution mới. Chỉ gọi cả hai là “chạy lại” sẽ dễ bỏ sót cost/effect.

## 8. Atomic transaction

terminal_transaction exit 84 ở giữa transaction. Checkpoint finished vừa INSERT chưa được commit/head update nên lần mở tiếp không công bố nó; state cũ vẫn phase planning. Resume bỏ dở lượt cũ, reserve lượt mới rồi commit final.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch,inspect
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'atomic';setup(p)
    assert launch(p,crash_at='terminal_transaction')['exit']==84
    b=inspect(p)
    assert b['state']['phase']=='planning'
    assert all(c['kind']!='finished' for c in b['checkpoints'])
    launch(p,owner='worker2',takeover_epoch=b['header']['epoch'])
    assert inspect(p)['state']['turns']==6
```

Phép thử kiểm worker death và SQLite recovery trên môi trường cụ thể; không mô phỏng OS/disk power failure. Không suy guarantee phần cứng từ một subprocess test.

## 9. Mất kết quả planner

planning_reserved đã tính lượt đầu nhưng chưa commit proposal/action. Sau restart, runtime ghi abandoned_planning rồi reserve thêm lượt. Cùng policy fixture đi tới answer với 6 lượt thay vì 5. Không reset counter về 0.

Với LLM thật, response cũ có thể đã được tạo/billed nhưng chưa persist; cần attempt IDs, usage known/unknown và retry policy riêng. Không ép model tái sinh token y hệt. Nếu proposal đã commit và pending action còn nguyên thì dùng action đó, không gọi lại planner trước khi giải quyết nó.

## 10. Pause và cached final

Pause xảy ra sau observation commit, không giữ một unknown tool call; owner được release. terminal_committed đã có final trên đĩa và owner null, process chết trước stdout. Lần sau claim mới, validate current rights/sources rồi trả cùng state/result mà không planning/tool thêm.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch,inspect
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'final';setup(p)
    assert launch(p,crash_at='terminal_committed')['exit']==83
    b=inspect(p)
    assert b['state']['phase']=='answered' and b['header']['owner'] is None
    launch(p,owner='worker2')
    a=inspect(p)
    assert a['state']==b['state'] and len(a['acknowledgements'])==4
```

Client có thể dedup theo run/result version để không tạo hai thông báo. Bài chưa có delivery acknowledgement/outbox/UI. Execution expiry không tự là answer TTL; code cho đọc final hợp lệ sau expires_at, nhưng vẫn phải qua source/permission checks.

## 11. Ngân sách và deadline

attempt_budget đã dành đủ 2 attempts trước crash nên không gửi lại S2; per_op_budget đã dùng 1 attempt/op nên cũng dừng. turn_budget đã dành một lượt trước crash nên không được sinh proposal mới. expired resume ở now=161 với deadline 160 dừng trước bước tiếp.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch,inspect
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'budget';setup(p,limits=dict(attempts=2))
    launch(p,crash_at='after_response')
    b=inspect(p)
    launch(p,owner='worker2',takeover_epoch=b['header']['epoch'])
    a=inspect(p)
    assert a['state']['result']['code']=='tool_attempt_budget'
    assert a['state']['attempts']==2 and len(a['acknowledgements'])==2
```

Clock tiêm vào suite là giả; không có hard total-run deadline giữa từng bước. Muốn vận hành phải dùng clock/cancellation/time accounting đúng contract. Link timeout chỉ cục bộ, không proof remote cancellation hoặc effect không xảy ra.

## 12. Hai worker cùng ghi

Revision so head để chặn một update dựa trên snapshot cũ. Epoch chặn worker thế hệ cũ sau takeover. Actor, owner và epoch là ba khái niệm khác nhau. Controller phải đợi process cũ chết; nếu cho takeover worker còn sống thì external calls chưa được fence bởi server.

```python
from pathlib import Path
import tempfile
from journal51 import Journal,JournalError
with tempfile.TemporaryDirectory() as d:
    j=Journal(Path(d)/'state.sqlite');binding={'demo':True}
    try:
        j.create('run','actor',binding)
        old=j.claim('run','actor',binding,'worker1')
        state=j.load('run','actor')['state']
        j.claim('run','actor',binding,'worker2',takeover_epoch=old['epoch'])
        try:
            j.save('run','actor',old,state,'late_write')
        except JournalError as e:
            assert str(e)=='stale_owner'
        else:
            raise AssertionError('stale owner committed')
    finally:
        j.close()
```

Tăng epoch không tự thu hồi một request đã bay tới remote service. Bài không có lease/heartbeat/remote fencing token hoặc multi-node lock service.

## 13. Quyền hiện tại và binding

Task/limit/health release khác làm binding mismatch; inactive actor bị unauthorized; source revoke chặn current evidence trước export. Cached final cũng phải tái kiểm nguồn. Không dùng quyền/thông tin cũ trong transcript làm authorization.

```python
from pathlib import Path
import tempfile
from experiment51 import setup,launch
from worker import work
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'revoke';setup(p);launch(p)
    result=work(p,owner='worker2',revoke=True)
    assert result==dict(ok=False,error='source_no_longer_allowed')
```

DB/probe/backup không bị xóa vì export bị chặn. Cần ACL/retention/encryption riêng khi có dữ liệu thật. Bộ chia sẻ chỉ chứa source và hình synthetic, không checkpoint runtime.

## 14. Corruption và migration

Hash/head không khớp phải reject, không đoán sửa checkpoint rồi tiếp tục pending action. Hash chain không có secret nên người kiểm soát DB có thể sửa và tính lại; nó không là chữ ký hoặc chứng minh nguồn dữ liệu đúng.

Migration cần schema/release nguồn và đích, validate plan/operation semantics, giữ IDs/counters/unknown status, kiểm quyền, ghi audit và rollback plan. Không chỉ thay version string. Bài chọn reject on mismatch, chưa migration runner hoặc importer cho DB không tin cậy.

## 15. Đọc counters và giới hạn thí nghiệm

54 là planning reservations, gồm các lượt mất trước commit; không là 54 LLM requests. 44 là tool attempt reservations; một reservation chết trước dispatch. 43 là acknowledgements được probe ghi sau dispatch return; không bao gồm mọi RPC/SQL/backend action. 25 là worker launches ở 13 case, gồm initial/recovery, chưa tính setup connections hoặc subprocess của inherited checks.

9 answered và 4 stopped đều đúng expected policy. Không có LLM sampling, chưa đo quality trên yêu cầu thật, RTO/RPO, storage throughput hoặc lỗi phần cứng. Các gate/guard tests đánh giá contract implementation, không chứng minh agent hiểu khái niệm như con người.

## 16. Mở rộng sang hành động ghi

Một thiết kế có thể dùng draft → reviewed/approved → durable intent → attempted → confirmed/unknown. Approval bind đúng payload/target/version/expiry, không câu “đồng ý” chung. Persist operation key trước dispatch; hệ đích phải dedup theo key/payload hoặc có status API. Retry sau unknown chỉ khi contract cho phép; nếu không, reconcile/review.

Local checkpoint/approval/outbox có thể chung transaction nếu cùng database, nhưng remote effect vẫn có gap. Recheck actor/source/target/current version khi resume; không reset budgets, không tự đổi key để vượt dedup, không compensate trước khi biết effect nào đã xảy ra. Test crash trước/giữa/sau call, concurrent owner, timeout, key reuse khác payload, stale approval/revoke và delayed status.

Bài 51 chưa có tool ghi, approval, outbox delivery, remote dedup/status hoặc compensation. Đây là thiết kế cho bài 53 dự kiến, không tính những tính năng ấy vào kết quả đã chạy.
