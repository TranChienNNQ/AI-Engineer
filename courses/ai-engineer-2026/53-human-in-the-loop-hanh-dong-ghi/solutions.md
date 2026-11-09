# Lời giải bài 53 — Human-in-the-loop và hành động ghi có kiểm soát

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Đào sâu](deep-dive.md)

Snippet độc lập, dùng dữ liệu synthetic và process con. Không gọi LLM hoặc SaaS. reviewed đóng vai reviewer bằng fixture/controller, không là phản hồi của một người thật.

## 1. Những mốc không được gộp

Answered chỉ nói investigation có hướng dẫn/quan sát đủ cho gate; proposed có bản nháp; approved cấp quyền đúng draft trong hạn; in_flight đã dành attempt; unknown chưa biết chắc remote effect; confirmed có receipt/status khớp operation.

Receipt của append note chỉ hỗ trợ “note của operation này đã ghi”. Nó không chứng minh database là root cause, health đã phục hồi hoặc remediation đã chạy. Cần tool/evidence khác và tiêu chí nghiệp vụ riêng cho các kết luận đó.

## 2. Review view đủ thông tin

Cần action/đối tượng/current và expected version, body đầy đủ, nguồn, phần chưa biết, tác động internal-only, draft hash/version, local revision và expiry. Quyết định phải bind đúng view để phát hiện stale update. Một câu approve chung không cấp quyền cho mọi action tương lai.

Hash chứng minh identity theo canonicalization, không usability/human comprehension. Cần đánh giá reviewer có phát hiện lỗi thật, UI có trung thực và scope có rõ không. Demo chỉ escape JSON thành artifact đọc offline.

## 3. Reviewer và actor scope

Agent không được tự approve, reviewer tenant B không được đọc draft A. Role/principal do trusted auth registry cung cấp; model output hoặc memory nói approved không làm thay role. Target API còn kiểm độc lập khi ghi.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare')
    view=launch(case,'view',token='demo-reviewer-a')['output']['result']
    assert launch(case,'approve',expected_revision=view['workflow_revision'],
                  display_hash=view['display_hash'])['output']['error']=='reviewer_required'
    assert launch(case,'view',token='demo-reviewer-b')['output']['error']=='operation_scope_denied'
```

Identity registry và tokens trong lab đều synthetic, chưa OAuth/SSO hoặc authentication của reviewer trên một web UI.

## 4. Sửa draft và stale view

Edit làm draft version/hash đổi, approval về null và phase awaiting_review. Local revision tăng ở mỗi transition, còn target version chưa tăng khi chưa ghi. View cũ không được dùng cho bản mới.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed,inspect
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare');old,approval=reviewed(case)
    launch(case,'edit',token='demo-reviewer-a',expected_revision=approval['output']['result']['revision'],
           comment='Cần trace_id trước bước xử lý tiếp.')
    bad=launch(case,'approve',token='demo-reviewer-a',expected_revision=old['workflow_revision'],
               display_hash=old['display_hash'])
    assert bad['output']['error']=='stale_review_view'
    state=inspect(case)['local']['state']
    assert state['draft']['version']==2 and state['approval'] is None
```

Sau attempt, edit bị cấm để không thay payload của operation đang unknown. Một thay đổi mới cần giải quyết operation cũ và có proposal/review phù hợp.

## 5. Expiry và source freshness

Policy kiểm now>=expires_at là hết hạn, nên 159 còn hạn, 160/161 không được fresh write nếu expiry=160. Source memory có TTL riêng; hết ở 150 có thể chặn write ở 151 dù approval chưa hết. SourceGate dùng clock command hiện tại để kiểm selection/binding.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed',warm=True);case=setup(root/'case',root/'seed')
    launch(case,'prepare');reviewed(case)
    result=launch(case,'execute',now=151)
    assert result['output']['error']=='source_binding_changed'
```

Đây là clock boundary synthetic, không wall-time TTL scheduler hoặc hard deadline giữa mọi microstep. Source cached final và execution expiry cũng là policy khác nhau; không suy chúng tự bằng nhau.

## 6. Commit trước/sau POST

Local transaction ghi in_flight/counter trước I/O. Target transaction atomically ghi note/version/receipt. Local result transaction lưu receipt/confirmed sau response hợp lệ. Không có transaction chung giữa chúng.

Local rollback không xóa target effect. Giữ DB lock qua I/O/human wait vừa kéo dài lock vừa không tạo distributed atomicity. Cần biểu diễn unknown và cơ chế recovery cụ thể thay vì bỏ khoảng trống đó khỏi state machine.

## 7. Mất acknowledgement

lost_ack có target note/receipt nhưng local chưa nhận receipt, phase unknown. Status applied với receipt đúng cho phép confirmed_by_status mà không POST lại. Client không được suy no effect chỉ từ thiếu local record hoặc một generic server error.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed,inspect
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare');reviewed(case);launch(case,'execute',fault='after_commit')
    before=inspect(case)
    assert before['local']['state']['phase']=='unknown' and len(before['target']['notes'])==1
    launch(case,'reconcile');after=inspect(case)
    assert after['local']['state']['last_code']=='confirmed_by_status'
    assert sum(x['method']=='POST' for x in after['target']['requests'])==1
```

Nếu status chưa có hoặc không truy cập được thì vẫn unknown. Không tự refund budget hay tạo key mới vì chưa nhìn thấy effect.

## 8. Status not_found và retry cùng key

status_missing là fault visibility: receipt đã có nhưng query không trả nó. Local vẫn unknown. Explicit retry giữ key/body, target dedup trả lại receipt cũ, một note effect. Khác payload dưới cùng key phải conflict, không tùy tiện trả thành công cho nội dung khác.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed,inspect
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare');reviewed(case);launch(case,'execute',fault='after_commit')
    launch(case,'reconcile',fault='status_missing')
    assert inspect(case)['local']['state']['phase']=='unknown'
    launch(case,'execute',retry_same_key=True)
    a=inspect(case)
    assert len(a['target']['notes'])==1
    assert sum(x['method']=='POST' for x in a['target']['requests'])==2
```

Same-key retry còn cần current approval/quyền/source/budget. Dedup contract/retention là của target cụ thể, không guarantee mọi API có hành vi này.

## 9. Target version đổi

Target authoritative phải kiểm expected_version khi effect chưa tồn tại. Stale response không cho phép client tự tăng số version dưới approval cũ. Source world bài 52 chỉ là snapshot/mirror trong lab; không thay current target check.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed,inspect
from target53 import Target
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare');reviewed(case)
    t=Target(case/'target.sqlite')
    try:
        t.db.execute('UPDATE tickets SET version=version+1 WHERE id=1') # External actor fixture.
    finally:
        t.close()
    launch(case,'execute');a=inspect(case)
    assert a['local']['state']['last_code']=='stale_target_version'
    assert a['target']['notes']==[]
```

Lab dừng ở conflict, chưa refresh authoritative snapshot rồi chạy investigation/review mới tự động. Không ghi đè cập nhật của người khác để làm test “thành công”.

## 10. Kiểm receipt identity

Receipt phải đúng op_id/draft_hash/ticket_id/version effect và state applied. bad_receipt cố tình đổi ticket_id, local giữ unknown thay vì tin nhãn applied. Reconcile sau đó có thể xác nhận receipt đúng từ target.

Version 2 là version của effect đã commit tại thời điểm đó, không guarantee hiện ticket vẫn version 2. Một external actor có thể đã cập nhật tiếp. Confirmation của operation và current resource state là hai loại dữ liệu.

## 11. Cancel và compensation

Cancel trước attempt có thể dừng ý định chưa gửi theo protocol. Sau unknown/in_flight, đổi flag local không thu hồi request/effect; code đòi reconcile_before_cancel. UI phải nói rõ chưa xác nhận hủy effect, không hứa “không có thay đổi”.

Compensation là operation mới, ví dụ ghi một correction note có review thay vì âm thầm xóa audit cũ. Cần target/version/payload/quyền/audit và điều kiện nghiệp vụ. Bài chưa có compensation, outbox delivery hoặc customer notification.

## 12. Crash và ngân sách bền vững

crash_before_send đã tiêu attempt 1 nhưng chưa POST. Resume tra status, retry cùng key dùng attempt 2, không reset về 0. crash_after_response target đã có note còn local in_flight, tra status đủ xác nhận, không cần gửi thêm.

```python
from pathlib import Path
import tempfile
from experiment53 import template,setup,launch,reviewed,inspect
with tempfile.TemporaryDirectory() as d:
    root=Path(d);template(root/'seed');case=setup(root/'case',root/'seed')
    launch(case,'prepare');reviewed(case)
    assert launch(case,'execute',crash_at='before_send')['exit']==83
    before=inspect(case)
    assert before['local']['state']['attempts']==1 and before['target']['notes']==[]
    launch(case,'reconcile');launch(case,'execute',retry_same_key=True)
    after=inspect(case)
    assert after['local']['state']['attempts']==2 and len(after['target']['notes'])==1
```

Write attempts tối đa 2, status checks tối đa 3 cho operation. Các counter persist; không chứng minh network calls và backend executions luôn bằng reservation. Suite chưa đo thời gian khôi phục/RTO hoặc power loss.

## 13. Quyền hiện tại và effect lịch sử

Fresh write kiểm actor/reviewer/source/memory/release/expiry. Revoke/expiry không đảo ngược note đã commit. Reconcile có thể ghi nhận receipt metadata lịch sử nếu owner còn quyền đọc status, kể cả approval cũ hết hạn hoặc nguồn đã revoke.

Không export lại note/source body chỉ vì được đọc receipt metadata. Actor inactive vẫn bị status auth chặn. Các database/review artifacts cũ chưa tự bị xóa; logging/retention/backup/deletion cần policy riêng.

## 14. Demo permit và review artifact

Sai signature hoặc body không khớp permit bị target từ chối; thiếu envelope fields bị schema chặn. Key demo công khai chỉ minh họa message binding, không chống người sở hữu source/key giả mạo issuer. Hệ thật cần key isolation và authenticated reviewer/controller/service.

```python
from pathlib import Path
import tempfile
from workflow53 import render_review
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'review.html'
    render_review({'body':'</pre><script>alert(1)</script>'},path)
    text=path.read_text()
    assert '<script>' not in text and '&lt;script&gt;' in text
```

Escape giúp text không thành active HTML trong artifact, chưa là security toàn bộ web UI. Review hash cũng chưa chứng minh reviewer đã đọc kỹ hoặc UI không bị compromise.

## 15. Audit và evaluation

Audit nên nối local proposal/review/edit/reserve/result events với đúng target op/payload/receipt, kiểm monotonic counters, phase/binding/CAS và target effect một lần theo key. Giữ cả rejected/expired/unknown/failed và case có target effect nhưng chưa local confirmation.

26 scenario và 546 checks đánh giá contract/fixtures. Chưa LLM output quality, human false-approval rate, usability/latency/cost hoặc end-to-end exactly-once guarantee. Một investigation snapshot được copy để kiểm protocol; không phải 26 mẫu độc lập trên population.

## 16. Tích hợp thật vào agent

LLM có thể đề xuất draft qua schema hẹp, nhưng host bind target/source/release/expiry và điều phối review/execute/status. Human UI phải hiển thị concrete payload, lấy authenticated decision, kiểm stale version/hash, cho edit/reject/cancel rõ. External memory/retrieved text không tự nâng quyền thành approval.

Cần adapter target thực với auth, operation status/idempotency/retention/error contract đã đọc và thử; keys/attempts bền vững, source/identity checks, audit/data minimization, unknown/reconciliation/compensation policy. Phải thử actual model/human tasks và lỗi network/permission/target concurrency tương ứng.

Bài chưa live LLM/SaaS, chưa authenticated interactive UI, production key isolation, distributed revocation, broker/outbox, compensation hoặc human evaluation. Không được công bố những tính năng đó là đã triển khai chỉ vì state machine có tên gần giống.
