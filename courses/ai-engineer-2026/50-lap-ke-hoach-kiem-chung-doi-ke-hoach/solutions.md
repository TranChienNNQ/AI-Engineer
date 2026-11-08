# Lời giải bài 50 — Lập kế hoạch, kiểm chứng và đổi kế hoạch

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Đào sâu](deep-dive.md)

Các đoạn Python độc lập, chạy từ thư mục bài 50 và thư mục tạm. Không gọi provider; proposal scripted, MCP memory/stdio chạy thật.

## 1. Plan khác lời kể

Một plan minh họa: S1 đọc ticket 1 để có report; S2 tìm runbook bằng query, phụ thuộc S1 và cần evidence; S3 đọc health cho triage, phụ thuộc runbook. Args/scope cụ thể và postconditions giúp host quyết định bước nào ready/satisfied, không chỉ thấy câu “đã làm”.

RAG là cách lấy kiến thức, workflow là graph do code chọn, agent có bước lựa chọn theo observation; planned agent thêm biểu diễn/lineage/guard cho những lựa chọn ấy. Chúng có thể dùng chung tools. Có plan không tự là formal planning hoặc tăng chất lượng model.

## 2. Thực thi và tiến triển

Search ok=true nghĩa tool trả kết quả nghiệp vụ thành công; evidence=[] làm postcondition runbook_available chưa thỏa, step empty. Health unknown khác tool unavailable: một bên có snapshot nhưng status chưa đủ, bên kia chưa cung cấp được dữ liệu. Runbook có evidence mới thỏa nghĩa vụ theo rule nhỏ của lab.

Task outcome còn cần ticket/health/current scope/final gate và chất lượng nội dung. Không cộng mọi tool response thành task success, cũng không xóa empty/error khỏi cost/trace.

## 3. DAG và schema

Cycle khiến không có thứ tự thỏa dependencies, ID trùng không rõ step nào và dependency lạ không được giả completed. Extra status=done không thuộc schema: host mới tính trạng thái từ observations. DAG hợp lệ chưa chứng minh kế hoạch đúng domain hoặc tối ưu.

```python
from bridge50 import task
from policy50 import initial_plan
from plan_guard import PlanGuard,PlanError
p=initial_plan(task());p['steps'][0]['depends_on']=['S3']
try:
    PlanGuard().accept(p,[],task())
except PlanError as e:
    assert str(e)=='cyclic_plan'
else:
    raise AssertionError('cycle accepted')
```

## 4. Bước ready và authority

Chỉ S1 ready ban đầu. S2 cần ticket, S3 cần runbook theo dependency. Action phải trùng step/tool/args của plan và step ready. Đúng plan cũng chưa có quyền: backend/MCP tiếp tục kiểm target/current principal/source/role.

```python
from bridge50 import task
from policy50 import initial_plan,tool_action
from plan_guard import PlanGuard,PlanError
p=initial_plan(task());g=PlanGuard();g.accept(p,[],task())
try:
    g.authorize({'plan':None,'action':tool_action(p['steps'][-1])},[],task())
except PlanError as e:
    assert str(e)=='step_not_ready'
else:
    raise AssertionError('dependency skipped')
assert g.states([])['S1']=='ready'
```

Sửa args action thành ticket 2 khác accepted S1 bị action_not_in_plan; đưa ticket 2 ngay trong plan/task 1 bị plan_scope_denied. Cả hai không cho model đổi principal hoặc tenant.

## 5. Sửa query nhưng không sửa quá khứ

S2 original query đã chạy và empty. Revision 2 thêm S4 query từ symptom O1, dựa vào empty O2. Khi S4 có source, nghĩa vụ runbook thỏa; S2 vẫn là lần tìm không có kết quả. S3 chưa chạy nên có thể sửa dependency sang S4 trong revision hợp lệ.

Xóa S2 hoặc đổi args S2 sẽ làm trace dường như query mới đã chạy từ trước và bỏ mất cost/failure. Guard giữ executed-prefix identity để ngăn điều đó. Lưu quá khứ còn giúp chấm recovery thay vì chỉ nhìn output cuối đẹp.

## 6. Parent và trigger của revision

Parent hash phải khớp bản plan hiện hành, revision tăng 1, based_on có thật và đúng loại trigger. Missing_evidence chỉ trỏ O1 ticket successful chưa là lý do thiếu bằng chứng ở tool search/health. ID O999 không được model tự tạo.

```python
from bridge50 import task
from policy50 import initial_plan,revised
from plan_guard import PlanGuard,PlanError
from experiment50 import CASES,execute
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as d:
    row=execute(Path(d),next(c for c in CASES if c['id']=='recover_memory'))
    obs=row['result']['observations'][:2]
    g=PlanGuard();g.accept(initial_plan(row['task']),[],row['task'])
    p=revised(g.current(),'0'*64,'missing_evidence',['O2'],'request_clarification')
    try:
        g.accept(p,obs,row['task'])
    except PlanError as e:
        assert str(e)=='stale_plan_parent'
    else:
        raise AssertionError('stale parent accepted')
```

No-op revision và reused trigger bị chặn riêng. Những guard này chưa phát hiện mọi semantic no-op; query khác chuỗi vẫn có thể vô ích, nên cần budget và đánh giá tác vụ.

## 7. Discrepancy và thời gian

Cùng mốc fixture, report degraded/snapshot ok được gắn discrepancy, cần làm rõ chứ chưa quyết định report hay sensor đúng. Khác mốc là not_comparable: service có thể đã thay đổi. Cả hai chưa chứng minh database là root cause.

```python
import tempfile
from pathlib import Path
from experiment50 import execute,CASES
with tempfile.TemporaryDirectory() as d:
    for name,status,state in [('discrepancy','discrepancy','handoff'),
                              ('different_time','not_comparable','answered')]:
        r=execute(Path(d),next(c for c in CASES if c['id']==name))['result']
        check=next(c for c in r['verification'] if c['check']=='report_consistency')
        assert check['status']==status and r['state']==state
        assert next(c for c in r['verification'] if c['check']=='root_cause_established')['status']=='unknown'
```

Guidance ở different_time chỉ dùng runbook/current snapshot template, không arbitrary causal diagnosis. Nếu sản phẩm đòi giải quyết discrepancy theo lịch sử, cần tools và rubric mới thay vì dùng status hiện tại để kể lại quá khứ.

## 8. Giả thuyết và kết luận chưa được hỗ trợ

Không có causal diagnostic tool nên root_cause_confirmed không được nhận. Ticket report, runbook và status snapshot có thể cần thiết cho điều tra, nhưng không đủ chứng minh nguyên nhân hoặc đã sửa. Unknown là chưa biết, không đồng nghĩa đã chứng minh claim sai.

```python
import tempfile
from pathlib import Path
from experiment50 import execute,CASES
with tempfile.TemporaryDirectory() as d:
    c=next(c for c in CASES if c['id']=='claim_root')
    r=execute(Path(d),c)['result']
    assert r['code']=='plan_unverified_conclusion' and not r['answer']
    assert r['mcp']['tools_calls_sent']==3
```

Verifier hiện kiểm rule shape/graph/fixture postconditions và report/snapshot comparability. Nó không là entailment model, formal planner đầy đủ, source truth hoặc bằng chứng nhân quả vận hành.

## 9. Dữ liệu không có quyền sửa plan state

Model không có field status/budget trong Plan; tool chỉ trong allowlist, args còn qua strict scope. Plan sửa executed step bị chặn, còn runtime 49 giữ authorization ngoài plan. Schema cho authority không được “học lại” từ ticket text hay tool metadata.

```python
from bridge50 import task
from policy50 import initial_plan
from plan_guard import PlanGuard,PlanError
for field,value in [('status','done'),('budget',999)]:
    p=initial_plan(task());p[field]=value
    try:
        PlanGuard().accept(p,[],task())
    except PlanError as e:
        assert str(e)=='invalid_plan_schema'
    else:
        raise AssertionError('model authority field accepted')
```

Những tests này chứng minh boundary khi nhận proposal sai, không đo khả năng LLM chống prompt injection. Tool results/plan text vẫn là dữ liệu không tăng quyền.

## 10. Đọc phép so baseline

Straight cùng answered, 4 turns/3 calls. Recover: baseline handoff 3/2, planned answered 5/4, thêm 2 lượt/2 calls. Discrepancy: baseline 49 đưa guidance 4/3, planned 50 handoff 4/3 do thêm discrepancy policy. Baseline 49 không tuyên bố causal root cause trong answer nên không nên gắn nhãn rằng nó đã bịa nguyên nhân chỉ vì không handoff theo rule 50.

Phép so đổi scripted policy/representation/gate, không giữ mọi biến chỉ đổi planning. Nó minh họa logic/recovery cost, chưa định lượng LLM planning hoặc population risk. Muốn so thật cần cùng rubric/gates/task scope/model/sampling và paired evaluation được thiết kế trước.

## 11. Giá trị thông tin giả định

Expected net=.6×10−2=4 đơn vị giả định. Một query mới từ symptom có thể tìm đúng runbook và đổi quyết định. Một lần đọc lại cùng health snapshot không có thêm freshness/khả năng phân biệt giả thuyết có thể ít giá trị.

Cần dữ liệu để ước lượng p/gain/cost, tính cả time/context/tool/review/risk và calibration khi dùng xác suất. Không lấy confidence model viết ra làm xác suất đã đo. Scope/authorization vẫn là ràng buộc trước lợi ích giả định.

## 12. Counter và contract success

Plan versions 31 là số đề xuất được chấp nhận; planning turns 77 gồm action/final/rejected proposals; MCP calls 52 là các lần thực thi gửi qua MCP. Revision accepted có thể chưa có tool effect nếu runtime budget cạn. Các mẫu số này không thay nhau.

25/25 có 14 expected failures/2 stops; không 100% LLM task success. Port 50 không HTTP/model nên usage not_applicable, không model token savings hoặc zero-price live inference. Live cần semantic/trajectory quality, false confirmation, unnecessary calls, early handoff, uncertainty và actual cost/latency.

## 13. Budget và chống thrash

Revision mới không reset runtime/MCP counters. Tool_budget case chấp nhận revision 2 nhưng chưa dispatch S4 vì đã dùng 2 reservations. Revision_budget chỉ cho một plan thì repair revision 2 bị stop. Thrash dùng lại trigger O2 sau repair bị chặn dù có parent mới và thêm step khác.

```python
import tempfile
from pathlib import Path
from experiment50 import execute,CASES
with tempfile.TemporaryDirectory() as d:
    expected={'revision_budget':('plan_revision_budget',2),
              'tool_budget':('tool_budget',2),
              'thrash':('plan_reused_revision_trigger',3)}
    for name,(code,calls) in expected.items():
        r=execute(Path(d),next(c for c in CASES if c['id']==name))['result']
        assert r['code']==code and r['mcp']['tools_calls_sent']==calls
```

Chặn reused trigger không chứng minh mọi plan hợp lệ có progress thực. Query khác chữ, source stale hoặc verifier thiếu điều kiện vẫn có thể gây lãng phí; cần evaluations/budgets/freshness semantics bổ sung.

## 14. Thu hồi quyền và export

Planning trace có thể chứa terminal quote hoặc query từ ticket, plan mới có thể giữ symptom làm search args. Chỉ redact observations chưa đủ. PlannedAgent không export plans/trace/states/ledger khi result blocked; regression có revoke actor sau lượt đọc đầu.

RAM contexts hoặc dữ liệu đã gửi tới nơi khác không được thu hồi chỉ bằng cách xóa field trong response. Các artifacts/log/provider handling sau này cần policy riêng. Không coi hash/ID là anonymization tổng quát.

## 15. Reproduce và MCP stdio

Stdio case dùng MCP server process thật; planned policy vẫn scripted. Plan parent/hash có thể đối chiếu; SQLite chỉ giữ dataset/world, không plan checkpoint.

```python
import tempfile
from pathlib import Path
from experiment50 import execute,CASES
from bridge50 import digest
with tempfile.TemporaryDirectory() as d:
    r=execute(Path(d),next(c for c in CASES if c['id']=='recover_stdio'))['result']
    assert r['state']=='answered' and r['mcp']['closed']
    assert len(r['plans'])==2
    assert r['plans'][1]['plan']['parent_hash']==digest(r['plans'][0]['plan'])
    assert r['effects']==dict(tickets=2,runs=0,proposals=0,receipts=0,audit=0)
```

Trusted setup UPDATE body là tạo fixture trước run, có seed hash để đối chiếu. Model không có tool ghi và không thực hiện UPDATE ấy. Reopen DB chưa resume accepted plan/counters/pending actions; đó là phạm vi bài 51.

## 16. Thiết kế live planning experiment

Cần adapter nhận explicit view 50/prompt/Proposal schema, bounded parse/time/input-output/usage, actual model identity/sample/attempt IDs và evidence provenance. Dùng corpus synthetic trước, tool scope hữu hạn, cùng baseline/rubric/gates, new annotations theo output thật; giữ cả failed/rejected/extra-plan costs và uncertainty theo unit phù hợp.

Không chỉ lấy prompt 48 cũ vì rule no-source→handoff xung đột repair policy 50. Không dùng author profile labels cho mọi output LLM mới. Kiểm planner/DAG/verification/authorization từng lớp, human-audit closed verifier assumptions và plan/answer completeness. Bài chưa có live collector/provider call hoặc formal/causal proof; các bước mở rộng phải ghi đúng trạng thái và không bắt buộc API có phí để học bài này.
