# Lời giải bài 52 — Context engineering và memory cho agent

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Đào sâu](deep-dive.md)

Các snippet độc lập, dùng TemporaryDirectory, không gọi LLM hoặc tool ghi ticket. Reviewer và source mappings là fixtures, không đánh giá của người thật.

## 1. State, context và memory

Checkpoint giữ đủ dữ liệu để recovery/audit/final validation; planner chỉ cần một view để chọn bước hiện tại. Vì thế state 40 KB và packet 7 KB có thể hợp lý nếu không mất thông tin thiết yếu. Không được dùng số nhỏ hơn để tự khẳng định summary đúng.

Memory phục vụ tái dùng qua nhiều run, không giữ thay pending op/counters/ownership của một run cụ thể. External memory cũng không sửa model weights. Context được dựng từ state và các dữ liệu còn được policy cho sử dụng tại thời điểm đó.

## 2. Ngân sách và đơn vị

Available=14.000−2.000=12.000 bytes. JSON syntax, Unicode UTF-8, metadata, refs và tool/response contracts đều được đếm trong packet của lab. Tokenization/model request envelope/output limit có đơn vị và overhead riêng; không có phép quy đổi chung 1 byte=1 token.

Nếu mandatory packet quá lớn, code fail required_context_budget. Sản phẩm có thể lấy dữ liệu hẹp hơn hoặc handoff theo policy, nhưng không tự cắt mất instruction/source/unknown rồi tiếp tục như đầy đủ. Reserve bytes là teaching convention, chưa guarantee cho một API thật.

## 3. Candidate và review

Candidate đã qua schema/source checks nhưng chưa active. Reviewer có role/scope hợp lệ approve, agent không tự approve trong policy này. Review ở lab là trusted fixture call, không human study/consent UI hoặc chứng minh nội dung nguồn là chân lý.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'review';setup(path);x=publish(path,approved=False);p=principals()
    m=MemoryStore(path/'memory.sqlite')
    try:
        assert m.recall(p['agent'],'gateway','XQ7')['items']==[]
        m.approve(p['reviewer'],x['id'])
        assert m.recall(p['agent'],'gateway','XQ7')['items'][0]['id']==x['id']
    finally:
        m.close()
```

Trong code thực tế, principal phải do host authentication cung cấp; không nhận một dict role=reviewer do model tự viết rồi coi đó là quyền.

## 4. Namespace trước ranking

A chỉ thấy rows đúng tenant/owner/service. Candidate pool không gồm B; diagnostics không cần công khai ID hay lý do loại của B. Scope sau top-k có thể để dữ liệu B ảnh hưởng thứ tự hoặc làm A nhận ít kết quả hơn dù kho còn nhiều item A hợp lệ.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'scope';setup(path);a=publish(path);b=publish(path,other=True,suffix='-b');p=principals()
    m=MemoryStore(path/'memory.sqlite')
    try:
        x=m.recall(p['agent'],'gateway','XQ7')
        assert [v['id'] for v in x['items']]==[a['id']]
        assert b['id'] not in [v['id'] for v in x['excluded']]
        assert [v['id'] for v in m.recall(p['other'],'gateway','XQ7')['items']]==[b['id']]
    finally:
        m.close()
```

Đây là kiểm payload pool trong local application, chưa production multi-tenant service/SQL user isolation hoặc OAuth architecture.

## 5. Lineage và payload

Query phải khớp symptom field của report source, trigger khớp alias source; refs phải đúng scope/version/hash và cùng origin run. Ref giả hoặc ghép nguồn từ hai run bị chặn, không cho một proposal tự bịa evidence. Source registry là trusted controller boundary.

Exact equality chỉ chứng minh quan hệ chuỗi theo rule. Việc alias XQ7 thật sự chỉ cùng sự cố với symptom còn phụ thuộc nguồn/nhãn của domain. Memory đã được review vẫn có thể sai hoặc vô ích; agent còn phải lấy current runbook/health và final gate kiểm lại.

## 6. TTL, source update và revoke

expires_at<=now thì không recall. Source update/revoke cũng có thể vô hiệu trước TTL; cả hai kiểm vấn đề khác nhau. TTL còn hạn không chứng minh fact còn đúng, chỉ là một điều kiện cần của policy sử dụng.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'ttl';setup(path);publish(path,expires_at=150);p=principals()
    m=MemoryStore(path/'memory.sqlite')
    try:
        assert len(m.recall(p['agent'],'gateway','XQ7',149)['items'])==1
        assert m.recall(p['agent'],'gateway','XQ7',150)['items']==[]
        m.revoke_source(p['reviewer'],'report')
        assert m.recall(p['agent'],'gateway','XQ7',149)['excluded'][0]['reason']=='source_revoked'
    finally:
        m.close()
```

Clock trong ví dụ tiêm vào để thử boundary, không wall-time TTL scheduler tự xóa data. Expired rows còn trên đĩa cho đến policy retention khác.

## 7. Conflict và recency

Hai active payload khác dưới cùng key bị loại cả hai, dù item mới hơn có timestamp/ID lớn hơn. Đây là conflict theo logical key của ứng dụng; hai query có thể đều tìm được cùng tài liệu ngoài đời, nên không gọi đó là logical contradiction đã được chứng minh.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'conflict';setup(path);publish(path);winner=publish(path,query='gateway latency upstream',suffix='2');p=principals()
    m=MemoryStore(path/'memory.sqlite')
    try:
        before=m.recall(p['agent'],'gateway','XQ7')
        assert before['items']==[] and {x['reason'] for x in before['excluded']}=={'conflict'}
        m.resolve(p['reviewer'],winner['id'],m.epoch)
        assert [x['id'] for x in m.recall(p['agent'],'gateway','XQ7')['items']]==[winner['id']]
    finally:
        m.close()
```

Expected epoch chặn update dựa trên snapshot cũ. Việc winner hữu ích/đúng cần đánh giá ngoài CAS; SQL không tự làm semantic arbitration.

## 8. Summary có nguồn và unknown

Summary giữ empty, unknown, timestamp và nguyên quote có phủ định, cùng origin refs về raw observation. Packet bị sửa phải reject trước consumer. Không tin status “đã kiểm tra xong” thay cho các trường này.

```python
from pathlib import Path
import tempfile,copy
from experiment52 import setup,launch,inspect,principals
from bridge52 import restore
from memory52 import MemoryStore
from context52 import compose,validate_bundle,ContextError
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'summary';cfg=setup(path);launch(path,create=True,pause_after=2)
    state=inspect(path)['state'];g=restore(state,cfg['limits'],cfg['task']);m=MemoryStore(path/'memory.sqlite')
    try:
        recalled=m.recall(principals()['agent'],'gateway',cfg['task']['question'])
        bundle=compose(cfg['task'],state,g,recalled,cfg['policy'])
        bad=copy.deepcopy(bundle)
        bad['packet']['working'][1]['observation']['data']['evidence']=[{'quote':'fabricated'}]
        try:
            validate_bundle(bad,cfg['task'],state,g,recalled,cfg['policy'])
        except ContextError as e:
            assert str(e)=='context_projection_changed'
        else:
            raise AssertionError('modified summary accepted')
    finally:
        m.close()
```

So lại cùng projection phát hiện packet tamper theo rule, chưa phát hiện mọi omission do chính thiết kế projection hoặc LLM summarizer gây ra. Cần critical-fact/semantic evaluation khi mở domain mới.

## 9. Đọc phép so compact/full

26.206−25.079=1.127 bytes; tỷ lệ 1.127/26.206≈4,30%. Lượt đầu chênh +3 vì chưa có observation để bỏ metadata, còn nhãn compact dài hơn full. Không phải mọi lượt đều rút ngắn.

Cùng actions/final trong script giúp kiểm cơ chế projection/packing trên fixture, chưa đo model quality hoặc khả năng sử dụng long context. Bytes JSON chưa là tokens, request wire bytes, bill hoặc measured latency. Không dùng biểu đồ này như một benchmark tối ưu LLM.

## 10. Poison và authority

Extra instructions/approved không thuộc Proposal/typed payload nên bị schema chặn. Episode root_cause=database không thuộc enum unknown. Nhưng string query có instruction-shaped text vẫn có thể khớp source và qua schema; có nguồn không làm nó trở thành authority.

Script chỉ dùng query trong read tool, còn host giữ allowlist/args/scope/plan/final validation. Không chạy write từ một câu trong memory. Vì không có LLM sampling, đây là boundary test, không đánh giá injection resistance của model. Khi live phải thử actual malicious content và chấm cả tool trajectory lẫn final.

## 11. Bỏ memory hay cắt instruction?

Memory tùy chọn được bỏ khi không còn byte budget; mandatory instructions/task/contracts/state không bị cắt âm thầm. Không nên bỏ refs để nhét nhiều item hơn vì khi đó mất khả năng kiểm provenance; cũng không tăng max_memories thay cho kiểm tổng serialized bytes.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from bridge52 import State,restore
from memory52 import MemoryStore
from context52 import compose
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'packing';cfg=setup(path);publish(path);m=MemoryStore(path/'memory.sqlite')
    try:
        state=State().model_dump();g=restore(state,cfg['limits'],cfg['task'])
        recalled=m.recall(principals()['agent'],'gateway',cfg['task']['question'])
        full=compose(cfg['task'],state,g,recalled,cfg['policy'])
        policy=cfg['policy']|dict(window_bytes=full['receipt']['mandatory_bytes']+cfg['policy']['reserve_bytes'])
        packed=compose(cfg['task'],state,g,recalled,policy)
        assert packed['packet']['memory_data']==[]
        assert packed['receipt']['omitted']==[dict(id=1,reason='memory_byte_budget')]
        assert packed['packet']['instructions']==full['packet']['instructions']
    finally:
        m.close()
```

Nếu ngay mandatory cũng vượt cap, cần xử lý lỗi/đổi chiến lược có kiểm; code trả required_context_budget. Không có automatic model summarization fallback trong lab.

## 12. Memory đổi trong lúc resume

Run pin epoch/selection khi tạo; forget đổi epoch, nên checkpoint cũ không còn binding hợp lệ. Nó không được nạp plan cũ rồi âm thầm tiếp tục query từ memory đã bị bỏ. Store update giữa build và consume cũng bị current check chặn.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,launch,inspect,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'resume';setup(path);publish(path)
    launch(path,create=True,pause_after=1);before=inspect(path)
    m=MemoryStore(path/'memory.sqlite')
    try:
        m.forget(principals()['agent'],'hint')
    finally:
        m.close()
    result=launch(path,owner='worker2')
    assert result['output']==dict(ok=False,error='run_binding_changed')
    assert inspect(path)['state']==before['state']
```

Epoch toàn store có thể invalid một run vì update không liên quan; chưa selective dependency invalidation. Binding mismatch hiện fail closed, chưa tự refresh/replan/migrate pending state. Remote call và memory lifecycle chưa atomic chung, nên không tuyên bố zero race window.

## 13. Logical forget và xóa dữ liệu

Memory rows cùng key được logical clear, tombstone chặn proposal cũ cùng key; sources và artifacts khác chưa bị xóa. Cần nói rõ phạm vi, không dùng “đã quên” như lời hứa bytes trong backup đã biến mất.

```python
from pathlib import Path
import tempfile
from experiment52 import setup,publish,principals
from memory52 import MemoryStore
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'logical-delete';setup(path);publish(path);m=MemoryStore(path/'memory.sqlite')
    try:
        m.forget(principals()['agent'],'hint')
        row=m.db.execute('SELECT status,payload,refs FROM memories WHERE id=1').fetchone()
        assert tuple(row)==('forgotten','{}','[]')
        assert m.db.execute('SELECT COUNT(*) FROM tombstones').fetchone()[0]==1
        assert m.db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==2
    finally:
        m.close()
```

Semantic aliases, derived summaries, checkpoint queries, logs, replicas và backup cần dependency/retention/deletion policy riêng. Tombstone một key chưa phát hiện cùng thông tin được ghi bằng key khác. SQLite UPDATE không secure erase storage.

## 14. Chọn gì để nhớ, khi nào lấy lại

Một mapping alias thường dùng đã review hoặc preference người dùng xác nhận có thể tái dùng; health snapshot của incident hiện tại thường chỉ là dữ liệu có thời điểm, không current truth ở run sau. Cần source/expiry và re-fetch khi quyết định đòi trạng thái hiện hành.

Hot-path write đơn giản hơn về thời điểm nhưng thêm latency; background cần xử lý late writes, dedup và forget/revoke giữa các bước. Lab dùng controller synchronous giữa hai run, chưa background worker, embedding retrieval, consolidation model hoặc consent UI. Không mặc định mọi nội dung chat đều nên được nhớ.

## 15. Thiết kế đánh giá

Promotion: tỷ lệ memory sai/không cần thiết được active. Recall: precision/recall trên authorized pool và coverage của facts cần thiết. Compaction: critical-fact/negation/unknown/ref retention. Lifecycle: expired/revoked/forgotten inclusion, resurrection và stale cache. Outcome: false confirmation, extra calls, handoff hợp lý, task quality và cost/latency toàn run.

20 cases + cross-run demo và 476 checks kiểm implementation/fixtures, không LLM population study. Các case failed/error có thể là kết quả đúng để chặn thao tác không đủ cơ sở. Không chỉ báo warm_compact thành công rồi bỏ conflict/budget/forget hoặc chi phí metadata/review.

## 16. Mở rộng sang LLM thật

Cần adapter gửi đúng explicit packet với prompt/tools/response schema versioned, tokenizer/model identity và input/output/timeout/usage accounting thực. Memory write proposal phải qua source/scope/policy/review; summary sinh bởi model cần validators và semantic/human audit, không chỉ tự chấm.

Giữ quyền, target, plan readiness, persistent budgets, current source và final evidence gates ở host. Thiết kế receipts/logs theo data minimization, cache invalidation theo dependency, retry/unknown provider usage và deletion scope. So strategies trên cùng tasks/models/snapshots/rubric, giữ cả failures và raw output lineage phù hợp.

Bài chưa chạy LLM/embedding/judge, chưa chứng minh token/cost savings, chống poisoning tổng quát, human-reviewed quality, secure deletion, selective invalidation hoặc production durability. Những claim đó chỉ được thêm sau khi có thực nghiệm tương ứng.
