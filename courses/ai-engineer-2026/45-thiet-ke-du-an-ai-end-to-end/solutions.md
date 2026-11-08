# Lời giải bài 45 — Thiết kế dự án AI end-to-end

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Thiết kế đào sâu](deep-dive.md)

Các đoạn Python độc lập, chạy từ thư mục bài 45 sau khi cài requirements. Những đáp án kiến trúc khác vẫn hợp lệ nếu chỉ ra rõ assumption, contract, quyền, failure boundary và bằng chứng cần đo.

## 1. Viết product brief

Ví dụ: nhân viên hỗ trợ tốn effort tìm runbook đúng và nhập lại nội dung vào ticket. Prototype hỗ trợ đọc tài liệu có source, tạo draft ticket và đề xuất note/resolve rõ target. Ngoài phạm vi: tự sửa production, tự xác nhận root cause, xóa ticket hoặc gửi dữ liệu ra ngoài tenant.

“Giảm effort tìm/nhập liệu” là hypothesis cần baseline/time study. “Không execute proposal pending”, “retry cùng key không tạo ticket thứ hai” là acceptance criteria có thể kiểm bằng API/DB. “Thông minh” chưa xác định task population, metric hoặc hành vi khi sai, nên chưa đủ để review thiết kế.

## 2. Vẽ kiến trúc và trust boundary

Client input→schema/identity→workflow. Workflow đọc nguồn qua eligibility/retrieval→context→answer validator. Nhánh ghi chỉ từ intent có cấu trúc→proposal có target/version/hash→reviewer→execute gateway→transaction ticket/receipt/audit. Text tài liệu và model output là dữ liệu chưa được cấp quyền; identity/policy/gateway là authority.

Baseline không thông dịch instruction trong tài liệu thành operation. Với LLM, tool proposal vẫn phải qua allowlist/schema/current user scope và review phù hợp. Không đưa DB handle/credential cho generator rồi kỳ vọng prompt tự kiểm quyền.

## 3. Input contract và authority

Note hợp lệ có action, ticket_id và expected_version. Schema chặn những lỗi hình thức; chỉ backend biết ticket có thuộc tenant, role hiện tại còn được ghi, version/state có còn phù hợp. Fixture token là mapping công khai trong RAM, không OAuth/JWT validation hoặc durable revoke.

```python
from api import RunIn
from pipeline import request
from pydantic import ValidationError
valid=request(action='note',ticket_id=1,expected_version=1,key='note')
assert RunIn.model_validate(valid).expected_version==1
for bad in [valid|{'expected_version':None},valid|{'tenant':'b'},valid|{'max_steps':True}]:
    try:
        RunIn.model_validate(bad)
    except ValidationError:
        pass
    else:
        raise AssertionError('invalid contract accepted')
```

## 4. Retrieval và context budget

Lọc sau top K có thể để source cấm chạm context và loại mất source hợp lệ đã bị đẩy khỏi top K. IDF toàn corpus còn cho dữ liệu không được phép ảnh hưởng ranking; lab tính trên eligible corpus. Source quyền/service/active/approved/revoked được kiểm trước scoring.

600 characters là số code points theo Python len, không phải byte hoặc tokenizer tokens. Paragraph không vừa phần còn lại bị bỏ nguyên; điều này giữ quote không bị đứt nhưng có thể abstain dù một câu ngắn bên trong hữu ích. Muốn chunk nhỏ hơn cần version/offset/index và eval mới, không cắt bừa rồi giữ offset cũ.

## 5. Kiểm citation

Gateway hiện hành là v2, start 0/end=len(text), hash là SHA-256 canonical JSON string của toàn text trong helper. Hash không chỉ là SHA-256 raw UTF-8 text nên cần giữ đúng cách serialize. Sửa quote/offset bị validator từ chối. Source ID có thật chưa nói version, evidence membership hoặc quyền đúng; cached result phải revalidate current source scope, không trả quote đã bị revoke.

```python
import copy
from corpus import documents,TOKENS
from retrieval import retrieve,validate
p=TOKENS['demo-agent-a'];docs=documents()
evidence,_=retrieve(docs,p,'gateway','timeout 504')
assert evidence[0]['version']==2 and evidence[0]['start']==0
assert validate(evidence,evidence,docs,p,'gateway')
bad=copy.deepcopy(evidence);bad[0]['quote']+=' sai'
assert not validate(bad,evidence,docs,p,'gateway')
bad=copy.deepcopy(evidence);bad[0]['end']-=1
assert not validate(bad,bad,docs,p,'gateway')
assert not validate(evidence,evidence,docs,p,'gateway',revoked={'gateway'})
```

## 6. Baseline và bước thêm LLM

Hiện tại: lexical retrieval sau ACL, overlap gate, whole paragraph pack, quote exact và explicit action workflow. Chưa có free-form reasoning, semantic paraphrase retrieval, learned intent, natural-language tool planning hay quality benchmark LLM.

Generator mới có thể nhận query/evidence có ID/version/offset và trả structured answer/claims/citations/abstain. Backend vẫn quyết định auth/quyền/action scope/review/version/receipt. Eval mới gồm paraphrase cần source, câu không có nguồn, conflict giữa nguồn, source chứa instruction, claim trộn supported/unsupported và thiếu ý quan trọng. Không nhất thiết đổi Store transaction khi chỉ thay generator, nhưng phải giữ payload/citation contract và đánh giá end-to-end behavior mới.

## 7. Proposal, approval và effect

Ticket count ban đầu 2, proposal create không đổi count. Execute trước approve báo approval_required; reviewer thấy operation, target/version, title/body/service, citation và release cùng payload_hash. Không cho execute body thay args vì approval đã gắn việc cụ thể. Đoạn dưới xem payload trước approve:

```python
import tempfile
from pathlib import Path
from pipeline import Engine,request
from store import Fault
with tempfile.TemporaryDirectory() as directory:
    e=Engine(Path(directory)/'project.sqlite')
    pid=e.ask('demo-agent-a',request(action='create',key='draft'))['proposal_id']
    assert e.store.db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]==2
    try:
        e.commit('demo-agent-a',pid,'effect')
    except Fault as err:
        assert err.code=='approval_required'
    else:
        raise AssertionError('unreviewed mutation')
    preview=e.read_proposal('demo-reviewer-a',pid)
    assert preview['payload']['operation']=='create'
    e.review('demo-reviewer-a',pid)
    receipt=e.commit('demo-agent-a',pid,'effect')
    assert receipt['ticket_id']==3
    e.close()
```

Đây là fixture automation cho kiểm thử, không phải một người thật đã review hoặc bằng chứng review UX hiệu quả.

## 8. Version và semantic state

Note đầu từ version 1 tạo version 2; note sau được duyệt trên version 1 bị stale_ticket_version, không có effect. Nếu muốn rebase tự động, phải xác định merge semantics và reviewer đã duyệt ý nghĩa gì; không mặc định version mới luôn tương đương snapshot cũ.

Lab chặn note sau resolved. Một hệ khác có thể cho comment sau close hoặc có transition reopen, nhưng cần quyền/expected version/state/eval tương ứng. Giữ code đúng policy đã công bố thay vì sửa guard chỉ để request nào cũng thành công.

## 9. Mất phản hồi sau commit

After_commit có effect và receipt rồi mới ném lỗi. Reopen DB không mất receipt; cùng execute key trả chính result cũ. Proposal có unique receipt nên key khác không được execute lần hai. Cùng execute key nhắm proposal khác báo receipt_key_conflict. Local transaction không bao remote API, nên chưa có exactly-once trên downstream ngoài máy.

```python
import tempfile
from pathlib import Path
from pipeline import Engine,request
from store import Fault
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'recover.sqlite';e=Engine(path)
    pid=e.ask('demo-agent-a',request(action='create',key='draft'))['proposal_id']
    e.review('demo-reviewer-a',pid)
    try:
        e.commit('demo-agent-a',pid,'write',fault='after_commit')
    except Fault as err:
        assert err.code=='injected_lost_response'
    e.close();e=Engine(path)
    a=e.commit('demo-agent-a',pid,'write');b=e.commit('demo-agent-a',pid,'write')
    assert a==b and e.store.db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]==3
    assert e.store.db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]==1
    e.close()
```

## 10. Lỗi trước commit và pending run

Before_commit rollback ticket mutation, receipt, proposal executed transition và effect audit; proposal vẫn approved để retry nếu còn hợp lệ. After_commit không rollback sự thật đã commit chỉ vì caller không nhận response.

Pending run trước finish chưa có result, retry trả run_in_progress. Một thiết kế mở rộng có owner/lease, input hash và checkpoint/status để operator/recovery worker kiểm. Với remote effect, trước khi chạy lại cần status/receipt theo operation ID. Không dùng “lease hết” làm bằng chứng chắc chưa có side effect. Receipt recovery của local effect đã kiểm không phải full process-crash resume của mọi stage.

## 11. Release và quyền thay đổi

Đổi min_terms đổi manifest/release, nên một approved proposal của release cũ không tạo effect dưới release mới. Receipt cũ chỉ báo effect đã commit, không chạy lại; có thể đọc sau khi kiểm identity/owner. Actor disable phải chặn endpoint hiện tại; source revoke phải chặn answer replay và effect mới dựa nguồn đó. Fixture RAM không làm revoke bền vững qua Engine restart.

```python
import tempfile
from pathlib import Path
from pipeline import Engine,request
from store import Fault
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'release.sqlite';e=Engine(path,min_terms=2)
    p1=e.ask('demo-agent-a',request(action='create',key='one'))['proposal_id']
    e.review('demo-reviewer-a',p1);done=e.commit('demo-agent-a',p1,'one-effect')
    p2=e.ask('demo-agent-a',request(action='create',key='two'))['proposal_id']
    e.review('demo-reviewer-a',p2);e.close()
    newer=Engine(path,min_terms=1)
    assert newer.commit('demo-agent-a',p1,'one-effect')==done
    try:
        newer.commit('demo-agent-a',p2,'two-effect')
    except Fault as err:
        assert err.code=='release_changed'
    else:
        raise AssertionError('stale approval')
    newer.close()
```

## 12. Đọc metric fixture

Always abstain trả đúng 4 unanswerable trên 8 test ⇒ exact=.5, answer rate 0, conditional source accuracy không xác định, answerable recall 0. Selected baseline trả 4 answerable đúng source và abstain 4 câu còn lại⇒exact 1/answer rate=.5/conditional 1/answerable recall 1.

Chưa có LLM/human task study nên không biết semantic generation quality, incident resolution hay productivity. Holdout khó hơn có nhiều source cùng service, hard negatives, paraphrase cùng incident được group, future version, conflict, query dài/trộn scope và role changes; tách dev/test theo câu hỏi generalization thực sự cần đánh giá.

## 13. API thực hành

HTTPX ASGI dùng app thật trong process, không TCP/TLS hoặc load generation. Engine được tạo/đóng tường minh; nếu thêm lifespan-managed worker, ASGITransport không tự chạy lifecycle nên cần test startup/shutdown đúng. Ví dụ đầy đủ:

```python
import asyncio,tempfile
from pathlib import Path
import httpx
from pipeline import Engine,request
from api import create_app
async def example():
    with tempfile.TemporaryDirectory() as directory:
        e=Engine(Path(directory)/'http.sqlite');h={'Authorization':'Bearer demo-agent-a'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(e)),
                                     base_url='http://lesson45.local',trust_env=False) as client:
            answer=await client.post('/v1/runs',headers=h,json=request(key='ask'))
            assert answer.json()['state']=='answered'
            proposed=await client.post('/v1/runs',headers=h,json=request(action='create',key='create'))
            pid=proposed.json()['proposal_id']
            preview=await client.get(f'/v1/proposals/{pid}',headers={'Authorization':'Bearer demo-reviewer-a'})
            assert preview.json()['payload']['operation']=='create'
            approved=await client.post(f'/v1/proposals/{pid}/approve',headers={'Authorization':'Bearer demo-reviewer-a'})
            done=await client.post(f'/v1/proposals/{pid}/execute',headers=h,json={'request_key':'effect'})
            retry=await client.post(f'/v1/proposals/{pid}/execute',headers=h,json={'request_key':'effect'})
            assert approved.status_code==done.status_code==200 and done.json()==retry.json()
            assert (await client.post('/v1/runs',json=request(key='noauth'))).status_code==401
            assert (await client.get('/v1/tickets/2',headers=h)).status_code==404
            assert (await client.post('/v1/runs',headers=h,json=request(key='bad')|{'tenant':'b'})).status_code==422
        e.close()
asyncio.run(example())
```

Server TCP recipe có trong bài chính, nhưng socket bind của phiên biên soạn bị sandbox từ chối; không dùng kết quả ASGI để tuyên bố đã kiểm TCP hoặc production capacity.

## 14. Budget, effort và chi phí

Conditional chain .9 ×.95 ×.98=.8379. Không được nhân ba marginal metrics khác population rồi coi là end-to-end success. Effort 100 task=`100*(.6*2+.4*8)=440phút`; cần effort baseline và các lỗi/rework để tính saving. Giá .002+.3 ×.01=.005 là đơn vị giả định, không phải giá provider.

max_steps chỉ đếm stage logic. Nó không đo wall-clock/queue deadline, CPU/RAM, model tokens, provider billing, downstream quotas hoặc effort reviewer. Cần đo và kiểm từng ngân sách thật nếu thêm model/network/traffic.

## 15. Review DB và audit

Query join proposal/receipt cho binding operation; ticket pointer nằm trong JSON receipt, nên giải mã rồi lookup target. Dữ liệu body/query còn trong các bảng khác dù audit chỉ có kind/entity/actor/code. Cần phân quyền/retention toàn bộ artifacts và backup, không chỉ log mask. Digest không có key/chữ ký, không xác thực nguồn gửi.

```python
import json,tempfile
from pathlib import Path
from pipeline import Engine,request
with tempfile.TemporaryDirectory() as directory:
    e=Engine(Path(directory)/'audit.sqlite')
    pid=e.ask('demo-agent-a',request(action='create',key='draft'))['proposal_id']
    e.review('demo-reviewer-a',pid);e.commit('demo-agent-a',pid,'effect')
    row=e.store.db.execute('''SELECT p.state,p.payload_hash,r.result
        FROM proposals p JOIN receipts r ON p.id=r.proposal_id WHERE p.id=?''',(pid,)).fetchone()
    receipt=json.loads(row['result'])
    ticket=e.store.db.execute('SELECT tenant,status,version FROM tickets WHERE id=?',(receipt['ticket_id'],)).fetchone()
    assert row['state']=='executed' and ticket['tenant']=='a' and ticket['version']==1
    assert e.store.db.execute("SELECT COUNT(*) FROM audit WHERE kind='effect' AND entity=?",(pid,)).fetchone()[0]==1
    e.close()
```

Log tối thiểu có run/operation ID, actor/tenant theo scope phù hợp, stage/status/code, release/policy version và timestamps/latency khi đo; tránh token/raw secrets. Quyền xem log và correlation IDs cần thiết kế, hash của query cũng có thể nhạy cảm nếu không có threat model phù hợp.

## 16. Lập kế hoạch triển khai có bằng chứng

Ví dụ ADR thêm LLM answer để giải đáp paraphrase và tổng hợp nhiều paragraph. Giữ ACL/context/citation/action boundary; thêm adapter/schema, token/cost limits và eval semantic. So baseline bằng dev mới, giữ test mới tách incident/version, kiểm unsupported/conflicting/injected context. Chạy read-only shadow trước; không gửi writes “thử” tới ticket thật.

Nếu thay remote ticket adapter, cần status/idempotency downstream và fault test qua ranh giới mạng; local receipt không đủ. Staging writes có reviewer/scope thật, pilot nhỏ đo quality/cost/latency/review effort, rồi mới tăng phạm vi. Rollback có thể đổi model/config/traffic route, nhưng không tự đảo ticket đã commit, dữ liệu đã lộ hoặc migration không tương thích. Bằng chứng lab này chưa hoàn thành các bước rollout đó.
