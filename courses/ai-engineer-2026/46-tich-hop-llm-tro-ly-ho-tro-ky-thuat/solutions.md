# Lời giải bài 46 — Tích hợp LLM vào trợ lý hỗ trợ

[← Bài chính](README.md) · [Đề bài](exercises.md) · [Đào sâu](deep-dive.md)

Mỗi đoạn Python độc lập, chạy từ thư mục bài 46 trong process sạch. Không có đoạn Python nào dưới đây gọi model thật hoặc cần credential/network.

## 1. Xác định trách nhiệm của model

User request có action/service/message/target/version; backend auth/ACL/retrieval→evidence→model candidate→parse/ground→answer hoặc proposal→review→gateway/receipt. Model viết claim text/quote/evidence ID và title/body draft; không viết tenant/role/operation/ticket_id/expected_version/approved.

UI báo effect succeeded từ committed receipt, không từ câu model nói đã làm. Generated prose có thể bịa ticket ID hoặc nhầm trạng thái. Trong flow hiện tại, model còn chưa thấy execute result; nó không có cơ sở xác nhận action đã xảy ra.

## 2. Phân biệt fixture, smoke và evaluation

Fixture kiểm handler/contract/fault path theo output đã đặt. Live smoke xác minh ít nhất một account/model/request có thể hoạt động ở thời điểm chạy, chưa đo population quality. Evaluation có dataset/rubric/split/repeats phù hợp mới hỗ trợ chọn model và định lượng failure modes.

12 cases là test vector người biên soạn, không samples từ LLM. Nhánh live còn cần account/model access, strict schema acceptance, actual envelope/usage, factuality, cost và latency. Một HTTP 200 không bảo đảm status completed, schema hợp lệ hoặc answer đúng.

## 3. Xem prompt và schema

User payload chỉ có query/intent_ceiling/service/support_policy/evidence; mỗi evidence có id/text. Không có app bearer/tenant/role/target. Nếu query tự chứa sensitive text thì nó vẫn được gửi; đây chưa là PII filter. Nullable draft luôn có key với object hoặc null; provider wire schema required, local Pydantic strict chặn extra/coercion.

```python
import json
from bridge45 import documents,TOKENS,retrieve,request,Fault
from contracts import prompt_input,parse
from fixtures import reply_for
req=request();evidence,_=retrieve(documents(),TOKENS['demo-agent-a'],'gateway',req['message'])
messages=prompt_input(req,evidence,'exact_quote');payload=json.loads(messages[1]['content'])
assert set(payload)=={'query','intent_ceiling','service','support_policy','evidence'}
reply=reply_for(payload)
for bad in [reply|{'tenant':'b'},reply|{'claims':[{'text':42,'quote':'x','evidence_id':'E1'}]}]:
    try:
        parse(json.dumps(bad))
    except Fault as err:
        assert err.code=='invalid_model_schema'
    else:
        raise AssertionError('invalid reply accepted')
```

## 4. Prompt injection và cấu trúc request

Serialization bảo vệ JSON/message structure: query không tự trở thành một message role mới trong HTTP body. Nó không bảo vệ ngữ nghĩa model khỏi instruction bên trong text. Một model vẫn có thể bị lừa chọn output không mong muốn.

Test boundary độc lập: trả function_call resolve_ticket hoặc extra authority từ transport fixture; adapter/schema phải chặn và DB receipt count không đổi. Với draft hợp lệ nhưng sai nội dung, chỉ được tạo proposal, chưa effect trước review. Không dùng “prompt đã nói đừng làm” làm lớp authorization duy nhất.

## 5. Tính citation offset ở server

Câu thứ hai của gateway nằm sau `. ` đầu, start là index trong source, end=start+len(quote). Hash vẫn là canonical full source text, không hash substring. Unique substring tránh ambiguity; model không tự chọn một URL/offset ngoài context. Revoke sau khi gửi context không thu lại bytes từ provider, nhưng backend có thể chặn answer/draft/effect mới.

```python
import json
from bridge45 import documents,TOKENS,retrieve,request
from contracts import prompt_input,parse,ground
from fixtures import reply_for
req=request();docs=documents();p=TOKENS['demo-agent-a']
evidence,_=retrieve(docs,p,'gateway',req['message'])
reply=reply_for(json.loads(prompt_input(req,evidence,'exact_quote')[1]['content']))
quote=docs[0]['text'].split('. ')[1]
reply['claims'][0].update(text=quote,quote=quote)
result=ground(parse(json.dumps(reply)),evidence,docs,p,req,set())
cite=result['citations'][0]
assert cite['start']==docs[0]['text'].find(quote)>0
assert docs[0]['text'][cite['start']:cite['end']]==quote
```

## 6. Schema đúng nhưng nội dung sai

Reply answer có claims đúng field/type, evidence_id E1 có thật, quote đúng runbook, nhưng text “retry mọi request ghi vô hạn”. Shape và citation vẫn hợp lệ; semantic support sai vì quote đặt điều kiện idempotency. Exact_quote chặn text khác quote, citation_only có thể cho qua.

Cần kiểm riêng schema, source/evidence identity/scope, và claim support/relevance/completeness. Exact_quote chưa kiểm source truth hoặc đã lấy đủ cảnh báo/đúng câu hỏi. Một source sai hoặc quote không liên quan vẫn có thể qua phép so string.

## 7. So hai support policy

Exact_quote chặn cả paraphrase lẫn unsupported; citation_only cho cả hai qua với semantic not_established. Từ suite: default TP=3/FP=0/FN=1⇒precision 1/recall .75; citation_only TP=4/FP=1/FN=0⇒precision .8/recall 1. Cùng 11/12 đúng quyết định nhưng chi phí false acceptance/rejection khác nhau. Không suy ra thống kê risk production từ các fixture này.

```python
import tempfile
from pathlib import Path
from bridge45 import request
from fixtures import mock_adapter
from support import SupportEngine
with tempfile.TemporaryDirectory() as directory:
    for policy in ['exact_quote','citation_only']:
        for event in ['paraphrase','unsupported']:
            adapter,script=mock_adapter([event])
            e=SupportEngine(Path(directory)/f'{policy}-{event}.sqlite',adapter,policy)
            out=e.ask('demo-agent-a',request(key='case'))
            assert (out['state']=='answered')==(policy=='citation_only')
            if policy=='citation_only':assert out['stats']['semantic_check']=='not_established'
            else:assert out['code']=='unsupported_synthesis'
            e.close()
```

## 8. Response envelope và trạng thái

HTTP 400 không retry;429/503 có thể retry nếu còn attempt/time budget. Incomplete không dùng partial output, refusal thành abstained, malformed JSON fail không repair loop, reasoning item không render. Function_call bất ngờ bị reject vì app không cấu hình dispatcher/tool loop này.

Không nối mọi output thành text vì có typed items như reasoning/tool call/refusal, nhiều structured blocks có thể không là một reply. Side effect cần output hoàn chỉnh/được validate và policy gateway, không được khởi chạy khi đang thấy từng mảnh token JSON.

## 9. Retry, deadline và usage

503→good:2 attempts, known 120 input/40 output fixture tokens,1 attempt unknown, fake backoff 50 ms. Hai timeout:2 unknown attempts, sum known 0 chỉ nghĩa chưa có usage đã biết, không nghĩa bill 0. Total .04 nhỏ hơn delay .05 nên không có attempt thứ hai. Provider clock guard chưa là hard API cancellation.

```python
from fixtures import mock_adapter
from llm_adapter import ProviderError
from contracts import prompt_input
from bridge45 import documents,TOKENS,retrieve,request
req=request();evidence,_=retrieve(documents(),TOKENS['demo-agent-a'],'gateway',req['message'])
messages=prompt_input(req,evidence,'exact_quote')
a,s=mock_adapter(['timeout','timeout'])
try:
    a.generate(messages)
except ProviderError as err:
    assert err.code=='provider_retry_exhausted' and err.meta['unknown_usage_attempts']==2
else:
    raise AssertionError('expected exhausted retries')
assert len(s.calls)==2;a.close()
a,s=mock_adapter([503,'good'],total_seconds=.04)
try:
    a.generate(messages)
except ProviderError as err:
    assert err.code=='provider_deadline'
else:
    raise AssertionError('expected deadline')
assert len(s.calls)==1;a.close()
```

## 10. Dự trù budget trước model call

Ask có evidence 4 stages, write 5, no-evidence 3. max_steps 3 cho ask bị chặn sau retrieval trước generate, không provider call. Context chars là nguồn được pack; input chars còn prompt/query/serialization; decoded response bytes là buffer phía client; max_output_tokens là budget model request. Không đổi trực tiếp các đơn vị này thành nhau hoặc thành bill.

Nếu tổng stage đủ nhưng network chậm, step counter chưa chặn được wall-clock. Nếu token cap nhỏ gây incomplete, không được chỉ tăng vô hạn hoặc execute partial response. Cần budget phù hợp model và evaluation của thay đổi đó.

## 11. Draft, review và receipt

Model draft chỉ trở thành proposal, DB vẫn 2 ticket seed. Reviewer đọc đúng title/body/operation/target/version/citation/release/hash rồi approve; gateway 45 execute, retry receipt và run replay không gọi model nữa. Draft không là fact đã xác minh, cần reviewer kiểm phù hợp nghiệp vụ. Version/idempotency nằm ở backend nên không biến mất khi thay generator.

```python
import tempfile
from pathlib import Path
from bridge45 import request,Fault
from fixtures import mock_adapter
from support import SupportEngine
with tempfile.TemporaryDirectory() as directory:
    adapter,script=mock_adapter();e=SupportEngine(Path(directory)/'draft.sqlite',adapter)
    out=e.ask('demo-agent-a',request(action='create',key='draft'));pid=out['proposal_id']
    assert e.store.db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]==2
    preview=e.read_proposal('demo-reviewer-a',pid)
    assert preview['payload']['operation']=='create'
    try:
        e.commit('demo-agent-a',pid,'write')
    except Fault as err:
        assert err.code=='approval_required'
    else:
        raise AssertionError('unreviewed effect')
    e.review('demo-reviewer-a',pid);done=e.commit('demo-agent-a',pid,'write')
    assert e.commit('demo-agent-a',pid,'write')==done and len(script.calls)==1
    e.close()
```

## 12. Function calling khác structured draft ở đâu?

App gửi tools→model trả function call→app kiểm/thực thi→gửi tool result gắn call_id→model trả tiếp. Lab dùng text JSON claims/draft, không đăng ký provider tools; function call bất ngờ bị chặn. Không tự cấp quyền cho tên hàm model bịa.

Muốn thêm dispatcher: allowlist/name+args schema, principal/current permission, target/version, review binding, call/step/token/time budgets, tool call ID/receipt mapping và stop conditions. Tool result là observation có provenance, không phải instruction có quyền tăng scope. Xử lý unknown effect khi timeout cần status/idempotency downstream, không để model tự kết luận đã failed rồi retry key mới.

## 13. API-ASGI thực hành

Envelope 45 đã có answer/citations/state/trace/stats; model metadata thêm trong stats nên không cần thay mọi client. Request operation/target vẫn theo schema 45. ASGI kiểm app in-process, không TCP/TLS/proxy/client latency và không remote provider.

```python
import asyncio,tempfile
from pathlib import Path
import httpx
from bridge45 import request,RunOut
from fixtures import mock_adapter
from support import SupportEngine,app_for
async def example():
    with tempfile.TemporaryDirectory() as directory:
        adapter,script=mock_adapter();e=SupportEngine(Path(directory)/'api.sqlite',adapter)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(e)),
                                     base_url='http://lesson46.local',trust_env=False) as client:
            h={'Authorization':'Bearer demo-agent-a'}
            a=await client.post('/v1/runs',headers=h,json=request(key='ask'))
            b=await client.post('/v1/runs',headers=h,json=request(key='ask'))
            assert a.json()==b.json() and RunOut.model_validate(a.json()).state=='answered'
            assert len(script.calls)==1
            assert (await client.post('/v1/runs',json=request(key='noauth'))).status_code==401
            assert (await client.post('/v1/runs',headers=h,json=request(key='bad')|{'tenant':'b'})).status_code==422
        e.close()
asyncio.run(example())
```

Nếu sau này thêm lifespan workers/provider client lifecycle, cần test startup/shutdown đúng thay vì giả ASGITransport tự chạy lifecycle. Prototype hiện tạo/đóng Engine/adapter tường minh.

## 14. Đọc kết quả và thiết kế model evaluation

Paraphrase được tác giả gắn nhãn supported nhưng exact_quote chặn khác chuỗi; unsupported được citation_only serve vì chưa semantic verifier. Known 1320/440 là số fixture, có 1 attempt unknown, nên không model bill thật và cũng chưa full bill ngay cả nếu counts lấy từ provider.

Thiết kế live eval có query domain đã được phép dùng, answerable/unanswerable/conflict/injection cases, group split theo incident/source/version, dev chọn prompt/model/policy và holdout mới. Chốt rubric đúng/đủ/có nguồn/hữu ích, human-check một subset và audit judge errors. Repeats cho variability, nhưng báo cả cost/failure của mọi sample; không cherry-pick một output đẹp rồi báo như single-run success.

## 15. Release và reopen DB

Generated run/receipt lưu bền, reopen cùng release trả output cũ với 0 call provider. Prompt/schema/model/policy/config/source thay đổi phải có release mới. Run 46 hash gồm release nên key legacy 45 không được giả current generation; receipt old committed vẫn báo sự thật cũ theo scope. Hash không bảo đảm deterministic generation hoặc authenticated publisher.

```python
import tempfile
from pathlib import Path
from bridge45 import request
from fixtures import mock_adapter
from support import SupportEngine
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'durable.sqlite'
    a,s=mock_adapter();e=SupportEngine(path,a)
    req=request(action='create',key='draft');out=e.ask('demo-agent-a',req);pid=out['proposal_id']
    e.review('demo-reviewer-a',pid);receipt=e.commit('demo-agent-a',pid,'effect');e.close()
    a,s=mock_adapter(['unsupported']);e=SupportEngine(path,a)
    assert e.ask('demo-agent-a',req)==out and e.commit('demo-agent-a',pid,'effect')==receipt
    assert not s.calls
    e.close()
```

Receipt replay không resample model. Nếu muốn regenerate, dùng run mới và làm rõ đây là ứng viên khác; không sửa âm thầm draft đã được approve của proposal cũ.

## 16. Kế hoạch live có thể review

Ví dụ smoke: model tài khoản có quyền dùng Responses+strict schema, corpus synthetic,1 câu có nguồn/1 câu no-evidence; default exact_quote, max 2 attempts/output 768 tokens/provider budget 10 s. Ghi requested/returned model, response ID/status/usage khi có, schema/grounding result và output được chấp nhận; không raw key/error body. No-evidence nên bypass provider.

Chưa có live call ở bài này; smoke tương lai cần kiểm server-side model/schema/access và data policy thực tế. Sau smoke còn thiếu evaluation đủ đại diện/repeats, human relevance/support/draft review, usage/latency thật và remote ticket adapter faults nếu thêm downstream. Staging/pilot có scope và rollback rõ; hash/review/local receipt không thay tất cả các bằng chứng đó.
