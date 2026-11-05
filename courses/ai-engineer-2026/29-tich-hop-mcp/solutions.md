# Bài 29 — Lời giải

[← Bài chính](README.md) · [Bài tập](exercises.md)

Snippet chạy độc lập từ thư mục bài với requirements.txt. Đây là dữ liệu/process tự tạo, không cấu hình ứng dụng MCP bên ngoài.

## 1. Kiến trúc

Host giữ task/model/context/policy và client manager. Client chuyển request theo protocol; server bọc backend, thực thi kiểm resource/workspace trước đọc. Agent state/checkpoint vẫn thuộc host hoặc ứng dụng được thiết kế riêng.

Server không bắt buộc có LLM; server.py chỉ có Python/SDK/domain. REST có thể là downstream, nhưng muốn nói MCP cần đúng message semantics/discovery/metadata/primitives theo revision. Không phải đặt tên endpoint `/mcp` là tự đạt.

## 2. Các version

2026-07-28 là revision protocol hiện hành khi biên soạn; 2025-11-25 thuộc đời handshake; 2.3.0 là version package. Client auto của bài nhận modern; forced legacy trên cùng server nhận 2025-11-25. Chưa kiểm toàn bộ server cũ hoặc auto fallback trong mọi lỗi/môi trường.

Modern per-request metadata không cần initialize trên mọi connection. Pin version không là discovery; capability/software name không là auth. Nên log version thực dùng, không đoán từ ngày package được cài.

## 3. Tools/resources/prompts

get_ticket là tool: tools/list rồi tools/call. Catalog là concrete resource: resources/list/read. Runbook URI template ở resources/templates/list, rồi read URI cụ thể. Triage prompt được prompts/list/get; output là các message template, không tự gọi tool.

```python
import asyncio
from host import connect
async def main():
    async with connect('memory') as (client,_):
        resources=await client.list_resources()
        templates=await client.list_resource_templates()
        prompts=await client.list_prompts()
        assert [str(r.uri) for r in resources.resources]==['kb://catalog']
        assert [str(r.uri_template) for r in templates.resource_templates]==['kb://runbooks/{doc_id}']
        assert [p.name for p in prompts.prompts]==['triage_ticket']
asyncio.run(main())
```

Prompt thường do user/workflow chọn; host quyết định mức thẩm quyền và cách đặt vào conversation. Một body tự ghi role=system không đổi role metadata của message.

## 4. Wire và correlation

```python
import asyncio
from wire import wire_demo
async def main():
    r=await wire_demo();rows=r['transcript']
    assert len(rows)==13 and r['exit_code']==0
    assert all(x['request']['id']==x['response']['id'] for x in rows)
    assert all('io.modelcontextprotocol/protocolVersion' in x['request']['params']['_meta'] for x in rows)
    rpc_errors=[x for x in rows if 'error' in x['response']]
    tool_errors=[x for x in rows if x['response'].get('result',{}).get('isError')]
    assert len(rpc_errors)==2 and len(tool_errors)==2
    assert len(rows)-len(rpc_errors)-len(tool_errors)==9
asyncio.run(main())
```

Method course/unknown_method không tồn tại trả -32601. Tools/call là RPC method hợp lệ nhưng name=missing tool được bản SDK này trả tool error result. Unsupported version -32022; request version hợp lệ kế tiếp vẫn dùng được. ID không là tenant/run/approval token.

Observer chưa xử lý notifications xen kẽ, concurrent pending IDs, MRTR, cache hoặc subscriptions. Chỉ cần server gửi notification trong lúc nó chờ response là giả định “dòng tiếp theo chính là response” có thể sai; production dùng SDK dispatcher.

## 5. Số tính tay

12 và 7 theo sơ đồ N×M/N+M lý tưởng, chưa đo công sức. Mapping backend, auth, intent, schema version, testing, UX và vận hành vẫn cần làm. MCP giảm phần giao thức tự viết, không xóa nghiệp vụ.

110 ms chỉ là cộng duration giả định của 6 request liệt kê; chưa có prompt render/host checks/SDK extra calls/model/network queue. Không dùng nó làm latency benchmark hoặc quảng cáo MCP nhanh hơn REST.

## 6. Stdio thật

```python
import asyncio
from agent import demo
async def main():
    r=await demo('stdio')
    assert r['transport']=='stdio' and r['protocol_version']=='2026-07-28'
    assert r['ticket']['data']['ticket_id']=='T-BLUE-1'
    assert len(r['documents'])==2
asyncio.run(main())
```

SDK client launch server.py bằng Python hiện tại và args đã biết, qua stdin/stdout pipe; context manager đóng stdin và chờ/cleanup child. Chạy server.py một mình chờ request không in gì là bình thường. Stdio không cần port, nhưng process vẫn cần OS permission/sandbox phù hợp; nó không tự an toàn chỉ vì không có TCP.

Logging tới stderr giữ stdout là JSON-RPC line framing. Command từ model hoặc cấu hình không tin cậy có thể là arbitrary code execution; lab chỉ chạy source đi kèm đã được yêu cầu biên soạn/kiểm tra.

## 7. HTTP-ASGI

```python
import asyncio
from agent import demo
async def main():
    local=await demo('memory')
    http=await demo('http-asgi')
    assert http['protocol_version']==local['protocol_version']=='2026-07-28'
    assert http['ticket']==local['ticket']
    assert http['sources']==local['sources']
asyncio.run(main())
```

ASGI kiểm HTTP request/endpoint/serialization với SDK thật, nhưng không dùng socket, DNS, TLS, reverse proxy hoặc OAuth. URL loopback trong test được ASGITransport xử lý trong process, không là service đang listen trên port 8000.

Streamable HTTP mới dùng POST, JSON hoặc SSE response theo request; SSE cũng có thể phục vụ subscription. Không đồng nhất với legacy HTTP+SSE `/sse`, GET stream hoặc protocol-level session. Optional server CLI có thể bind local để người học thử, nhưng không nằm trong phạm vi socket verification của bài.

## 8. Đóng input đúng tầng

```python
import asyncio
from host import connect,GuardedHost,HostError
async def main():
    async with connect('memory') as (client,backend):
        host=GuardedHost(client);await host.review_catalog()
        cases=[({'request':{'ticket_id':'T-BLUE-1'},'actor':'carol'},'invalid_outer_arguments'),
               ({'request':{'ticket_id':'T-BLUE-1','actor':'carol'}},'invalid_arguments')]
        before=len(backend.reads)
        for args,expected in cases:
            try:await host.call('backend_blue__get_ticket',args)
            except HostError as e:assert e.code==expected
            else:raise AssertionError('Invalid args accepted by host')
        assert len(backend.reads)==before
asyncio.run(main())
```

Nested request là closed/strict Pydantic; limit bool/out of 1..3 bị host/schema chặn. Direct SDK có thể bỏ outer actor thay vì reject; callback vẫn dùng workspace server đã cấu hình, không dùng actor đó. Host tự đóng outer args, không suy từ type hints rằng mọi tầng đều extra=forbid.

Schema đúng vẫn có thể request resource khác quyền. Server workspace guard cần kiểm riêng dù một host hợp lệ có pinned task chặt hơn.

## 9. Output và ý nghĩa

Wire structuredContent/isError/outputSchema dùng camelCase; Python structured_content/is_error/output_schema dùng snake_case. Content có thể là text/media; structured output cho code vẫn cần schema và semantic identity checks.

Wrong_ticket fault có shape hợp lệ nhưng khác ticket đã pin: host wrong_result_resource. Tool error không dùng như kết quả thành công. Hash so text nhất quán với digest trả kèm, không là chữ ký nguồn và không kiểm entailment/root cause; một server có thể tự trả cả text sai và hash tương ứng.

## 10. Nguồn và version

```python
import asyncio
from host import connect,GuardedHost,HostError
async def main():
    async with connect('memory') as (client,backend):
        host=GuardedHost(client);await host.review_catalog()
        await host.call('backend_blue__search_runbooks',{'request':{'query':'retry','limit':3}})
        doc=await host.read_document('kb://runbooks/RB-BLUE-RETRY')
        assert doc['authority']=='data' and doc['data']['version']==1
        backend.docs['RB-BLUE-RETRY']['version']+=1
        try:await host.read_document('kb://runbooks/RB-BLUE-RETRY')
        except HostError as e:assert e.code=='resource_snapshot_changed'
        else:raise AssertionError('Stale selected source accepted')
asyncio.run(main())
```

Unselected URI bị host chặn trước read. Đổi workspace ở server làm read bị từ chối dù URI đã được chọn; select không phải grant vĩnh viễn. Post-receive size cap chỉ kiểm app payload đã nhận, chưa hạn chế frame/stream/bandwidth trước khi cấp phát; production cần limit phù hợp ở các tầng transport/proxy/client.

## 11. Routing và drift

```python
import asyncio
from mcp import Client
from server import build_server
from host import GuardedHost,merge_descriptors,HostError
async def main():
    sa,_=build_server('blue');sb,_=build_server('green')
    async with Client(sa) as ca,Client(sb) as cb:
        a=GuardedHost(ca,alias='blue');b=GuardedHost(cb,alias='green',ticket_id='T-GREEN-1')
        await a.review_catalog();await b.review_catalog()
        assert ca.server_info.name==cb.server_info.name
        assert len({d['name'] for d in merge_descriptors([a,b])})==4
        sa.remove_tool('search_runbooks')
        try:await a.refresh()
        except HostError as e:assert e.code=='catalog_changed_review_required'
        else:raise AssertionError('Catalog drift ignored')
asyncio.run(main())
```

Hash giữ descriptor đã review, không chứng minh code backend không đổi. Description/annotations cũng là dữ liệu external; một hash của một mô tả xấu không làm nó tốt. Host đang poll tools/list; chưa subscribe listChanged và không duyệt pagination, gặp cursor thì báo lỗi rõ.

## 12. Lỗi và observability

Outside_task là host policy, local timeout do host timer, unsupported protocol -32022 ở peer, not_permitted trong tool là isError. Prompt thiếu args ở SDK ghim là -32603 dù ví dụ docs hiện tại -32602. Unknown tool cũng là tool error result của bản ghim dù spec minh họa protocol error.

Ghi đúng khác biệt để tránh gán nhầm local/peer/layer; không lấy behavior SDK làm định nghĩa chuẩn hoặc công bố conformance. Client nên dựa vào envelope/type và chủ động normalize cho workflow, giữ thông tin layer cần thiết. Không parse tùy tiện error message string như một schema nghiệp vụ ổn định.

## 13. Server kiểm độc lập host

```python
import asyncio
from host import connect
async def main():
    async with connect('stdio','blue') as (client,_):
        r=await client.call_tool('get_ticket',{'request':{'ticket_id':'T-GREEN-1'}})
        assert r.is_error and r.structured_content is None
        missing=await client.call_tool('get_ticket',{'request':{'ticket_id':'T-MISSING-9'}})
        assert missing.is_error
asyncio.run(main())
```

Cả cross-workspace và missing trả not_permitted ở domain path, không trả body/marker riêng tư. Workspace chọn ở deployment là fixture service account, không thay OAuth/authn nhiều user. ClientInfo/serverInfo/URI/handle không là credential; không dùng model args actor để xác thực.

## 14. Injection và residual

Poison resource có thể được recall vì đúng workspace/relevance và còn budget. Host gắn data và chỉ có allowlist read tools; không parse body thành command. Prompt là template, không tự nâng role. Policy Python không làm theo language instruction như LLM nên chưa là ASR test.

Residual BANANA_OVERRIDE được ghi blocked=False, semantic_correct=False; lab không có verifier hoặc model judge. Cần đo task quality/grounding và adversarial outcomes riêng nếu thêm LLM. Không lấy “MCP exchange hoàn tất” làm “câu trả lời đúng”.

## 15. Đếm đúng operation

5 ops thuộc workflow demo: render prompt, get ticket, search, đọc 2 nguồn. Review/refresh catalog, capability discovery và các list primitives không nằm trong counter này. Wire observer là một chương trình khác với 13 request cố định, gồm cả negative cases; hai mẫu số không mâu thuẫn.

Trace nên giữ trusted alias, remote tool/method, request ID khi phù hợp, status/layer, latency và version. Credential/raw source không nên đổ vào log tùy tiện. Request ID không là idempotency key; nếu write timeout cần stable business intent/key, receipt/reconciliation và review binding như bài 27, không blind retry.

## 16. Tiêu chí dự án

Người khác chạy lại được discovery/list/call/resource/prompt qua transport đã nêu; xác định server/client version; thấy host và server denial riêng, resource version và catalog drift. Ghi rõ API nào chỉ giải thích và chưa triển khai, như OAuth/MRTR/subscription/Tasks.

Nếu thêm LLM, giữ adapter/gateway và đo thêm tool selection/args/task quality, false refusal, source support, memory/permission behavior. Không dùng 33 deterministic checks như 33 lần model thành công; không tự cài server bên thứ ba hoặc mở dữ liệu thật để chứng minh bài học.
