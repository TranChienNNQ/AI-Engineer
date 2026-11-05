# Bài 22 — Lời giải

[Bài học](README.md) · [Bài tập](exercises.md) · [Phụ lục](deep-dive.md)

## 1. Hai API

Client gửi ticket theo contract nghiệp vụ. Server render prompt, gọi adapter và chuẩn hóa output. Tách adapter giúp không ép client phụ thuộc wire format/model config của một vendor. HTTP 200 upstream vẫn có thể có refusal, length hoặc output sai schema/task; ứng dụng phải kiểm envelope và candidate.

## 2. Required/null

Route hợp lệ: decision=route, team=NET, evidence_ids=[L 1], schema_version='1'. Abstain hợp lệ: decision=abstain, team=null, evidence_ids=[], schema_version='1'. Route+null, route+evidence rỗng, abstain+team NET đều không hợp lệ. Thiếu team khác null; lab yêu cầu key team ở cả hai nhánh. Nullable một mình không ràng buộc tương quan với decision.

## 3. Parse/schema

Root string `"NET"` là JSON nhưng sai root object. Object thiếu team và có extra action có thể parse được nhưng sai schema. Fence không là JSON theo parser lab. Duplicate key bị parser policy từ chối dù parser mặc định khác có thể giữ key cuối. Không cắt/sửa candidate bằng regex rồi che lỗi.

## 4. OneOf

Một object phải hợp lệ đúng một nhánh. Decision const route/abstain tách hai nhánh rõ ràng. Nếu thiết kế nhánh chồng lấn, một object có thể hợp lệ cả hai và bị oneOf loại; nếu không có quan hệ đủ chặt, mâu thuẫn có thể vẫn lọt qua. Snippet phụ lục kiểm JSON Schema và Python validator; file output-schema.json được so với schema sinh lại.

## 5. Semantic

NET, L 1 có thể qua completion, parse, schema và ID existence. Nhưng L 1 nói timeout đã được loại trừ và denied mới là event, theo reference phảiAUTH. Validator không có bộ suy luận semantic hoặc ground truth tại runtime; scorer fixture mới biết expected. Đừng gọi reference validity là entailment hoặc accuracy.

## 6. Finish

Length báo generation bị cắt theo envelope, không chắc nội dung đủ nhiệm vụ dù đoạn hiện tại parse được. Lab reject trước parse theo policy. Refusal là trạng thái upstream, không cùng abstain hợp lệ do contract nghiệp vụ trả về. Đổi mọi lỗi thành abstain sẽ làm mất khả năng phân biệt sự cố hạ tầng với thiếu dữ kiện và che availability/error rate.

## 7. Budget

Hai timeout:400+50+400=850 ms, sau đó hết 2 attempt. Rate-limit rồi thành công:20+max(50,100)+30=150 ms. Total 100 ms thì sau 20 ms không đủ backoff 100 ms; nên dừng deadline_exceeded, không reset budget. FakeClock làm phép tính này kiểm được mà không sleep thật.

## 8. Retry

Lab retry timeout, transient, rate_limit khi cònattempt/budget. Fatal không retry; schema, JSON, length, refusal cũng không tự retry. Repair là call mới với prompt/context khác, có thể đổi semantic và chi phí, nên cần policy/eval riêng. Hai lớp mỗi lớp 3 attempt có thể thành 9 call. Timeout của client không đồng nghĩa provider đã hủy hoặc không tính phí.

## 9. Idempotency

Request ID chủ yếu để trace, không tự là khóa dedup nghiệp vụ. Endpoint chỉ đề xuất không ghi dữ liệu; nếu thêm ghi assignment cần xử lý idempotency/transaction/state và quyền ở executor. Timeout sau khi upstream ghi xong nhưng response mất vẫn có thể tạo side effect lặp khi retry. Không dựa cùngprompt hoặc temperature 0 để suy hành động chỉ một lần.

## 10. Metric

Acceptance 8/20=40%; task success 5/20=25%; accepted accuracy 5/8=62.5%; route coverage 7/20=35%; route accuracy 4/7≈57.1429%. Chúng khác mẫu số và mục đích. Chỉ báo accuracy trên accepted sẽ bỏ qua 12 case không có response hợp lệ. Abstain đúng được tính task success nhưng không là route coverage.

## 11. Slice

Clean là subset dễ, chưa mô tả semantic/fault. Cần bảng slice, n, đúng/accepted và metric tổng; không suy 100%clean là 100%hệ thống. ERROR là không có output accepted; ABSTAIN là quyết định hợp lệ của ứng dụng. Gộp hai loại làm sai confusion, coverage và triage lỗi.

## 12. Cost/latency

23/20=1.15 attempt/case, trung bình thêm 15%call so với một attempt mỗi case. Không tự là 15%chi phí tiền vì mỗi call có thể dài/giá/usage khác nhau. Usage thiếu phải báo unknown/partial; FakeClock là duration fixture, không đo CPU/network, không đại diện p 95 production. Muốn có latency thật phải đo boundary và workload thật.

## 13. Test/eval

Unit kiểm parser/schema; integration kiểmHTTP và fault orchestration; model eval cần output model thật trên dữ liệu/cấu hình được công bố; end-to-end đo cả pipeline theo task. Fixture suite mở không là holdout chưa xem. Giữ cùngincident/hội thoại/tài liệu qua mọi split/training, không tách bản sao gần trùng sang test. Scorer biết reference, provider không được nhậnreference khi đo.

## 14. Rubric

Ví dụ email hỗ trợ: (1) đúng trạng thái đơn/không bịa hoàn tiền, (2) đủ bước tiếp theo, (3) giọng lịch sự, dễ hiểu. ID, số tiền, schema và từ cấm cụ thể có thể kiểm bằng rule khi có dữ liệu chuẩn; độ đủ/giọng văn cần rubric/anchor và đối chiếu người chấm. Judge có thể sai hoặc thiên lệch, nên cần đo agreement/ổn định. Confidence 0.9 tự viết chưa có calibration thì không suy 90%đúng.

## 15. Endpoint

Đoạn độc lập dùng ASGI TestClient và clock giả, không mở mạng ngoài:

```python
from fastapi.testclient import TestClient
from api import create_app
from service import FakeClock, ScriptedProvider, Event, Reply, proposal_json
payload = {'schema_version':'1','ticket_id':'api-check','lines':[{'id':'L1','text':'event=timeout.'}]}

def client_for(events):
    return TestClient(create_app(lambda req,clock: ScriptedProvider(events,clock),FakeClock))

with client_for([Event(reply=Reply(proposal_json()))]) as client:
    assert client.post('/v1/route-proposals',json=payload).status_code == 200
    assert client.post('/v1/route-proposals',json={**payload,'extra':'x'}).status_code == 422
    assert '/v1/route-proposals' in client.get('/openapi.json').json()['paths']
with client_for([Event(reply=Reply(proposal_json('NET',['L9'])))]) as client:
    response = client.post('/v1/route-proposals',json=payload)
    assert response.status_code == 502
    assert response.json()['error'] == 'unknown_evidence'
with client_for([Event(kind='transient'),Event(kind='transient')]) as client:
    assert client.post('/v1/route-proposals',json=payload).status_code == 503
with client_for([Event(duration_ms=500),Event(duration_ms=500)]) as client:
    assert client.post('/v1/route-proposals',json=payload).status_code == 504
```

Schema-valid wrong-team vẫn có thể trả 200. Đó là giới hạn chủ đích, không được sửa evaluator để mọi accepted đều thành correct. Test bổ sung extra field bằng candidate phải 502, extra field trong request phải 422; hai lỗi ở hai hợp đồng khác nhau.

## 16. Eval model thật

Giữ fault suite làm regression deterministic. Tạo quality dataset riêng với ID/group/slice/reference, rà nhãn và tách dev/validation/test. Adapter model thật nhận request đã render, không nhận expected. Công bố model/prompt/schema/scorer version, decode policy, số lần lặp và dataset hash; chọn config bằng validation, đo holdout sau khi chốt.

Báo syntax/schema/reference/task, coverage/error, các slice, latency/usage thật với phần missing rõ ràng. So rule hoặc prompt baseline trên cùng case; đọc regression theo từng case, không chỉ một mean. Với task mở dùng rubric phù hợp và đối chiếu human/judge, có kiểm scorer. Đề bài không yêu cầu gọi provider trả phí; thiết kế rõ và code adapter interface đã đủ cho phần mở rộng chưa chạy. Không bịa kết quả model để lấp chỗ trống.
