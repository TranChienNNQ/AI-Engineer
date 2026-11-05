# Bài 22 — Đào sâu contract, adapter và eval

[Bài học](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## A. Schema là đặc tả tập giá trị hợp lệ

Một field có type đúng chưa đủ làm cả object hợp lệ. Ở nhánh route, team phải khác null và evidence phải có phần tử; ở abstain, team=null và evidence rỗng. Nếu chỉ khai báo team nullable mà không có quan hệ với decision, bạn cho phép các object mâu thuẫn.

`Proposal` có cùng quan hệ ở Python validator và JSON Schema oneOf để cả validation phía server lẫn schema xuất ra mô tả được quy tắc. Đây là hai biểu diễn có thể lệch nếu sửa thiếu một bên; bộ kiểm cần thử cả hai và so file schema lưu với schema sinh lại. Required team và nullable team không mâu thuẫn: key bắt buộc có mặt, giá trị được phép null ở nhánh phù hợp.

```python
from contracts import OUTPUT_VALIDATOR, Proposal
from pydantic import ValidationError
cases = [
    ({'schema_version':'1','decision':'route','team':'NET','evidence_ids':['L1']}, True),
    ({'schema_version':'1','decision':'abstain','team':None,'evidence_ids':[]}, True),
    ({'schema_version':'1','decision':'route','team':None,'evidence_ids':['L1']}, False),
    ({'schema_version':'1','decision':'abstain','team':'NET','evidence_ids':[]}, False),
    ({'schema_version':'1','decision':'route','team':'NET','evidence_ids':['L1','L1']}, False),
]
for obj, expected in cases:
    assert OUTPUT_VALIDATOR.is_valid(obj) == expected
    try:
        Proposal.model_validate(obj)
        accepted = True
    except ValidationError:
        accepted = False
    assert accepted == expected
print('PASS Python/schema agreement')
```

Schema file và prompt version nên được version cùng quy trình review của ứng dụng. Nếu đổi thêm field bắt buộc hoặc đổi ý nghĩa enum, client cũ có thể không còn tương thích dù vẫn là JSON. Không sửa âm thầm một contract rồi giữ nguyên version chỉ vì endpoint path chưa đổi.

## B. JSON parser policy cũng là contract

Python json có thể giữ giá trị cuối của key lặp và chấp nhận hằng số không chuẩn ở chế độ mặc định; lab chủ động dùng hooks để từ chối chúng. Một regex lấy nội dung giữa dấu ngoặc đầu/cuối có thể sai khi có object lồng, ngoặc trong string hoặc nhiều object. Không biến regex extraction thành parser JSON tổng quát.

Whitespace JSON bên ngoài object vẫn hợp lệ. Markdown fence lại là ký tự ngoài grammar JSON của contract này. Nếu sản phẩm chọn hỗ trợ fence, phải xác định rõ format, giữ raw output và đo tác động; lab không tự sửa để tránh che lỗi upstream.

Giới hạn raw candidate 16,000 byte được kiểm trước parse; độ dài field được kiểm sau parse. Đó không là tổng request-body limit hoặc bảo đảm chống mọi tài nguyên xấu ở tầng mạng. Trong lab, schema và reference của validator được tạo từ code tin cậy, không nhận schema tùy ý từ người dùng để validator tự truy cập tài nguyên ngoài.

## C. Adapter cần chuẩn hóa điều gì khi có provider thật?

Giao diện nội bộ tối thiểu của bài là `complete(prompt, timeout_ms) -> Reply`. Adapter thật cần ánh xạ riêng cho dịch vụ đang dùng:

- Kết quả text/parsed object và raw envelope theo khả năng của API.
- Trạng thái hoàn thành, refusal, cắt vì output/context limit và lỗi transport.
- Usage/token counters có sẵn, trường hợp thiếu và cách provider báo retry-after.
- Model/version, timeout của HTTP client, retry policy của SDK nếu có.

Không để cả SDK và application đều retry mà không tính số lần gọi nhân lên. Hai lớp mỗi lớp tối đa 3 attempt có thể thành 9 call cho một request. Điều đó không được nhìn thấy nếu chỉ log application attempts. Hãy chọn một nơi chịu trách nhiệm hoặc công bố rõ ngân sách của cả hai.

Timeout của caller không tự khẳng định provider đã dừng tính hoặc không tính phí. Bài dùng clock giả để kiểm state machine; không có cancellation token xuyên tới một model server thật. Field usage_complete=False trên success sau retry chỉ nói thông tin usage chưa đầy đủ, không là cờ tác vụ sai.

## D. Kiểm retry bằng fake clock

Thay vì sleep thật 850 ms ở mọi lần test, FakeClock chỉ tăng giá trị khi fixture mô phỏng duration/backoff. Cùng input cho trace giống nhau, giúp kiểm boundary deadline nhanh. Nó không đo thời gian CPU hoặc latency mạng thật.

```python
from contracts import RouteRequest
from service import FakeClock, ScriptedProvider, Event, Reply, Policy, execute, proposal_json, ServiceFailure
request = RouteRequest(schema_version='1',ticket_id='retry',lines=[{'id':'L1','text':'event=timeout.'}])
clock = FakeClock()
provider = ScriptedProvider([
    Event(kind='rate_limit',duration_ms=20,retry_after_ms=100),
    Event(reply=Reply(proposal_json()),duration_ms=30)
],clock)
result, trace = execute(request,provider,clock)
assert result.metrics.attempts == 2
assert result.metrics.elapsed_ms == 150.
assert not result.metrics.usage_complete

clock = FakeClock()
provider = ScriptedProvider([Event(kind='transient',duration_ms=20,retry_after_ms=100)],clock)
try:
    execute(request,provider,clock,Policy(total_ms=100))
except ServiceFailure as exc:
    assert exc.code == 'deadline_exceeded'
    assert provider.calls == 1
else:
    raise AssertionError('Deadline was reset or ignored')
```

Đây là test tổng budget và không gọi attempt mới khi backoff đã vượt deadline. Nó không chứng minh một adapter blocking không tuân thủ timeout sẽ bị hủy. Khi nối adapter thật, cần integration test riêng với timeout/cancel ở transport và boundary end-to-end.

## E. Evidence tồn tại không đồng nghĩa evidence hỗ trợ

Input có L 1 ghi “timeout đã được loại trừ; event=denied”. Candidate trả NET, evidence_ids=[L 1]. Schema và kiểm ID đều pass, nhưng nhãn không được hỗ trợ theo reference AUTH của ví dụ.

Có ít nhất ba phép đo khác nhau:

1. Citation/reference validity: ID có tồn tại trong dữ liệu request được phép dùng không?
2. Support/entailment: nội dung được dẫn có hỗ trợ claim/nhãn không?
3. Task correctness: câu trả lời có giải quyết nhiệm vụ theo ground truth/rubric không?

Lab chỉ tự động validate mục 1 và scorer có nhãn expected để đo mục 3; không có mô hình entailment hoặc RAG retrieval. Trong hệ thống thật, một nguồn tồn tại nhưng đã cũ/ngoài quyền cũng có thể không phù hợp. Các bài RAG/phân quyền sau sẽ nối thêm các ràng buộc đó.

## F. Một eval model thật cần khác fixture suite thế nào?

Fixture suite mở để kiểm branch, có nhiều ca lỗi cố ý và phân phối label lệch. Vì vậy tỷ lệ 25% task success ở đây không là ước lượng về người dùng thực, không có confidence interval được suy như mẫu ngẫu nhiên của production.

Với model thật, tạo dataset theo nhóm nguồn/incident, có reference được review và tách phần development khỏi holdout. Với task mở, viết rubric từng tiêu chí và ví dụ anchor. Judge khác model được đánh giá vẫn có thể chia sẻ lỗi, thiên lệch hoặc bị nội dung output ảnh hưởng; không coi việc đổi tên model judge là đủ độc lập.

So phiên bản A/B trên cùng case. Giả sử A đúng 16/20, B đúng 17/20: chênh lệch tổng là một case, nhưng cần xem case nào cải thiện/case nào regression, số lần lặp và mức quan trọng của lỗi. Case hiếm nhưng nghiêm trọng có thể cần gate riêng; không gộp mọi thứ và o một score đẹp.

Có thể báo expected cost theo token/call thực và latency thực theo boundary đo. Nếu retry timeout không trả usage, báo missing/known subtotal thay vì bịa 0. Nếu muốn tính chi phí tiền, cần giá và quy tắc billing hiện hành của provider thực; lab không dùng giá dịch vụ nào.

## G. Auth, idempotency và tác động bên ngoài

API localhost trong bài chưa có authentication/authorization hoặc tenant isolation. Các dòng evidence đến từ request, không được đọc từ kho dữ liệu riêng. Không triển khai nguyên demo lên public endpoint rồi coi các quyền đã được xử lý.

Một proposal khác với hành động ghi. Nếu sau này nối `create_assignment` hoặc `refund`, cần kiểm quyền/điều kiện và chống lặp ở tầng thực thi. Request_id theo dõi một request không thay business idempotency key. Một lỗi timeout sau khi side effect xảy ra là trường hợp cần thiết kế reconciliation, không giải quyết chỉ bằng retry prompt.

JSON serialization của input giúp giữ biểu diễn dữ liệu, nhưng lời “input là data” không là một bằng chứng chống prompt injection. Nội dung model sinh cần đi qua contract và lớp quyền/tác vụ; không chạy code, URL hoặc tool name chỉ vì nó xuất hiện trong một field lạ. Lab từ chối field action và không có executor.

## H. Đọc báo cáo đã chạy

```python
# Sau: python lab.py --mode all --out /tmp/lesson22-run-01
from pathlib import Path
import json
run = Path('/tmp/lesson22-run-01')
report = json.loads((run/'report.json').read_text())['evaluation']
rows = report['rows']
assert len(rows) == 20
assert sum(r['accepted'] for r in rows) == 8
assert sum(r['correct'] for r in rows) == 5
assert sum(r['attempts'] for r in rows) == 23
assert report['summary']['accepted_accuracy'] == 5/8
assert report['summary']['route_accuracy'] == 4/7
print(report['fixture_sha256'])
```

Hash định danh chính xác fixture inputs/events/reference được dùng, không là dấu chứng nhận chất lượng. Cần giữ cả scorer/code version và dependency version để giải thích kết quả. Nếu thay fixture hay scorer, không chỉ ghi đè report cũ rồi so hai con số không cùng định nghĩa.
