# Bài 25 — Lời giải

[← Bài chính](README.md) · [Đề bài](exercises.md)

Snippet chạy từ thư mục bài sau khi cài `requirements.txt`. Phần đọc file report cần chạy `lab.py --mode all --out /tmp/lesson25-run-01` trước, hoặc thay path bằng thư mục run của bạn. Số policy là dữ kiện giả định của bộ học.

## 1. Kết hợp theo nhu cầu

RAG cấp policy bên ngoài theo request/snapshot; fine-tuning có thể giúp hành vi/format/citation nếu có dữ liệu và kiểm chứng phù hợp. Fine-tuning không tự cập nhật mọi policy tuần sau. Context dài không xác định ai có quyền, tài liệu nào được duyệt, bản nào hiệu lực hay nguồn nào có authority cao hơn.

Có thể dùng model đã phù hợp định dạng cùng retrieval và kiểm nguồn. Cần đo lợi ích trước khi thêm huấn luyện; không có yêu cầu phải fine-tune để được gọi là RAG.

## 2. Xác suất

Marginal=.6×.7+.3×.2+.1×.1=.49. Nó là xác suất sinh y theo mô hình giả định, không là độ đúng của y. Score BM 25 của lab không là prior xác suất ấy.

Hai token: Sequence=.5×.9×.1+.5×.1×.9=.09; Token=(.45+.05)×(.05+.45)=.25. Sequence dùng một document tiềm ẩn cho toàn chuỗi trong từng số hạng; Token cộng qua document trong từng bước. Không hoán đổi tổng và tích như thể hai mô hình bằng nhau. Đây là ví dụ toán, không là kết quả model inference trong lab.

## 3. Nguồn và offset

Source/version định danh snapshot; chunk trỏ một khoảng của source; evidence ID E1 chỉ định danh một đoạn đã cấp trong một request. `[200+10,200+40)=[210,240)` là span nguồn. File hiển thị thêm header/fence nên không được dùng offset đó như line number trong file.

```python
from data import dataset
from rag import run
sources, _, _ = dataset()
result = run('policy idempotency key lưu bao nhiêu giờ',sources)
by_source = {(s['source_id'],s['version']):s for s in sources}
for c in result['validation']['citations']:
    assert c['valid']
    s = by_source[c['source_id'],c['version']]
    assert s['text'][c['source_start']:c['source_end']] == c['quote']
print('Every quote maps to the original source body')
```

## 4. Nguồn ngoài context

Không chấp nhận citation nằm ngoài mapping context của request, kể cả source/chunk thật có trong index. ID bịa là không tồn tại; ID thật bị packing loại là tồn tại nhưng chưa được cấp cho generator. Cả hai đều chưa đáp ứng giao thức cite chỉ evidence đã nhận.

Tra nguồn khác cùng từ rồi đổi ID không chứng minh entailment, còn có thể che lỗi đầu tiên. Nếu cần thêm nguồn, hãy gọi retrieval/packing như bước có trace và kiểm lại câu trả lời, không sửa tên citation một cách tùy ý.

## 5. Kiểm cơ học không suy nghĩa

Fixture `real_quote_wrong_claim` có schema hợp lệ, quote thật, đúng span/evidence. Bộ cơ học pass; annotation cho thấy quote deadline không hỗ trợ claim thời hạn key. Với phủ định, “không retry lỗi thiếu quyền” không hỗ trợ “retry mọi lỗi thiếu quyền”, dù gần như trùng term.

```python
import json
from data import dataset
from fixtures import setup, citation
from rag import validate_candidate
from evaluation import evaluate_candidate
ctx, allowed, facts = setup(['no_permission'])
candidate = dict(status='answer',reason='',claims=[dict(
    text='Mọi lỗi đều có thể retry vô hạn.',
    citations=[citation('no_permission',ctx,facts)])])
validation = validate_candidate(json.dumps(candidate,ensure_ascii=False),ctx,allowed)
query = dict(required=['no_permission'],expected_status='answer')
semantic = evaluate_candidate(query,validation,facts)
assert validation['accepted']
assert semantic['supported_claims'] == 0
assert semantic['task_success'] == 0
```

Claim được dùng trong snippet là mẫu đã được annotation không có tập hỗ trợ trong catalog. Với câu paraphrase mới ngoài catalog, scorer sẽ ghi `known=False` và chưa cho credit; đó là chưa được xác minh, không là một bộ phân loại mọi câu đúng/sai.

## 6. Ba claim/bốn edge

Attachment=3/3=1; grounded claim rate=2/3; citation precision=2/4=.5; required fact recall=2/2=1; task success=0 do C3 không có hỗ trợ.

Bỏ C3: còn 2 claim đều hỗ trợ,3 edge trong đó 2 hữu ích. Attachment=1, grounded rate=1, precision=2/3, required recall=1. Task success theo giao thức hẹp của bài trở thành 1 vì mọi claim hỗ trợ và đủ ý/đúng trạng thái, dù precision chưa hoàn hảo. Điều này nhắc rằng một điểm task không thay mọi tiêu chí citation.

## 7. AND/OR và đổi đơn vị

Chỉ cite policy key chưa đủ cho composite claim có cả deadline; cần citation bao phủ cả tập `{'key_72','deadline'}`. Cite đủ cả hai thì full support. Nếu có một tài liệu khác chứa đủ toàn bộ, annotation có thể có tập hỗ trợ thay thế, nhưng không tự giả định nguồn ấy tồn tại.

Fact thời hạn 72 giờ và phép chia 72/24=3 hỗ trợ phát biểu 3 ngày theo quy ước ngày 24 giờ của ví dụ. Catalog đã gán quy tắc này cho một câu cụ thể; nó không hiểu mọi đơn vị, mệnh đề hay quan hệ suy luận. Với claim mới cần thêm annotation hoặc evaluator phù hợp.

```python
from fixtures import evaluate_fixtures
r = evaluate_fixtures()
by_name = {x['name']:x for x in r['rows']}
assert by_name['joint_support_incomplete']['validation']['accepted']
assert by_name['joint_support_incomplete']['semantic']['supported_claims'] == 0
assert by_name['joint_support_complete']['semantic']['task_success'] == 1
assert by_name['derived_72h_to_3days']['semantic']['task_success'] == 1
```

## 8. Budget

Context còn 232. Greedy theo rank giữ 150, dư 82 nên không giữ 120/112. Bộ 120+112 vừa 232; nếu có ích lợi phù hợp có thể là lựa chọn tốt hơn. Chưa biết utility thì không thể kết luận bộ nào tốt hơn chỉ theo chi phí. Dùng gold required facts khi serving để chọn sẽ làm rò rỉ nhãn và không đại diện vận hành.

Header/source label/version/separator đều vào input model. Lab đếm serialized context và kiểm `prompt_count+output_reserve≤window`; word-level count này không thay tokenizer/message overhead của LLM thật. Reserve chưa tự chặn output sinh, adapter cần output limit riêng.

## 9. Theo thời điểm và authority

2026-06-01: P01@1; 2026-09-01 và 2026-11-04: P01@2. Bản 1 hết hiệu lực đúng lúc bản 2 bắt đầu, vì `valid_to` là exclusive. Quote 24 giờ thật nhưng không là câu trả lời policy hiện hành ở query có as_of sau chuyển bản.

P07/P08 là hai source khác nhau, cùng có hiệu lực/approved nhưng mâu thuẫn về cùng dịch vụ giả định. Khi chưa có quy tắc authority/scope giải quyết, báo conflict và dẫn cả hai, không tự lấy con số lớn hơn/nhỏ hơn hoặc chọn nguồn đứng đầu BM 25. Code default extractive chưa làm được điều đó, nên T09 fail.

## 10. Trạng thái

Answer đủ ý có nguồn; partial chỉ trả phần đủ căn cứ và nêu thiếu; abstain chưa đủ căn cứ để trả lời; conflict nêu các nguồn chưa hòa giải. Schema cho hình dạng output, không tự xác định trạng thái nào phù hợp.

Partial trung thực tránh bịa nhưng có thể chưa hoàn thành query đầy đủ, nên task completeness 0 vẫn có grounded rate 1. Abstain không có claim/cite nên tỷ lệ không có mẫu số, đểNone. T11 OOV trả abstain đúng; T12 có term quen khiến baseline trả TTL của hệ thống khác phạm vi, task sai dù quote đúng.

## 11. Tầng đầu tiên làm mất dữ kiện

Pool/candidate đủ còn packed thiếu thì lỗi đầu tiên là lựa chọn context/budget, không phải embedding thiếu tri thức. Packed đủ mà answer thiếu/sai thì xem generator, instruction, khả năng tổng hợp hoặc contract; chỉ tăng M có thể không giúp.

Oracle context cấp đúng gold evidence cho cùng generator là thí nghiệm tách retrieval khỏi answering. Nếu oracle còn sai, sửa retrieval đơn thuần khó giải quyết. Không được báo oracle như kết quả serving hoặc đưa required labels vào generator rồi gọi nó tự suy ra bằng chứng.

## 12. Mẫu số

Micro=1/(1+9)=.1. Macro=(1/1+0/9)/2=.5. Một cách cho mỗi claim trọng số bằng nhau; cách kia cho mỗi answer trọng số bằng nhau.

Parse/schema failure vẫn thuộc 22 case task. Claim/citation micro của lab chỉ đếm cấu trúc schema-valid, gồm cả candidate bị reject cơ học. Vì vậy duplicate citation có thể được scorer biết là cùng hỗ trợ về nội dung nhưng candidate vẫn không được phát hành/task credit. Báo cả kiểm cơ học lẫn semantic, không dùng metric có điều kiện để che lỗi đầu vào.

## 13. Report thực tế

Run mặc định:11 snapshot,10 source ID,8 source public hiện hành,13 chunk,25 fact;4 dev/12 test. Test có 10 query với required facts (gồm conflict),2 no-answer. Main cap 110 giữ đủ fact/context ở10/10, có 22 claim/edge hỗ trợ; task 10/12. T09 cần conflict nhưng output answer; T12 quote đúng nhưng sai câu hỏi.

Fixture 22 case: parse 21, schema 19, mechanically accepted 12, task 5;20 claim/22 edge từ schema-valid, attachment 19/20, grounded 11/20, precision 13/22. Đây là candidate do lập trình tạo, không là tỷ lệ lỗi model sinh tự nhiên.

Cap 40: packed required recall.45, context complete 3/10, task 4/12; cap 110 và 180: recall 1, complete 10/10, task 10/12. Tăng budget sửa mất bằng chứng nhưng chưa sửa conflict/relevance của baseline.

```python
import json
from pathlib import Path
r = json.loads(Path('/tmp/lesson25-run-01/report.json').read_text())
assert r['source_snapshots'] == 11
assert r['current_public_chunks'] == 13
test = r['primary']['splits']['test']
assert test['task_successes'] == 10 and test['cases'] == 12
assert test['grounded_claim_rate'] == 1
assert r['fixtures']['aggregate']['mechanically_accepted'] == 12
print(test, r['fixtures']['aggregate'])
```

## 14. Renderer

Model chỉ trả ID/span/quote, server giữ mapping URL/file path hợp lệ. Extra field URL bị schema reject. Renderer còn escape Markdown/HTML trong claim text thay vì để text model trở thành markup có quyền tự tạo link; hệ thống web thật cần policy render phù hợp, chưa có web service trong lab.

```python
import copy
import json
from fixtures import setup,citation
from rag import validate_candidate,render_answer
ctx,allowed,facts = setup(['key_72'])
base = dict(status='answer',reason='',claims=[dict(text=facts['key_72']['text'],citations=[citation('key_72',ctx,facts)])])
for field,value in [('quote','quote bịa'),('evidence_id','E999')]:
    bad = copy.deepcopy(base)
    bad['claims'][0]['citations'][0][field] = value
    v = validate_candidate(json.dumps(bad,ensure_ascii=False),ctx,allowed)
    assert not v['accepted']
malicious = copy.deepcopy(base)
malicious['claims'][0]['text'] = '[nhấp](https://untrusted.invalid/) <script>alert(1)</script>'
v = validate_candidate(json.dumps(malicious,ensure_ascii=False),ctx,allowed)
assert v['accepted']  # Mechanics only; claim is outside the closed semantic catalog.
rendered = render_answer(v)
assert '<script>' not in rendered and '&lt;script&gt;' in rendered
assert '\\[nhấp\\]' in rendered
print(rendered)
```

`accepted=True` chỉ là qua kiểm cơ học, không phải semantic truth. Vì vậy thật sự thay bằng LLM còn cần lựa chọn/kiểm evaluator phù hợp; không quảng cáo validator này là bộ chống hallucination hoàn chỉnh.

## 15. Adapter và holdout

Generator nhận question, context đã lọc/pack và prompt. Nó không nhận required facts, gold support sets, expected status hoặc câu trả lời tham chiếu. Adapter LLM cần ghi model/revision, prompt/contract, tokenizer, sampling/seed nếu có, input/output budget, timeout/retry và usage; giữ raw response cho chẩn đoán theo policy dữ liệu.

Lab không có LLM/tokenizer provider, không benchmark latency/cost hoặc fluency. Metric fixture không là xác suất lỗi LLM. Sau khi dùng lỗi test để chỉnh hệ thống, coi test ấy là dữ liệu phát triển và có holdout mới phù hợp để đánh giá độc lập; khác query text chưa bảo đảm khác intent/source.

## 16. Tiêu chí sản phẩm

Bộ nhỏ đạt yêu cầu khi người khác tái lập được lệnh, nguồn và mapping span; thấy rõ nhãn không đi vào serving; đọc được trace packing/candidate/citation; biết mẫu số metric và ba lỗi cụ thể. Source version/quyền phải gắn với từng request, không chỉ một link chung có thể đổi.

Nếu mất required fact ở packing, thử budget/selection và đo lại. Nếu quote/claim không hỗ trợ, sửa prompt/generator/verification, không đổi ID cho đẹp. Nếu grounded nhưng irrelevant hoặc không báo conflict, sửa logic trả lời và bài đo tương ứng. Một cải tiến có ích phải nêu cả kết quả và giới hạn, không chỉ tăng số link hoặc làm JSON hợp lệ hơn.
