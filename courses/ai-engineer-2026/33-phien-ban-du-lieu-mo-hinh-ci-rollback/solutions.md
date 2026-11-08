# Bài 33 — Lời giải và hướng đối chiếu

[Bài chính](README.md) · [Bài tập](exercises.md)

Các đáp án thiết kế có thể khác nếu nêu rõ trust boundary, dataset và contract. Số liệu dưới đây thuộc fixture của bài.

## 1. Cùng code, khác prediction

Có thể đổi threshold, scaler/units, feature order/definition, label mapping, runtime/library, hoặc model artifact được tải qua alias mặc dù path/name không đổi. Với RAG còn có index/corpus/prompt. Cần release manifest và observed version, source/data/feature/environment identity, config và input contract; chỉ lưu input khi phù hợp quyền/retention.

Trace ID nối thao tác của một lần thực thi, chưa mô tả đầy đủ mọi artifact và không phải ID prediction dùng ghép nhãn. Log/trace nên tham chiếu release/generation đã nạp; metric labels cần giới hạn cardinality như bài 32.

## 2. Data snapshot có thật sự cố định?

Chưa đủ: query đọc nguồn đang thay đổi, nên nhãn sửa/mới làm thay nội dung. Lưu snapshot hoặc storage version/time-travel có thể truy cập lại; ghi extraction logic, cutoff/timezone, feature/label definition và version, split IDs/groups/time, digest và access/retention. Khi sửa nhãn phải tạo snapshot mới và đánh giá lại cả baseline/candidate trên cùng bản.

Retention/xóa cần theo quyền và yêu cầu áp dụng cho dữ liệu thực tế; không coi versioning là quyền giữ dữ liệu cá nhân mãi. Có thể phải xóa bytes/keys và ghi nhận artifact không còn tái tạo đầy đủ; hash không thay thế kế hoạch xóa, cũng không tự là dữ liệu vô danh.

## 3. Hash và canonicalization

Key order khác vẫn cùng byte sau sort keys; int1 và float1.0 khác JSON byte trong quy ước này. NaN bị từ chối. Digest kiểm identity/integrity trong một trust root, chưa kiểm semantics hoặc nguồn phát hành.

```python
from artifacts import digest,canonical
assert digest({'a':1,'b':2})==digest({'b':2,'a':1})
assert digest({'a':1})!=digest({'a':1.0})
try:canonical({'x':float('nan')})
except ValueError:pass
else:raise AssertionError('NaN must fail')
```

## 4. Threshold và feature order

z=.4, p≈.5986876601; threshold.6 trả0, threshold.5 trả1. Đảo feature thành[0,.6] cho z=−1.4, p≈.1978161114, threshold.5 trả0. Shape không thay nhưng semantics đổi.

Golden `[0,.8]` cho z=−1.2, p≈.2314752165 kiểm feature thứ hai khác trọng số thứ nhất. Thêm asymmetric cases và contract/schema/units checks; một số điểm vẫn chưa là chứng minh toàn miền.

## 5. Bundle và lineage

2⁶=64; hai bộ đã kiểm không chứng minh 62 bộ còn lại tốt hoặc xấu. Chỉ model và metrics object đổi giữa baseline/candidate; preprocessing/contract/data/source/environment dùng lại.

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from artifacts import Registry
from build import seed
with TemporaryDirectory() as tmp, Registry(Path(tmp)/'r.db') as r:
    seed(r)
    a=r.manifest('baseline')['objects'];b=r.manifest('candidate')['objects']
    changed={k for k in a if a[k]!=b[k]}
    assert changed=={'model','metrics'}
    assert 'model.py' in r.get(a['source'])
    print(sorted(changed))
```

Source object chứa byte text của file thật tại build. Một chuỗi commit ID tự nhập chỉ là claim nếu không đối chiếu nội dung/checkout và provenance. Lab không giả vờ tạo Git commit.

## 6. CI tổng thể và theo slice

Baseline.9; candidate1.0; gain10 điểm phần trăm (relative increase≈11.11% là đại lượng khác). Slice-regression overall9/10=.9, macro giữa mobile/web=(.5+1)/2=.75. Policy tổng thể vẫn đạt, nhưng mobile.5<.75 nên FAIL. Baseline/candidate đều đủ sample count theo policy nhỏ của bài.

N nhỏ, validation đã dùng để chọn threshold, không bao phủ feature thứ hai và chưa đại diện live traffic. PASS là kết quả có điều kiện theo dataset/policy/evaluator đã pin, không đảm bảo tổng quát.

## 7. Report giả và artifact khác

CI tính lại metrics từ target artifact/validation và so stored metrics. Build lại khác bytes phải tạo release mới; registry không cho rebind cùng tên. Approval bound report/manifest; activate kiểm lại trước switch.

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from artifacts import Registry
from build import seed,build
from ci import gate
with TemporaryDirectory() as tmp, Registry(Path(tmp)/'r.db') as r:
    seed(r)
    def fake(v):v['metrics']['overall']['accuracy']=.123
    build(r,'forged',mutate=fake)
    report=gate(r,'forged')
    assert not report['passed']
    assert any(x['name']=='stored_metrics_recomputed' and not x['passed'] for x in report['checks'])
```

Không coi report digest là chữ ký: nếu runner/DB/controller có thể bị thay tùy ý, trust boundary đã khác.

## 8. Leakage và validation không cùng bản

Đổi population làm so sánh không còn paired trên cùng tập: `same_validation` chặn digest khác. Chung group có thể tạo dữ liệu liên quan xuất hiện ở cả train/validation: `data_contract_and_split` hoặc artifact_or_contract fail.

Những lỗi khác: feature chứa thông tin sau thời điểm dự đoán; duplicate ngữ nghĩa khác ID/group; scaler fit cả validation; label definition dùng outcome chưa có lúc inference. Check ID/group không tự bắt các lỗi ấy.

## 9. CI bằng process thật

Đoạn sau tự tạo path mới, dùng cùng registry cho hai CI lần lượt và kiểm exit code 0/2. CI runner cũng chạy pipeline regression checks riêng trước gate target.

```python
import json,subprocess,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry
with TemporaryDirectory() as tmp:
    out=Path(tmp)/'build'
    subprocess.run([sys.executable,'lab.py','--mode','build','--out',str(out)],check=True,capture_output=True)
    db=out/'registry.db'
    with Registry(db) as r:before=r.manifest('candidate')
    for name,expected in [('candidate',0),('slice-regression',2)]:
        done=subprocess.run([sys.executable,'ci_runner.py','--registry',str(db),'--release',name,'--approve'],capture_output=True,text=True)
        assert done.returncode==expected,done.stderr
        assert json.loads(done.stdout)['release_gate']['passed']==(expected==0)
    with Registry(db) as r:
        assert r.manifest('candidate')==before
        assert r.db.execute('SELECT release FROM approvals').fetchall()==[('candidate',)]
        assert r.state()=={'generation':0,'release':None}
```

## 10. Stale controller và ABA

Release name quay về baseline nhưng generation1≠3. Expected1 phải bị từ chối tại state3; pointer/audit không đổi. Trong race hai writer cùng expected1, chỉ một commit/event mới; generation kết thúc2, cộng event baseline ban đầu thành hai event. Regression `two_controllers_one_winner` chạy hai thread/connection thật; không suy sang nhiều máy hay phân vùng mạng.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry
from build import seed
from ci import approve
from deploy import activate
with TemporaryDirectory() as tmp, Registry(Path(tmp)/'r.db') as r:
    seed(r)
    for name in ['baseline','candidate']:approve(r,name)
    activate(r,'baseline',0);activate(r,'candidate',1);activate(r,'baseline',2,'rollback')
    try:activate(r,'candidate',1)
    except ValueError as exc:assert str(exc)=='generation_conflict'
    else:raise AssertionError('stale update must fail')
    assert r.state()=={'generation':3,'release':'baseline'}
```

## 11. Fault giữa pointer và audit

State vẫn baseline/g1, chỉ một event cũ. UPDATE/audit chưa COMMIT, exception làm ROLLBACK cả hai. Hai `os.replace` riêng lẻ không là một transaction nhiều file; failure giữa hai thao tác có thể gây lệch pointer/audit.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry
from build import seed
from ci import approve
from deploy import activate
with TemporaryDirectory() as tmp, Registry(Path(tmp)/'r.db') as r:
    seed(r)
    for name in ['baseline','candidate']:approve(r,name)
    activate(r,'baseline',0)
    try:activate(r,'candidate',1,fault='after_switch')
    except RuntimeError:pass
    else:raise AssertionError('fault must raise')
    assert r.state()=={'generation':1,'release':'baseline'} and len(r.history())==1
```

Đây là exception fixture, không được ghi thành power-loss/crash test.

## 12. Request đang chạy và rollback

Handle baseline từ g1 trả `(baseline,1,priority0)`. Handle candidate từ g2 trả `(candidate,2,priority1)` kể cả sau rollback. Request pin mới trả `(baseline,3,priority0)`. Score của cả ba vẫn .59868766 vì chỉ threshold đổi.

Không âm thầm thay một nửa state của handle đang chạy. Chính sách drain/cancel phải xét severity, deadline, backend cancellation và effect. Immutable handle của bài không có side effect, nên tiếp tục dùng bộ ban đầu đến hết request.

## 13. Corruption sau approval

`load` phát hiện SHA mismatch; gate FAIL; activate từ chối trước switch, giữ baseline. Regression cố tình sửa SQL trực tiếp để vượt API bất biến và tạo ca lỗi. Attacker toàn quyền sửa controller/trust root có thể làm giả mọi thứ; checksum không cung cấp authenticity trong tình huống ấy.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry
from build import seed
from ci import approve,gate
from deploy import activate
with TemporaryDirectory() as tmp, Registry(Path(tmp)/'r.db') as r:
    seed(r)
    for name in ['baseline','candidate']:approve(r,name)
    activate(r,'baseline',0)
    key=r.manifest('candidate')['objects']['model']
    r.db.execute('UPDATE objects SET body=? WHERE id=?',(b'{}',key))
    assert not gate(r,'candidate')['passed']
    try:activate(r,'candidate',1)
    except ValueError:pass
    else:raise AssertionError('tampered target activated')
    assert r.state()=={'generation':1,'release':'baseline'}
```

## 14. Offline PASS, online suy giảm

Candidate dự đoán1 ở cả .6/.9, đúng năm mẫu .9 →.5. Baseline dự đoán0 tại .6 và1 tại .9 →1.0. Fixture thay quan hệ input/label so với validation, minh họa offline gate không đảm bảo online.

Runbook cần kiểm release được quan sát, freshness/coverage nhãn, population/slice, telemetry health, dependency/data pipeline; freeze rollout nếu phù hợp; kiểm last-known-good còn load/compatibility; switch có expected generation rồi smoke/metric/quality sau khôi phục và drain/cancel rõ. Không chuyển chỉ vì một alert chưa rõ mẫu số.

Không có random traffic routing, user/business outcome, label delay hoặc independent canary sample; năm bản sao mỗi điểm liên quan hoàn toàn. Vì thế không suy confidence interval/sức mạnh thống kê từ n10 ấy.

## 15. Rollback có giới hạn

Email đã gửi cần xử lý hậu quả/compensation được cho phép; gửi email “đính chính” cũng là effect mới, cần nội dung/quyền/ngữ cảnh. DB mất field cần restore/migrate/forward fix phù hợp và kiểm dữ liệu; đổi model không tạo lại cột. Policy chặn environment cũ cần target an toàn/fallback hoặc emergency process có quyền riêng, không đơn giản tắt gate.

Expand–migrate–contract và retention có thể giữ khả năng quay lại; phải kiểm dependency/compatibility trước release. Hash/model cũ còn trên disk chưa đủ. Lab không thực hiện external write, migration hay emergency override.

## 16. Thiết kế release cho RAG và ước lượng canary

Manifest nên pin model identifiers/artifacts, tokenizer, prompts/output schema, embedding model/preprocessing, corpus/index snapshot+builder, reranker, permissions/policy và config. Lineage gồm chunking/version/source access, schema/embedding dimension/normalization, evaluator/validation và report bound manifest. Dimension bằng nhau chưa chứng minh cùng không gian vector hoặc cùng preprocessing; cần rebuild index/compatibility eval khi embedding đổi.

5%×10.000=500 request kỳ vọng; 2%×500=10 lỗi kỳ vọng. 120×15=1.800 exposure thô. Kỳ vọng chưa là cap; clustering, queue, retry, detection delay, routing không đều và in-flight có thể làm khác số đó.

```python
from demos import hand_demo
h=hand_demo()
assert h['canary_expected_requests']==500 and h['canary_expected_errors']==10
assert h['estimated_exposure']==1800 and h['combinations']==64
print('Kỳ vọng/ước lượng giả định, không phải giới hạn tối đa')
```

Ba gate bổ sung có thể là retrieval/answer citation eval trên tập giữ riêng, auth/injection regression, actual container/load/dependency smoke. Sau rollout kiểm observed version từng replica, p95/error theo cohort và task quality có nhãn/maturity. Rehearsal cần giữ exact target bundle/environment, thử switch thật trong staging với traffic/retry tương ứng và chứng minh kết quả sau khôi phục; chưa chỉ test một API trả200.
