# Bài 33 — Đào sâu thiết kế phiên bản và phát hành

[Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Identity, lineage, integrity và authenticity khác nhau

**Identity** là cách gọi chính xác một artifact. **Lineage** là quan hệ “được tạo từ đâu/bằng cách nào”. **Integrity** kiểm artifact không khác nội dung gắn với digest đã biết. **Authenticity** kiểm nguồn phát hành/ủy quyền. Một hệ thống cần phân biệt bốn câu hỏi đó.

Ví dụ: URI `s3://bucket/model/latest.bin` mô tả chỗ tìm, không đủ bất biến nếu key bị ghi đè. Digest mô tả byte, không tự mô tả label semantics. Một manifest nói model được train từ snapshotD nhưng thiếu record pipeline đáng tin thì mới là claim lineage. Signature xác nhận một chủ thể ký claim, chưa chứng minh claim đúng hoặc model tốt.

Lab dùng SQLite tin cậy và các API không ghi đè; người có quyền SQL tùy ý có thể phá cả object/pointer/approval. Canary corruption cố tình sửa một object để kiểm phát hiện thay đổi đơn lẻ, không phải bài kiểm an ninh trước attacker toàn quyền. Source snapshot/approval không có chữ ký hoặc tài khoản người duyệt.

## 2. Snapshot bằng byte, logic hay version của storage?

Có thể lưu nguyên byte đã trích xuất, tham chiếu object version bất biến hoặc dùng snapshot/time-travel của database. Cần kiểm retention và semantics của sản phẩm cụ thể. Query + ngày chạy không đủ nếu dữ liệu cũ được update/delete và query không đọc lịch sử.

Hai dataset có cùng tập row nhưng thứ tự khác có thể khác hash; training cũng có thể bị ảnh hưởng bởi thứ tự. Nếu muốn một semantic digest độc lập thứ tự, phải định nghĩa sort key, duplicate policy, encoding, float representation và version của phép chuẩn hóa. Không âm thầm sort mọi dataset: sequence/time series có ý nghĩa thứ tự.

Lab hash toàn snapshot theo đúng thứ tự rows và label hiện có, không chuẩn hóa Unicode hoặc CSV. Python JSON sort keys không biến nó thành một chuẩn canonical JSON dùng được giữa mọi ngôn ngữ.

Train snapshot và validation snapshot là hai object riêng. Sửa label tạo object/digest mới, kể cả feature không đổi; approval dựa trên snapshot cũ chưa tự mở rộng sang snapshot mới. Group split kiểm một lớp leakage; không tự bắt semantic duplicate, future feature hay preprocessing fit nhầm validation.

## 3. Model artifact nên load được theo cách nào?

Weight cần kiến trúc, input shape/dtype, preprocessing, vocabulary/tokenizer, output mapping và runtime phù hợp. Với LLM/RAG, release còn có thể pin prompt/template, adapter, tokenizer, retrieval corpus/index, embedding model, reranker và policy. Tùy hệ thống, một phần là deployment config riêng nhưng vẫn cần liên kết version để kiểm tương thích.

Lab dùng JSON logistic đóng format để tránh việc ví dụ vô tình load pickle không tin cậy. Không có tokenizer hoặc neural checkpoint; không suy rằng JSON luôn an toàn trước kích thước quá lớn hay resource exhaustion. Object store chưa có quota tổng, schema validator không thay body-size guard.

`pin()` chỉ tạo handle sau khi contract/preprocessing của bài được kiểm; handle hỗ trợ identity scale đúng hai feature. Nếu mở rộng scaling, phải mang preprocessing thật vào handle cùng version thay vì chỉ nới validator. Đổi input units cần adapter/contract version và test numerical equivalence, không chỉ thêm một string version mới.

## 4. Build once và policy không do candidate tự quyết

Một CI report cần bound ít nhất: candidate manifest digest, baseline, validation identity, policy/config, evaluator/source và environment. Nếu report chứa `accuracy=1` nhưng thiếu artifact identity, nó có thể bị gắn nhầm sang release khác.

Trong lab, manifest trỏ source/environment/metrics; gate report còn ghi candidate manifest, baseline và policy digest. Approval trỏ report object và controller tính lại toàn gate trước activation. Validation phải cùng digest với baseline. Approval row trong DB là record nội bộ tin cậy, chưa có expiration/actor/audit riêng của lần duyệt.

Không để manifest candidate tự đặt `min_accuracy=0` rồi controller dùng theo. `POLICY` nằm trong code CI/controller. Nếu đổi policy, approval cũ bị từ chối. Nếu chính code controller không đáng tin, cách bố trí một biến constant không tạo security boundary.

Recompute mọi thứ khi activation phù hợp fixture rất nhỏ. Hệ thống lớn thường tách CI evaluator, report/provenance được bảo vệ và verifier nhẹ, có policy freshness/revocation/expiry. Bài chưa triển khai mô hình trust đó.

## 5. Thiết kế CI jobs mà không cần thuộc tên platform

Một cấu trúc job có thể dùng:

```text
checkout exact revision
  → syntax/unit/data-contract checks
  → build candidate once
  → evaluate baseline/candidate on pinned validation
  → integration/load/compatibility checks
  → publish artifact + provenance/report, fail on any required gate
  → deployment stage consumes the exact approved digest
```

Job sau cần phụ thuộc kết quả job trước, lấy artifact đúng run và kiểm digest; không tải “artifact mới nhất” của branch. Với PR không tin cậy, tách quyền đọc/build khỏi quyền deploy/credentials; không chạy code PR tùy ý cùng secret production. Những nguyên lý này không yêu cầu bạn dùng GitHub Actions; lab chạy bằng process Python cục bộ.

Thực hành cụ thể: `lab.py --mode build --out ...` tạo registry; `ci_runner.py --registry ... --release candidate --approve` kiểm chính registry đó. Runner compile mọi `.py`, chạy 52 pipeline checks trong registry tạm riêng rồi gate/approve target có sẵn. Các checks tạm không sửa production slot của registry đầu vào. Kiểm slice-regression trả exit2 và không thêm approval.

Sáu mode của `lab.py` dùng workspace mới cho mỗi lần để dễ tái tạo. `--mode ci` một mình build fixture trong workspace của lần đó; muốn kiểm artifact đã build từ lệnh trước phải dùng `ci_runner.py --registry`. Không nhầm hai cách dùng.

Không có file `.github/workflows` đang hoạt động, không push, không gọi remote runner. Tự động hóa thật cần cấu hình checkout/runtime, permissions, artifact retention và secret isolation theo platform đang dùng.

## 6. CI data/quality gate không thay thiết kế đánh giá

Accuracy tổng thể cao có thể che minority slice; macro average là góc nhìn khác, chưa thay kiểm từng nhóm. Một slice có hai row cho phép tính 1/2 nhưng không cho độ tin cậy cao. Trong thực tế cần đủ n, label maturity/coverage, cost-sensitive metric và cách kiểm uncertainty phù hợp dependency giữa mẫu.

Baseline/candidate cần cùng population/labels và evaluator. Nếu mỗi model chấm trên tập khác nhau, chênh lệch có thể do độ khó dataset. Khi nhãn được sửa, chạy lại cả hai và giữ snapshot cũ cho audit; không so metric lịch sử cũ với metric mới rồi gọi là improvement.

Validation của lab dùng để chọn/so threshold nên được gọi là validation. Lặp tối ưu trên một “test set” sẽ khiến nó thành validation trên thực tế. Một final holdout hoặc temporal evaluation bổ sung phải được thiết kế trước, không sửa nhãn/tập để vượt gate.

Golden case `.5→p=.5` và contract catches giúp bắt một số incompatibility nhưng không bao phủ feature thứ hai; validation có error_ratio=0 hoàn toàn. Đây là khoảng trống có chủ đích để nhìn ra test coverage. Bài tập yêu cầu tự đề xuất golden/property checks mới; không được suy PASS thành model đúng toàn miền.

## 7. CAS, transaction và vấn đề ABA

Nếu chỉ so `current_release=='A'`, controller đọc A trước một chuỗi A→B→A có thể tưởng không thay đổi. Generation làm state thành `(A,1)→(B,2)→(A,3)`. Controller giữ expected1 phải từ chối state3.

`BEGIN IMMEDIATE` vào write transaction trước khi check rồi update, nên hai writer không cùng kiểm/commit một generation. Pointer và audit cùng DB/transaction; failure sau UPDATE trước COMMIT làm cả hai rollback. `Registry` dùng explicit SQL transaction với connection isolation_level=None, đóng connection tường minh.

Trong lab, object build/approval diễn ra trước activation; activation đọc object/approval trong write transaction. Race test dùng hai connection trên hai thread thật, mỗi connection tạo trong thread của nó. Không dùng một connection shared để tránh thread guard rồi gọi đó là concurrent correctness.

Transaction SQLite local không biến các external effect thành atomic. Nếu controller update DB rồi gọi router từ xa, hai việc đó có thể lệch nhau. Cần desired/observed state, reconciler, idempotency, outbox hoặc protocol phù hợp. Các cơ chế ấy nằm ngoài triển khai bài này.

## 8. Durable state, filesystem atomicity và crash windows

SQLite lưu deployment/events trên disk, được kiểm đóng/mở lại. Exception rollback không đồng nghĩa đã kiểm crash/power-loss durability; OS/filesystem/hardware, journal và backup còn ảnh hưởng. Bài chưa bật/tuning WAL hoặc thử database corruption.

Nếu tự làm registry bằng file, ghi đè `production.json` trực tiếp có thể khiến reader gặp file dang dở. Có thể ghi file tạm cùng filesystem rồi dùng atomic rename/replace, nhưng vẫn cần CAS/lock và kế hoạch durability/flush đúng. [Python os.replace](https://docs.python.org/3.14/library/os.html#os.replace) nêu atomicity khi thành công và hạn chế khác filesystem. Một rename atomic không tự gom pointer và audit ở hai file vào một transaction. Lab chọn một SQLite transaction cho cả hai nên không triển khai file-pointer switch.

Đừng xóa old artifacts ngay khi promote: target rollback có thể mất object hoặc environment. Ngược lại, giữ mọi thứ mãi cũng có chi phí/retention/privacy. Garbage collection nên dựa trên graph reference, rollback window, in-flight handles/jobs và audit needs. Lab không có GC hoặc quota; chỉ tạo ít fixture trong temp workspace.

## 9. Rollback của desired state và observed serving state

Pointer trong registry nói controller muốn phiên bản nào. Một service thật có nhiều replica/cache có thể đang dùng nhiều release trong rollout. Sau switch phải kiểm version được báo từ process phục vụ, không chỉ đọc lại desired pointer.

Handle trong lab chính là snapshot đã nạp vào request. Sau rollback, handle candidate cũ vẫn tồn tại tới khi hết scope; request mới pin baseline. Không có frontend/load balancer hoặc queue drain policy; chưa chứng minh mọi replica đã đổi trong một thời gian xác định.

Nếu phải cancel in-flight, cần xét backend có cooperative cancellation không, effect đã commit chưa, retry có lặp effect không, job có bị orphan không. Một rollback model read-only khác hẳn rollback agent đã write ticket.

## 10. Rollback window và khả năng tương thích dữ liệu

Expand–migrate–contract giữ đường cũ trong thời gian cần quay lại: thêm field/new reader trước, chuyển producer/consumer từng bước, chỉ bỏ field cũ khi đã qua window và xác nhận dependency. Dual-write cũng có rủi ro inconsistency; cần cơ chế kiểm/reconcile, không tự coi là atomic.

Với feature store/index/schema, model cũ có thể cần dữ liệu đã không còn tạo. Retain weight thôi không đủ. Với security patch, quay về environment cũ có thể không được chấp nhận; emergency response có thể là tắt tính năng hoặc phục vụ fallback được duyệt.

Lab không miễn gate khi reason='rollback'. Baseline approval vẫn được recheck; đổi source/environment/policy có thể block target. Đây là hành vi fail closed có chủ đích. Emergency path trong hệ thống thật cần quyền, phạm vi, bằng chứng và audit riêng, chưa được triển khai ở đây.

## 11. Liên hệ sang LLM và agent

Một RAG release có thể là tuple `(generator,tokenizer,prompt,embedding,index,reranker,policy)`. Đổi embedding nhưng dùng index tạo bởi model cũ có thể phá không gian vector dù dimension trùng. Đổi prompt schema nhưng downstream parser cũ có thể làm lỗi contract. Bài 24–26 cung cấp ngữ cảnh để tự lập manifest tương ứng.

Với hosted model, không phải lúc nào bạn kiểm soát/truy xuất được cùng weight/runtime mãi. Ghi model identifier/provider/config/date, dùng version pin khi API hỗ trợ và regression eval; không hứa bitwise reproducibility khi provider không cho cam kết ấy. Lab không gọi provider API.

Agent rollback policy không hoàn tác tool writes. Giữ operation/receipt/journal để xác định effect, rồi dùng compensation có quyền và version precondition; không tự phát lệnh nghịch đảo cho mọi effect.

Quay lại bài chính và chỉ ra ba lớp kết quả riêng: **artifact đủ identity/integrity**, **gate đạt trên dataset đã chọn**, **hệ thống đang phục vụ đúng release và đạt outcome mong muốn**. Có lớp trước chưa tự chứng minh lớp sau.
