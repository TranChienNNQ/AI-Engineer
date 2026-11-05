# Bài 28 — Đào sâu state, recovery và memory

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Một vòng agent có nhiều tầng quyết định

Model chọn action trong không gian host cấp. Host chọn schema, danh tính, task boundaries, stop conditions và effect permissions. Một action sequence linh hoạt không có nghĩa agent có quyền thay đổi mục tiêu hoặc tự cấp năng lực mới.

Policy Python cho kết quả deterministic theo observation; không cần training hay API. Nó giúp tách lỗi runtime khỏi lỗi model, nhưng bỏ qua khó khăn tool selection và ngôn ngữ thật. Nối LLM phải có adapter, version/sampling, parse contract và eval riêng; không đổi policy thành model rồi coi các số fixture cũ là số mới.

ReAct là nguồn nghiên cứu cho interleaving reasoning/acting, không buộc production lưu toàn bộ suy nghĩ nội bộ. Log hữu ích là action/args đã kiểm, observation/provenance, lỗi, budget và lý do chuyển trạng thái có thể giải thích công khai. Không dùng trace như chứng minh model reasoning luôn trung thực.

## 2. Working state không là “tất cả những gì agent biết”

Observations là dữ kiện được cấp tại một thời điểm, có quyền và TTL. Summary được rút từ observation hiện đang hợp lệ, không là một bảng chân lý vĩnh viễn. Context là view mà policy nhận ở turn hiện tại; long-term memory nằm ngoài state, được recall lại mỗi turn.

Lab giữ observation mới nhất cho mỗi loại tool. Nếu bài toán cần so timeline, nhiều hypotheses hay evidence mâu thuẫn, cấu trúc đó quá đơn giản; cần records riêng có ID/time/status thay vì ghi đè. Event log hiện chỉ giữ metadata, không đủ tái dựng đầy đủ mọi payload cũ.

Một thread/run ID là định danh scope, không là credential. Server phải xác thực actor rồi kiểm owner/tenant trước load. SQLite file có plaintext/checkpoint là dữ liệu được tin ở boundary hệ điều hành, không được trao cho model/client như một resume token.

## 3. Versioning và checkpoint compatibility

State có schema_version, model_id và revision. Sai contract/version bị từ chối thay vì load tùy ý. Tuy nhiên model_id là nhãn do người viết adapter giữ, không là hash toàn bộ code/model/config. Thay hành vi nhưng giữ nguyên nhãn có thể phá assumptions; production cần migration/version protocol rõ hơn.

ScriptPolicy phục vụ fixture dùng turn index để chọn item. Sau crash, một turn đã reserved chưa chắc đã có response; replay chuỗi script theo turn không tương đương phục hồi model session. Những case recovery của bài dùng AdaptivePolicy tái dựng quyết định từ observation. Một adapter stateful thật phải persist/khôi phục state của chính nó hoặc dùng protocol stateless phù hợp.

Runtime không nạp pickle hay exec state. JSON/schema giảm ambiguity nhưng không xác thực nguồn DB. Người có quyền sửa database có thể sửa giới hạn/nội dung; checksum không thay HMAC/signature, auth và access control. Lab không là sandbox cho Python cùng process.

## 4. Reservation trước call là một lựa chọn bảo thủ

Nếu ghi budget sau call, crash ở giữa có thể làm mỗi lần restart nhận quota mới. Lab ghi reservation trước model/tool, giữ counter qua restart. Đổi lại reservation có thể được tính dù call chưa xảy ra; nó là upper bound kế toán bảo thủ, chưa là billing usage thật.

Sau crash ở model phase, turn cũ vẫn được tính, response chưa commit bị bỏ và turn mới được reserve. Sau crash ở read phase, có thể phải đọc lại. At-least-once invocation không đồng nghĩa observation giống hệt: nguồn có thể đổi; host kiểm quyền/snapshot hiện tại rồi commit cái mới.

Không có network trong lab. Một client thật cần provider request ID, billed usage, deadline/cancellation và retry policy; không suy call chưa bill chỉ vì chưa có checkpoint. Nếu thêm write cần idempotency/receipt như bài 27; state save và remote side effect không tự nằm cùng transaction.

## 5. CAS ngăn lost update, không phải một distributed execution protocol

UPDATE theo expected revision và insert event cùng SQLite transaction giúp checkpoint không bị ghi đè bởi writer cũ. Nó chưa chứng minh chỉ có một remote call in flight: hai worker có thể chạy ngoài transaction nếu protocol sở hữu run không đủ chặt.

Lab không tự takeover THINKING/PENDING_TOOL, yêu cầu recover tường minh khi worker cũ đã dừng. Production có thể cần lease, fencing token, task ownership và queue có quy tắc retry. Chỉ thêm TTL lease không tự chặn zombie worker tạo effect nếu backend không hiểu fencing/idempotency.

SQLite chỉ một writer cùng thời điểm; write có thể trả busy theo workload. Test CAS trên hai connection kiểm lost update, không benchmark throughput hoặc HA. File commit qua restart bình thường không chứng nhận chịu mọi lỗi storage/power loss.

## 6. Cancellation và lịch sử kết quả

Cancel là sự kiện coordinator; runtime cũ save theo revision cũ sẽ Conflict và phải đọc lại state. Nó không hoàn tác read đã chạy, bill hoặc dữ liệu đã gửi. Demo không có remote write, nên không có compensation.

DONE trong checkpoint phản ánh final gate tại thời điểm ấy. Khi quyền/TTL thay đổi, snapshot có thể không còn được trả answer dù status lịch sử vẫn DONE. Không tự mở lại một run terminal để tránh thay đổi ngầm semantics; ứng dụng cần lựa chọn rõ giữa query lịch sử, refresh task và run mới.

Store.load là internal checkpoint API, có thể chứa plaintext cũ. Agent.snapshot/build_view revalidate trước cấp cho caller/policy. Tránh dùng raw checkpoint như response HTTP hay đưa toàn bộ event/state vào prompt.

## 7. Memory admission, trust và scope

Memory candidate cần kiểm nội dung, nguồn, scope và mục đích. Helper remember_reviewed giả định coordinator đã xét ý định; nó không hiểu ngữ nghĩa để biết note có đúng hay chứa secret ngoài refs không. Origin metadata do luồng tin cậy giữ; body tự khai source ID không đủ.

Namespace actor/tenant ngăn đường recall chéo trong code. Một memory trộn nguồn phải còn đọc tất cả; từ chối theo lineage chưa chứng minh thông tin trong body thật sự chỉ đến từ những nguồn ấy. Khi dùng model tạo summary cần kiểm cả content/provenance, không chỉ gắn tag.

Memory poisoning có thể bền hơn injection trong một prompt vì được recall nhiều lần. Không cho recalled text đổi role/tool/limits; ngay cả vậy LLM vẫn có thể bị dẫn sai answer-only. Fixture policy Python không chịu rủi ro ngôn ngữ như model thật, nên chỉ kiểm host boundary, không là tấn công thành công/thất bại của LLM.

## 8. Update, conflict và quên không chỉ là SQL DELETE

Cùng memory key/version update tạo version mới; writer stale bị conflict. Cần quyết định key nào đại diện cùng fact, cách xử lý nguồn mâu thuẫn và khi nào tạo item mới. Lexical top-k không tự chọn fact mới/đúng hơn nếu metadata hoặc key design sai.

Forget hiện đánh tombstone và tăng version. Recall không trả item, nhưng file pages, backup, run checkpoint, log hoặc bên nhận có thể còn dữ liệu. Secure deletion/retention cần xét toàn bộ vòng đời, không hứa “quên sạch” từ một UPDATE.

Preference không có origin KB có thể dùng typed store riêng, như language=vi sau user instruction rõ; code bài này yêu cầu note có origin nên chưa hỗ trợ mọi loại memory. Procedural memory sửa policy là thay control, cần thẩm quyền riêng; đừng đưa prompt từ memory thành system instruction tự động.

## 9. Compaction cần giữ nghĩa, không chỉ giảm độ dài

Summary phải giữ task, phủ định, điều kiện, facts, uncertainty, open questions và nguồn. Lab tạo summary có cấu trúc từ observations hợp lệ nhưng vẫn gửi cả observation; đây không là thuật toán nén toàn context hoặc một LLM summarizer.

Nếu thay bằng summary do model sinh, cần một bộ truy vấn kiểm hiểu trước/sau compaction và nhãn những chi tiết bắt buộc. Có thể giảm token mà tăng sai sót hiếm; cần đo chứ không chỉ xem văn bản có vẻ trôi chảy.

Repeated action heuristic xét data/ref fingerprint, tránh observed_at mới đánh lừa progress. Nó vẫn có false positive với polling hợp lệ và false negative với chuỗi biến thể action. Budget, task-specific progress predicate và policy polling phải bổ sung cho nhau.

## 10. Đánh giá khi thêm LLM

25 fixture gồm terminal status đúng kỳ vọng, trong đó semantic_residual được cho DONE nhưng answer sai. Các phép đo cần tách: task correctness, evidence/grounding, unauthorized action, memory contamination, recall quality, resume fidelity, duplicate invocation/effect và cost/latency.

Run resumed nên so task/result với uninterrupted khi nguồn/memory/model/config tương ứng; khi nguồn thay đổi thì result khác có thể đúng hơn. Không ép byte-identical output của model nondeterministic làm tiêu chuẩn duy nhất. Ghi input/source/memory versions, runtime/model/sampling và policy quyền tại thời điểm đo.

Memory có thể tăng continuity nhưng cũng tăng token, stale facts hoặc privacy risk. Lab không chứng minh gain hiệu suất/chất lượng. Muốn biết có đáng dùng, cần baseline không memory và metric task cụ thể với nhãn độc lập, tách test khỏi dữ liệu đã ghi nhớ.
