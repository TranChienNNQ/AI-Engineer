# Đào sâu bài 52 — Context là view có thời điểm và quyền

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. View không thay authoritative state

Một database có thể giữ full execution state để audit/recovery, trong khi planner chỉ nhận một projection. Những phần không gửi tới planner vẫn tồn tại ở host và có thể cần cho authorization/final validation. Context compaction không đồng nghĩa xóa lịch sử hoặc đánh dấu bước đã xong.

Lab giữ raw observations trong checkpoint bài 51. PlanGuard và final gate dùng bản raw, còn scripted planner chỉ nhận packet. Cách chia này cho phép kiểm consumer thực sự nhìn gì. Nếu nói model “được rút gọn context” nhưng vẫn âm thầm cho script đọc toàn history bên ngoài packet, phép thử về context sẽ không còn đúng phạm vi.

Các quyền có thẩm quyền không được lấy từ summary hoặc memory. Một memory nói “người dùng đã approve” không thay approval object bind đúng target/args/version/expiry. Summary có thể mô tả một sự kiện approval cũ để audit; muốn thực thi vẫn phải kiểm approval hiện hành theo protocol riêng. Bài chưa có write action/approval workflow production.

## 2. Loại memory và cách truy xuất là hai trục khác nhau

Semantic memory nói về nội dung tri thức; episodic memory nói về sự kiện/trải nghiệm; procedural memory nói về quy tắc/cách làm. SQL, keyword matching, embedding hoặc graph traversal là phương pháp lưu/truy xuất. Một row SQLite có thể chứa semantic memory, và một vector index có thể truy xuất episode.

Lab chỉ có ba schema hẹp. Hint giúp search, preference và episode được đưa vào context như dữ liệu; không có học ngôn ngữ tự động, prompt tự sửa hoặc consolidation qua model. Chỉ hint tác động trực tiếp tới quyết định script. Không dùng kiểu dữ liệu “episode” để khẳng định đã triển khai reasoning về nhiều sự cố.

Procedural policy cần trust cao hơn một ghi chú tự sinh. Nếu muốn tự cập nhật prompt/code, đó là thay đổi hành vi được version/evaluate/review, không một thao tác nhớ thông tin thông thường. Không cho retrieval text tự biến thành policy chỉ vì nó được lưu ở database có tên memory.

## 3. Chọn gì để nhớ: giá trị tái dùng và chi phí sai

Không phải mọi câu chat hoặc tool output đều nên promote thành memory. Một triệu chứng chỉ đúng cho một incident có thể làm nhiễu task sau; một sở thích rõ ràng có thể hữu ích lâu hơn; một khẳng định nguyên nhân chưa kiểm chứng có thể tạo lỗi có tác động lớn. “Có thể nhớ” khác “nên nhớ”.

Một policy thực tế cần mục đích, owner, nguồn, consent nếu áp dụng, độ nhạy cảm, TTL và bằng chứng sửa/xóa. Nên xem memory là dữ liệu dẫn xuất có lifecycle: candidate → kiểm → active → expired/superseded/forgotten, cùng lineage về nguồn. Review không tự bảo đảm nội dung đúng; reviewer cũng cần rubric và dữ liệu.

Demo dùng mapping alias/symptom synthetic do controller xác lập, không diễn giải một cuộc hội thoại thật rồi tự cho là người dùng muốn lưu lâu dài. Không có memory của chính người học được ghi vào hệ thống cá nhân hoặc gửi ra dịch vụ ngoài. Các bản source/world/memory trong lab đều là artifact local được chương trình tạo.

## 4. Hot path, background và thời điểm kiểm lại

Có thể ghi memory ngay trên hot path sau tác vụ hoặc qua worker background. Hot path dễ thấy nguồn mới và phản hồi lỗi ghi nhưng tăng latency; background tách latency nhưng cần queue, retry, ordering, dedup và xử lý revoke/forget xảy ra trước khi worker chạy. Không có một lựa chọn đúng cho mọi ứng dụng.

Bài ghi candidate/review bằng trusted controller sau run, không có background scheduler. Cross-run demo đợi run A xong, đăng ký source/candidate/approve, rồi tạo run B. Không hứa tự cập nhật memory trong nền. Muốn async hóa phải xử lý trường hợp một task ghi cũ đến muộn và tái tạo dữ liệu đã forget.

Memory read cũng cần thời điểm: đọc store rồi gửi context sau một thời gian có thể dùng dữ liệu không còn được phép. Code kiểm epoch/selection trước consumer, dispatch và export; đó là các boundary đã kiểm, không một transaction atomic bao trùm việc gửi tới external provider. Hệ live cần xác định race window và semantics thực tế.

## 5. Projection có kiểm không là general summarization verifier

Structured summary của lab chỉ bỏ một số metadata và giữ những trường script cần. Origin hash liên kết projected item với raw observation. validate_bundle so lại đúng projection, nên phát hiện ai sửa packet sau build; nó không tự đánh giá rằng projection đã giữ mọi chi tiết quan trọng cho một domain mới.

Nếu dùng LLM summarizer, summary có thể bỏ phủ định, đổi uncertainty thành certainty, trộn nguồn hoặc gộp hai thời điểm. Cần schema cho claims/unknowns/refs, kiểm nguồn, điều kiện không được mất và human/semantic evaluation trên output thật. Không chỉ dùng một model khác nói “summary tốt” rồi coi đó là bảo đảm.

Một kỹ thuật thực dụng là giữ summary phục vụ điều hướng và ID để mở lại source khi cần trích dẫn hoặc quyết định quan trọng. Progressive disclosure giảm việc nhét mọi thứ vào context ngay đầu, nhưng tạo tool calls/latency và cần kiểm quyền mỗi lần fetch. Lab có memory hint → search source hiện hành, chưa arbitrary context-fetch tool hoặc hierarchical summarizer.

## 6. Không gian, recency và conflict

Capacity limit, recency và factual validity là các khái niệm khác nhau. Một item mới hơn chưa chắc đáng tin hơn; một item vừa hết TTL không tự là sai mà là không được policy hiện tại cho dùng. Hai query mappings khác payload dưới cùng key được lab coi là conflict ứng dụng, không chứng minh chúng mâu thuẫn logic ở thế giới thực.

Scope/filter phải đi trước ranking/top-k. Sau đó còn phải quyết định tính liên quan, nguồn, deadline và conflict. Trong lab, hint trigger khớp lexical, score 100/50/20 là priority do tác giả đặt. Không đo embedding similarity, confidence hoặc khả năng tìm đúng memory trên kho dữ liệu thật.

Resolve chọn winner với expected epoch nhằm tránh một controller dựa trên snapshot cũ ghi đè cập nhật mới. Nó không tự chứng minh winner đúng. Với nhiều source có độ tin cậy khác nhau, có thể cần giữ nhiều giả thuyết/kèm provenance thay vì loại cả nhóm; đó là policy/evaluation mới, không đã có trong code.

## 7. Memory poison: đúng schema vẫn có thể nguy hiểm

Schema hẹp chặn một số cách nhét instructions/authority hoặc causal claim không được hỗ trợ vào memory. Payload khớp source giúp tránh tự bịa source fields. Nhưng source có thể sai, bị đầu độc hoặc chứa instruction-shaped text trong một trường vốn được phép là string.

Case untrusted_query cố ý cho thấy một chuỗi như vậy có thể đi qua lineage review của fixture và được dùng làm query đọc. Script không tự làm theo câu ra lệnh vì nó không phải LLM và action space chỉ đọc. Không lấy kết quả này làm benchmark injection resistance. Data section, escaping và exact hash không biến nội dung không đáng tin thành instruction đáng tin.

Một đánh giá live cần các giai đoạn ingest, promote, retrieve, context serialization và action/final, vì poisoning có thể xuất hiện ở nhiều bước. Chấm cả việc nhớ sai, lấy nhầm, memory vượt quyền và claim không có nguồn; không chỉ xem câu trả lời cuối có chứa một marker tấn công hay không.

## 8. Forget phải có phạm vi, dependency và retention

Logical forget theo key chỉ chặn những record được gắn đúng key đó. Nó không tự tìm mọi alias hoặc bản sao tương đương ngữ nghĩa. Lab bỏ payload/refs của row memory, nhưng source registry, checkpoint query, telemetry, SQLite free pages và backup có thể còn dữ liệu. Đây không secure erasure hoặc tuyên bố tuân thủ pháp lý.

Tombstone giúp không tái nhập cùng key qua proposal cũ. Hệ thật cần entity/subject identity, dependency graph và policy cho derived summaries/caches để tránh resurrection dưới một key khác hoặc qua một worker đến muộn. Chặn output sau revoke và xóa dữ liệu vật lý là hai công việc khác nhau.

Bài pin toàn-store epoch, nên đơn giản nhưng bảo thủ: một update ở namespace khác cũng có thể làm run hiện tại không dùng được. Nó chưa selective invalidation, multi-tenant service hardening hoặc retention worker. Không đưa database runtime vào ZIP; bản chia sẻ chỉ có code/tài liệu/hình synthetic.

## 9. Nối memory với durable execution

Memory có thể tồn tại qua nhiều run, nhưng execution state của run A không tự trở thành state của run B. Bài dùng một memory store chung, hai run IDs/checkpoint chains riêng. op IDs/counters/plan lineage giữ theo từng run, không copy final của A rồi nói B đã đọc tool hiện tại.

Run binding có context policy, implementation/source lineage, memory epoch và selection fingerprint. Thay policy hoặc memory snapshot làm checkpoint cũ không resume được theo binding ấy. Đây là lựa chọn fail closed, chưa workflow migrate/refresh một pending action phụ thuộc memory đã thay đổi. Một số lỗi vẫn giữ owner trên checkpoint; recovery controller cần quy trình riêng nếu muốn tiếp tục.

Loop 52 mới dùng State/Journal/restore và final/MCP gates từ 51/50; không monkeypatch global planner của bài cũ. Full inherited checks kiểm source cũ, còn các tests mới kiểm memory/context boundaries và pause/crash paths của loop mới. Không suy 409 hồi quy thành mọi tính năng loop mới đã được formal verify.

## 10. Đánh giá chiến lược context/memory theo nhiều lớp

Bytes nhỏ hơn là một metric hệ thống, chưa là quality. So compact/full trong lab giữ task/memory/script và actions giống nhau, giảm tổng packet bytes trong fixture; nó chưa đo tokenization, provider wire payload, latency, model accuracy hay critical omission trên domain mới. Lượt đầu compact thậm chí lớn hơn 3 bytes.

Một evaluation ứng dụng có thể đo memory precision/recall trên pool đã được phép, candidate promotion false positives, stale/revoked inclusion, critical fact retention sau compaction, unnecessary tool calls, wrong-memory influence, task outcome và chi phí toàn trajectory. Phải giữ case không nhớ gì, conflict, forget và budget fail; không chỉ chọn ví dụ memory hữu ích.

Nếu thêm model thật, lấy outputs/usage thực, cùng rubric/budgets/tools/snapshots giữa các strategy và annotation độc lập theo task. Các bài nghiên cứu cũ hoặc hướng dẫn nhà cung cấp giúp đặt giả thuyết, không thay đánh giá hiện tại. Bài 53 về human review/write effects và bài 54 về agent evaluation vẫn đang ở mức đề cương.
