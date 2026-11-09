# Đào sâu bài 53 — Duyệt một ý định, xác minh một effect

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Con người tham gia ở quyết định nào?

Có thể hỏi con người để làm rõ task, chọn phương án, sửa draft, cấp quyền thực thi hoặc đánh giá kết quả. Những quyết định này có dữ liệu vào và tác động khác nhau. “Đồng ý với câu trả lời” chưa tự là “cho phép gửi nó cho khách hàng”, và feedback sau khi ghi không thay approval trước ghi.

Policy của bài chọn append internal note là action cần review. Điều đó không là một luật chung rằng mọi tool đọc/ghi đều cần cùng một gate. Hệ thật cần phân loại tác động, khả năng đảo ngược, scope và mức bất định; xác định khi nào được tự động làm và khi nào phải escalation. Không lấy một confidence score chưa hiệu chuẩn làm quyền hành động.

Reviewer cần quyết định trên nội dung và đối tượng cụ thể. Nếu không hiểu draft, phải có đường reject/edit/clarify thay vì chỉ một nút approve. Lab kiểm protocol qua fixture reviewer, chưa đo chất lượng quyết định của người thật hoặc tự xây risk classifier.

## 2. Approval object, view và payload là ba thứ liên quan

Payload là điều target sẽ nhận. View trình bày payload cùng current target metadata, nguồn, bất định và expiry cho reviewer. Approval object ghi lại quyết định bind đúng draft hash/version/reviewer/TTL. Hash view còn gồm workflow revision, nên một quyết định trên view cũ có thể bị phát hiện.

Canonical JSON giúp cùng object có cùng digest theo quy ước của lab; nó không chứng minh UI thực sự hiển thị trung thực hoặc người dùng đã hiểu nội dung. Một UI bị compromise vẫn có thể nói một đằng và gửi digest của một object khác. Trust boundary của phần render/controller phải được thiết kế và kiểm riêng.

Trong bài, HTML chỉ escape JSON để đọc offline. Nó chưa có authenticated session, click handling, CSRF defense hoặc chống clickjacking. Không triển khai một endpoint “approved=true” rồi coi việc thêm hash là đủ cho security. Reviewer identity phải do hệ thống xác thực cung cấp, không từ model output.

## 3. Sửa bản thảo và thay đổi điều kiện thực thi

Edit body trước attempt làm proposal version/hash đổi, hủy approval cũ và trở lại awaiting_review. Payload/version/source/release đổi đều có thể ảnh hưởng quyết định. Reviewer phải được thấy bản cuối cùng trước khi cấp quyền; không âm thầm sửa payload sau approve để “làm câu văn đẹp hơn”.

Lab chỉ thêm phần reviewer comment bên cạnh base body đã kiểm nguồn. Comment là nhận xét của reviewer, không tự được nâng thành causal proof. Khi có attempt, payload bị freeze, vì status/idempotency đang nói về cùng một operation. Nếu muốn thay nội dung sau unknown, phải giải quyết operation cũ trước.

Target version có thể đổi dù body draft giữ nguyên. Compare-and-set ở target bảo vệ precondition hiện hành. Không tăng expected_version theo phản xạ để vượt lỗi stale, vì reviewer đã cấp quyền dựa trên phiên bản trước. Demo chưa có automatic refresh/re-investigation sau conflict.

## 4. Pause/resume không có nghĩa bắt đầu lại từ đầu

Một workflow chờ review nên lưu proposal và trạng thái chờ, không giữ process sống bằng một input() vô hạn. Khi nhận quyết định, kiểm lại scope/version/hash/expiry rồi tiếp tục cùng operation. Resume không reset budget hoặc sinh một operation key mới chỉ để tránh conflict.

Các framework có thể chạy lại một node chứa interrupt khi resume. Nếu effect không idempotent đặt trước interrupt, nó có thể bị lặp. Bài dùng các command/transaction riêng để nhìn rõ boundary; không dùng LangGraph runtime. Bố trí effect sau approval vẫn chưa xử lý crash sau commit, nên còn cần dedup/status contract ở target.

Local workflow revision/CAS chặn ghi checkpoint dựa trên head cũ. Nó không tự ngăn một request đã gửi tới target. Loop ghi của bài không có lease/heartbeat hoặc remote fencing token. Payload freeze và target dedup giúp cùng operation/key không append hai lần theo contract đã định, chưa là distributed workflow engine tổng quát.

## 5. Ba transaction và khoảng trống giữa chúng

Local approval/intent transaction, target effect+receipt transaction và local result transaction thuộc các ranh giới khác nhau. Có thể commit local in_flight, chết trước POST; hoặc target đã commit, chết trước local result. Không thể suy target state chỉ từ local phase.

Bài 45 commit effect/receipt trong một SQLite database. Bài 53 vẫn giữ atomicity ở target, nhưng target DB tách khỏi local approval DB. Đây là thay đổi quan trọng: một rollback local không thu hồi note đã ghi. Target transaction chứa note, version update và receipt để không công bố nửa effect theo contract fixture.

Tests dùng worker exit và injected target error; chưa cắt điện, multi-region replication hoặc distributed transaction. synchronous FULL là cấu hình, không bằng chứng chống mọi lỗi phần cứng. Nếu target API không có receipt/status hay idempotency, recovery cần policy/human reconciliation khác.

## 6. Một HTTP code không thay một outcome model

Response bị mất, timeout hoặc 5xx có thể đi cùng effect chưa xảy ra, đang xảy ra hoặc đã commit. Fixture before_commit có thể biết chắc rollback vì người viết test điều khiển target; client generic không được tự dùng kiến thức đó để kết luận mọi 5xx không có effect.

Status not_found cũng cần contract: có thể chưa gửi, chưa commit, status index lag hoặc request cũ vẫn đang chạy. Case status_missing_retry cố ý che receipt có thật. Lab vẫn giữ unknown; explicit same-key retry chỉ được phép sau reconcile, với approval/quyền/source còn hợp lệ và budget còn.

Receipt identity phải khớp op_id, draft hash, ticket và version effect. Một JSON có state=applied nhưng nhầm ticket không đủ để confirmed. Confirmation nói note đã ghi theo operation đó, không nói health đã tốt hoặc reviewer comment đúng sự thật.

## 7. Idempotency key, payload freeze và retention

Một key ổn định giúp nhận diện cùng ý định, nhưng target phải enforce namespace/payload/dedup transaction. Lab dùng tenant+actor+op_id, trả receipt cũ cho cùng payload trước kiểm stale version; cùng key khác payload bị conflict. Không có retention expiry của dedup rows trong phạm vi thí nghiệm.

Không thể tự sinh key mới sau timeout rồi gọi đó là retry an toàn. Với target đã tăng version, việc đổi cả key và expected version có thể tạo note thứ hai mà reviewer chưa duyệt. Dữ liệu không được phép tự nâng scope hoặc rewrite operation identity từ một memory/summary.

API thật có thể lưu failed responses, có thời hạn key, operation status eventual hoặc error semantics khác. Cần đọc contract cụ thể và test với adapter đó. Bài không tích hợp Stripe/ticket SaaS, không đưa phép thử synthetic thành guarantee exactly-once end-to-end cho mọi dịch vụ.

## 8. Expiry/revoke khác lịch sử effect

Approval hết hạn chặn fresh write; nó không chứng minh effect cũ đã bị hủy. Reconcile có thể xác nhận applied sau expiry/source revoke nếu người truy vấn vẫn có quyền đọc status của operation. Workflow chỉ ghi nhận receipt metadata, không cần gửi lại body hoặc phát hành permit mới.

Actor/reviewer inactive, source revoked hoặc memory selection hết hạn trước lần ghi mới phải làm gate fail closed. SourceGate dùng clock hiện tại của command để kiểm memory, thay vì dùng lại mốc lúc investigation bắt đầu. Một memory TTL có thể hết ở 150 dù approval còn hạn tới 160; hai expiry đó không thay nhau.

Permit ở target có expiry/binding nhưng chưa là cơ chế thu hồi ngay một permit đã được phát hành/đang bay trên mạng. Key demo công khai chỉ phục vụ bài tập. Production cần issuer/key isolation, identity/rotation/revocation và auth boundary đúng hệ đích; không mô hình nào được quyền tự phát hành permit bằng cách viết token vào text.

## 9. Cancel, compensation và outbox

Cancel trước attempt có nghĩa dừng một ý định chưa gửi theo protocol local. Cancel sau unknown không bảo đảm target chưa làm; lab từ chối và yêu cầu reconcile. UI không nên hiển thị “đã hủy, không có thay đổi” chỉ vì đã đổi một local flag.

Compensation là hành động nghiệp vụ mới: một note đã đọc có thể không được “xóa khỏi lịch sử”, và gửi thông báo đính chính khác với đảo byte về giá trị trước. Cần quyền, payload, audit và approval phù hợp. Bài chưa thực hiện compensation hoặc customer notification.

Một outbox local giúp giao việc bền vững nhưng không tự giải quyết remote dedup. Worker outbox vẫn có thể chết sau send/trước ack; cần stable key, status và persistent attempts. Bài dùng explicit execute command/in_flight marker, chưa broker/outbox dispatcher, retry backoff service hoặc scheduled escalation.

## 10. Đánh giá cả protocol lẫn con người

Các tests mới kiểm source/identity/review/version/expiry/CAS, receipt/dedup, unknown/reconciliation và crash. Expected rejects/error exports cũng là pass. Một investigation template được copy vào nhiều case để kiểm protocol, không phải nhiều model samples độc lập hoặc independent human judgments.

Human study còn cần đo reviewer có phát hiện target/body/source sai không, approval fatigue, thời gian chờ, tỷ lệ sửa/từ chối và critical false approvals. Đánh giá agent phải tính cả review effort, failed/retried attempts và những case không hoàn tất. Không chỉ báo tỷ lệ note đã ghi rồi coi workflow tốt hơn.

Chưa có LLM live, human usability study, TLS/OAuth/ticket SaaS, compensation, hardware failure hoặc latency/cost benchmark. Bài 54 sẽ nối các ranh giới này vào đánh giá outcome/trajectory/recovery end-to-end; hiện bài đó vẫn là đề cương.
