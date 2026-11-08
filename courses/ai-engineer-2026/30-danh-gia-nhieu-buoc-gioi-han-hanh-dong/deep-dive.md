# Bài 30 — Đào sâu cách chấm và giới hạn hệ thống

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Tách thông số bài đo khỏi câu trả lời của hệ thống

Một evaluation specification cần ít nhất: trạng thái đầu, public instruction, miền tool, oracle, nguồn quyền/review, fault schedule, điều kiện dừng, metric và version. Chỉ lưu prompt cùng câu trả lời chưa đủ để tái lập effect nhiều bước.

Manifest lab ghi 12 case và 4 profile. Hash giúp phát hiện cấu hình/dữ liệu case thay đổi, không tự hash mọi dòng source, dependency, OS hoặc tool implementation. Muốn tái lập đầy đủ cần thêm source revision/artifact digest, phiên bản môi trường và snapshot dữ liệu. Bài ghi runtime trong report và đóng gói source, chưa có signed supply chain hoặc model revision.

Case.expected là annotation ngoài model view. Nhưng policy và grader nằm cùng Python process để học; một plugin Python tùy ý vẫn có thể đọc source/memory nếu được chạy trong process ấy. Interface tách oracle không là sandbox chống code độc hại.

## 2. Invariant theo thời gian và partial order

Nếu write xảy ra tại t, cần các sự kiện read ticket/runbook/health đã hoàn tất trước t. Reviewer phải duyệt đúng proposal, version và target; review còn hiệu lực tại thời điểm commit. Final state không tự chứng minh các điều kiện này.

Có thể mô tả một invariant như:

\[
write(t)\Rightarrow authorized(t)\land intent(t)\land reviewed(t)\land evidence\_before(t).
\]

Đây là điều kiện về mọi write, không phải “trong trace có ít nhất một review”. Review của ticket khác hoặc proposal cũ không đủ. Lab lưu approval snapshot cùng write event để grader kiểm binding và thời hạn; trusted list journal chưa chống giả mạo.

Không nên áp exact sequence nếu nhiệm vụ cho phép nhiều đường đi. Ngược lại, nếu business bắt buộc xác thực rồi mới hiển thị dữ liệu, thứ tự ấy là yêu cầu thực, không thể bỏ vì một đường khác cũng có cùng snapshot cuối.

Path grader lab kiểm read tiền đề trước write và đủ read trước REPORT. Nó chưa chứng minh model thật đã hiểu evidence, chưa kiểm source entailment hay mọi quan hệ event trong một hệ thống phân tán. HANDOFF chưa chấm lý do hoặc người nhận.

## 3. Negative control và kiểm chính bộ chấm

Một grader có thể luôn trả true, so sai ticket, quên kiểm collateral hoặc lỡ lấy label do agent tự khai. Cần test grader bằng output/world cố ý sai:

- Final NOTE nhưng không có note.
- REPORT đúng schema nhưng sai retry_limit.
- Đoán đúng REPORT facts mà không thực hiện read bắt buộc.
- Ghi green rồi restore snapshot; journal vẫn phải chỉ ra vi phạm.
- Xóa read khỏi journal nhưng giữ evidence claim trong gateway; path phải trượt.

`fixtures.checks()` chứa các negative control này. Đó là kiểm implementation của grader/harness, khác phép đo chất lượng policy. Cũng nên có nhiều positive control hợp lệ để tránh grader quá cứng; lab chấp nhận hai thứ tự đọc và retry unavailable.

## 4. Thống kê: chọn đúng đơn vị và estimand

**Estimand** là đại lượng muốn ước lượng. “Tỷ lệ trial thành công khi request được lấy từ traffic thực” khác “tỷ lệ task có mọi trial đều thành công trong bộ khó”. Mẫu số và cách lấy mẫu quyết định ý nghĩa con số.

Lab có 4 profile cố định/task và số trial bằng nhau nên trial mean bằng macro mean của c/n. Tuy vậy các profile không i.i.d.; confidence interval nhị thức không tự có ý nghĩa bao phủ cho traffic/model deployment.

Nếu một task được chạy nhiều lần, nên cân nhắc cluster theo task/family. Ví dụ bootstrap cả task và giữ các trial bên trong, hoặc thiết kế mô hình phân cấp, tùy câu hỏi. Không bootstrap từng assertion như chúng là người dùng độc lập. Bài chưa cài bootstrap, significance test hoặc suy luận traffic.

Ví dụ task A có p=1, B có p=0. Với k=2, trung bình p^2 là 0,5; lấy p trung bình=0,5 rồi bình phương được 0,25. Vì vậy phải tính metric từng task rồi tổng hợp với trọng số đã nêu. Nhiều trial dễ không nên tự làm task ấy nặng hơn nếu mục tiêu là macro theo task.

Wilson trong code dùng z=1,96, xấp xỉ hai phía 95%; upper bound 0 event dùng công thức nhị thức một phía 95%. Hai phương pháp không cùng xây dựng hay cùng số phía, không cần cho cùng upper endpoint. Chúng chỉ minh họa trên bài toán i.i.d. giả định ở phần tính tay.

## 5. Repeated trials và ý nghĩa vận hành

pass@k giả định có cách nhận ra và chọn trial thành công nếu dùng như một chiến lược sản phẩm. Có k lần thực hiện mà không biết lần nào đúng không tự đem lại chất lượng tương ứng. Đặc biệt với write, trial đã sai có thể gây effect không thể hoàn tác trước khi có lần đúng.

pass^k tập trung vào tính nhất quán, nhưng không thay tỷ lệ vi phạm, severity và distribution shift. Một tác vụ hiếm nhưng tác hại lớn vẫn cần đo riêng; không gộp nhẹ với hàng nghìn task đọc đơn giản.

Công thức subset cho n=4,k=4 chỉ có đúng một subset; một task có một lần trượt thì all-success=0. Đây là mô tả dữ liệu ít, không là bằng chứng chắc chắn xác suất thật bằng 0. Với profile được lập trình, không được gọi nó là consistency sampling của model.

## 6. Đối chiếu A/B có ghép cặp

Trong compare(), mọi policy/boundary nhận cùng Case và profile trên world mới. Bảng paired gồm both/strict_only/schema_only/neither. Chênh lệch accepted là:

\[
\Delta=\frac{n_{strict\ only}-n_{schema\ only}}{N}.
\]

Không suy chỉ có n tăng mới quan trọng; phải đọc những ca baseline mới làm hỏng. Khi model có randomness, cùng seed chưa chắc bảo đảm hai dịch vụ có random stream tương đương, nhất là transcript khác nhau. Cần ghi version/config và lặp có thiết kế phù hợp.

`schema_only` là ablation bỏ business guards ở lớp duy nhất đang thực thi quyền. Nó không tách đóng góp từng guard, chưa là thử nghiệm factorial, và không đo mức bảo vệ của hệ thống có backend authorization độc lập. `strict` giữ nguyên parser và budget nên vẫn có thể thất bại do loop hoặc sai facts.

## 7. Giới hạn trước dispatch, trong I/O và sau completion

Trong lab, duration/cost đã biết và mỗi operation tuần tự. reserve() kiểm projected ms/units/call/write trước tăng counter. Tất cả counter cập nhật trong cùng hàm, nhưng chưa có lock/transaction cho concurrency. Model turn cũng được charge trước parse output; invalid output không có refund.

Với I/O thật, deadline cần monotonic clock và timeout trên actual await/network layers; ngoài budget total có thể cần per-call timeout. Timeout caller không rollback downstream write. Nếu server hoàn tất sau cancellation, outcome phải được đối soát thay vì báo “chưa ghi”. Lab dùng đồng hồ integer giả, không đo wall clock/network cancellation.

Reservation budget cũng không bằng billing: API có thể báo usage sau stream, retry có thể phát sinh phí, cache/batch có đơn giá khác. Cần usage thật + upper bound/reserve + reconcile; ledger phải bền và chia sẻ giữa nhánh/agent nếu có.

Context cap lab đo byte UTF-8 public view trước model turn. Nó chặn việc đưa view lớn vào policy, không chặn toàn bộ memory allocation hoặc response DoS; chưa tokenizer, token window, trace retention hay log redaction.

## 8. Giới hạn hành động gồm cả quyền và ý định

Người dùng có thể có quyền ghi ticket nhưng task hiện tại chỉ yêu cầu đọc. Việc commit khi intent=REPORT là ngoài ý định dù ticket nằm đúng tenant. Tương tự, có ý định ghi không thay review, version hoặc health condition.

Approval trong bài giữ task/ticket/version/hash/expiry; không thể đổi facts sau review rồi dùng lại. Hash không là chữ ký và ID draft không là bearer credential. Backend/gateway là host-trusted; chưa multi-user identity hoặc grant service. Mã policy không được cấp access trực tiếp vào World để bypass gateway.

Giới hạn write cap=1 đếm reservation, không đếm effect. Một commit bị từ chối vẫn có thể dùng hết write budget. Thay chính sách này cần phân tích việc refund có tạo replay/DoS/liveness bug không; không sửa counter cho đẹp thống kê.

## 9. Điểm tổng, utility và cost

accepted là conjunction ở miền lab. Một dashboard tốt vẫn công bố outcome, safety, path, budget, positive completion, handoff và cost riêng. Lấy trung bình trọng số có thể cho một action ngoài quyền bị bù bằng văn phong tốt; nếu đó là hard constraint thì phải giữ gate riêng.

Có thể tính total units/accepted để so chi phí mỗi trial được chấp nhận, nhưng đó không là tiền hoặc marginal API cost. Cần ghi mẫu số gồm cả expected handoff hay chỉ positive. Baseline luôn HANDOFF rất rẻ nhưng không hoàn thành positive, nên cost thấp tự nó không đủ để chọn hệ thống.

Task success chỉ có thể dùng để chọn release nếu task/oracle phản ánh sản phẩm cần làm. Bộ nhỏ, synthetic hoặc public holdout có thể hữu ích cho regression nhưng không tự chứng minh chất lượng production.

## 10. Nối với MCP và bước triển khai tiếp

MCP tools/call thành công chỉ cho biết một exchange/operation theo contract; grader task còn phải kiểm domain effect và constraints qua nhiều exchange. Tool adapter có thể cung cấp alias, URI/version/hash, receipt và error layer làm dữ liệu cho evaluator. Không dùng protocol complete hoặc agent DONE làm nhãn đúng duy nhất.

Lab bài 30 gọi Python trực tiếp để cô lập phần đánh giá; không import SDK MCP, dùng lại checkpoint SQLite hoặc gọi model bài trước. Muốn nối thật cần một adapter, capture trusted tool/effect events, reset test backend và thêm integration tests cho cả model/harness/tool/server version.

Bài 31 sẽ tiếp tục API phục vụ, hàng đợi và timeout. Những phần chưa triển khai ở đây: model judge, user simulator, live traffic, MCP/network, auth/review UI, durable journal, distributed budget/concurrency, OS sandbox và CI service.
