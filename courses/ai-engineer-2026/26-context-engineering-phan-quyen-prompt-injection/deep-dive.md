# Bài 26 — Đào sâu context, quyền và ranh giới thực thi

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Context selection là tối ưu có ràng buộc, không phải mua quyền bằng score

Đặt `authorized(item,principal)` là một điều kiện bắt buộc. Chỉ trong tập item được phép mới xét relevance, freshness, novelty, cost và task utility. Không nên gộp “rủi ro thiếu quyền” thành một số phạt nhỏ để item quá liên quan có thể bù điểm và được chọn.

Utility trong ví dụ knapsack là giả định do người soạn đặt, không là score đã học hay xác suất model trả lời đúng. Lab dùng thứ tự cố định theo kind rồi BM25, không dùng gold task label để tìm tập context tốt nhất. Một thí nghiệm hợp lý có thể thay thứ tự hoặc budget rồi đo retrieval/task riêng, nhưng chưa có LLM/task-quality measurement trong bài.

Không cần dùng hết window. Một tool result cũ/dài có thể chiếm chỗ của bằng chứng mới; summary ngắn có thể bỏ mất điều kiện. “Ít token hơn” chỉ là một trục chi phí, không chứng minh đủ ý hay an toàn hơn.

## 2. Một summary vừa có confidentiality label vừa có authority label

Summary của nguồn internal vẫn có thể là internal. Nó còn phải giữ rằng nội dung gốc là dữ liệu, không phải trusted instruction. Hai thuộc tính này độc lập: một ghi chú user-private không tự được sửa system policy; một chỉ dẫn server-authoritative cũng không phải nơi để đặt mọi secret.

Nếu summary lấy thông tin từ nhiều nguồn, policy visibility cần xét các origin thực sự đóng góp. Lab dùng quy tắc bảo thủ: giữ toàn bộ origin và yêu cầu quyền trên tất cả; owner scope còn hẹp hơn. Nó chưa theo dõi đóng góp từng câu để redaction/declassification tinh vi.

M-LAUNDER cho thấy lineage hợp lệ vẫn có thể kèm nội dung độc hại/sai. Hash/version chứng minh đang dùng đúng payload đã gắn provenance trong luồng tin cậy; không chứng minh summary trung thành hoặc instruction trong đó đúng thẩm quyền.

Với preference như “trả lời tiếng Việt”, không nhất thiết có origin KB. Có thể lưu typed preference bằng policy riêng, chặn các field role/tenant/tool-capability; code hiện chỉ minh họa các record nội dung có origin, không phải toàn bộ hệ thống memory.

## 3. Xác thực nằm ngoài mô hình session fixture

`session_alice` là khóa tra bảng do test harness cung cấp. Nếu HTTP API thực tế nhận một chuỗi như vậy từ client rồi tin ngay thì chưa có authentication. Cần nguồn danh tính đáng tin và cơ chế kiểm session/token phù hợp; bài không xây SSO, cookie, JWT hay middleware.

Tương tự, thư viện Python không là sandbox: một adapter Python được toàn quyền chạy trong cùng process có thể truy cập World/grant store nếu được trao reference hoặc đọc bộ nhớ. Threat model ở đây là model chỉ trả proposal/data qua giao diện, không chạy Python tùy ý, không được gọi controller cấp grant.

Các invariant của code chỉ có ý nghĩa trong ranh giới đó. Khi thêm code execution/browser/network tools phải xem lại toàn bộ đường effect/egress, không thể suy an toàn chỉ từ việc hai tool hiện tại được kiểm tốt.

## 4. Vì sao schema vẫn cần, dù không đủ?

Schema cho một ngôn ngữ đề xuất hữu hạn: hai tên tool, argument cụ thể, type/length/pattern và chặn extra field. Nó ngăn một số ambiguity và mass assignment; ví dụ `roles=['operator']` không được truyền xuống domain object để vô tình đổi quyền.

Nhưng proposal `close_ticket(T-GREEN-1)` hoàn toàn có thể đúng schema và sai quyền. Validation cú pháp trả lời “code hiểu object này không”; authorization trả lời “principal có được thực hiện không”; intent grant trả lời “hành động cụ thể này đã được workflow cho phép chưa”. Không thay một câu hỏi bằng câu khác.

JSON parse còn phải nhất quán: duplicate key có thể bị các tầng hiểu khác nhau; NaN/Infinity không thuộc JSON contract; bool không được xem như integer version. `parse_proposal` chặn các trường hợp ấy trước dispatch. Code không eval/exec text model, không dựng SQL/shell từ reason.

## 5. Grant là state bên ngoài context, không phải câu “đã đồng ý”

Grant store giữ binding subject/tenant/principal revision/ACL revision/proposal hash/resource ref/expiry/used. Bearer token được trao cho trusted coordinator, không là field trong tool proposal. Model có thể đoán một chuỗi hoặc tự tính hash nhưng không tạo được entry hợp lệ bằng cách viết text.

Hash proposal canonicalize JSON key order; nó không làm ngữ nghĩa tự đúng. Trusted review/controller phải xét target, action, reason và các tham số có ý nghĩa trước khi cấp grant. Phê duyệt “hãy xử lý lỗi này” không tự là một grant cho mọi action tương lai.

Controller `issue_after_review` là một API nội bộ tin cậy, chưa kiểm một sự kiện UI review thật. Nếu vô tình expose nó như tool mà model tự gọi, ranh giới demo đã bị phá. Khi scope người dùng đã rõ và policy cho phép tự động, controller có thể cấp theo ý định đã có; không có quy tắc bắt buộc hỏi lại mọi action.

## 6. TOCTOU và phạm vi của RLock

Giữa review và execution có thể có update ticket, revoke quyền hoặc đổi trạng thái session. Vì vậy gateway kiểm lại live permission và snapshot ngay tại execution, không chỉ dùng boolean “đã kiểm” từ bước trước.

Một `RLock` trong World đảm bảo các thread cùng dùng **chính World đó trong một process** không đồng thời dùng một grant hai lần qua đoạn critical section này. Nó không là distributed lock, không đồng bộ nhiều worker/process, không có durability sau crash và không thay database transaction.

Production cần định nghĩa trạng thái nào kiểm/cập nhật nguyên tử, thường gồm kiểm version/unused rồi consume grant và đổi resource, cùng cơ chế idempotency/retry phù hợp. Chỉ dùng TTL không ngăn replay trong thời gian token còn sống. [OWASP Transaction Authorization](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html) là nguồn đọc cho binding/TOCTOU/expiry/uniqueness; code này là ví dụ in-memory riêng.

## 7. Cache revalidation mạnh hơn một key, nhưng chưa là toàn bộ isolation

Key của lab giữ nhiều revision nên khá bảo thủ: update ở tài nguyên khác có thể làm cache miss, kể cả principal không đọc tài nguyên đó. Điều này tốt cho tính đơn giản nhưng chưa tối ưu throughput; cũng chưa đánh giá timing/traffic side channel. Phép thử noninterference của bài chỉ kiểm score/result của retrieval khi thêm dữ liệu không được đọc, không chứng minh isolation mọi kênh phụ.

Cache lưu data items và trace, không lưu grant. Trước dùng, item phải còn đọc được, origin/version/hash/text khớp và còn vừa budget; payload trả cho caller được deep-copy. Nếu quên bump epoch, dependency recheck của item đang giữ vẫn bắt revoke/version đổi trong những ca đã kiểm.

Cơ chế đó chưa chứng minh cache luôn tối ưu độ mới: nếu có tài liệu mới đáng lẽ được tìm nhưng đường cập nhật không bump corpus revision, các dependency cũ có thể vẫn hợp lệ mà câu trả lời đã thiếu thông tin mới. API update của lab bump revision; ứng dụng thật cần giao thức invalidation và observability tương ứng.

## 8. Revoke sau disclosure không thể thu hồi tri thức đã cấp

Context được kiểm theo snapshot tại lúc build. Sau khi trả plaintext cho model/provider/user, đổi ACL không khiến bên nhận quên dữ liệu. Một request đang chạy có thể cần cancellation, lease hoặc policy xử lý kết quả sau revoke; tradeoff tùy dữ liệu và hệ thống.

Không cần giả định giữ một lock trong suốt thời gian gọi LLM là giải pháp: điều đó có thể chặn cập nhật lâu và vẫn không kiểm được retention ở bên ngoài. Cần xác định semantics thời điểm cấp quyền, boundary egress và phần trách nhiệm của bên nhận. Lab không thực hiện network egress, nên không dùng kết quả của nó để chứng nhận một tích hợp cloud.

## 9. Detector/guardrail nên là một phép đo riêng

Regex có thể bắt một số mẫu quen nhưng có false positive với nội dung đang phân tích chính attack string, và bỏ sót cách diễn đạt mới. NLI/LLM classifier có thể thêm tín hiệu nhưng cũng sai; nội dung được chấp nhận bởi detector vẫn không được tự cấp role hoặc grant.

Nếu thêm detector, giữ riêng tỷ lệ phát hiện trên các case đã gán nhãn, false positive trên benign và end-to-end utility/outcome. Đừng gộp việc detector báo động với việc harmful effect thật sự xảy ra; cũng đừng coi một blocked tool call là model đã hiểu đúng toàn bộ nhiệm vụ.

Bài không triển khai detector hoặc benchmark LLM. Nó chủ ý cho phép một số data item có injection đi vào context để thử ranh giới effect ngay cả khi giả định model bị dẫn sai.

## 10. Mẫu số và một cách dùng xác suất đúng giới hạn

Chain rule `P(F∩A)=P(F)P(A|F)` không cần độc lập. Nếu F là model đề xuất ngoài quyền, A là executor cho qua trong điều kiện đó, ví dụ .4 ×.1=.04 chỉ là giả định minh họa. Các case scripted không cho ước lượng đáng tin về P(F) của model thật.

Trong lab,17 adversarial proposal cố định có 13 event ở bản schema-only và 0 ở gateway. Metric đo predicate trên execution/data event của các proposal ấy.5 benign cho 5 allowed outcomes. Một answer-only residual không dùng tool được giữ ngoài các mẫu số action và confidentiality; nó vẫn sai và chưa có bộ semantic verifier trong code.

Nếu đo LLM thật, nên ghi benign task completion không attack, task completion khi bị attack và attack goal success riêng; reset trạng thái, sampling và budget rõ. [AgentDojo](https://arxiv.org/abs/2406.13352) là nguồn định hướng từ abstract, chưa được chạy trong bài. Không dùng 0 lỗi fixture để suy risk 0 hoặc áp confidence bound đòi mẫu IID cho tập case tự thiết kế.

## 11. Khi nối sang workflow nhiều bước

Mỗi bước tool là một lần vượt boundary mới. Output get_ticket là data có origin, không phải instruction cấp quyền cho close_ticket. Kết quả một tool hợp lệ có thể chứa comment độc hại; đưa vào history/summary không làm nó trở thành trusted control.

Khi xây vòng agent sau này, cần giữ task/intent, dependency, auth context, resource version và effect trace qua từng bước. Grant cho một bước không mặc nhiên là quyền cho chuỗi thao tác chưa review. Bài 27 sẽ triển khai gọi công cụ/workflow chi tiết hơn; ở đây mới có gateway tối thiểu để học ranh giới.
