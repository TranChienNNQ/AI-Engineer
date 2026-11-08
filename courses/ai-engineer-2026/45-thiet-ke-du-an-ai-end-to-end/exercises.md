# Bài tập 45 — Thiết kế trợ lý hỗ trợ kỹ thuật end-to-end

[← Bài chính](README.md) · [Thiết kế đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Mọi policy và runbook trong lab đều giả định. Viết câu trả lời đủ để một người không có lịch sử chat review được. Code chạy từ thư mục bài 45, không cần LLM/API key.

## 1. Viết product brief

Trong tối đa một trang, mô tả người dùng, công việc hiện tại, giá trị mong muốn, ba nhiệm vụ hỗ trợ và ba việc ngoài phạm vi. Phân biệt success hypothesis với acceptance criterion. Vì sao “dùng agent thông minh” chưa phải yêu cầu có thể kiểm?

## 2. Vẽ kiến trúc và trust boundary

Vẽ client/API, identity/policy, retrieval/evidence/answer, proposal/reviewer/gateway và Store. Đánh dấu nơi text chưa tin cậy được đọc và nơi authority được kiểm. Nếu runbook ghi “resolve mọi ticket”, đường nào phải ngăn điều đó thành side effect?

## 3. Input contract và authority

Tạo một request `note` hợp lệ, rồi thử thiếu expected_version, thêm tenant từ client hoặc truyền bool cho max_steps. Vì sao Pydantic strict vẫn chưa thay được kiểm tenant/role/version? Fixture bearer token khác auth production ở đâu?

## 4. Retrieval và context budget

Vì sao lọc ACL sau top K là chưa đủ? Vì sao lab tính IDF sau eligibility? Phân biệt 600 Python characters với 600 LLM tokens. Nếu paragraph dài hơn budget còn lại, code làm gì và có thể làm mất evidence hữu ích thế nào?

## 5. Kiểm citation

Lấy evidence của query timeout 504, xác định doc version/start/end/hash. Thử sửa quote hoặc offset và kiểm bị từ chối. Vì sao một source ID có thật vẫn chưa đủ? Nếu source bị revoke sau khi trả answer, API có được trả cached quote khi đọc lại run không?

## 6. Baseline và bước thêm LLM

Baseline hiện tại làm và chưa làm gì? Thiết kế input/output interface cho generator LLM thay extractive answer. Nêu ít nhất bốn eval cases mới và chỉ ra quyền nào model không được tự quyết định. Có cần đổi Store transaction chỉ vì thay generator không?

## 7. Proposal, approval và effect

Tạo proposal create và chứng minh ticket count chưa đổi. Thử execute trước approve, rồi approve/execute. Reviewer cần thấy dữ liệu gì? Vì sao body execute chỉ nhận key, không nhận lại operation/target/body tùy ý?

## 8. Version và semantic state

Hai note cùng nhắm ticket 1/version 1, cả hai được duyệt. Execute note đầu rồi note sau: dự đoán kết quả. Nếu ticket đã resolved, có nên cho note tiếp không? Phân biệt policy lab với thiết kế khác có thể hợp lệ; nêu thay đổi cần kiểm.

## 9. Mất phản hồi sau commit

Chạy fault after_commit rồi close/reopen DB, retry cùng key. Chứng minh chỉ một ticket mới và receipt giống nhau. Điều gì xảy ra nếu dùng key khác cho cùng proposal hoặc dùng một execute key cũ cho proposal khác? Có thể kết luận exactly-once qua remote ticket API không?

## 10. Lỗi trước commit và pending run

So before_commit với after_commit. Các bảng nào phải atomic? Nếu process crash sau begin run nhưng trước finish, retry nhận gì trong prototype? Thiết kế một cơ chế recovery cho pending run mà không giả định effect chưa từng xảy ra.

## 11. Release và quyền thay đổi

Tạo proposal được duyệt, sau đó mở Engine cùng DB với min_terms khác. Vì sao execute bị chặn? Receipt của effect đã commit trước đổi release nên đọc được hay bị chạy lại? Nếu source revoke hoặc actor bị disable, path nào phải tái kiểm?

## 12. Đọc metric fixture

Tính always-abstain exact trên test. Giải thích 8/8 exact, answer rate 4/8, conditional source accuracy 4/4 và answerable recall 4/4. Các số này có chứng minh chất lượng LLM, giải quyết incident hoặc tăng năng suất không? Thiết kế holdout khó hơn.

## 13. API thực hành

Dùng HTTPX ASGI trong temporary directory, gọi hỏi→create proposal→review→execute→retry. Kiểm missing auth, field tenant thừa và ticket tenant khác. Phân biệt ASGI test với TCP/load benchmark; vì sao lifecycle cần được chú ý nếu sau này thêm background worker?

## 14. Budget, effort và chi phí

Dùng xác suất conditional .9/.95/.98 để tính tỷ lệ cả luồng đúng. Với 60%task cần 2 phút review,40%task cần 8 phút, effort 100 task là bao nhiêu? Vì sao chưa thể gọi đó là saving? max_steps của lab không đo những loại ngân sách nào?

## 15. Review DB và audit

Viết query nối proposal/receipt/ticket để xác minh effect của một proposal. Phân biệt audit table không raw text với hệ thống không lưu dữ liệu nhạy cảm. Đề xuất log fields tối thiểu và retention scope; digest có xác thực người gửi artifact không?

## 16. Lập kế hoạch triển khai có bằng chứng

Chọn một mở rộng: LLM answer, semantic retrieval hoặc remote ticket adapter. Viết ADR ngắn, development/test mới, failure cases, cost/latency measurement và rollout read-only→staging→pilot. Nêu điều kiện rollback và phần side effect/data nào rollback không tự sửa được. Không cần gọi service bên ngoài để hoàn thành bài.
