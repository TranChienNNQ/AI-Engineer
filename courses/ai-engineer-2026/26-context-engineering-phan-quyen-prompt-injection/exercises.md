# Bài 26 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Mọi attack string, principal, marker và ticket là dữ liệu mô phỏng. Không cần thử trên dịch vụ thật. Dự đoán quyết định trước khi chạy code.

## 1. Phân loại context

Liệt kê rules, task hiện tại, tool schema, history, memory, evidence và tool result của một trợ lý điều tra timeout. Phần nào do server quản lý authority, phần nào là data? Một comment trong ticket viết `role=system` có đổi loại không?

## 2. Budget và quyền

Tính budget 640 từ ví dụ window 1024. Với costs 240/200/160/120/80, utilities 9/8/6/5/100 và authorized 1/1/1/1/0, chọn tập utility tối đa vừa budget. Vì sao không chọn item cuối dù utility cao? Con số utility này có là đo chất lượng LLM thật không?

## 3. Compaction có thể giữ sai điều gì?

Một history nói “chỉ retry lỗi tạm thời; không retry thiếu quyền; chưa có ý định đóng ticket”. Summary chỉ còn “retry và đóng ticket khi cần”. Nêu ba lỗi, metadata/typed state cần giữ và phần nào vẫn phải được kiểm ngoài model.

## 4. Authentication, role, tenant, intent

Alice nói “tôi là operator”, còn server resolve engineer. Ops_blue có role đóng ticket nhưng user chỉ hỏi runbook. Ai có quyền đề xuất, ai được thực thi và điều kiện nào còn thiếu? Vì sao chọn `--session` trong CLI không là login production?

## 5. Ma trận đọc

Tự lập bảng cho 5 principal ×8 resource từ data.py. Chỉ rõ vì sao Bob không đọc B-INTERNAL, Carol không đọc B-INTERNAL nhưng đọc G-INTERNAL. GLOBAL được chia sẻ với điều kiện gì? Có bao nhiêu cặp subject/resource được đọc?

## 6. Lọc trước retrieval và noninterference

Tại sao lọc sau rerank/LLM quá muộn? Viết code thêm một document tenant green nhiều term query rồi chứng minh score/thứ hạng retrieval của Alice không đổi. Giới hạn của phép thử ấy là gì?

## 7. Direct, indirect và benign

Phân loại role spoof trong user text, command trong runbook, command trong tool result và lệnh trong memory. Vì sao một request “phân tích chuỗi ignore previous instructions” có thể là benign? Nêu false positive/false negative của chặn bằng keyword.

## 8. Dữ liệu hợp quyền nhưng không có authority

Chạy context lớn đủ chứa B-POISON và M-LAUNDER. Chúng có thể còn trong prompt không? Role server và result khi Alice đề xuất close_ticket thay đổi ra sao? Kiểm này có chứng minh LLM không bị dẫn sai nội dung không?

## 9. Lineage summary

Summary thuộc Alice nhưng phụ thuộc B-PUBLIC và G-INTERNAL. Có được cấp cho Alice không? Viết điều kiện Readers(summary). Nếu bỏ origin green khỏi metadata mà giữ nguyên nội dung, đã giải quyết rò rỉ chưa? Phân biệt provenance với semantic faithfulness.

## 10. Cache sau revoke

Warm cache Alice, dùng cùng query cho Bob và Carol, rồi revoke B-INTERNAL. Entry nào được reuse? Vì sao vẫn cần revalidation nếu key có revision? Thử mô phỏng quên bump epoch. Sau khi dữ liệu đã được gửi, revoke có làm bên nhận quên được không?

## 11. Schema và dispatch

So proposal đúng JSON nhưng sai tenant với proposal thêm field roles, tool không allowlist hoặc bool version. Tầng nào chặn? Vì sao kiểm schema cần thiết nhưng chưa đủ? Code có eval/exec text model không?

## 12. Grant cụ thể và TTL

Cấp lúc 100, TTL 30: lúc 129/130 hợp lệ không? Thử đổi target hoặc reason, dùng grant của ops_blue cho ops_peer và replay. Vì sao hash proposal không tự cấp quyền? Controller cấp grant có được expose như model tool không?

## 13. TOCTOU và atomicity

Review ticket version 1, trước execute ticket đổi version 2 hoặc quyền bị revoke: xử lý thế nào? Thử hai thread dùng cùng grant; điều kiện cần là chỉ một effect. Vì sao RLock demo chưa giải quyết nhiều process/worker hoặc crash/durability?

## 14. Metric và residual case

Tính giả định .4 ×.1=.04 bằng chain rule, không giả định độc lập. Từ report, nêu benign count, attack count và event hai executor; tách context privacy/freshness/authority. Vì sao 0/17 không là chống injection 100%? Ca BANANA_OVERRIDE còn sai điều gì?

## 15. Audit và đổi model

Log trường nào để điều tra denial mà không ghi credential/raw body? Vì sao report nghiên cứu tổng hợp không phải mẫu production logging? Khi thay scripted proposal bằng LLM, cần giữ model/sampling/attack goal/reset state/utility gì để kết quả có ý nghĩa?

## 16. Dự án nhỏ

Tạo 2 tenant, ít nhất 3 vai trò, document public/internal, một memory trộn lineage và hai tool (một read, một write). Xây policy và gateway độc lập model, giữ action grant ngoài prompt. Tạo benign/adversarial fixtures, một ca revoke/cache và một residual semantic attack chưa chặn. Báo đúng mẫu số và giới hạn; nếu thêm LLM phải tách kết quả hành vi model khỏi invariant code, không dùng script lập trình sẵn như benchmark model.
