# Bài tập 52 — Context engineering và memory cho agent

[← Bài chính](README.md) · [Đào sâu](deep-dive.md) · [Lời giải — mở sau khi làm](solutions.md)

Code chạy từ thư mục bài 52, process sạch, dữ liệu tạm synthetic. Không dùng API key hoặc dữ liệu công ty.

## 1. State, context và memory

Một run có checkpoint 40 KB nhưng packet input 7 KB. Vì sao điều đó có thể hợp lý? Long-term memory có thay checkpoint được không? Memory có phải model weights đã được train thêm không?

## 2. Ngân sách và đơn vị

Tính available input từ window_bytes=14.000 và reserve_bytes=2.000. Vì sao không gọi 12.000 bytes là 12.000 tokens? Metadata/JSON/tool schema có được tính không? Nếu phần bắt buộc quá lớn thì nên làm gì?

## 3. Candidate và review

Tạo một hint candidate. Kiểm nó chưa được recall, agent không tự approve được, reviewer hợp lệ có thể approve. Review fixture đã chứng minh nội dung đúng và có human consent thật chưa?

## 4. Namespace trước ranking

Tạo memory cho agent A và agent B, cùng trigger XQ7. Khi A recall, candidate pool/diagnostics nên chứa gì? Vì sao rank toàn store rồi lọc top-k sai scope có thể vừa thiếu kết quả vừa lộ dữ liệu?

## 5. Lineage và payload

Đề xuất query không khớp report source, ref hash giả hoặc refs từ hai origin run khác nhau. Guard cần từ chối vì sao? Hash/substring khớp có chứng minh mapping alias → symptom đúng ngữ nghĩa không?

## 6. TTL, source update và revoke

Hint hết hạn ở 150: recall tại 149 và 150 khác gì? Source đổi version hoặc bị revoke trước 150 có được dùng memory không? TTL còn hạn có chứng minh dữ liệu vẫn đúng không?

## 7. Conflict và recency

Tạo hai active mappings cùng key nhưng query khác. Record mới hơn có tự được chọn không? Resolve bằng winner/expected epoch hoạt động thế nào? Đây là conflict ứng dụng hay đã chứng minh contradiction logic?

## 8. Summary có nguồn và unknown

Chạy một run không memory tới sau search rỗng, compose working summary và sửa evidence/quote trong packet. Validate phải làm gì? Vì sao cần giữ empty/unknown/phủ định cùng observation refs? So lại cùng projection có phát hiện mọi semantic omission không?

## 9. Đọc phép so compact/full

Từ tổng 25.079 và 26.206 bytes, tính chênh tuyệt đối và tỷ lệ theo full. Vì sao lượt đầu compact lớn hơn 3 bytes? Có thể kết luận giảm token/cost hoặc model quality tăng không?

## 10. Poison và authority

Phân tích proposal thêm instructions/approved, episode tự khẳng định root cause, và case query chứa “close_ticket”. Trường hợp nào bị schema chặn, trường hợp nào vẫn có thể là data string hợp lệ? Vì sao script không thực thi write chưa là LLM injection benchmark?

## 11. Bỏ memory hay cắt instruction?

Đặt budget chỉ vừa mandatory packet. Item tùy chọn nên bị xử lý ra sao? Thử max_memories=0 và required_context_budget. Vì sao không cắt raw string cuối cùng cho vừa hoặc bỏ source refs để lấy thêm nhiều memory hơn?

## 12. Memory đổi trong lúc resume

Pause run sau O1, forget hint rồi resume cùng run ID. Kết quả nào đúng với binding policy? So với forget xảy ra sau compose nhưng trước consume. Epoch toàn store có nhược điểm gì khi update không liên quan?

## 13. Logical forget và xóa dữ liệu

Kiểm payload/refs của memory row sau forget, tombstone và source registry còn gì. Proposal cũ cùng key có quay lại được không? Checkpoint, backup và alias mới có tự bị xóa không? Cần thêm gì cho deletion policy thực tế?

## 14. Chọn gì để nhớ, khi nào lấy lại

Trong trợ lý ticket, chọn một thông tin đáng nhớ lâu và một thông tin chỉ nên dùng trong run. So hot-path write với background consolidation. Khi nào nên re-fetch nguồn thay vì tin memory/summary cũ? Phần nào chưa có trong lab?

## 15. Thiết kế đánh giá

Đề xuất metric cho memory promotion, retrieval trong scope, compaction, stale/revoke/forget, wrong-memory influence và task outcome. 20 cases + cross-run demo/476 checks cho biết gì và chưa cho biết gì? Vì sao không chỉ giữ warm_compact thành công?

## 16. Mở rộng sang LLM thật

Thiết kế adapter nhận đúng packet, tokenizer/request budget, memory write proposal/review, semantic summary checks, context receipts, cache invalidation và evaluation. Nêu quyền/tool/final gate nào phải giữ ở host, dữ liệu nào không nên persist, và những claim chưa được phép đưa ra trước khi thử thật.
