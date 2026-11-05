# Bài 25 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Giải thích trước khi chạy. Số policy trong bài là giả định; dùng đúng snapshot, tokenizer và metric đã nêu, không coi đó là thông số một dịch vụ thật.

## 1. Chọn thiết kế theo nhu cầu

Một trợ lý cần trả lời policy nội bộ đổi hàng tuần, theo định dạng cố định và có nguồn. Phần nào RAG giúp, phần nào fine-tuning có thể giúp? Vì sao đưa cả corpus vào context hoặc chọn model context dài chưa tự giải quyết quyền/nguồn đúng?

## 2. Xác suất tính tay

Ba document có prior .6/.3/.1 và xác suất sinh cùng y là .7/.2/.1. Tính marginal RAG-Sequence. Vì sao số đó không là xác suất câu trả lời đúng? Với ví dụ hai token/two-document trong phụ lục, tính lại .09 và .25, giải thích tổng–tích khác nhau.

## 3. Theo dấu provenance

Phân biệt source ID/version, chunk ID và evidence ID cục bộ. Nếu chunk bắt đầu ở200, citation `[10,40)` thì span nguồn nào? Viết kiểm chứng mọi citation hợp lệ trong một result về source text; vì sao offset không tính header file hiển thị?

## 4. ID thật nhưng không có trong context

Một source có trong index, nhưng chunk bị packing loại. Model viện dẫn ID đó có nên được chấp nhận không? So với ID hoàn toàn bịa, lỗi khác nhau ở đâu? Vì sao tìm source khác cùng từ để thay ID không phải kiểm chứng semantic?

## 5. Quote đúng, claim sai

Dùng fixture `real_quote_wrong_claim`, chỉ ra điều gì pass và điều gì fail. Tạo thêm một phản ví dụ phủ định: một quote thật “không retry thiếu quyền” nhưng claim bảo retry mọi lỗi thiếu quyền. Vì sao token overlap hoặc schema chưa đủ?

## 6. Ba claim, bốn citation

Giải lại ví dụ C1/C2/C3: attachment, grounded claim rate, citation precision, required fact recall và task success. Giải thích vì sao required recall=1 vẫn task failure. Nếu bỏ claim C3, các số thay đổi thế nào khi vẫn giữ cite TTL thừa ở C2?

## 7. Multi-source support và suy luận

Claim tổng hợp cần cả fact thời hạn key và deadline, nhưng chỉ cite policy key. Có được full support không? Nếu cite cả hai thì sao? Với claim 3 ngày, phải dùng fact nào và phép tính nào? Vì sao rule đổi đơn vị trong catalog không biến scorer thành NLI tổng quát?

## 8. Budget và greedy packing

Window 512, instructions 96, question 24, history 64, output 96: tính phần context. Chi phí ứng viên 150/120/112 theo rank: greedy giữ gì và bộ nào vừa budget nếu chọn lại? Có thể dùng gold facts để chọn lúc serving rồi báo đó là chất lượng thực không? Vì sao phải đếm metadata/separator?

## 9. Phiên bản và mâu thuẫn

Chọn P01 version ở2026-06-01,2026-09-01 và 2026-11-04. Vì sao câu trích 24 giờ có thể đúng ở query này nhưng sai ở query khác? Hai runbook upload cùng hiệu lực ghi 20 MB/25 MB cần hành vi gì nếu chưa có authority policy?

## 10. Trạng thái và no-answer

Phân biệt answer/partial/abstain/conflict bằng ví dụ. Có query chưa đủ context để trả lời đủ: partial trung thực có thể tốt hơn bịa thế nào dù task completeness chưa đạt? Vì sao abstain không được tự có grounded rate=1? So T11/T12 của lab.

## 11. Tách các tầng khi điều tra lỗi

Với cùng query, pool chứa đủ facts, candidate đủ, packed thiếu một fact, answer chỉ nêu phần còn lại có cite đúng. Lỗi đầu tiên ở tầng nào? Nếu packed đủ mà generator vẫn sai thì sao? Thiết kế một thí nghiệm oracle context và nói rõ điều không được suy từ nó.

## 12. Micro, macro và mẫu số

AnswerA có 1 claim/support 1, answerB có 9 claim/support 0. Tính micro và macro grounded rate. Fixture parse/schema lỗi có nằm trong mẫu số task không? Các claim metric của lab có đếm candidate bị reject cơ học nhưng schema-valid không?

## 13. Chạy và đọc report

Chạy all, ghi số snapshot/current public chunk, split,22 fixture và kết quả chính. Mở requests/T09 và T12: vì sao quote/grounding tốt vẫn task failure? So cap 40/110/180, chỉ ra tác dụng và giới hạn của việc tăng context.

## 14. Renderer và contract

Vì sao server nên dựng URL từ mapping thay vì tin URL model? Viết code chứng minh quote bịa/ID lạ bị reject, trong khi quote thật cho claim sai vẫn có thể pass. Nếu claim text chứa Markdown link/HTML, renderer nên xử lý thế nào? `accepted=True` có bảo đảm nội dung đúng không?

## 15. Thay generator bằng LLM

Nêu những dữ liệu adapter được nhận và những nhãn đánh giá không được cấp cho nó. Cần ghi model/tokenizer/sampling/budget gì? Vì sao các con số cap/token/fixture của lab chưa là benchmark LLM? Sau khi chỉnh pipeline theo lỗi test, cần xử lý holdout thế nào?

## 16. Dự án nhỏ có thể chia sẻ

Tự tạo hoặc dùng nguồn có quyền sử dụng: ít nhất 6 source, một thay version theo thời gian, một query cần hai nguồn, một conflict và hai no-answer. Gán required facts và support spans, tách serving/evaluation. Chạy baseline, giữ prompt/context/candidate/snapshot/trace, đo riêng retrieval, packing, grounding, task success. Viết ba lỗi và một thay đổi có thể kiểm chứng. Nếu chưa có LLM, báo rõ baseline; nếu có, đừng dùng annotation oracle để tạo câu trả lời được gọi là serving.
