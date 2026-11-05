# Bài 24 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán trước khi chạy. Các bài số dùng đúng tokenizer, khoảng nửa kín và quy ước metric của bài; không thay bằng số token của một model khác mà không ghi rõ.

## 1. Thiết kế chunk cho runbook

Một trang có 401,403 và hướng dẫn cấp scope. Query hỏi “đã login nhưng bị cấm xem tài nguyên”. Nêu một lỗi của cách index cả trang, một lỗi của cách tách từng câu và một cách giữ ngữ cảnh mà không đưa cả trang vào mỗi chunk. Vì sao đúng source vẫn có thể thiếu bằng chứng?

## 2. Cửa sổ và chi phí

Tính toàn bộ cửa sổ cho L=13,C=6,O=2; tổng lượt token body và tỷ lệ so nguồn. Làm lại với L=10. Chứng minh công thức số chunk; giải thích vì sao không sinh thêm đuôi sau khi đã chạm end. O=C phải xử lý thế nào?

## 3. Coverage và offset

Bằng chứng token `[5,9)` có được giữ nguyên khi C=6,O=0 và O=2 không? Nếu bằng chứng dài hơn C thì sao? Offset trong manifest là ký tự, byte hay token? Viết kiểm chứng text slice và evidence quote cho tất cả source/chunk/query.

## 4. Ba loại budget

Reranker limit128, query17, special3, prefix9: body còn bao nhiêu? Chunk body110 thì vượt bao nhiêu? Tại sao embedding model còn chỗ không chứng minh reranker/context cuối còn chỗ? Một API gọi `stride=2` có chắc bước tiến bằng 2 không?

## 5. Masked mean

H=`[[2,0],[0,2],[100,100]]`, mask=`[1,1,0]`. Tính mean và vector normalize. Nếu bỏ mask hoặc chia cho 3 thì có gì đổi? Vì sao key padding mask trong Transformer chưa thay thế mask khi pooling?

## 6. Contrastive loss và gradient

Score `[.8,0,−.6]`, positive cột 0, τ=.2: tính logits, xác suất softmax, loss và gradient theo score. Giải thích dấu gradient. Tính triplet loss khi margin=.2, positive=.8, negative=.65. Hàm loss nào thực sự được dùng để train bi-encoder của lab?

## 7. Positive trùng và false negative

Hai query cùng trỏ một chunk. Vì sao tạo ma trận vuông và chỉ nhận đường chéo là positive có thể gây sai? Với logits `[2,1,0]`, positive ở cột 0 và 1, tính loss đa positive ở phụ lục; so với chỉ cột 0 positive. Khi nào hard negative lại là nhãn sai?

## 8. Trần của reranker

Một query chỉ có B chứa span. Candidate `[A,B,C]`, final trước rerank `[A,B,C]`, sau `[B,A,C]`: tính CandidateHit@3, EvidenceHit@1 và RR@3 trước/sau. Nếu candidate `[A,C,D]`, reranker tốt nhất có đạt Hit=1 được không? Chứng minh bất đẳng thức theo từng query.

## 9. Logit và BCE

Reranker chấm A=−.2,B=1.4,C=−.7, chỉ B positive. Tính BCE không trọng số cho A và B. Vì sao logit 1.4 hợp lệ? Pos_weight=5 có cho phép coi sigmoid(logit) là xác suất relevance trên traffic thật không? Tính nghiệm weighted BCE lý tưởng với p=.2,w=5.

## 10. Đọc metric chunking

Từ report, lập bảng số chunk, lượt token, coverage và SourceHit cho ba cấu hình. Vì sao SourceHit=1 mà EvidenceHit có thể chỉ .5556? Nêu ít nhất hai lý do bảng này chưa chứng minh section chunking luôn tốt nhất.

## 11. Luồng tensor và tương tác

Viết shape từ query/document IDs đến contrastive score của lab. Cross-encoder nhận gì và output gì? Nó có cần causal mask không? Bi-encoder mean-word có phân biệt hai câu cùng term count nhưng đổi thứ tự không? Viết kiểm chứng điều cuối.

## 12. Chẩn đoán bằng kết quả thật

Chạy train, ghi best epoch và dev MRR của hai model. So train loss cuối với checkpoint được dùng. Tìm query test về cache invalidation và connection pool; xem ứng viên/rank trước/sau. Với số đo lab, bạn có chọn triển khai reranker để tăng chất lượng không? Vì sao?

## 13. Bundle và lỗi vector cache

Load bundle, đối chiếu retrieve/rerank với pipeline vừa train hoặc load độc lập. Trong một bản sao tạm, đổi một tọa độ vector cache mà giữ model: loader nên phát hiện gì? Nếu chỉ đổi reranker, vector bi-encoder có nhất thiết cần encode lại không? Bundle có đủ để resume train chính xác không?

## 14. Split và rò rỉ

Vocabulary fit từ dữ liệu nào? Nhãn/corpus nào được dùng để mine negative? Khi positive train không vào candidate thì code làm gì, dev/test có làm vậy không? Query test khác text nhưng cùng intent/source/span với train có kiểm được tổng quát hóa sang intent mới không?

## 15. No-answer và các hướng nối tiếp

So query toàn OOV với “cache trên sao Hỏa”: rỗng là model tự biết không có đáp án hay policy gate? Phân biệt parent–child retrieval, late chunking và overlap. Tại sao late chunking không thể được tái hiện bằng cách cộng các vector chunk độc lập?

## 16. Dự án nhỏ

Tạo corpus có heading/code/điều kiện phủ định, ít nhất 12 query answerable và vài no-answer; giữ quyền sử dụng dữ liệu. Gán evidence theo source/version/offset trước khi chia; thử ít nhất hai chunker. Đo coverage, CandidateHit, final EvidenceHit/MRR, token count và ba lỗi cụ thể. Nếu dùng neural model, tách train/dev/test và ghi giới hạn intent/source; nếu quá ít nhãn, giữ baseline đã đủ để làm phần chunking. Nêu điều kiện nào sẽ khiến bạn thay chunker, retriever hoặc reranker.
