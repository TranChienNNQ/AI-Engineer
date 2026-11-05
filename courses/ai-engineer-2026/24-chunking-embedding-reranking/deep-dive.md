# Bài 24 — Đào sâu chunking, mục tiêu học và vòng đời index

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Một lần đổi chunker thực sự đổi những gì?

Đổi size/overlap không chỉ đổi số vector: nó đổi độ dài document, df/avgdl của BM25, nội dung đầu vào encoder, prefix lặp, tập ứng viên và các negative mà miner chọn. Vì vậy một bảng “size A thắng size B” cần nói rõ những thành phần còn lại có fit lại, có giữ cùng quyền/corpus/query và có chọn tham số trên dev hay không.

Nếu muốn tách riêng ảnh hưởng ranh giới, có thể giữ cùng prefix, encoder đã đóng băng, metric và nguồn; chỉ thay cách chia. Nếu muốn đánh giá pipeline tốt nhất của mỗi cách chia, có thể fit/chọn tham số lại nhưng phải dùng quy trình và budget phát triển công bằng. Hai câu hỏi khác nhau; đừng diễn giải phép đo thứ hai như một ablation cô lập.

Bảng lab giữ BM25 làm thuật toán chung nhưng thay cả size, ranh giới và heading. Coverage dựa span cho biết bằng chứng có còn nguyên hay không; phần xếp hạng vẫn chịu ảnh hưởng của title/heading/statistics khác nhau. Kết quả không chứng minh mọi yếu tố cải thiện đều đến từ cấu trúc section.

## 2. Công thức số cửa sổ và chi phí

Với L>0, C>0, 0≤O<C, đặt S=C−O. Chunk đầu phủ tối đa C token. Nếu còn L−C token, cần thêm `ceil(max(0,L−C)/S)` bước. Dừng ngay khi end=L; không tiếp tục sinh các đoạn đuôi đã nằm trong chunk trước.

Cửa sổ không có overlap cũng có thể có chunk cuối rất ngắn; section chunking còn có đuôi ngắn ở nhiều section. Những đuôi này có thể chứa ít bằng chứng nhưng prefix lại chiếm tỷ lệ lớn. Có thể ghép đuôi vào chunk trước nếu không vượt model budget, hoặc đổi ranh giới câu; đó là phương án cần đánh giá, chưa có trong code.

Với index float32, N vector chiều d tốn `4Nd` byte thô. Lab 25×32 chỉ là 3,200 byte vector. Thực tế còn metadata, ANN structure, tokenizer/model weights, cache và bản sao; số byte vector không là toàn bộ RAM hệ thống. Overlap tăng N và lượt token encode, chưa chắc tăng số token **mới** đem lại bằng chứng trong context cuối.

## 3. Nhiều positive: vì sao CE đường chéo có thể sai?

Trong bài, 36 query có 25 cột chunk, target là cột đúng của từng query. Không tạo ma trận vuông chỉ vì một hướng dẫn in-batch loss thường dùng đường chéo. Hai query cùng hỏi refund có thể chung positive; khi duplicate positive thành hai cột rồi xem cột kia là negative, loss tự tạo mâu thuẫn.

Nếu một query có tập positive P và một tập ứng viên A, một mục tiêu đa positive khả dĩ là:

$$L=-\ln\frac{\sum_{j\in P}\exp(s_j/\tau)}{\sum_{j\in A}\exp(s_j/\tau)}.$$

Công thức này khuyến khích tổng xác suất positive lớn; chưa buộc mọi positive có score đều cao. Nếu cần tất cả loại bằng chứng được lấy, có thể cần mục tiêu và metric khác. Không có một sửa đổi loss duy nhất đáp ứng mọi định nghĩa relevance.

Với logits `[2,1,0]`, positive là cột0 và cột1: loss=`logsumexp(2,1,0)−logsumexp(2,1)≈.094344`. Nếu chỉ cho cột0 là positive, loss≈.407606. Chênh lệch không là “model tốt lên” vì đã đổi nhãn/mục tiêu.

```python
import torch
x = torch.tensor([2.,1.,0.], dtype=torch.float64)
multi = torch.logsumexp(x,0) - torch.logsumexp(x[:2],0)
single = torch.logsumexp(x,0) - x[0]
assert 0 < multi < single
print(float(multi), float(single))
```

Lab dùng một canonical span/query và chỉ chấp nhận train khi có đúng một chunk chứa span. Khi bạn đổi chunker sang overlap, điều kiện này có thể không còn đúng; cần sửa mục tiêu đa positive hoặc cách lấy mẫu có chủ đích, không tắt assert để train tiếp một cách mơ hồ.

## 4. Chọn negative là một phần thiết kế dữ liệu

Random negative dễ tạo loss thấp nhưng ít rèn phân biệt các trường hợp gần nhau. Hard negative từ retriever hiện tại thường sát phân bố lúc serving hơn, nhưng có thể là false negative do nhãn chưa đủ. Hãy rà các cặp “cùng mã khác điều kiện”, “cùng trang khác section”, “phiên bản cũ/mới” trước khi coi chúng là âm.

Miner trong lab là hybrid đã đóng băng sau khi chọn bi-encoder theo dev. Nó chỉ đọc query/nhãn train để tạo cặp; không mine gold từ test. Nếu positive không vào top-6 train, code thêm nó vào tập train, nhưng dev/test chỉ dùng ứng viên thật. Làm hai việc khác nhau này không sai nếu nêu rõ; đánh giá bằng candidate có chèn gold mới làm mất ý nghĩa CandidateHit.

Lab không cập nhật miner theo từng epoch cross-encoder. Đổi retriever sau khi đã train reranker có thể đổi phân bố negative và chất lượng; cần kiểm lại cả pipeline. Đừng chỉ giữ điểm reranker trên một bộ candidate cũ rồi coi đó là điểm hệ thống mới.

## 5. Pointwise, pairwise và listwise

**Pointwise** chấm từng pair query–document với label, ví dụ BCE của lab. **Pairwise** học positive cao hơn negative, chẳng hạn `softplus(s_negative−s_positive)`. **Listwise** dùng một danh sách của query để học phân bố/thứ hạng; contrastive CE cũng có thể nhìn theo tập ứng viên như vậy.

BCE train đo phân loại pair, trong khi MRR@3 đo vị trí positive đầu tiên. Hai mục tiêu liên quan nhưng không trùng. Model có thể giảm lỗi trên nhiều negative dễ trong train mà vẫn đổi sai thứ tự các candidate khó ở dev. Train loss giảm còn dev không tốt không tự xác định được duy nhất lỗi nằm ở overfit, nhãn, kiến trúc hay optimizer.

Cross-encoder của lab là một lớp attention hai chiều trên chuỗi ghép, pooling contextual token. Trọng số ngẫu nhiên và 36 query không cung cấp tri thức ngôn ngữ phong phú từ pretraining. Kết quả kém không bác bỏ cross-encoder; nó cho thấy cần baseline và dữ liệu/kiểm chứng trước khi dùng một tầng học thêm.

## 6. Logit, sigmoid và pos_weight

BCE của lab có 36 positive, 180 negative; pos_weight=5. Sigmoid(logit) trong model được train có trọng số **không tự là xác suất relevance theo traffic thật**. Ngay cả trong một bài toán lý tưởng tối ưu weighted BCE, việc tăng trọng số positive làm đổi điểm tối ưu.

Nếu xác suất positive thật ở một x là p và trọng số positive w, nghiệm sigmoid lý tưởng là:

$$r=\frac{wp}{wp+(1-p)}.$$

Chẳng hạn p=.2,w=5 cho r=.5556; giá trị lớn hơn .5 không có nghĩa p>.5. Đây là phân tích lý tưởng của loss, không phải hiệu chỉnh xác suất đã đo cho model lab. Negative mining còn làm phân bố train khác online, nên không dùng công thức này như một giải pháp calibration đầy đủ.

```python
p, w = .2, 5
r = w*p/(w*p+1-p)
assert abs(r-5/9) < 1e-12
print(r)
```

Nếu ứng dụng phải abstain, cần mẫu có đáp án và không có đáp án, định nghĩa chi phí, chọn ngưỡng bằng dev và đánh giá trên holdout đúng phân bố. Lab không có quy trình calibration hay abstention model.

## 7. Parent–child, late chunking và late interaction

Các kỹ thuật sau là hướng nghiên cứu nối tiếp; chưa triển khai trong lab:

| Kỹ thuật | Thay đổi chính | Điều cần kiểm |
|---|---|---|
| Parent–child retrieval | Tìm child nhỏ, trả parent/đoạn lân cận lớn hơn | Lợi ích ngữ cảnh so với nhiễu, quyền và context budget |
| Contextualized text | Thêm title/heading hoặc mô tả ngữ cảnh vào text trước encode | Chi phí, thông tin sai do bước tạo mô tả, version/provenance |
| Late chunking | Contextualize token trên văn bản dài trước, pool theo span sau | Model long-context, offset/token alignment, giới hạn nguồn |
| Late interaction | Giữ nhiều vector/token và tương tác query–document lúc chấm | Dung lượng index, phép tổng hợp và chi phí truy xuất |

**Late chunking không phải thêm overlap.** Abstract [Late Chunking, bản v3 năm 2025](https://arxiv.org/abs/2409.04701) mô tả encode token với ngữ cảnh dài rồi mới chia/pool thành chunk embedding. Phần đọc ở đây là abstract/metadata, chưa dùng kết quả thực nghiệm của paper để khẳng định ưu thế cho corpus này. Không thể áp dụng ý tưởng đó bằng cách mean-pool vector của các chunk độc lập rồi giả định đã có attention qua toàn nguồn.

Late interaction cũng không phải cross-encoder truyền thống: nó giữ biểu diễn đa vector có thể tính trước ở document, rồi mới tương tác theo query. Bảng trên chỉ phân biệt khái niệm, chưa là hướng dẫn triển khai hay lựa chọn model cụ thể.

## 8. Ghép bằng chứng vào context là bước mới

Truy xuất lấy các đoạn; ghép context cần xử lý trùng lặp, thứ tự, nguồn dẫn và budget riêng. Ví dụ hai chunk `[0,100)` và `[80,180)` cùng source/version có 20 ký tự chồng, không nên ghép như 200 ký tự mới. Nếu muốn hợp nhất span, phải xác nhận cùng snapshot và văn bản liên tục; không ghép các source khác nhau rồi mất ranh giới trích dẫn.

Đúng nguồn chưa bảo đảm đủ cả điều kiện, ngoại lệ, phủ định hoặc mã phiên bản. Bài đo single-span của lab hữu ích để thấy lỗi biên nhưng chưa kiểm câu hỏi nhiều bằng chứng. Câu hỏi “khi nào retry và khi nào không” có thể cần hai section; EvidenceHit một span chưa đủ đo câu trả lời đầy đủ. Xem [bài 25 — RAG và nguồn dẫn](../25-rag-nguon-dan/README.md) để tiếp tục luồng trả lời có evidence và citation.

## 9. Version và cache invalidation

Cache document vector theo nội dung chunk + encoder/tokenizer/pooling/normalize version. Cache query vector thêm query preprocessing/instruction. Cache rerank cần query, candidate content/version và reranker version. Một key chỉ gồm source ID dễ dùng vector cũ khi nguồn đổi.

Bundle của lab lưu state dict, chunks/vocabulary/config, analyzer version, model hash và vector cache. Load encode lại chunks để kiểm vector tương thích, nên có chi phí startup. Không fit lại model, không resume optimizer. Hash giúp bắt nhầm revision trong luồng tin cậy; không phải chữ ký số xác thực nguồn hay cơ chế chống người tấn công thay cả file lẫn manifest.

Chỉ số dimension bằng nhau không đủ: model mới có thể đổi hẳn hệ tọa độ. Reranker đổi riêng thì thường không cần encode lại vector bi-encoder, nhưng score cache, evaluation và rollout policy vẫn phải đổi. Chunker đổi thì phải cập nhật chunk ID/text, vector và qrels mapping từ span tương ứng.

## 10. Chọn thí nghiệm tiếp theo từ lỗi quan sát được

Nếu coverage thấp, sửa parser/chunker hoặc cơ chế ghép lân cận trước khi kỳ vọng reranker cứu. Nếu coverage tốt mà CandidateHit thấp, kiểm analyzer/model/M/ANN/filter. Nếu candidate chứa bằng chứng nhưng final Hit thấp, kiểm rerank/data/mục tiêu/selection. Nếu retrieval tốt mà câu trả lời sai, cần đánh giá bước đọc và tạo câu trả lời; tăng M không tự sửa mọi lỗi.

Khi đã xem test và bắt đầu điều chỉnh theo lỗi của nó, ghi nhận test đó là dữ liệu phát triển; chuẩn bị holdout mới cho kết luận độc lập. Với bộ lab nhỏ cùng ý định giữa các split, một bản báo cáo có ý nghĩa là “cơ chế này chạy và có những giới hạn này”, không phải “đã giải quyết semantic retrieval trong production”.
