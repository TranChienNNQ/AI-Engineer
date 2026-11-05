# Bài 23 — Đào sâu biểu diễn, index và đánh giá

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Sparse và dense là cách biểu diễn, không phải mức độ thông minh

Vocabulary 100.000 term tạo vector TF-IDF có 100.000 chiều, nhưng một đoạn ngắn chỉ có vài chục tọa độ khác 0: đó là **sparse — thưa**. Có thể lưu cặp `(term_id, weight)` thay vì cả mảng. Vector dày 384 chiều thường lưu đủ 384 số; mỗi chiều không tương ứng rõ với một từ. Các con số này chỉ là ví dụ, không phải yêu cầu bắt buộc của thuật toán.

Một vector nhiều chiều vẫn có thể lưu thưa; vector ít chiều vẫn có thể kém cho tác vụ. Một số retriever học vector sparse bằng neural model; vì vậy sparse/dense và learned/non-learned là hai trục khác nhau. BM25 dựa trên term không có nghĩa hệ thống chỉ tìm chuỗi bằng `in`: analyzer, field, phrase, synonym và ranking đều ảnh hưởng.

Hãy đặt câu hỏi cụ thể: biểu diễn giữ được mã lỗi không? hiểu phủ định không? xử lý tiếng Việt/Anh trộn lẫn không? phân biệt phiên bản API cũ và mới không? “Embedding hiện đại” chưa trả lời các câu đó.

## 2. Phép chiếu tính tay: giảm chiều có thể mất gì?

Đây là **phép chiếu giả định**, không phải components fit từ corpus 20 document. Ba chiều đầu vào lần lượt là `[refund, hoàn_tiền, token]`. Chọn hai hàng trực chuẩn:

$$C=\begin{bmatrix}1/\sqrt2&1/\sqrt2&0\\0&0&1\end{bmatrix}.$$

Với q=`[1,0,0]`, d1=`[0,1,0]`, d2=`[0,0,1]`, trước chiếu q·d1=q·d2=0. Sau chiếu z=xCᵀ:

```text
q_z  = [1/√2, 0]  -> normalize -> [1,0]
d1_z = [1/√2, 0]  -> normalize -> [1,0]
d2_z = [0,    1]  -> normalize -> [0,1]
```

Cosine q–d1 thành 1, q–d2 vẫn bằng 0. Ta đã chọn C để gộp hai từ liên quan; đây chỉ giải thích phép toán. Một SVD thật chỉ tìm cấu trúc của X, **không được cung cấp ý nghĩa “refund=hoàn tiền”**. Nếu corpus thiếu mẫu đồng xuất hiện, phép chiếu học được chưa chắc gộp chúng. Tương tự, nếu gộp nhầm `token_bucket` và `access_token`, độ chính xác có thể giảm.

Ở SVD, hàng components là các hướng trực chuẩn trong không gian term. `X @ components.T` tương ứng tọa độ document trong không gian giảm chiều trước bước normalize; với phân rã cùng X, nó bằng U_rΣ_r. Lab không chia cho singular value để tạo tọa độ U_r, vì làm vậy là đổi cách cân trọng số các chiều. Không thể thay lẫn các quy ước rồi coi cosine chắc chắn giữ nguyên.

Một chiều SVD có thể đồng thời đổi dấu trong document và query mà dot không đổi. Đổi dấu document nhưng giữ query ở hệ tọa độ cũ sẽ làm hỏng score. Khi singular value trùng/gần nhau, các hướng có thể khác giữa môi trường; so thứ hạng/score và lưu cùng components hữu ích hơn so từng dấu tọa độ.

```python
import numpy as np
from lab import normalize_rows
C = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 1]])
np.testing.assert_allclose(C @ C.T, np.eye(2))
x = np.eye(3)
z = normalize_rows(x @ C.T)
np.testing.assert_allclose(z @ z[0], [1, 1, 0])
print(z)
```

## 3. Vì sao BM25 không chỉ là TF-IDF đổi tên?

Cả hai dùng thống kê term/corpus, nhưng code bài có khác biệt quan trọng:

| Thành phần | TF-IDF trong lab | BM25 trong lab |
|---|---|---|
| TF trong document | Số lần xuất hiện tuyến tính | Hàm bão hòa |
| Độ dài | L2 normalize vector | `|d|/avgdl` trong mẫu số |
| IDF | `ln((N+1)/(df+1))+1` | `ln(1+(N-df+.5)/(df+.5))` |
| TF trong query | Đếm rồi normalize | Chỉ duyệt term duy nhất |
| Thang điểm | Cosine không âm của TF-IDF | Tổng trọng số BM25, không giới hạn 1 |

Đặt `L=k1(1-b+b|d|/avgdl)` cố định. TF factor `tf(k1+1)/(tf+L)` tiến đến `k1+1` khi tf tăng rất lớn. Đạo hàm theo tf là `(k1+1)L/(tf+L)^2`, dương nhưng giảm dần. Đây là lý do lặp từ thêm vẫn có thể tăng điểm nhưng lợi ích nhỏ dần. Nếu b=0, L=k1: hiệu ứng độ dài biến mất, bão hòa vẫn còn.

Không có một ngưỡng score BM25 dùng chung cho mọi corpus. Thêm tài liệu làm đổi N/df/avgdl; analyzer mới làm đổi term/độ dài; query dài có thể cộng thêm nhiều phần điểm. Với cosine cũng vậy, phân bố score phụ thuộc model, corpus và query. Nếu cần quyết định abstain, cần tập nhãn có no-answer và tiêu chí chi phí của ứng dụng; không chọn 0.8 chỉ vì trông “cao”.

## 4. Candidate generation và reranking là hai tầng khác nhau

Retriever tạo vài chục/vài trăm ứng viên; reranker chấm lại các ứng viên bằng tín hiệu đắt hơn. Nếu tài liệu đúng không vào tập ứng viên, reranker không thể tạo nó từ hư không. Vì vậy cần đo Recall@candidate_depth ở tầng đầu, rồi NDCG/MRR sau rerank; cải thiện một tầng có thể không làm metric cuối tăng nếu tầng khác đang là nút thắt.

RRF dùng vị trí, nên một nhánh có top-1 yếu vẫn đóng góp bằng trọng số top-1 của nhánh tốt. Hai nhánh gần giống nhau cũng có thể “bỏ phiếu” lặp cho cùng một thiên lệch. Weighted RRF hoặc fusion sau hiệu chỉnh score là các lựa chọn nghiên cứu tiếp; thêm độ phức tạp phải có bằng chứng dev và kết quả test độc lập.

## 5. Exact top-k recall và relevance recall

Giả sử exact vector search trả top-3 `[A,B,C]`, ANN trả `[A,D,E]`. ANN recall so với exact là 1/3. Nhưng qrels chỉ đánh A và D liên quan thì relevance recall của ANN là 2/2, exact là 1/2. Hai phép đo có mẫu số khác nhau, không mâu thuẫn.

Exact là đúng với **hàm khoảng cách và vector đã chọn**, chưa phải chân lý ngữ nghĩa. Tối ưu ANN recall giúp tái hiện exact; nó không chứng minh exact đã chọn model đúng. [Faiss](https://faiss.ai/) là nguồn định hướng các đánh đổi exact/approximate; lab mới triển khai exact bằng NumPy, không đo tốc độ ANN.

## 6. Thiết kế evaluation có ích hơn một con số đẹp

Report của lab lưu query split/slice, top-3 và qrels. Hãy phân tích ít nhất các slice: mã lỗi chính xác; câu diễn đạt lại; từ đa nghĩa; no-answer. Macro mean cho mỗi query trọng số như nhau; nếu ứng dụng có query phổ biến hơn nhiều, có thể bổ sung traffic-weighted metric và vẫn giữ bảng theo slice. Không có log thực tế trong lab, nên chưa có trọng số traffic có cơ sở.

Một tập 10 query answerable quá nhỏ để suy rộng. Chưa có nhãn độc lập từ nhiều người, time split, tenant split, kiểm dữ liệu gần trùng hoặc bootstrap uncertainty. Dev/test không trùng query text chưa bảo đảm tách ý định hoàn toàn. Nếu sau khi đọc lỗi test bạn chỉnh analyzer/corpus/model, hãy coi test đó đã tham gia phát triển và dùng tập test mới cho báo cáo độc lập.

P@3 thấp trong bảng không hẳn do truy xuất sai tài liệu đứng đầu: nhiều query chỉ có 1 tài liệu relevant, nên P@3 tối đa theo nhãn cũng chỉ là 1/3. Đừng đổi mẫu số sang “số kết quả đã trả” chỉ để làm đẹp. Nếu cần thêm metric đó, đặt tên và công thức khác, báo song song.

Query no-answer cần bài đo riêng: tỷ lệ trả nhầm ứng viên, tỷ lệ hệ thống cuối trả lời khi không đủ nguồn, độ chính xác của abstention và chi phí bỏ lỡ query có đáp án. Những phép đo này liên quan nhưng không đồng nhất. Lab chỉ đo `returned_any` ở tầng truy xuất.

## 7. Checklist vận hành dùng kiến thức backend

Đây là hướng thiết kế để tự phân tích, chưa là tính năng đã có trong code:

| Khi xảy ra | Cần xác định |
|---|---|
| Đổi model/query prompt | Tương thích doc vector, metric, dimension, normalize, version |
| Sửa/xóa tài liệu | Source ID, document version, trạng thái index, vô hiệu hóa cache |
| Deploy index mới | Tạo snapshot, kiểm chất lượng, chuyển phiên bản, rollback |
| Thêm tenant/scope | Quyền từ server, cách ly fit/index/cache/log, lọc cả ứng viên rerank |
| Độ trễ tăng | Thời gian analyze/encode/filter/search/fusion riêng, không chỉ tổng |
| Query không có kết quả | OOV, quyền, index chưa cập nhật, nhu cầu ngoài corpus, lỗi hạ tầng |

`corpus_hash` giúp nhận biết nội dung corpus khác nhau, không chứng minh dữ liệu đúng hay chưa bị kẻ tấn công sửa. Report còn có `queries_hash`, cấu hình, analyzer version, Python và NumPy. Hash không thay thế provenance, quyền truy cập hoặc đánh giá nhãn.

Nếu chuyển lab thành API, không nhận `allowed_scopes` nguyên xi từ client như quyền có hiệu lực. API cần suy ra quyền từ danh tính đã xác thực; truy xuất chỉ nhận tập quyền do server cung cấp. Bài chưa làm auth, chưa cách ly thống kê corpus, chưa có concurrency/update service. Không đánh đồng demo filter với hoàn thành bảo mật đa tenant.

## 8. Các nhận định cần tránh

“Vector luôn hiểu đồng nghĩa” thiếu điều kiện về model, dữ liệu và tác vụ. “BM25 lỗi thời” bỏ qua truy vấn có identifier và khả năng làm baseline. “Hybrid chắc thắng” bị chính Q12 phản ví dụ. “Cosine cao là câu trả lời đúng” nhầm độ tương tự biểu diễn với chứng cứ. “Top-k lúc nào cũng có k kết quả” bỏ qua filter và việc không pad điểm 0. “Không cần test vì search không train classifier” bỏ qua các tham số analyzer/index/encoder được chọn qua thử nghiệm.

[Trở lại bài chính](README.md) để chạy lab, rồi dùng [bài tập](exercises.md) kiểm tra từng phân biệt này.
