# Bài 23 — Lời giải

[← Bài chính](README.md) · [Đề bài](exercises.md)

Các snippet Python chạy từ thư mục bài sau khi cài `requirements.txt`. Số làm tròn dùng để đọc; code so số thực bằng tolerance, không đòi chuỗi thập phân khớp tuyệt đối.

## 1. Nhu cầu trước thuật toán

`ERR_BILL_042` cần giữ mã nguyên vẹn và phân biệt với 043; analyzer làm mất underscore hoặc vector gom các mã gần nhau đều có thể gây sai. Query “gửi lại không tạo thêm đơn” cần tìm idempotency; từ khóa có thể hụt khi corpus chỉ dùng tiếng Anh, vector có thể nhầm retry với bảo đảm không trùng hành động. `token` thiếu ngữ cảnh: cả access token và token bucket đều có thể hợp lý; có thể cần làm rõ ý định hoặc trả nhiều nhóm kết quả.

Chưa thể kết luận phương pháp thắng khi chưa có corpus/query/qrels và điều kiện đo. Một query diễn đạt lại vẫn có thể trùng nhiều term hữu ích nên BM25 tìm tốt; tên slice “paraphrase” không chứng minh đã kiểm suy luận ngữ nghĩa mạnh.

## 2. Analyzer và df

Kết quả `['err_bill_042','api','v1/refunds']`. Postings: api→A1,B1; timeout→A2; token→B1,C1. df tương ứng 2,1,2. df đếm document, nên hai lần `timeout` trong A chỉ góp 1 vào df.

Nếu chỉ đổi analyzer query, term `errbill042` chẳng hạn không khớp vocabulary đã lưu `err_bill_042`. Cần version hóa và đồng bộ cách phân tích, kiểm lại index thay vì chỉ sửa một phía.

## 3. TF-IDF

Đặt a=1.2876820725, b=1.6931471806. IDF `[a,b,a]`. A=`[a,2b,0]`, B=`[a,0,a]`, C=`[0,0,a]`, q=`[a,b,0]`. Sau chia chuẩn, cosine A≈0.959146, B≈0.428046, C=0. Timeout chỉ có trong một document nên b lớn hơn a. Thay cả ba IDF bằng một hằng số sẽ giải một bài toán khác.

```python
import numpy as np
from lab import SearchIndex, Config
small = [dict(id=i, title='', text=t, scope='public') for i,t in
         [('A','api timeout timeout'), ('B','api token'), ('C','token')]]
index = SearchIndex(small, Config(lsa_dim=2))
np.testing.assert_allclose(index.df, [2,1,2])
np.testing.assert_allclose(index.scores('api timeout','tfidf'), [0.9591463953,0.4280460351,0], atol=1e-9)
print(index.idf)
```

## 4. BM25

Avgdl=2. IDF api≈0.4700036292, timeout≈0.9808292530. A≈1.5725612027, B≈0.4700036292, C=0. A cộng hai đóng góp term; B chỉ có api, C không khớp.

Với TF=1: độ dài 2 cho factor=1; độ dài 6 cho `2.2/(1+1.2×(.25+.75×3))=2.2/4≈0.55`. Khi b=0, cả hai cùng `2.2/(1+1.2)=1`. Vì thế tắt hiệu chỉnh độ dài không đồng nghĩa tắt bão hòa TF.

## 5. Hình học

Dot `[1,1,4]`, cosine `[1/√2,1/√2,1]`. Thay d3 bằng `[20,20]` làm dot=40, cosine vẫn 1. Với unit vectors u,v:

$$\|u-v\|^2=u\cdot u+v\cdot v-2u\cdot v=2-2\cos(u,v).$$

Điều kiện unit vector là thiết yếu. Không normalize q/d thì độ dài có thể làm đổi thứ hạng L2 hoặc dot; chọn metric phải đúng với cách model được huấn luyện và cách index lưu vector.

## 6. Chiếu không tạo tri thức từ hư không

q_z và d1_z đều `[1/√2,0]`, sau normalize là `[1,0]`; d2_z=`[0,1]`. Cosine lần lượt 1 và 0. Nhưng C được người soạn đặt để gộp hai term, không phải chứng cứ một mô hình đã tự học được quan hệ đó. SVD thật cần cấu trúc dữ liệu thích hợp và vẫn có thể gộp sai.

Query toàn OOV → TF-IDF zero → chiếu zero → normalize zero → score zero → kết quả rỗng. Không có neural encoder để suy biểu diễn từ byte/subword trong lab LSA này.

## 7. RRF

A=`1/11+1/13≈.167832`; B=`1/12+1/11≈.174242`; C=`1/13+1/12≈.160256`: B>A>C. Z không có trong hợp danh sách nên không được xét. ID trùng giữa hai nhánh là bình thường, còn trùng trong một nhánh phải sửa/dedup trước hoặc reject; hàm `rrf` của lab chọn reject để không âm thầm che lỗi.

## 8. Metric

`[B,X,A]`: Hit=1, P=2/3, R=1, RR=1. DCG=`1+0+7/log2(4)=4.5`. IDCG=`7+1/log2(3)≈7.6309297536`; NDCG≈.5897053367.

`[A]`: Hit=1, P=1/3, R=1/2, RR=1. DCG=7; IDCG vẫn tính trên hai tài liệu relevant, NDCG≈.9173194127, chưa bằng 1. Không xóa B khỏi qrels chỉ vì hệ thống chưa lấy B ra.

```python
import numpy as np
from lab import retrieval_metrics
full = retrieval_metrics(['B','X','A'], {'A':3,'B':1})
short = retrieval_metrics(['A'], {'A':3,'B':1})
np.testing.assert_allclose([full['precision'], full['ndcg']], [2/3,.5897053367440438])
np.testing.assert_allclose([short['precision'], short['recall'], short['ndcg']], [1/3,.5,.9173194127129571])
print(full, short)
```

## 9. No-answer

R rỗng khiến Recall có mẫu số 0; IDCG=0 khiến NDCG không xác định theo công thức bài. Gán 0 sẽ lẫn “truy xuất kém trên câu có đáp án” với “đúng là không có đáp án”; gán 1 tự động cũng che lỗi trả tài liệu không liên quan.

Q17 rỗng ở cả bốn nhánh; Q18 vẫn có ứng viên ở cả bốn nhánh, dù nhu cầu không được đáp ứng. Ví dụ thêm: `zzqv999 nebula999` toàn OOV; “TTL của cache ở một dịch vụ ngoài corpus tên Sao Kim” có term quen nhưng yêu cầu chưa có nguồn. Cần kiểm analyzer để xác nhận ca thứ nhất thật sự OOV, và xem lại qrels ca thứ hai thay vì giả định mọi từ quen tạo đáp án.

## 10. Filter

Lấy `[P1,R1,P2]` rồi lọc còn `[P1,P2]`. Lọc trước cho `[P1,P2,P3]`. `None` dùng mặc định public, tập rỗng không cho scope nào. Không dùng toán tử `or` để thay mặc định cho cả hai.

Caller tự khai scope không chứng minh có quyền đó. Production phải suy quyền từ server và bảo vệ cả ứng viên/metadata, không chỉ response cuối. Lab còn fit IDF/LSA toàn corpus nên chưa cách ly thống kê tenant.

## 11. Đọc kết quả theo query

Các số dưới thuộc cấu hình cố định của bài, có thể đổi khi bạn đổi corpus/analyzer/thư viện:

| Query | TF-IDF | BM25 | LSA | Hybrid |
|---|---|---|---|---|
| Q10 top-3 | D19,D03,D13 | D19,D13,D03 | D19,D03,D13 | D19,D03,D13 |
| Q10 NDCG | .955831 | 1 | .955831 | .955831 |
| Q12 top-3 | D07,D03,D15 | D07,D03,D15 | D07,D08,D20 | D07,D15,D17 |
| Q12 NDCG | .917319 | .917319 | 1 | .917319 |

Hybrid thua BM25 ở Q10 và LSA ở Q12. RRF phụ thuộc hai thứ hạng và depth, không chọn “phương pháp đang đúng nhất cho query” bằng một oracle. D19 hoặc D07 đều ở hạng 1, nên Hit/RR=1 không nhìn thấy vị trí tài liệu relevant thứ hai. NDCG có grade thấy chênh lệch đó.

## 12. Round-trip và không refit từ query

Snippet dùng thư mục tạm tự dọn và chặn SVD lúc load. Nếu load cố fit SVD, kiểm chứng sẽ thất bại. Đây là kiểm tra vòng đời index, ngoài việc so một con số xếp hạng.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from data import documents, queries
from lab import SearchIndex
index = SearchIndex(documents())
vocab = list(index.vocab)
idf, components = index.idf.copy(), index.components.copy()
expected = {(q['id'],m): index.search(q['text'],m) for q in queries()
            for m in ['tfidf','bm25','lsa','hybrid']}
assert index.vocab == vocab
np.testing.assert_array_equal(index.idf, idf)
np.testing.assert_array_equal(index.components, components)
with TemporaryDirectory() as tmp:
    path = Path(tmp)/'index'
    index.save(path)
    with patch('numpy.linalg.svd', side_effect=AssertionError('Unexpected refit')):
        loaded = SearchIndex.load(path)
    for q in queries():
        for method in ['tfidf','bm25','lsa','hybrid']:
            before, after = expected[q['id'],method], loaded.search(q['text'],method)
            assert [d['id'] for d in before] == [d['id'] for d in after]
            np.testing.assert_allclose([d['score'] for d in before], [d['score'] for d in after])
print('All 72 query-method round-trips OK; no SVD during load')
```

## 13. Recall khác mẫu số

ANN recall vs exact=`|{A,D,E}∩{A,B,C}|/3=1/3`. Relevance recall exact=1/2, ANN=2/2=1. ANN bỏ một số hàng xóm theo metric vector nhưng vô tình lấy được D liên quan; không có mâu thuẫn. Muốn chất lượng ngữ nghĩa tốt, phải đánh giá cả biểu diễn và nhãn relevance, không chỉ mức giống exact.

## 14. Dev/test

Đặt trước tập cấu hình, chọn theo metric/chi phí trên dev, đóng băng rồi đánh giá test một lần cho kết luận độc lập. Nếu cần thử tiếp sau khi thấy lỗi test, ghi rõ test đã thành dữ liệu phát triển, tạo test mới hoặc một holdout phù hợp hơn trước báo cáo cuối. Không biến một bảng thử nhiều test thành một điểm “unseen” duy nhất.

Corpus chứa tài liệu đích vì hệ thống phải tìm trong đó; fit IDF/LSA từ corpus phục vụ là một phần xây index. Rò rỉ có thể xảy ra khi chỉnh corpus/nhãn/feature theo câu hỏi test, dùng truy vấn tương lai trái điều kiện thời gian hoặc đưa thông tin mà hệ thống thật chưa có vào corpus. Phải xác định giao thức tác vụ trước khi gọi một thao tác là leakage.

## 15. Tương thích và quyền

Ngoài dimension: model ID/revision, tokenizer/preprocessing, query/document prompt hoặc encoder task, normalize, metric, chunking, corpus version, analyzer và index config. Khi đổi model, thường cần encode lại document và đánh giá index mới cùng query path tương ứng. Hai model cùng dimension không đủ tương thích.

Chỉ lọc output D18 chưa cách ly IDF/SVD, cache hay log. Có thể cần index/fit riêng theo tenant hoặc miền quyền phù hợp, cộng auth từ server, filter trước mọi downstream và kiểm snapshot/cache. Kiến trúc cụ thể phụ thuộc ranh giới tin cậy; bài chưa triển khai những thay đổi đó.

## 16. Tiêu chí dự án nhỏ

Báo cáo đạt yêu cầu nếu người khác có corpus/query/version, chạy lại được lệnh, hiểu grade, thấy tách dev/test/no-answer, và truy được từng metric về top-k. Ba lỗi nên có nội dung cụ thể, ví dụ sai mã, nhầm token, trả cache cho câu ngoài phạm vi, không chỉ ghi “accuracy còn thấp”.

Chọn baseline đơn giản hơn khi chất lượng đáp ứng yêu cầu, hybrid/LSA không tạo cải thiện ổn định trên phép đo phù hợp, hoặc chi phí cập nhật/độ trễ/phức tạp cao hơn lợi ích. Một bộ 12 query chưa đủ chứng minh rộng; cần nêu điều đó. LSA trong bài là cầu nối toán học, không phải phép so sánh hoàn chỉnh với neural retrieval hiện đại.
