# Bài 23 — Bài tập

[← Bài chính](README.md) · [Lời giải](solutions.md)

Làm phần dự đoán trước khi chạy. Các câu 1–10 có thể giải bằng giấy; câu 11–16 cần quan sát code hoặc thiết kế kiểm chứng. Corpus A/B/C và các công thức dùng đúng quy ước của bài chính.

## 1. Chọn cách biểu diễn nhu cầu

Ba query: `ERR_BILL_042`, “gửi lại không tạo thêm đơn”, `token`. Với mỗi query, giải thích tín hiệu nào cần giữ, một lỗi từ khóa có thể gặp và một lỗi vector dày có thể gặp. Có thể kết luận trước phương pháp thắng không?

## 2. Analyzer và postings

Phân tích `ERR_BILL_042 API /v1/refunds` bằng analyzer của lab. Với A=`api timeout timeout`, B=`api token`, C=`token`, lập postings và df cho từng term. Vì sao df(timeout) không bằng 2? Nếu đổi analyzer chỉ lúc query thì có rủi ro gì?

## 3. TF-IDF tính tay

Dùng vocabulary `[api,timeout,token]`, tính IDF, ba vector document và query `api timeout` trước chuẩn hóa. Tính cosine với A và B đến 6 chữ số thập phân. Giải thích vì sao không thể coi IDF của ba term bằng nhau.

## 4. BM25 tính tay

Dùng `k1=1.2,b=.75`. Tính avgdl, IDF BM25, điểm A/B/C với query trên. Giữ TF=1 và avgdl=2, so TF factor cho document dài 2 và dài 6. Nếu b=0, hai TF factor bằng nhau không?

## 5. Cosine, norm và dot

Cho q=`[1,1]`, d1=`[1,0]`, d2=`[0,1]`, d3=`[2,2]`. Tính dot/cosine. Thay d3 bằng `[20,20]` thì hai đại lượng thay đổi ra sao? Chứng minh quan hệ L2 với cosine khi hai vector đơn vị.

## 6. Phép chiếu và OOV

Dùng C trong phụ lục, chiếu và normalize q=`[1,0,0]`, d1=`[0,1,0]`, d2=`[0,0,1]`. Vì sao ví dụ này không chứng minh LSA thật đã học đồng nghĩa? Một query có mọi term ngoài vocabulary được lab mã hóa thế nào?

## 7. RRF

Hai nhánh `[A,B,C]` và `[B,C,A]`, c=10: tính điểm và thứ hạng. Nếu tài liệu Z không có ở cả hai danh sách, RRF có lấy Z ra được không? Nếu một ID lặp hai lần trong cùng nhánh thì nên làm gì?

## 8. Metric có grade

Kết quả `[B,X,A]`, qrels A=3,B=1; k=3. Tính Hit, P, R, RR, DCG, IDCG, NDCG với gain `2^g−1`. Sau đó đổi kết quả thành `[A]`: metric nào bằng 1, metric nào không? Dùng mẫu số P@3 cố định.

## 9. Không có đáp án

Vì sao không nên gộp qrels rỗng vào Recall/NDCG thông thường? Q17 và Q18 cho kết quả khác nhau thế nào? Thiết kế hai ca no-answer khác: một ca toàn OOV, một ca có từ quen nhưng nhu cầu ngoài corpus.

## 10. Lọc trước hay sau top-k

Thứ hạng toàn corpus `[P1,R1,P2,P3]`, P là public, R là restricted. Người gọi chỉ được public, k=3. So hai cách: lấy top-3 rồi lọc và lọc trước top-3. Giải thích `None` khác `set()` trong API demo, và vì sao đây chưa là auth thật.

## 11. So sánh theo query

Chạy `--mode all` với output mới. Đọc Q10 và Q12 ở cả bốn phương pháp, lập bảng top-3/NDCG. Có ca nào hybrid kém một nhánh không? Vì sao Hit/MRR=1 vẫn che mất khác biệt? Không chỉnh tham số để “cứu” kết quả test.

## 12. Kiểm save/load và không học từ query

Tạo index, chụp vocabulary, IDF và components; chạy nhiều query, kiểm ba thứ không đổi. Save vào thư mục tạm mới, load lại; so ID và score của cả bốn phương pháp trên toàn 18 query. Làm sao chứng minh load không gọi SVD lại?

## 13. Hai kiểu recall

Exact trả `[A,B,C]`; ANN trả `[A,D,E]`; qrels chỉ A,D liên quan. Tính ANN recall so exact và relevance recall của hai hệ thống. Vì sao nâng ANN recall chưa chắc nâng relevance recall?

## 14. Thử nghiệm đúng ranh giới dữ liệu

Bạn muốn đổi lsa_dim và candidate_depth. Hãy thiết kế quy trình chọn cấu hình bằng dev, đóng băng rồi đo test. Nếu đã xem test để chọn cấu hình, cần xử lý báo cáo thế nào? Vì sao fit IDF trên các tài liệu đích không tự là rò rỉ nhãn test?

## 15. Lỗi tương thích và vòng đời index

Bạn đổi embedding model nhưng model mới vẫn có 768 chiều. Nêu ít nhất bốn thông tin ngoài số chiều cần lưu/kiểm. Với D18 riêng tư, chỉ lọc output đã đủ cách ly tenant chưa? Nêu một thay đổi kiến trúc cần cân nhắc.

## 16. Dự án nhỏ và báo cáo trung thực

Chọn 10–20 đoạn tài liệu backend do bạn tự viết hoặc có quyền sử dụng. Tạo ít nhất 12 query gồm mã định danh, diễn đạt lại, đa nghĩa và no-answer. Tách dev/test trước khi thử; gán grade có hướng dẫn; giữ provenance/version. Chạy baseline từ khóa, LSA và hybrid với cùng corpus/quyền/top-k. Báo metric có đáp án, no-answer riêng, ít nhất ba lỗi và giới hạn. Chưa cần triển khai API hay model neural. Một kết quả nào sẽ khiến bạn chọn baseline đơn giản hơn?
