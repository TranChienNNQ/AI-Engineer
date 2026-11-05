# Bài 25 — Đào sâu RAG, attribution và evaluation

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. “RAG” là một họ thiết kế, không một sơ đồ bắt buộc

Paper gốc kết hợp mô hình sinh với retriever và marginalization qua tài liệu tiềm ẩn. Một ứng dụng retrieve–concatenate–prompt có thể dùng retriever/generator đóng băng, không train chung và không tính tổng xác suất qua từng document. Hai cách cùng kết nối truy xuất với sinh, nhưng không có cùng hàm xác suất hoặc quy trình train.

Với RAG-Sequence:

$$p(y\mid q)\approx\sum_d p_\eta(d\mid q)\prod_i p_\theta(y_i\mid q,d,y_{<i}).$$

Với RAG-Token:

$$p(y\mid q)\approx\prod_i\sum_d p_\eta(d\mid q)p_\theta(y_i\mid q,d,y_{<i}).$$

Đổi vị trí tổng và tích làm mô hình khác nhau. Ví dụ giả định hai document có prior 1/2; xác suất token đầu là .9/.1 và xác suất token thứ hai theo cùng prefix là .1/.9. Sequence=.5×.9×.1+.5×.1×.9=.09; Token=(.5×.9+.5×.1)×(.5×.1+.5×.9)=.25. RAG-Token có thể nhận đóng góp document khác nhau ở từng bước, không cố định một document cho cả chuỗi tiềm ẩn.

```python
prior = [.5,.5]
p_first = [.9,.1]
p_second = [.1,.9]
sequence = sum(w*a*b for w,a,b in zip(prior,p_first,p_second))
token = sum(w*a for w,a in zip(prior,p_first))*sum(w*b for w,b in zip(prior,p_second))
assert abs(sequence-.09) < 1e-12
assert abs(token-.25) < 1e-12
print(sequence,token)
```

Đây là số tự tạo để hiểu phép toán; không là xác suất từ một model đã chạy. Lab chính không marginalize và không gọi seq 2seq/LLM. Phần đã đọc của [Lewis et al.](https://papers.nips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) là abstract/methods, đặc biệt 2.1–2.3; không dùng số benchmark trong paper làm kết quả lab.

## 2. Nguồn có hỗ trợ khác nguồn có thật và khác sự thật tuyệt đối

Giữ ba câu hỏi riêng: (1) source/snapshot có tồn tại và hợp quyền không; (2) nội dung đó hỗ trợ claim không; (3) claim có đúng với thực tế/phạm vi/thời điểm không. Một runbook giả mạo có thể được quote chính xác; một runbook đáng tin có thể bị áp vào sai môi trường; một quote hoàn toàn đúng có thể không hỗ trợ kết luận.

Không dùng “faithfulness=1” như từ đồng nghĩa với “truth=1”. Trong lab, grounded claim rate dựa trên annotation về hỗ trợ theo nguồn; nó không là kiểm toán độc lập tính đúng của policy thật. Số 72 giờ,300 giây,20 MB,25 MB đều là dữ kiện giả định của ví dụ.

Một citation đúng còn không chứng minh model đã dựa vào nguồn đó để sinh. Bài đo attribution/entailment kiểm mối quan hệ nội dung mà người đọc có thể kiểm lại, không đo toàn bộ cơ chế nhân quả bên trong mạng neural.

## 3. Một claim cần nhiều nguồn: giữ quan hệ AND/OR

Catalog của lab cho mỗi claim một danh sách các **tập hỗ trợ đủ**. Trong một tập, các fact kết hợp theo AND; giữa các tập là các cách hỗ trợ thay thế theo OR. Claim “key giữ 72 giờ và các attempt cùng deadline” cần cả hai fact. Một cite chỉ đủ thời hạn không hỗ trợ toàn claim.

Gọi U là hợp các fact được những quote đã xác thực chứa trọn. Claim có hỗ trợ nếu tồn tại một tập S trong annotation sao cho S⊆U. Không dùng “có ít nhất một cite liên quan” thay điều kiện đủ.

```python
support_sets = [{'key_72','deadline'}, {'alternative_policy'}]
def supported(covered):
    return any(s <= set(covered) for s in support_sets)
assert not supported({'key_72'})
assert supported({'key_72','deadline'})
assert supported({'alternative_policy'})
```

Ví dụ set `alternative_policy` chỉ minh họa logic, không là fact tồn tại trong corpus. Trong dữ liệu thật, việc gán các tập bằng chứng đủ cần kiến thức tác vụ và rà soát; không suy quan hệ AND/OR chỉ từ số citation.

Một đoạn có chứa số 72 không tự suy ra mọi claim có số 72. Vì vậy scorer dùng annotation của toàn claim và span, không đối chiếu riêng số/từ khóa. Claim ngoài catalog được đánh dấu `known=False`; cách tính strict của bài chưa ghi nhận hỗ trợ cho nó. Đây là giới hạn của bộ tham chiếu, không là bằng chứng claim ngoài catalog sai.

## 4. Citation precision của lab và ALCE không nên được so máy móc

[ALCE §3.3](https://aclanthology.org/2023.emnlp-main.398.pdf) dùng NLI cho quan hệ hỗ trợ và phát hiện citation không liên quan. Lab chỉ mượn việc tách statement support với edge usefulness; nó dùng annotation đóng, không chạy NLI, không tái hiện mọi quy tắc/giới hạn của paper.

Trong scorer bài: claim phải có đủ hỗ trợ trước; một citation nhận credit nếu chứa ít nhất một fact hữu ích cho claim đó. Hai nguồn cùng xác nhận một fact có thể đều được credit; metric không yêu cầu bộ citation tối thiểu. Citation dư nhưng liên quan khác citation hoàn toàn không liên quan. Đếm nhiều link không tự tăng grounded claim rate.

Nếu một composite claim thiếu nửa bằng chứng, các edge của nó không nhận credit theo quy ước này, dù một edge có đóng góp đúng một phần. Muốn đo partial credit, cần định nghĩa metric khác và giữ tên/mẫu số rõ. Không thay quy tắc sau khi thấy điểm thấp rồi so trực tiếp với bảng cũ.

## 5. Mẫu số, micro/macro và abstention

Giả sử answerA có 1 claim, support 1; answerB có 9 claim, support 0. Macro mean grounded rate=(1+0)/2=.5; micro=1/10=.1. Cả hai tính đúng theo câu hỏi khác nhau: trọng số mỗi câu trả lời hay mỗi claim. Report lab dùng micro cho claim/citation, macro cho task.

Abstain không có claim: support rate/precision không có mẫu số nên đểNone. Chấm tự động 1 cho mọi abstain sẽ làm hệ thống “không bao giờ trả lời” có vẻ đạt chất lượng grounding tuyệt đối. Cần đo thêm tỷ lệ trả lời được, bỏ lỡ query có đáp án và trả claim khi không có đáp án.

Fixture parse/schema lỗi không góp claim/edge vì chưa có cấu trúc đáng tin để đếm, nhưng vẫn nằm trong toàn bộ 22 case của task success. Candidate schema-valid mà bị reject cơ học vẫn được báo metric chẩn đoán raw candidate; nó không được phát hành và không được task success. Không lấy conditional accuracy trên candidate đã được chấp nhận làm tỷ lệ thành công toàn pipeline.

## 6. Required fact recall có ích nhưng không đủ cho chất lượng câu trả lời

Nhãn required của lab chỉ liệt kê các ý phải có. Task success còn yêu cầu trạng thái và mọi claim có hỗ trợ; chưa phạt hết claim thừa nhưng đúng nguồn, chưa chấm độ dễ hiểu hoặc cách giải thích. Ví dụ một answer có thời hạn key đúng và thêm một quy tắc payload có nguồn vẫn có thể pass query chỉ hỏi thời hạn.

Một bộ evaluation đầy đủ hơn có thể cần nhãn relevance, completeness, contradiction, style và an toàn hành động; phải chọn theo tác vụ thật. Không dùng một con số tự đặt tên “RAG accuracy” để gộp mọi thứ rồi không biết sửa tầng nào.

Source-level match cũng chưa đủ: answer có thể cite đúng trang nhưng quote nhầm section. Quote match chưa đủ: quote có thể không suy ra claim. Claim support chưa đủ: claim có thể không trả lời câu hỏi. Bốn mức này cần case kiểm riêng.

## 7. Oracle context và các thí nghiệm chẩn đoán

Có thể chạy cùng generator với context phục vụ, oracle context và context rỗng. Nếu oracle vẫn sai, sửa retrieval đơn thuần khó giải quyết; nếu oracle đúng còn context phục vụ thiếu, xem retriever/packing. Context rỗng cho thấy model dựa vào tri thức trong trọng số đến đâu, nhưng một answer đúng khi không có nguồn vẫn chưa đáp ứng yêu cầu citation.

Oracle không là phép đo serving vì dùng thông tin gold. Đừng cấp required fact/gold span cho generator rồi báo model tự tìm được bằng chứng. Trong code, generator chỉ nhận question/context/prompt; `facts` và `required` nằm ở evaluator. Đây là tách dữ liệu theo luồng ứng dụng, chưa là sandbox chống một adapter Python cố tình đọc mọi tệp.

## 8. Packing theo giá trị thông tin và tính liên tục

Greedy whole-chunk của lab minh bạch nhưng có thể bỏ lỡ một cặp bằng chứng nhỏ. Có thể nghiên cứu sentence extraction, parent expansion, diversity hoặc dự đoán utility/cost; chúng đều có lỗi và phải giữ trace vị trí. Tóm tắt để nén context tạo thêm một bước có thể bịa hoặc bỏ điều kiện, nên summary cần mapping về nguồn gốc.

Khi hợp nhất hai span `[0,100)` và `[80,180)` cùng source/version, phần mới chỉ có 80 ký tự; không phải hai nguồn độc lập. Chỉ nối nếu đúng thứ tự/source/snapshot; không gộp text từ nhiều nguồn rồi làm người đọc tưởng một tài liệu nói tất cả.

[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) cung cấp một lý do để kiểm cả vị trí thông tin, không chỉ số token tối đa. Phần được dùng ở đây là abstract: ảnh hưởng trên model/tác vụ nghiên cứu, không là phép đo các model đang phổ biến vào 2026 hay kết quả của lab extractive.

## 9. Phiên bản, quyền và cache

Một cache key cho kết quả cuối thường cần query/normalization, phạm vi quyền, corpus/index snapshot, prompt, generator/model revision, sampling và contract version. Những gì cụ thể cần lưu tùy kiến trúc. `request_hash` của lab băm payload request/prompt/context/config để theo dấu, **chưa là cache key hoàn chỉnh cho mọi model/authorization policy**.

Citation E1 chỉ có nghĩa cùng mapping request. Nếu cache hoặc replay, phải mang mapping/snapshot tương ứng và kiểm lại quyền theo policy phục vụ. Link vẫn mở được không có nghĩa nội dung chưa đổi; source content hash giúp phát hiện nhầm snapshot trong luồng tin cậy, chưa là chữ ký số hoặc chứng nhận uy tín nguồn.

Default snapshot public được lọc trước thống kê BM 25, packing và generator. Caller tự chọn scope ở thư viện demo chưa là authentication/authorization. Hệ thống thật cần quyền do server suy ra, cách ly dữ liệu/log/cache và chính sách với response đã phát hành. Bài 26 sẽ đi sâu; bài 25 chưa triển khai auth service hay phòng vệ prompt injection toàn diện.

## 10. Khi thay baseline bằng LLM

Adapter cần giữ cùng contract dữ liệu và validator, nhưng phép đo thay đổi: tokenizer/messages overhead, output token limit, sampling, nondeterminism, refusals, partial response, timeout và retry. Cần log model revision/parameters, usage và trace đủ để chẩn đoán. Không dùng số token word-level hay thời gian extractive để suy chi phí/độ trễ của API thật.

Với claim paraphrase mới, catalog đóng sẽ không bao phủ; cần annotation người hoặc evaluator có khả năng entailment và được kiểm bằng tập người gán. NLI/LLM judge có thể sai ở phủ định, số lượng, thời điểm, nhiều nguồn hoặc văn bản thiếu bối cảnh. Giữ riêng kết quả judge và ground truth đã rà soát; đừng để generator chấm chính câu trả lời của nó như một bằng chứng chắc chắn.

Một bước retry nhằm “sửa citation” cũng có thể đổi claim và nguồn; phải validate lại toàn bộ candidate và giữ budget. Tự động thay ID không có trong context bằng một ID có thật là sửa triệu chứng, không chứng minh nội dung được hỗ trợ. Các nguyên tắc deadline/bounded retry ở bài 22 vẫn áp dụng.
