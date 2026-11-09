# Bài 20 — Tiền huấn luyện và tinh chỉnh: từ mô hình nền đến hành vi cho tác vụ

[Mục lục](../README.md) · [Bài 19](../19-tokenizer-mo-hinh-ngon-ngu/README.md) · [Đào sâu và đọc code](deep-dive.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Dành cho người có nền backend và đã học Transformer, tokenizer, next-token loss ở bài 18–19. Ví dụ và dữ liệu trong bài tự tạo, không dùng log khách hàng hoặc tải pretrained model. Lab chạy thật trên CPU; “mô hình nền” trong lab là mô hình rất nhỏ tự train, không đại diện chất lượng LLM thương mại.

## 1. Mục tiêu và cách đọc

Ở bài 19, model đã biết viết tiếp văn bản nhưng có thể sinh câu lặp hoặc không làm điều người dùng muốn. *(Giải thích tường minh: Mô hình ngôn ngữ tiền huấn luyện gốc — Base Model — chỉ được học một nhiệm vụ duy nhất là "dự đoán từ tiếp theo" trên kho văn bản thô. Nó giống như một người đọc rất nhiều sách và có khả năng nối chữ rất tài tình, nhưng chưa được huấn luyện để đóng vai một trợ lý, không biết trả lời đúng trọng tâm câu hỏi hay tuân thủ các quy tắc định dạng).* Bài 20 hỏi: có thể dùng weight đã học rồi dạy một hành vi cụ thể không? Nếu được, thay đổi dữ liệu, loss và những parameter nào? *(Khái niệm Chuyển giao trọng số / Thích nghi mô hình - Weight Adaptation / Transfer Learning: Thay vì phải huấn luyện lại một mô hình mới toanh từ đầu với chi phí khổng lồ, ta tái sử dụng toàn bộ tri thức ngôn ngữ đã được học và lưu trong các ma trận trọng số (weights) của mô hình nền, sau đó chỉ cần cập nhật nhẹ bằng một tập dữ liệu nhỏ chuyên biệt để định hình hành vi mong muốn).*

Sau bài này, bạn cần:

- Phân biệt pretraining, continued pretraining, supervised fine-tuning (SFT), instruction tuning và học từ ưu tiên. *(Nắm vững bản đồ các giai đoạn huấn luyện một mô hình ngôn ngữ từ dữ liệu thô đến khi trở thành trợ lý AI hoàn chỉnh).*
- Biến một tình huống backend thành dữ liệu prompt–answer, xác định rõ output được mong đợi. *(Biết cách thiết kế tập dữ liệu tinh chỉnh có giám sát như một bộ "đặc tả hành vi" chuẩn hóa cho hệ thống phần mềm).*
- Tự tạo loss mask sau shift, giữ EOS, bỏ PAD và kiểm prompt vẫn có thể nhận gradient qua context. *(Hiểu bản chất toán học của việc che mặt nạ mất mát: chỉ chấm điểm câu trả lời, không chấm điểm câu hỏi nhưng vẫn cho phép câu hỏi truyền gradient qua cơ chế Attention).*
- Đọc thí nghiệm trước/sau cùng control scratch và head-only, nhận ra chi phí thêm và khả năng bị quên. *(Hiểu hiện tượng "quên thảm khốc" - Catastrophic Forgetting - khi mô hình học kỹ năng mới nhưng suy giảm năng lực cũ).*
- Giải thích full fine-tuning khác freezing/LoRA và lưu artifact đủ cho inference. *(Nắm rõ sự đánh đổi giữa việc cập nhật toàn bộ tham số, chỉ huấn luyện tầng đầu ra, hoặc chèn các ma trận hạng thấp LoRA).*

Ba lượt đọc: phần 2–6 cho bức tranh và ví dụ thực tế; phần 7–10 cho tính tay/cơ chế; phần 11–15 cho lab, đánh giá và thực hành. Không cần GPU, tài khoản hay quyền truy cập model bên ngoài.

## 2. Ví dụ thực tế: đọc log khác với định tuyến ticket

Một corpus có nhiều dòng:

```text
service=api op=read event=timeout.
service=auth op=check event=expired-token.
```

Pretraining theo next-token có thể dạy pattern `service=`, tên dịch vụ, thứ tự các trường và cách kết thúc dòng. *(Định nghĩa Tiền huấn luyện - Pretraining: Là giai đoạn mô hình đọc hàng tỷ dòng văn bản thô mà không có nhãn của con người, tự học các quy luật thống kê, từ vựng và cấu trúc cú pháp của log hệ thống).* Nhưng khi ứng dụng hỏi “chuyển ticket này cho đội nào?”, model có thể chỉ tiếp tục viết thêm log. Không có gì trong mục tiêu viết tiếp bắt buộc câu trả lời phải là nhãn đội. *(Giải thích tường minh: Nếu bạn đưa vào prompt `Route: service=api op=read event=timeout.\nTeam: `, một mô hình Base chưa qua tinh chỉnh rất có thể sẽ viết tiếp một dòng log mới như `service=auth op=check event=denied.` thay vì dừng lại và trả lời tên đội kỹ thuật. Mô hình không có ý thức rằng đây là một câu hỏi cần giải đáp; đối với nó, câu prompt chỉ là phần mở đầu của một file log dài bất tận).*

Ta tạo một hợp đồng cụ thể:

```text
Prompt:
Route: service=api op=read event=timeout.
Team: 

Answer mong muốn: NET
```

*(Định nghĩa Hợp đồng Giao tiếp (Input-Output Contract):
- **Prompt (Câu lệnh / Ngữ cảnh đầu vào):** Đoạn văn bản mô tả thông tin sự cố cần giải quyết.
- **Answer (Câu trả lời mục tiêu):** Kết quả chính xác và duy nhất mà hệ thống backend mong muốn nhận được (ở đây là mã đội `NET` - viết tắt của Network Team)).*

Trong ví dụ giả định của bài, timeout/connection-reset → NET; denied/expired-token → AUTH; deadlock/duplicate-key → DATA. Đây là **quy tắc tác vụ do ta định nghĩa**, không phải chính sách vận hành chuẩn cho mọi tổ chức. Trong production, cần đội nghiệp vụ xác nhận nhãn; timeout có thể có nhiều nguyên nhân và không luôn thuộc đội network.

SFT cho model xem input cùng answer mẫu và tăng xác suất answer đó ở đúng context. *(Định nghĩa Tinh chỉnh có giám sát - Supervised Fine-Tuning (SFT): Là phương pháp huấn luyện tiếp mô hình trên các cặp dữ liệu mẫu `(Prompt, Answer)` chuẩn xác. Mục tiêu toán học của SFT là: khi nhìn thấy `Prompt`, mô hình phải tối đa hóa xác suất sinh ra đúng các token của `Answer`, sau đó phát ra ngay token kết thúc `EOS`).* Nó không tự gửi ticket, thay đổi database hoặc chứng minh nhãn mẫu đúng. Ví dụ đủ cụ thể để đo: output phải đúng nhãn và kết thúc EOS; `NET vì...` không đạt hợp đồng exact dù chứa chữ NET. *(Ứng dụng thực tế trong Backend: Một API phân loại ticket cần một mã định danh ngắn gọn và chuẩn xác (như enum `'NET'`) để điều phối vào hàng đợi RabbitMQ hoặc Kafka cho đội Network. Nếu mô hình sinh ra câu lan man như `NET vì hệ thống bị timeout ở gateway`, parser của backend sẽ bị lỗi `ValueError: Invalid team identifier`. SFT giúp rèn giũa mô hình tuân thủ tuyệt đối quy ước đầu ra).*

## 3. Các giai đoạn học: cùng công cụ tối ưu, khác dữ liệu và mục tiêu

| Giai đoạn | Khởi đầu | Dữ liệu/mục tiêu | Điều không được suy ra |
|---|---|---|---|
| Pretraining từ đầu | Weight khởi tạo ngẫu nhiên | Học dự đoán văn bản/tín hiệu trên corpus; lab dùng causal LM | Model tự có instruction following hoặc hiểu đúng mọi sự thật |
| Continued pretraining (CPT) | Checkpoint đã được pretrain | Tiếp tục mục tiêu LM trên corpus phù hợp | Chỉ đổi domain là tự học đúng format trả lời |
| Supervised fine-tuning (SFT) | Thường từ model đã học | Input–output mẫu cho tác vụ; lab chấm completion + EOS | Mọi output mới sẽ đúng hoặc luôn giữ nguyên khả năng cũ |
| Instruction tuning | Thường từ base hoặc model đã thích nghi | Một dạng SFT với instruction và response mong muốn | Một tập route nhãn hẹp đã là trợ lý hội thoại tổng quát |
| Học từ ưu tiên/phản hồi | Model và dữ liệu theo phương pháp | Dùng so sánh/đánh giá hành vi thay vì chỉ answer mẫu | Đồng nghĩa mọi fine-tuning hoặc chỉ cần chấm CE trên chosen answer |

*(Giải thích chi tiết 5 giai đoạn cho người mới:
1. **Pretraining từ đầu (Tiền huấn luyện):**
   - *Khởi đầu:* Trọng số ma trận được khởi tạo ngẫu nhiên hoàn toàn.
   - *Dữ liệu:* Kho văn bản thô khổng lồ (hàng nghìn tỷ token từ sách báo, Wikipedia, mã nguồn GitHub, log hệ thống).
   - *Mục tiêu:* Dự đoán token tiếp theo ($P(x_t \mid x_{<t})$) bằng phương pháp tự giám sát.
   - *Chi phí:* Khổng lồ (hàng trăm GPU chạy nhiều tuần/tháng, tốn hàng triệu USD).
   - *Sản phẩm:* Mô hình nền (Base Model) có tri thức rộng nhưng chưa biết nghe lời hay làm theo chỉ dẫn.
2. **Continued Pretraining (CPT - Tiền huấn luyện tiếp diễn / Thích nghi miền):**
   - *Khởi đầu:* Từ một mô hình nền đã có sẵn tri thức ngôn ngữ chung.
   - *Dữ liệu:* Văn bản thô chuyên ngành sâu (ví dụ toàn bộ tài liệu y khoa, văn bản pháp luật, hoặc kho log nội bộ của một tập đoàn tài chính).
   - *Mục tiêu:* Vẫn là dự đoán từ tiếp theo, giúp mô hình làm quen với từ vựng và ngữ cảnh chuyên ngành trước khi học làm tác vụ cụ thể.
3. **Supervised Fine-Tuning (SFT - Tinh chỉnh có giám sát):**
   - *Khởi đầu:* Từ mô hình nền (hoặc sau CPT).
   - *Dữ liệu:* Hàng nghìn đến hàng trăm nghìn cặp `(Prompt, Answer)` chất lượng cao do chuyên gia biên soạn.
   - *Mục tiêu:* Dạy mô hình kỹ năng phản hồi: trả lời đúng trọng tâm câu hỏi, tuân thủ định dạng (JSON, Markdown, mã enum), và dừng đúng lúc với token `EOS`.
   - *Chi phí:* Tương đối rẻ (chạy trên 1 hoặc vài GPU trong vài giờ đến vài ngày).
4. **Instruction Tuning (Tinh chỉnh theo chỉ dẫn):**
   - Là một dạng mở rộng của SFT nhưng tập dữ liệu bao gồm hàng nghìn loại tác vụ đa dạng khác nhau (dịch thuật, tóm tắt, viết code, sửa lỗi, giải toán) kèm theo câu lệnh (instructions). Mục tiêu là giúp mô hình tổng quát hóa để có thể làm được cả những tác vụ mới mà nó chưa từng thấy lúc huấn luyện (*zero-shot generalization*).
5. **Học từ ưu tiên / phản hồi (RLHF / DPO - Preference Tuning):**
   - *Khởi đầu:* Sau khi mô hình đã hoàn thành giai đoạn SFT.
   - *Dữ liệu:* Các cặp câu trả lời được con người hoặc mô hình giám sát chấm điểm: câu A tốt hơn câu B (Chosen vs. Rejected).
   - *Mục tiêu:* Tinh chỉnh hành vi để mô hình an toàn hơn, hữu ích hơn, trung thực hơn và hạn chế tối đa việc tạo ra nội dung độc hại hoặc ảo giác).*

“Fine-tuning” trong cách nói rộng còn bao gồm các kiểu thích nghi khác; bài này gọi rõ **SFT** khi nói đến supervised prompt–answer. Pretraining không bắt buộc causal LM trong mọi kiến trúc: masked modeling cũng là một mục tiêu khác như bài 19. Lab chỉ triển khai causal LM → SFT.

SFT và học từ ưu tiên có thể nằm ở các bước khác nhau trong post-training. Công trình [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) mô tả dùng demonstration để supervised fine-tune rồi dùng ranking và RLHF. Ở đây chỉ đối chiếu sự khác nhau của dữ liệu/mục tiêu; không tái hiện pipeline hay kết quả của nghiên cứu.

![Các đường thích nghi và phạm vi lab](figures/01-stages.png)

## 4. Continued pretraining: thêm văn bản miền chưa phải thêm câu trả lời

Giả sử model ít quen log hệ thống, tên trường và thuật ngữ nội bộ. Có thể tiếp tục LM training trên tài liệu được phép dùng để mô hình thích nghi phân phối văn bản đó, rồi SFT bằng ví dụ thực hiện tác vụ. Domain-adaptive pretraining nhấn vào corpus miền; task-adaptive pretraining nhấn vào dữ liệu không nhãn gần tác vụ. Hai khái niệm này được nghiên cứu trong [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964).

Nguồn trên báo cáo lợi ích trong các miền/tác vụ phân loại được nghiên cứu; không chứng minh CPT luôn cải thiện mọi LLM hoặc mọi corpus log. Cần một phép thử riêng, kiểm chất lượng và ngân sách. **Lab bài này không chạy thêm nhánh CPT** sau base: nó chạy pretraining nhỏ từ đầu rồi SFT, nhằm giữ thí nghiệm dễ kiểm.

Ví dụ khác: tài liệu có câu “API hoàn tiền dùng endpoint /refunds” có thể giúp học văn bản miền. Nhưng muốn trả về JSON `{"action":"refund", "requires_approval":true}` cần ví dụ hành vi và phép kiểm riêng. *(Phân biệt rõ ràng giữa Tri thức miền và Hành vi tác vụ:
- CPT bổ sung "tri thức và từ vựng chuyên ngành": Mô hình học được các khái niệm như `/refunds`, `deadlock`, `read_replica`.
- Nhưng CPT KHÔNG dạy mô hình cách hành xử: Mô hình đọc tài liệu biết endpoint `/refunds`, nhưng khi nhận một ticket "khách đòi hoàn tiền", nó không tự biết phải sinh ra một JSON payload với schema hợp lệ hay cần gửi phê duyệt cho quản lý. Muốn có hành vi đó, bắt buộc phải có SFT với các ví dụ cụ thể về schema và luồng nghiệp vụ).* Đưa một file tài liệu vào training không tương đương có cơ chế tra cứu phiên bản mới nhất hoặc xác minh quyền hành động.

## 5. Chọn cách giải bài toán trước khi chọn cách train

Bảng dưới là suy luận thiết kế, không phải khuyến nghị sản phẩm/nhà cung cấp:

| Nhu cầu cụ thể | Thử nghiệm phù hợp để bắt đầu | Vì sao |
|---|---|---|
| Đổi format/giọng điệu, tác vụ đã có thể làm | Prompt và ví dụ trong context, có eval giữ riêng | Có thể chưa cần cập nhật weight |
| Nhãn được xác định bởi vài quy tắc ổn định | Rule engine hoặc classifier đơn giản | Có lời giải trực tiếp và dễ kiểm hơn cho quy tắc rõ |
| Câu trả lời cần dữ liệu mới hoặc theo quyền người dùng | Truy xuất dữ liệu/công cụ đúng quyền, kiểm nguồn | Weight không tự là bản sao tức thời của database |
| Hành vi lặp lại cần thích nghi theo mẫu | SFT cùng baseline và kiểm hồi quy | Dạy pattern input–output ở weight |
| Model chưa quen phân phối văn bản miền | Đánh giá CPT rồi SFT nếu có cơ sở | Cần phân biệt học miền và học tác vụ |

*(Tư duy kỹ thuật phần mềm (Engineering Mindset) khi chọn giải pháp:
1. **Nguyên tắc "Rule of Thumb":** Đừng vội vàng nhảy vào Fine-Tuning khi chưa thử các giải pháp nhẹ hơn! Fine-tuning đòi hỏi thu thập dữ liệu chuẩn, tốn tài nguyên GPU, phức tạp hóa quy trình CI/CD và tiềm ẩn rủi ro quên kiến thức cũ.
2. **Thứ bậc giải pháp (Hierarchy of Engineering Effort):**
   - **Mức 1 — Prompt Engineering & Few-Shot:** Đưa 2–3 ví dụ mẫu trực tiếp vào prompt. Tốn 0 chi phí huấn luyện, triển khai trong vài phút.
   - **Mức 2 — Rule Engine / Machine Learning truyền thống:** Nếu bài toán chỉ là `event='timeout' -> team='NET'`, một câu lệnh `switch-case` hoặc mô hình Logistic Regression/Random Forest chạy trong 1 mili-giây, chính xác 100%, tốn 0 MB VRAM. Tuyệt đối không dùng LLM hàng tỷ tham số chỉ để làm một bảng ánh xạ tĩnh!
   - **Mức 3 — RAG (Retrieval-Augmented Generation):** Khi dữ liệu thay đổi liên tục theo thời gian thực (giá sản phẩm, chính sách mới, số dư tài khoản). Trọng số mô hình (weights) là tĩnh, không thể cập nhật liên tục từng giây qua fine-tuning.
   - **Mức 4 — SFT:** Khi cần mô hình tuân thủ nghiêm ngặt một định dạng phức tạp (như JSON schema tùy biến), rút ngắn prompt (bỏ bớt các ví dụ few-shot dài dòng để tiết kiệm token và giảm độ trễ), hoặc rèn giũa phong cách phản hồi chuyên biệt).*

Lab route có rule đạt 100%, nên không phải lý do triển khai neural model cho bảng quy tắc này. Mục đích là nhìn rõ cơ chế chuyển weight đã học sang tác vụ. Các bài RAG/công cụ sau sẽ đi sâu nhu cầu dữ liệu riêng, không giả rằng fine-tuning giải quyết mọi trường hợp.

## 6. Dữ liệu SFT là đặc tả hành vi bằng ví dụ

Một record tối thiểu của lab có `group`, `text`, `prompt`, `answer`. `text` dùng mục tiêu LM; `prompt` và `answer` dùng SFT. Nhãn phải thống nhất: cùng loại event không bị gán hai đội chỉ vì copy lỗi. Tác vụ thật có trường hợp thiếu dữ liệu thì cần định nghĩa “chưa đủ thông tin”, không dạy model bịa một đội để luôn có answer. *(Ứng dụng thực tế: Trong kỹ thuật phần mềm, dữ liệu SFT đóng vai trò như các bản "Unit Test" của hành vi mô hình. Nếu câu hỏi không đủ thông tin để định tuyến, dữ liệu mẫu phải dạy mô hình trả lời: "Chưa đủ dữ liệu để kết luận". Nếu luôn ép mô hình phải chọn bừa một đội, bạn đang vô tình huấn luyện cho mô hình thói quen bịa đặt (ảo giác - hallucination)).*

Những việc cần xác định trước khi train:

1. Tác vụ/format và tiêu chí đạt: đúng nhãn, đúng JSON schema, hoặc rubric phù hợp. *(Cần có hàm kiểm tra tính hợp lệ cú pháp (schema validator) tự động trong pipeline đánh giá).*
2. Đơn vị split: incident, khách hàng, hội thoại, tài liệu hoặc mốc thời gian; không chia ngẫu nhiên các đoạn của cùng incident qua train/test. *(Cạm bẫy rò rỉ dữ liệu - Data Leakage: Nếu một sự cố server kéo dài 30 phút sinh ra 10 dòng log tương tự nhau, bạn không được bốc ngẫu nhiên 8 dòng vào train và 2 dòng vào test! Vì các dòng log này chia sẻ cùng mã lỗi và mốc thời gian, mô hình sẽ "học vẹt" sự cố đó và cho kết quả test cao giả tạo).*
3. Dữ liệu được phép dùng, loại thông tin cần bỏ và khả năng dữ liệu nhạy cảm đi vào output. Loại thông tin nhận diện không đồng nghĩa đã loại mọi cách ghi nhớ dữ liệu. *(Cần làm sạch thông tin định danh cá nhân (PII), secret key, token xác thực trước khi đưa vào tập huấn luyện).*
4. Template render và tokenizer version; inference phải dùng cùng hợp đồng. *(Nếu lúc train dùng prompt dạng `Route: ...\nTeam: ` thì lúc chạy thực tế (inference) bắt buộc phải dùng chính xác template này, không được đổi thành `Câu hỏi: ...\nĐội: `).*
5. Chính sách độ dài: sau truncation còn answer/EOS hay không; sample không còn target thì không có tín hiệu SFT hữu ích.

Đây là việc thiết kế tập học/eval, không phải nghi thức thay thế chạy kiểm. Một dataset nhiều ví dụ gần trùng có thể làm train loss đẹp và test bị lạc quan. Cần giữ test không tham gia tokenizer fit, pretraining, SFT hoặc chọn checkpoint khi muốn đánh giá độc lập theo split ấy.

## 7. Loss SFT: không dự đoán prompt nhưng vẫn đọc prompt

Đặt p là prompt, a là answer. Mục tiêu completion-only thường dùng:

\[
L_{SFT}=-\frac{1}{N_{answer}}\sum_{t\in answer+EOS}\log P_\theta(x_t\mid x_{<t}).
\]

*(Giải thích chi tiết công thức toán học cho người mới:
- $p$: Chuỗi token của Prompt (câu hỏi hoặc ngữ cảnh đầu vào).
- $a$: Chuỗi token của Answer (câu trả lời mục tiêu).
- $N_{answer}$: Tổng số lượng token thuộc về Answer cộng với token kết thúc `EOS`.
- $\sum_{t \in answer+EOS}$: Phép lấy tổng mất mát CHỈ thực hiện trên các vị trí thuộc câu trả lời và token dừng `EOS`, HOÀN TOÀN KHÔNG tính mất mát trên các token của Prompt!
- **Vì sao không chấm loss cho Prompt?**
  Trong ứng dụng thực tế, người dùng có thể gửi vào bất kỳ câu hỏi nào. Ta không cần mô hình học cách tự dự đoán lại câu hỏi của người dùng! Ta chỉ cần mô hình học cách: *khi đã nhìn thấy toàn bộ câu hỏi trước mặt, hãy sinh ra câu trả lời chuẩn xác nhất*. Nếu chấm loss cả trên prompt, mô hình sẽ lãng phí năng lực để học vẹt cách diễn đạt của câu hỏi thay vì tập trung vào câu trả lời).*

Mỗi query vẫn chỉ nhìn prefix qua causal mask. Ta bỏ loss ở các vị trí dự đoán prompt; không xóa prompt khỏi input. Mô hình phải đọc prompt mới biết answer cần gì. EOS là target thật, PAD là phần điền và bị bỏ loss.

Ví dụ token minh họa, không là tokenization cố định của một sản phẩm:

```text
prompt IDs = [10,11]; answer IDs = [20]
full = [BOS,10,11,20,EOS]
x    = [BOS,10,11,20]
y    = [IGNORE,IGNORE,20,EOS]
index=    0,     1,   2,  3
```

*(Mổ xẻ từng vị trí ghép cặp sau phép dịch chuyển token (Token Shift):
- Giả sử Prompt gồm 2 token: `[10, 11]` ($P=2$). Answer gồm 1 token: `[20]`.
- Chuỗi ghép đầy đủ: `full = [BOS, 10, 11, 20, EOS]` (tổng 5 token).
- Đầu vào mô hình ($x = full[:-1]$): `[BOS, 10, 11, 20]` (độ dài 4).
- Nhãn mục tiêu ban đầu ($y = full[1:]$): `[10, 11, 20, EOS]` (độ dài 4).
- Áp dụng mặt nạ Loss Mask cho nhãn $y$:
  - Tại vị trí index 0: Đầu vào là `BOS`, nhãn ban đầu là `10` (thuộc Prompt) $\to$ Đặt thành `IGNORE` (`-100`).
  - Tại vị trí index 1: Đầu vào là `10`, nhãn ban đầu là `11` (thuộc Prompt) $\to$ Đặt thành `IGNORE` (`-100`).
  - Tại vị trí index 2: Đầu vào là `11` (token CUỐI CÙNG của Prompt!), nhãn ban đầu là `20` (token ĐẦU TIÊN của Answer!) $\to$ GIỮ NGUYÊN nhãn `20` để tính loss!
  - Tại vị trí index 3: Đầu vào là `20`, nhãn ban đầu là `EOS` $\to$ GIỮ NGUYÊN nhãn `EOS` để tính loss!
- Kết quả thu được mảng nhãn: $y = [IGNORE, IGNORE, 20, EOS]$).*

Query tại prompt token cuối, index 2, dự đoán answer token 20. Nếu prompt có P token nội dung thì mask `y[:P]`, **không phải `y[:P+1]`**; mask thừa một vị trí sẽ bỏ token answer đầu. BOS làm chỉ số shift dễ lệch. *(Cạm bẫy chỉ số chết người: Trong ví dụ trên, Prompt có $P=2$ token nội dung. Do đó ta chỉ che đúng 2 vị trí đầu tiên của mảng nhãn (`y[:2] = IGNORE`). Nếu lập trình viên ngây thơ nghĩ rằng "Prompt gồm 2 token cộng thêm 1 token BOS là 3 phần tử, nên che 3 vị trí `y[:3]`", thì vị trí index 2 (chính là token `20` của answer) sẽ bị xóa sạch thành `IGNORE`! Mô hình sẽ bị mất hoàn toàn tín hiệu học cách sinh ra từ đầu tiên của câu trả lời).*

`ignore_index=-100` khiến gradient trực tiếp tới logits của vị trí ignore bằng 0. Nhưng embedding/hidden của prompt vẫn có thể nhận gradient vì answer query đọc chúng qua attention. “Prompt loss bị bỏ” khác “prompt bị detach”. Xem [phụ lục](deep-dive.md) để kiểm bằng code. *(Giải thích cơ chế lan truyền ngược: Dù hàm mất mát không tính loss trực tiếp trên các logits của prompt, nhưng để dự đoán được token `20` tại index 2, cơ chế Self-Attention buộc phải "nhìn lại" và truy vấn thông tin từ các vector biểu diễn ẩn của `BOS`, `10` và `11`. Khi đạo hàm lan truyền ngược chảy từ token `20` về, gradient sẽ đi xuyên qua các ma trận Attention và cập nhật các tầng biểu diễn của Prompt. Nhờ đó, mô hình vẫn học được cách "đọc hiểu prompt sâu sắc hơn" để phục vụ cho việc trả lời).*

Một thư viện có thể tự shift labels, template có thể chèn role hoặc EOS khác. Không chuyển nguyên chỉ số từ lab sang mọi Trainer. [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) phân biệt completion-only và assistant-only loss; assistant-only còn cần template tạo được mask thích hợp. Bài này tự tính loss bằng PyTorch, không cài hay chạy TRL.

![Shift và mask ở ranh giới prompt–answer](figures/02-loss-mask.png)

## 8. Ví dụ tính tay: đổi mục tiêu và một bước cập nhật

### 8.1 Loss toàn chuỗi có thể che mờ answer khó

Giả sử 8 token prompt có xác suất nhãn đúng 0.9 mỗi token, 2 token answer/EOS có xác suất 1/4 và 1/2:

\[
L_{full}=\frac{-8\ln 0.9-\ln(1/4)-\ln(1/2)}{10}\approx 0.292233,
\qquad L_{answer}=\frac{\ln 4+\ln 2}{2}\approx 1.039721.
\]

*(Giải thích trực giác toán học vì sao Loss toàn chuỗi gây ảo tưởng:
- Xét một prompt gồm 8 token: vì prompt là các câu văn hoặc đoạn log quen thuộc, mô hình đoán rất dễ dàng với xác suất cao $p=0.9$ cho mỗi token. Mất mát của mỗi token prompt rất nhỏ: $-\ln(0.9) \approx 0.10536$.
- Xét 2 token câu trả lời (Answer + EOS): mô hình đang rất lúng túng, chỉ đoán với xác suất thấp $p=0.25$ và $p=0.5$. Mất mát tương ứng là $-\ln(0.25) = \ln(4) \approx 1.386$ và $-\ln(0.5) = \ln(2) \approx 0.693$.
- Nếu tính trung bình trên toàn bộ 10 token ($L_{full}$):
  $L_{full} = \frac{8 \times 0.10536 + 1.386 + 0.693}{10} \approx 0.292$.
  Con số $0.292$ trông rất đẹp và nhỏ, khiến bạn lầm tưởng mô hình đã học rất tốt!
- Nhưng nếu chỉ tính trên 2 token câu trả lời thật sự ($L_{answer}$):
  $L_{answer} = \frac{1.386 + 0.693}{2} \approx 1.040$.
  Mất mát thực tế cao gấp gần 4 lần! Mất mát nhỏ của đoạn prompt dài đã "nhấn chìm" và che giấu hoàn toàn sự yếu kém của mô hình trên câu trả lời. Đó là lý do trong SFT, bắt buộc phải theo dõi riêng chỉ số **Answer-only Loss**).*

Loss nhỏ hơn không tự là model trả lời tốt hơn: hai số đang chấm các vị trí khác nhau. Không so loss pretraining toàn văn bản với loss SFT answer-only rồi kết luận chất lượng tăng chỉ từ độ lớn số. Lab giữ phép đo LM cũ và task riêng để so trước/sau đúng mục tiêu.

### 8.2 Cross-entropy cập nhật logits thế nào?

Tại một answer query có hai lớp, logits z=[ln 2,0], nhãn đúng là lớp index 1. Softmax=[2/3,1/3], loss=ln 3≈1.098612. Gradient theo logits:

\[
\nabla_z L=p-onehot(y)=[2/3, -2/3].
\]

Một bước SGD minh họa lr=0.1:

\[
z'=[\ln 2-1/15, \;1/15].
\]

*(Giải thích từng bước tính toán chi tiết:
1. **Logits đầu vào:** $z = [z_0, z_1] = [\ln 2, 0] \approx [0.693147, 0.0]$.
2. **Tính xác suất Softmax:**
   - $e^{z_0} = e^{\ln 2} = 2$.
   - $e^{z_1} = e^0 = 1$.
   - Tổng số mũ: $e^{z_0} + e^{z_1} = 2 + 1 = 3$.
   - Xác suất: $p_0 = \frac{2}{3} \approx 0.667$; $p_1 = \frac{1}{3} \approx 0.333$.
3. **Mất mát Cross-Entropy (với nhãn đúng là lớp 1, vector one-hot $y = [0, 1]$):**
   $L = -\ln(p_1) = -\ln(1/3) = \ln(3) \approx 1.098612$.
4. **Công thức đạo hàm của Softmax kết hợp Cross-Entropy:**
   $\nabla_z L = p - y$.
   - Với lớp 0 (lớp sai): $g_0 = p_0 - 0 = \frac{2}{3} > 0$ (gradient dương: tín hiệu yêu cầu GIẢM logit $z_0$ xuống).
   - Với lớp 1 (lớp đúng): $g_1 = p_1 - 1 = \frac{1}{3} - 1 = -\frac{2}{3} < 0$ (gradient âm: tín hiệu yêu cầu TĂNG logit $z_1$ lên).
5. **Cập nhật trọng số theo Gradient Descent với Learning Rate $\eta = 0.1$:**
   - $z_0' = z_0 - \eta \cdot g_0 = \ln(2) - 0.1 \times \frac{2}{3} = \ln(2) - \frac{1}{15} \approx 0.626480$.
   - $z_1' = z_1 - \eta \cdot g_1 = 0 - 0.1 \times \left(-\frac{2}{3}\right) = \frac{1}{15} \approx 0.066667$.
6. **Kiểm tra xác suất mới sau cập nhật:**
   $e^{z_0'} \approx 1.871$, $e^{z_1'} \approx 1.069 \to p_1' = \frac{1.069}{1.871 + 1.069} \approx 0.3636$.
   Xác suất của nhãn đúng đã tăng từ $33.3\%$ lên $36.4\%$. Mô hình đã dịch chuyển theo đúng hướng mong đợi!).*

Xác suất nhãn đúng tăng. Đây là cập nhật **logits coi như biến độc lập để hiểu hướng gradient**; khi train Transformer, optimizer cập nhật weight tạo ra logits, không lưu z của từng record như parameter. Tất cả layer trainable liên quan có thể nhận gradient qua chain rule.

```python
import math
import torch
from torch.nn import functional as F
z = torch.tensor([[math.log(2), 0.]], dtype=torch.float64, requires_grad=True)
loss = F.cross_entropy(z, torch.tensor([1]))
loss.backward()
torch.testing.assert_close(z.grad, torch.tensor([[2/3, -2/3]], dtype=z.dtype))
assert (z.detach() - .1*z.grad).softmax(-1)[0,1] > 1/3
print(loss.item(), z.grad)
```

Với nhiều nhãn answer và `reduction='mean'`, gradient của từng vị trí còn chia cho tổng số nhãn được chấm. Không đem gradient một token chưa chia so với gradient batch mean rồi coi khác là lỗi. [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html).

## 9. Những weight nào được cập nhật?

| Phương án | Weight học | Cần hiểu đúng |
|---|---|---|
| Full fine-tuning | Toàn bộ parameter được chọn của model | Không chỉ head; cần lưu weight đã cập nhật và kiểm hồi quy |
| Head-only trong lab | Linear head D→V và bias, freeze phần còn lại | Vẫn sinh chuỗi label qua LM head; không thay bằng classifier 3 lớp |
| LoRA | Các ma trận cập nhật hạng thấp trên module được chọn, thường giữ base weight cố định | Vẫn cần base checkpoint tương thích; không tự xóa chi phí base/activation |

*(So sánh chi tiết 3 phương án cập nhật trọng số:
1. **Full Fine-Tuning (Tinh chỉnh toàn bộ tham số):**
   - *Cơ chế:* Mở khóa toàn bộ các tầng trong mạng nơ-ron để gradient cập nhật mọi trọng số.
   - *Ưu điểm:* Mô hình có khả năng học và thích nghi mạnh mẽ nhất.
   - *Nhược điểm:* Tốn bộ nhớ VRAM lớn nhất (vì optimizer Adam phải lưu thêm 2 trạng thái động lượng $m$ và $v$ cho từng tham số, ngốn gấp 4–8 lần dung lượng mô hình); nguy cơ xảy ra hiện tượng "quên thảm khốc" cao nhất.
2. **Head-Only Fine-Tuning (Chỉ tinh chỉnh tầng đầu ra):**
   - *Cơ chế:* Đóng băng toàn bộ thân Transformer (*Backbone*), chỉ cho phép ma trận chiếu tuyến tính cuối cùng (`head.weight` và `head.bias`) nhận gradient.
   - *Ưu điểm:* Cực kỳ nhẹ và nhanh, tốn rất ít bộ nhớ GPU.
   - *Nhược điểm:* Khả năng học bị giới hạn nghiêm trọng vì thân mô hình không thể học thêm biểu diễn đặc trưng mới nào từ dữ liệu tác vụ.
3. **LoRA (Low-Rank Adaptation - Thích nghi hạng thấp):**
   - *Cơ chế:* Giữ nguyên và đóng băng 100% trọng số gốc $W$. Chèn thêm hai ma trận phụ $A$ và $B$ có kích thước rất nhỏ (hạng thấp $r \ll d$) chạy song song với $W$.
   - *Ưu điểm:* Tiết kiệm bộ nhớ GPU vượt trội (giảm 70–80% VRAM so với Full Fine-Tuning), cho phép huấn luyện mô hình lớn trên GPU phổ thông, dễ dàng lưu trữ và hoán đổi các adapter nhẹ nhàng (vài chục MB thay vì hàng chục GB)).*

`requires_grad_(False)` ngăn tạo gradient cho parameter đó. `.eval()` không freeze weight; `no_grad()` bao toàn forward có thể làm mất graph cần train. *(Phân biệt rõ 3 khái niệm thường nhầm lẫn trong PyTorch:
- `param.requires_grad_(False)`: Đóng băng tham số — PyTorch sẽ không tính đạo hàm và không cập nhật tham số này trong quá trình backward.
- `model.eval()`: Chuyển mô hình sang chế độ đánh giá — chỉ làm thay đổi hành vi của các lớp đặc thù như tắt Dropout và cố định thống kê của BatchNorm, hoàn toàn KHÔNG đóng băng trọng số!
- `with torch.no_grad():`: Tắt hoàn toàn việc ghi lại đồ thị tính toán (computational graph) trong khối lệnh — giúp tiết kiệm bộ nhớ khi chạy inference, nhưng nếu bọc hàm forward lúc đang huấn luyện adapter thì PyTorch sẽ không thể backward được).* Khi train head-only với backbone thật sự frozen và input không cần gradient, có thể tổ chức tính feature không graph; nhưng khi có adapter nằm sâu trong mạng, không được bao tùy ý cả backbone trong no_grad vì sẽ chặn gradient tới adapter.

Bài này thực sự train full/head-only, còn LoRA chỉ có demo tính tay/gradient, **chưa train adapter trên Transformer**. [PyTorch autograd: freezing và grad modes](https://docs.pytorch.org/docs/2.15/notes/autograd.html).

### Tính tay một phép LoRA

Theo vector cột, W có shape m×n, A có shape r×n, B có shape m×r:

\[
y=Wx+\frac\alpha r BAx.
\]

*(Giải thích các ký hiệu trong công thức LoRA:
- $W$: Ma trận trọng số gốc kích thước $m \times n$, được ĐÓNG BĂNG hoàn toàn (không tính gradient).
- $x$: Vector đầu vào kích thước $n \times 1$.
- $A$: Ma trận nén hạng thấp kích thước $r \times n$, ánh xạ không gian $n$ chiều xuống không gian nhỏ $r$ chiều ($r \ll n$).
- $B$: Ma trận giải nén kích thước $m \times r$, chiếu từ không gian nhỏ $r$ chiều trở lại không gian $m$ chiều ban đầu.
- $r$: Bậc hạng (*rank*) của LoRA, thường chọn các giá trị nhỏ như 4, 8, 16.
- $\alpha$: Hệ số tỷ lệ cố định (*scaling factor*) giúp ổn định quá trình học khi thay đổi rank $r$).*

Chọn r=α=1:

```text
W = [[1,0,-1], [0,2,1]]
A = [[1,0,1]]
B = [[0.2],[-0.1]]
x = [2,1,3]
```

Wx=[−1,5], Ax=5, BAx=[1, −0.5], nên y=[0,4.5]. Parameter cập nhật A/B có 3+2=5 phần tử; W nhỏ ở ví dụ chỉ có 6 nên không kỳ vọng tiết kiệm lớn ở kích thước này. Với W 1024×1024 và r=4, A/B có 8,192 phần tử thay vì 1,048,576 phần tử W; đó là đếm parameter, không phải benchmark tổng VRAM.

*(Từng bước tính toán chi tiết bằng số học:
1. **Nhánh gốc ($Wx$):**
   $Wx = \begin{bmatrix} 1\times 2 + 0\times 1 + (-1)\times 3 \\ 0\times 2 + 2\times 1 + 1\times 3 \end{bmatrix} = \begin{bmatrix} 2 - 3 \\ 2 + 3 \end{bmatrix} = \begin{bmatrix} -1 \\ 5 \end{bmatrix}$.
2. **Nhánh LoRA ($BAx$):**
   - Tính $Ax$: $A$ có shape $1 \times 3$, nhân với $x$ shape $3 \times 1$:
     $Ax = 1\times 2 + 0\times 1 + 1\times 3 = 5$ (kết quả là số vô hướng 5).
   - Tính $B(Ax)$: $B$ có shape $2 \times 1$, nhân với số 5:
     $BAx = \begin{bmatrix} 0.2 \times 5 \\ -0.1 \times 5 \end{bmatrix} = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$.
3. **Tổng đầu ra với $\frac{\alpha}{r} = \frac{1}{1} = 1$:**
   $y = Wx + BAx = \begin{bmatrix} -1 \\ 5 \end{bmatrix} + \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 4.5 \end{bmatrix}$).*

Nếu A khác 0, B=0 ở khởi tạo, update BA bằng 0 nên forward bằng base; ở bước đầu gradient A bằng 0 nhưng gradient B có thể khác 0. Không khởi tạo cả A lẫn B bằng 0 rồi kỳ vọng chúng tự học bình thường. *(Giải thích bí quyết khởi tạo ma trận LoRA:
- Trong kỹ thuật LoRA, ma trận $A$ được khởi tạo ngẫu nhiên theo phân phối chuẩn Gaussian, còn ma trận $B$ được khởi tạo bằng TOÀN SỐ 0.
- Vì $B=0$, tích $BA = 0 \times A = 0$. Điều này đảm bảo tại bước huấn luyện đầu tiên, nhánh LoRA hoàn toàn chưa can thiệp ($\Delta W = 0$), đầu ra của mô hình giống hệt 100% mô hình gốc.
- Khi lan truyền ngược, gradient của $B$ phụ thuộc vào $A \cdot x \neq 0$ nên $B$ lập tức nhận được gradient và bắt đầu học ngay từ bước đầu tiên!).* Khái niệm cập nhật hạng thấp và freeze base đối chiếu [LoRA](https://arxiv.org/abs/2106.09685); `--mode lora` kiểm ví dụ tự tạo này, không tái hiện số tiết kiệm phần cứng của nghiên cứu.

![Full, head-only và LoRA](figures/03-trainable.png)

## 10. Không chỉ “loss giảm”: forgetting và đánh giá hành vi

Sau SFT, model có thể làm task tốt hơn nhưng giảm khả năng viết tiếp loại văn bản cũ. Đây là dấu hiệu cần kiểm của **forgetting** (quên/giảm năng lực đã học), không tự chứng minh model mất mọi kiến thức hay không thể phục hồi. *(Định nghĩa Hiện tượng Quên thảm khốc - Catastrophic Forgetting: Khi mạng nơ-ron được tinh chỉnh quá mạnh trên một nhiệm vụ mới hẹp (như phân loại ticket), các trọng số ma trận bị xoay chuyển để tối ưu cho nhiệm vụ mới, vô tình làm xáo trộn các biểu diễn đặc trưng từng giúp mô hình hiểu văn bản tổng quát. Mô hình có thể trở nên rất giỏi phân loại ticket nhưng mất khả năng viết một câu văn hoàn chỉnh).* Đo một suite nhiệm vụ cũ cùng tokenizer/context thay vì suy từ vài câu nhìn khác.

Ví dụ route cần ít nhất ba phép đo:

- Answer CE dưới teacher forcing: model gán xác suất cho nhãn/EOS thật tốt không? *(Đo lường mức độ khớp xác suất toán học của mô hình trên tập câu trả lời mẫu).*
- Greedy exact và valid format: khi không có answer thật trong prefix, có tự sinh đúng nhãn rồi EOS không? *(Đo lường hành vi sinh tự do thực tế của mô hình: có tuân thủ định dạng và trả về chính xác chuỗi mong muốn hay không).*
- LM CE trên văn bản miền cũ: khả năng viết tiếp trước kia có bị giảm không? *(Đo lường xem mô hình có bị "quên" tri thức nền tảng đã học ở giai đoạn tiền huấn luyện hay không).*

Chọn checkpoint bằng validation với tiêu chí xác định trước, rồi mở test. Nếu dùng nhiều variant hoặc xem test nhiều vòng để sửa model, cần một holdout phù hợp cho kết luận cuối. Nhiều seed, tập ngoài miền và metric nghiệp vụ là bước tiếp theo; lab một seed chưa đủ để khẳng định phương pháp nào luôn tốt hơn.

## 11. Thực hành: pretrain nhỏ → SFT, có control

### 11.1 Dữ liệu và split

8 service ×6 operation tạo 48 nhóm `[service, operation]`. Shuffle seed 20, chia 32/8/8 nhóm; mỗi nhóm có 6 event →192 train,48 validation,48 test. Mỗi lớp có 2 event mỗi nhóm nên test có 16 AUTH,16 DATA,16 NET.

Các câu LM và record SFT cùng incident phải nằm chung split. Pretraining chỉ dùng trường `text` của train; không có prompt/answer của test trong base training. Mỗi service/operation/event riêng có thể xuất hiện ở nhiều split, nhưng **cặp service–operation** không giao nhau. Đây là holdout tổ hợp hẹp, không phải dữ liệu production hoặc holdout event chưa thấy.

BPE fit 64 merge **chỉ trên text pretrain train**; freeze tokenizer cho mọi nhánh. Answer viết hoa có thể chưa gặp lúc fit nhưng vẫn encode được nhờ base byte. Prompt kết thúc space, answer bắt đầu non-space; pre-tokenizer của lab tạo ranh giới nên encode(prompt)+encode(answer)=encode(prompt+answer). `batch()` kiểm hợp đồng này; không giả định mọi tokenizer/template đều có tính chất đó.

### 11.2 Kiến trúc và các nhánh

Kế thừa bài 19: vocabulary 323, D=32, H=4, F=64, một pre-norm causal block, learned position 128×32, final LN, head không tying, dropout=0. Tổng **33,699 parameter**. `core.py` lưu phần lõi ngay trong bài, nên không cần import file của bài trước để chạy.

```text
initial random weights
  ├─ pretrain LM → best base ─┬─ full SFT → best full-sft
  │                          └─ freeze backbone, SFT head → best head-sft
  └─ cùng initial weights ────── SFT từ đầu → best scratch-sft
```

*(Mô tả 4 nhánh thí nghiệm đối chứng:
1. **Base:** Mô hình chỉ qua tiền huấn luyện (Pretraining) trên văn bản thô, chưa từng học SFT.
2. **Base → Full SFT:** Lấy checkpoint Base tốt nhất, mở khóa toàn bộ tham số để huấn luyện SFT.
3. **Base → Head-only SFT:** Lấy checkpoint Base tốt nhất, đóng băng thân Transformer, chỉ tinh chỉnh tầng Linear Head cuối cùng.
4. **Random → SFT (Scratch):** Khởi tạo ngẫu nhiên từ đầu và nhảy thẳng vào học SFT mà không qua giai đoạn tiền huấn luyện).*

Mọi nhánh dùng clone để không làm thay đổi base. Pretraining 25 epoch lr=0.003; mỗi nhánh SFT 25 epoch lr=0.001; batch 16, Adam, clip norm 1, CPU float 32 một thread. 192/16=12 update/epoch, 300 update mỗi giai đoạn. Scratch và full có cùng ngân sách **SFT**, nhưng full còn có chi phí pretraining thêm; đây không phải so sánh cùng tổng compute. Toàn run có 1,200 update.

Mỗi stage dùng optimizer mới; đây là đổi mục tiêu/adaptation, không phải resume chính xác. Base chọn best theo validation LM CE; các SFT chọn best theo validation answer CE. Code không early stop, vẫn chạy hết số epoch; best epoch có thể sớm hơn epoch cuối.

![Split dữ liệu và ngân sách xuyên giai đoạn](figures/04-data-budget.png)

### 11.3 Kết quả đã chạy thật

| Nhánh | Best epoch | Parameter train trong stage | Test answer CE | Greedy exact, gồm EOS | Format hợp lệ | Test LM CE cũ |
|---|---:|---:|---:|---:|---:|---:|
| Base sau pretrain | 18 | 33,699 | 9.211146 | 0/48 =0% | 0% | 0.466955 |
| Base → full SFT | 25 | 33,699 | 0.034129 | 48/48 =100% | 100% | 2.531813 |
| Random → SFT | 25 | 33,699 | 0.231383 | 28/48 =58.3333% | 58.3333% | 8.464169 |
| Base → head-only SFT | 25 | 10,659 | 0.827183 | 6/48 =12.5% | 22.9167% | 1.073547 |

*(Phân tích chuyên sâu kết quả thực nghiệm:
- **Base sau pretrain (0% Exact):** Mô hình đạt $0\%$ chính xác trên tác vụ định tuyến vì nó chưa từng học cách trả lời. Khi nhận prompt, nó chỉ tiếp tục sinh log thay vì đưa ra nhãn đội. Tuy nhiên, nó có năng lực viết tiếp log cũ tốt nhất ($LM\ CE = 0.467$).
- **Base → Full SFT (100% Exact, nhưng LM CE tăng lên 2.532):** Mô hình trả lời hoàn hảo toàn bộ 48 câu test. Nhưng chỉ số mất mát trên dữ liệu log cũ ($LM\ CE$) vọt từ $0.467$ lên $2.532$. Đây là bằng chứng thực nghiệm rõ ràng nhất cho hiện tượng **Catastrophic Forgetting** (quên bớt năng lực cũ khi học tác vụ mới).
- **Random → SFT (58.33% Exact):** Khi không có tiền huấn luyện, mô hình phải mò mẫm học từ đầu cả ngữ pháp ngôn ngữ lẫn tác vụ phân loại trong cùng 25 epoch, dẫn đến độ chính xác kém xa so với mô hình có Pretraining ($58.3\%$ so với $100\%$). Điều này chứng minh giá trị nền tảng của giai đoạn Pretraining.
- **Base → Head-only SFT (12.5% Exact):** Việc đóng băng toàn bộ thân Transformer và chỉ cập nhật ma trận phân loại cuối cùng khiến mô hình bị thiếu năng lực biểu diễn, hầu như không thể trích xuất mối liên hệ giữa các từ trong prompt để ra quyết định đúng).*

Full SFT cải thiện routing trong thí nghiệm này, đồng thời LM CE cũ tăng từ khoảng 0.467 lên 2.532. Đây là sự đánh đổi đã đo trên văn bản template cũ, không phải bằng chứng về mọi năng lực ngôn ngữ. Scratch chưa từng pretrain nên LM CE cao của nó không gọi là “quên so với base”; nó có điểm bắt đầu khác.

Head-only giữ nguyên 23,040 parameter backbone, nhưng output head vẫn đổi nên LM CE cũng có thể đổi. Freeze backbone không bảo đảm hành vi đầu ra cũ được giữ. Ít parameter train hơn không tự là chất lượng tốt hơn.

Baseline luôn đoán một lớp chọn từ train (ba lớp hòa, chọn AUTH theo thứ tự cố định) đúng 16/48=33.3333%. Baseline rule theo mapping event đúng 48/48=100%. Task được định nghĩa bằng rule nên model không vượt lời giải trực tiếp; không lấy kết quả 100% trên 48 câu template làm lý do triển khai cho routing thực.

Mẫu test đầu, event timeout, answer NET:

```text
Base:        event=denied.\n       (viết tiếp log, sai format)
Full SFT:    NET                   (đúng, có EOS)
Scratch SFT: NET                   (mẫu này đúng; toàn test còn lỗi)
Head-only:   NET                   (mẫu này đúng; toàn test chỉ đúng 6/48)
```

Mẫu event denied của scratch có output `AT`, head-only có `DATH`. Không sửa thành AUTH để bảng đẹp hơn. Nhãn display không bị strip hay chuẩn hóa để che lỗi; exact kiểm ID answer chuẩn + EOS.

![Đường học từng mục tiêu và đánh giá trước/sau](figures/05-results.png)
![Confusion matrix gồm cột invalid](figures/06-confusion.png)

## 12. Đọc code và artifact

[lab.py](lab.py) chứa tạo split, batch LM/SFT, train từng stage, generation và đo metric. [core.py](core.py) chứa tokenizer/model kế thừa bài 19. Không có thư viện TRL/PEFT hoặc tải model ngoài.

Run lưu `tokenizer.json`, `data.json`, `report.json`, bốn file checkpoint `base.pt`, `full-sft.pt`, `scratch-sft.pt`, `head-sft.pt` và log `*-metrics.json`. Checkpoint gồm state_dict, config, tokenizer hash, stage, best epoch và hash weight trước stage; report có hyperparameter từng stage, số update, các parameter đổi và metric. Hash giúp truy dòng weight, không chứng minh nguồn checkpoint đáng tin.

`load_checkpoint` kiểm tokenizer hash rồi dựng model/tải weight cho inference; flags requires_grad không nằm trong state_dict. Muốn train lại head-only phải thiết lập freeze lại, không coi tên file head-sft tự freeze module. Checkpoint này không có optimizer/RNG để resume như bài 12.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import run, load_checkpoint, ByteBPE, generate
with TemporaryDirectory() as folder:
    out = Path(folder) / 'run'
    report = run(out, pre_epochs=1, sft_epochs=1)  # Kiểm pipeline, chưa kỳ vọng bảng 25 epoch.
    tok = ByteBPE.load(out / 'tokenizer.json')
    model, info = load_checkpoint(out / 'full-sft.pt', tok)
    print(generate(model, tok, 'Route: service=api op=read event=timeout.\nTeam: '))
    assert report['stages']['full-sft']['updates'] == 12
```

Generation greedy chặn PAD/BOS, dừng EOS hoặc budget 10. Không ràng buộc output chỉ được AUTH/DATA/NET nên có thể invalid; metric giữ cột INVALID. Không có KV cache, top-p hoặc logit bias phục vụ trả đúng label bằng mẹo. Bài 21 sẽ đi sâu decoding/context.

## 13. Những bẫy cần phát hiện

| Bẫy | Hậu quả / cách kiểm |
|---|---|
| Chấm prompt dài, answer ngắn mà chỉ xem loss tổng | Model viết prompt tốt che answer kém; log answer CE riêng |
| Mask cả token answer đầu do lệch shift | Mất tín hiệu chọn nhãn; in x/y và vị trí mask trước train |
| Mask theo `id == EOS` khi EOS dùng làm PAD | Có thể bỏ cả EOS thật; dùng length/attention metadata đúng hợp đồng |
| `no_grad` toàn model khi muốn train adapter bên trong | Chặn graph cần học; kiểm parameter nào có gradient/thay đổi |
| Nhánh head-only khởi tạo sau full SFT đã sửa base | So sai điểm bắt đầu; clone và kiểm parent hash |
| Đổi tokenizer/template tại inference | Shape có thể vẫn đúng nhưng ý nghĩa ID hoặc ranh giới khác |
| Chọn epoch bằng test hoặc pretrain trên incident test | Contamination; giữ split xuyên mọi stage |
| SFT làm format đúng nên tưởng nội dung đúng | Đo riêng exact/format và metric tác vụ thật |
| Train loss giảm, khả năng cũ không đo | Bỏ sót regression/forgetting |

*(Hướng dẫn chi tiết cách phòng tránh các bẫy thực tế:
- **Bẫy 1 — Giám sát nhầm Loss:** Luôn tách biệt hai độ đo: Loss trên Prompt (phải bằng 0 hoặc bỏ qua) và Loss trên Answer (độ đo cốt lõi để quyết định dừng huấn luyện).
- **Bẫy 2 — Lệch chỉ số Shift:** Luôn in ra màn hình chuỗi token của $x$ và $y$ trên một ví dụ cụ thể trước khi bấm nút train. Đảm bảo token đầu tiên của Answer nằm đúng vị trí có nhãn hợp lệ trong $y$, không bị gán `-100`.
- **Bẫy 3 — Đóng băng nhầm khi dùng LoRA:** Khi dùng adapter, hãy lặp qua danh sách `model.named_parameters()` và in ra cờ `p.requires_grad`. Đảm bảo rằng chỉ có các ma trận LoRA mang giá trị `True`, còn các ma trận gốc mang giá trị `False`.
- **Bẫy 4 — Ảo tưởng định dạng:** Mô hình sinh ra cấu trúc JSON rất đẹp mắt không đồng nghĩa với việc thông tin bên trong JSON là đúng sự thật. Luôn cần các bộ kiểm tra logic nghiệp vụ độc lập).*

## 14. Lệnh chạy và hình

Đã kiểm Python 3.14.6, torch 2.15.0, macOS arm 64 CPU; NumPy 2.5.4, Matplotlib 3.11.2 dùng kiểm/vẽ. Không cần GPU hay tài khoản. Từ thư mục bài 20:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode mask
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode lora
.venv/bin/python lab.py --mode all --out /tmp/lesson20-run-01
```

`--mode train` chỉ chạy các stage; `--pre-epochs 1 --sft-epochs 1` để kiểm ngắn. Output phải mới; bỏ --out sẽ in đường dẫn run tạm. Windows dùng `.venv\Scripts\python` cùng đường dẫn phù hợp. Phiên bản ghim cần wheel phù hợp; seed không bảo đảm giống bit giữa mọi backend.

Sáu hình có PNG/SVG, tái tạo từ công thức và artifact run thật:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py --run /tmp/lesson20-run-01
```

## 15. Tự đánh giá và bước tiếp

Làm [16 bài tập](exercises.md) rồi xem [lời giải](solutions.md). Trước khi đi tiếp, hãy tự giải thích vì sao SFT vẫn là next-token nhưng khác vị trí chấm loss; vì sao prompt không có direct loss vẫn có gradient; vì sao freeze backbone không giữ nguyên mọi output; và vì sao full SFT thắng scratch ở đây chưa chứng minh lợi thế trên mọi task/ngân sách.

[Phụ lục](deep-dive.md) có thêm gradient, dữ liệu đa lượt, đếm ngân sách và cách đọc artifact. **Bài tiếp theo:** [Bài 21 — Decoding, context và inference](../21-decoding-context-inference/README.md), chọn token, quản lý context và kiểm KV cache/streaming. [Nguồn và phạm vi đối chiếu](../SOURCES.md).
