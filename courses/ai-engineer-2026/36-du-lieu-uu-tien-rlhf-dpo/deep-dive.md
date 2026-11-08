# Phụ lục bài 36 — Đọc sâu cơ chế và các điểm dễ sai

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Đọc sau khi tự tính được một cặp DPO. Các phép suy luận và ví dụ số dưới đây do bài học biên soạn; nguồn lý thuyết chính là mục 3–5/Appendix A của [DPO](https://arxiv.org/pdf/2305.18290), loss reward đối chiếu với [TRL Reward Modeling](https://huggingface.co/docs/trl/reward_trainer). Không coi phép suy ra dưới giả định là kết quả đo chất lượng sản phẩm.

## 1. Tối ưu KL có thể viết thành một KL khác

Với `q(y)>0`, β>0 và Z hữu hạn, đặt `p*(y)=q(y)exp(r(y)/β)/Z`. Vì:

\[
\log p^*(y)=\log q(y)+r(y)/\beta-\log Z,
\]

nên mục tiêu của một prompt bằng:

\[
\sum_y p(y)r(y)-\beta\sum_yp(y)\log\frac{p(y)}{q(y)}
=\beta\log Z-\beta D_{KL}(p\|p^*).
\]

KL không âm nên nghiệm tối ưu là p=p*. Cách này còn cho một cách kiểm optimizer: objective gap bằng β nhân KL đến nghiệm đóng. Nếu finite-action lab cho objective cao hơn `βlogZ` ngoài sai số float, cần kiểm công thức hoặc normalization.

LLM có tập chuỗi khổng lồ nên không liệt kê Z trong thực tế. Cặp cùng prompt khiến hằng số chuẩn hóa triệt tiêu khi lấy hiệu; đây là bước giúp viết loss DPO mà không tính Z.

## 2. Support của reference là điều kiện có nội dung

Nếu `q(y)=0`, tỷ số và log trong bài không còn dùng trực tiếp được tại y. Finite KL yêu cầu p không dồn xác suất vào vùng q bằng 0. Softmax trên logits hữu hạn thường cho xác suất dương về mặt toán; trong số học, nhân tích token có thể underflow về 0, vì vậy dùng tổng log-probability.

Top-k/top-p trong decoding lại có thể đặt xác suất lấy mẫu một số token bằng 0. Không lấy xác suất sau bộ lọc đó làm reference log-probability trong lab DPO này. Chấm điểm full-vocabulary bằng logits gốc của mô hình theo cùng template. Nếu thay phân bố dùng để huấn luyện, cần nêu rõ mục tiêu mới.

`q(y)>0` cũng chưa là dữ liệu đủ tốt: một chuỗi xác suất cực nhỏ vẫn có thể nằm ngoài coverage mà reward model đã học đáng tin. Bandit chỉ cần q của verbose_false bằng .01 để thấy hậu quả ngoại suy.

## 3. Những thứ không xác định chỉ từ so sánh

Một cặp chỉ chứa thông tin về chênh lệch reward. Cộng `c(x)` vào tất cả reward cho cùng prompt không đổi xác suất ưu tiên và không đổi policy tối ưu có KL. Không thể dùng phép so sánh đơn thuần để nói “reward 7 ở prompt A tốt hơn reward 5 ở prompt B” nếu chưa thống nhất cách chuẩn hóa và ý nghĩa thước đo.

Scale là chuyện khác offset: nhân reward với c>0 làm sigmoid tự tin hơn, không bất biến. Tuy nhiên nếu cùng nhân β với c thì policy tối ưu `q exp(r/β)/Z` không đổi. Vì vậy so sánh raw reward của hai model hoặc raw β của hai recipe rất dễ sai.

Trong ví dụ ba phiếu thuận/một phiếu ngược, tỷ lệ .75 là thống kê trên bốn phiếu tổng hợp. Đó chưa là xác suất calibrated cho mọi người dùng. Bất đồng có thể do thiếu ngữ cảnh, khác ưu tiên hoặc rubric không rõ, không nhất thiết là noise cần loại bỏ.

## 4. DPO có thể giảm cả xác suất chosen lẫn rejected

Một phản ví dụ hợp lệ, có ba completion A/B/C:

```text
reference: A=.4, B=.3, C=.3
policy:    A=.2, B=.1, C=.7
pair:      A > B
```

Chênh lệch chuẩn hóa của cặp:

\[
m=\ln(.2/.4)-\ln(.1/.3)=\ln1.5>0.
\]

Loss β=1 giảm từ `ln2` xuống `−ln(.6)=.510826`, dù xác suất A giảm một nửa. Xác suất chuyển vào C không được cặp này trực tiếp ràng buộc. Phản ví dụ số này đủ bác bỏ kết luận “DPO loss giảm thì mọi chosen đều tăng xác suất tuyệt đối”.

Trong mạng chia sẻ tham số, gradient của một cặp còn ảnh hưởng các prompt khác. Hãy xem chosen/rejected logps, độ dài sinh, tính đúng và hành vi ngoài phân bố cùng preference margin. Tài liệu [TRL DPO, Logged metrics](https://huggingface.co/docs/trl/dpo_trainer) liệt kê logps và implicit reward/margin; tên “reward accuracy” trong trainer cần đọc đúng định nghĩa, không gọi là factual accuracy.

## 5. Sum hay mean log-probability không phải một chi tiết vô hại

Cho completion A có hai target token với log-probability `[-.1,-.1]`, B có ba token `[-.08,-.08,-.08]`. Sum cho A=−.2>B=−.24; mean cho A=−.1<B=−.08. Hai phép giảm chiều có thể đảo thứ tự.

DPO dùng hiệu log-ratio so với reference, nên ví dụ sum/mean trên một model chưa đủ mô tả mọi hành vi của DPO. Nhưng nó chứng minh chuẩn hóa độ dài thay phép tính, không thể đổi rồi giữ nguyên mọi diễn giải. Bài lab completion phần lớn cùng độ dài để cô lập cơ chế; test riêng dùng completion dài ngắn khác nhau để phát hiện lỗi mask và reduction.

Khi đọc recipe, kiểm loss_type thực sự, label smoothing, cách EOS, truncation, prompt mask và aggregation. Trang TRL hiện hành có nhiều biến thể; ví dụ `sigmoid_norm` được mô tả riêng với chuẩn hóa số token. Bài này chỉ cài sigmoid DPO với tổng completion logps, không chạy TRL Trainer. [TRL DPO, Loss types](https://huggingface.co/docs/trl/dpo_trainer).

## 6. Freeze, eval, no_grad và cache giải quyết các vấn đề khác nhau

- `requires_grad_(False)` tắt gradient trên tham số reference.
- `eval()` thay chế độ những layer như dropout; nó không tắt autograd.
- `torch.no_grad()` không tạo graph khi tính reference score; dùng cả ở bước cache.
- Cache chỉ hợp lệ cho đúng reference weights, input token IDs, mask, template và reduction đã tạo nó.

Policy và reference phải có bộ tham số riêng khi full fine-tuning. `reference=policy` chỉ thêm một tên trỏ cùng đối tượng, không clone. Reference vô tình trôi theo policy sẽ làm mất ý nghĩa mốc so sánh; nếu detach chưa đúng, gradient còn chảy vào nhánh sai.

Lab không có dropout nên các score cố định dễ kiểm. Với model có dropout, `eval()` trên reference tránh score thay theo mask dropout. Nếu policy dropout thay giữa lần chấm điểm, initial loss mỗi batch không nhất thiết bằng ln2 chính xác dù trọng số khởi tạo bằng nhau.

Bài 35 gợi thêm rủi ro khi ghép PEFT: “tắt adapter” chỉ cho base, chưa chắc cho reference mong muốn nếu reference bao gồm adapter SFT. Cần tái dựng đúng trạng thái SFT và đo score trước huấn luyện. Đừng mặc định base pretrained và SFT reference là cùng mô hình.

## 7. Một cặp sai nhãn có thể kéo gradient ngược rất mạnh

Với `L=softplus(−βm)`, `dL/dm=−βσ(−βm)`. Khi m rất âm, độ lớn gần β; khi m rất dương, độ lớn gần 0. Những cặp model hiện đánh giá ngược hướng được ưu tiên cập nhật mạnh. Nếu một nhãn sai và nhất quán với một mẫu hình thức, model có thể học mẫu hình thức đó.

Label smoothing trong logistic preference có thể viết `L=−(1−ε)logσ(βm)−εlogσ(−βm)`. Nó làm thay mục tiêu; ε không tự biết nhãn nào sai. Nếu bất đồng xuất phát từ nhiều nhóm người dùng có mục tiêu hợp lệ khác nhau, một scalar trung bình chưa giải quyết bài toán thiết kế sản phẩm.

Lab giữ synthetic labels rõ, không bật smoothing và không khẳng định robust trước noisy label. Phần bandit có bốn phiếu trái chiều cố ý để có nghiệm reward hữu hạn; phần TinyLM ưu tiên nhất quán để tiện quan sát.

## 8. Offline preference và dữ liệu do policy hiện tại sinh

DPO ở đây học lại một tập cặp cố định. Nếu policy sau huấn luyện tạo kiểu câu chưa có trong tập đó, metric offline có thể bỏ sót. Cải thiện coverage cần thu phản hồi cho các đầu ra mới và đánh giá lại trên tập giữ riêng; đó là một vòng dữ liệu, không chỉ tăng epoch.

RLHF dùng RM có thể tạo rollout mới mà không hỏi con người ở từng token; RM đang thay con người cung cấp tín hiệu trong vòng tối ưu. Khi policy đi tới vùng RM ngoại suy sai, reward hacking có thể xuất hiện. Nghiên cứu [Gao và cộng sự](https://arxiv.org/abs/2210.10760) khảo sát overoptimization trong thiết lập synthetic gold/proxy; bài này chỉ đọc abstract và không suy số định lượng từ nghiên cứu đó.

Ví dụ `verbose_false` của lab là phản ví dụ độc lập do bài học tự thiết kế. Nó cho thấy hoàn toàn có thể tăng proxy reward trong khi giảm utility ta quan tâm; không ước lượng tần suất hiện tượng ở một LLM thật.

## 9. Đánh giá phải tách ba câu hỏi

**Model có fit các nhãn ưu tiên không?** Xem pair NLL, raw completion ordering và relative reward ordering. Lab dùng ngưỡng `1e-6` cho relative tie; reference so với chính nó có mọi relative margin bằng 0. Không gọi trường hợp đó là 0% chất lượng.

**Output sinh ra có đáp ứng công việc không?** Lab kiểm route token đầu và exact sequence `LABEL CHECK EOS` theo một rule khác với tính loss. Rule này vẫn cùng ý đồ thiết kế dữ liệu, nên chưa là đánh giá độc lập về nguồn nhãn. Phải phân biệt với bandit: utility bandit là một hàm riêng không được dùng fit.

**Model có gây regression không?** Cần tập kiểm tra hành vi ngoài các cặp học: factuality, refusal khi thiếu quyền, follow-up, context dài, ngôn ngữ, nhóm người dùng và chi phí/độ trễ theo yêu cầu sản phẩm. Lab chưa kiểm các năng lực đó. Không suy rằng reference KL thấp sẽ giữ mọi khả năng.

Nếu dùng judge model, ẩn tên ứng viên, đảo thứ tự, có mẫu kiểm người thật và phân tích bất đồng. Báo CI trên đơn vị group phù hợp; bootstrap từng cặp trong cùng prompt rồi coi chúng độc lập sẽ lạc quan quá mức. 12 prompt/4 group của lab không đủ làm benchmark tổng quát.

## 10. LoRA, SFT và DPO là ba quyết định khác nhau

SFT/DPO nói về tín hiệu/loss. Full fine-tuning/LoRA nói về tham số được cập nhật. FP32/FP16/NF4 nói về biểu diễn tính toán/lưu trữ. Có thể kết hợp chúng khi implementation và phần cứng phù hợp, nhưng mỗi kết hợp cần kiểm reference đúng, mask đúng, gradient đúng và memory thực.

Ở DPO với full reference và policy, phải lưu đủ trọng số hai nhánh, hoặc thiết kế cache/adapter sharing được kiểm chứng. Cache giảm lượt forward reference trong mỗi epoch nhưng có chi phí lưu score và hợp đồng invalidation. Nếu reference đổi, score cũ không còn đúng.

Lab chỉ full-parameter CPU TinyLM để giảm biến gây nhiễu. Nó chưa đo tốc độ, VRAM, QLoRA+DPO, multi-GPU hay distributed accumulation. Với gradient accumulation, nhớ mean theo **số cặp**; mean của hai microbatch kích thước khác nhau rồi chia đều không phải mean toàn batch.

## 11. PPO: clip ratio và KL reference không đồng nghĩa

Hai ratio có thể xuất hiện:

\[
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)},\qquad
\ell_t=\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}.
\]

Cái đầu dùng trong surrogate PPO trên rollout đã thu; cái sau đo độ lệch khỏi reference của mục tiêu RLHF. Với chuỗi đầy đủ có EOS và phân bố phù hợp, tổng log-ratio token là log-ratio sequence. Với giới hạn chiều dài, cần xác định phân bố trên chuỗi kết thúc/cắt cụt trước khi đồng nhất tổng token với KL trên toàn văn bản.

Ví dụ surrogate trong bài chính chỉ tính hai số của công thức, chưa mô phỏng một lượt PPO. Một trainer PPO còn cần quy ước reward/value, advantage, masking, rollout sampling và các loss khác. Xem [PPO, Eq. 7 và thuật toán](https://arxiv.org/pdf/1707.06347) để phân biệt phần cốt lõi với một hệ huấn luyện hoàn chỉnh.

## 12. Một kế hoạch nâng lab thành dự án nhỏ

Giữ nguyên cách tách tài liệu và mã, thay toy triage bằng các câu hỏi backend đã được phép dùng. Viết rubric trước, thu ứng viên từ baseline SFT, lưu cả những cặp khó phân xử. Tạo held-out incident/time/source trước khi tune. Có baseline không train, SFT chosen-only và DPO với ngân sách phù hợp; ghi decoding giống nhau khi so sánh.

Điều kiện dừng không chỉ là preference loss: cần đạt tiêu chí công việc, không tăng lỗi nghiêm trọng, không giảm các lát cắt quan trọng và đáp ứng chi phí vận hành. Từng tiêu chí phải có phép đo cụ thể và ngưỡng do bối cảnh ứng dụng quyết định. Với lab hiện tại, rule lookup giải chính xác route/style; vì vậy huấn luyện model là phương tiện học cơ chế, chưa là lựa chọn sản phẩm hợp lý hơn rule.
