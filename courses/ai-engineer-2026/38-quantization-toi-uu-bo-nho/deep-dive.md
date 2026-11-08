# Phụ lục bài 38 — Sai số và bộ nhớ cần được theo dõi riêng

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Các phép suy luận dưới đây là diễn giải và ví dụ tự biên soạn. Nguồn cho linear quantization là [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html); nguồn cho calibration/PTQ là [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Format thực hành được định nghĩa bởi code của bài, không tương thích nhị phân với các định dạng ngoài chỉ nhờ cùng bit width.

## 1. Vì sao sai số weights nhỏ chưa bảo đảm output nhỏ?

Với một layer và input cố định, đặt Ŵ=W+ΔW. Khi chỉ đổi weights:

\[
\Delta y=x\Delta W^T.
\]

Với mỗi output j, Cauchy–Schwarz cho `|Δyj|≤||x||₂||Δwj||₂`. Vì vậy cùng một weight error có thể gây output error lớn hơn khi input có norm lớn. Nếu đổi cả activation, còn có `Δx Wᵀ` và `Δx ΔWᵀ`. Các layer tiếp theo cùng phi tuyến tiếp tục biến đổi sai số.

Đây là lý do kiểm weight MSE chưa đủ. Lab đo logit RMSE, KL phân bố và metric classification. Stress4× là một cách nhìn sensitivity của model và calibration; không chứng minh một định luật tỷ lệ sai số cố định cho mọi mạng.

## 2. Argmax, margin và cùng accuracy

Hai logit vector `[1,.99]` và `[.98,1.01]` rất gần nhưng đổi top1. Nếu margin giữa hai lớp cao nhất nhỏ, lượng sai số nhỏ cũng có thể đổi lựa chọn. Ngược lại, logit RMSE có thể khá lớn nhưng thứ tự lớp giữ nguyên.

Với classification, xem accuracy theo nhãn và agreement theo baseline. Với LM, còn phải kiểm token sequence/độ dài/stopping và metric công việc; lỗi một token có thể đổi toàn prefix sau đó. Không chuyển kết quả MLP sang một cam kết perplexity/generation quality của LLM.

Một metric tổng không cho biết mẫu nào đổi. Khi accuracy bằng nhau, vẫn có thể có mẫu bị mất và mẫu được sửa, hoặc hai model sai theo cách khác nhau. Cần đối chiếu từng prediction nếu muốn kết luận về mẫu lỗi.

## 3. Clipping range là một quyết định có trade-off

Scale absmax chứa outlier nhưng có thể dành phần lớn các mức code cho vùng hiếm. Chọn percentile/MSE/entropy calibration là những cách cân nhắc khác nhau về sai số đại diện. Khi cho phép clip, bound s/2 của rounding không giới hạn tổng error ngoài range nữa.

Ví dụ symmetric range `[−1,1]`, scale1/127: giá trị4giải mã thành1, error−3. Giảm step size không sửa lỗi do range chỉ đến1. Nếu tăng range tới4, step tăng4lần và các giá trị nhỏ có lưới thô hơn. Granularity hoặc một phép biến đổi tương đương có thể giúp, nhưng cần kiểm cost/quality.

Static calibration của lab dùng absmax FP32 input của ba Linear, chưa chạy percentile hoặc tối ưu range. Các input layer sau trong model quantized có thể đổi do lỗi layer trước, nên range đã thấy ở FP32 không bảo đảm bao hết nhánh quantized.

## 4. Symmetric range và zero-point không chỉ có một quy ước

Hardware INT8 signed có miền−128…127; lab dùng−127…127để absmax đối xứng. INT4 signed thường nói tới−8…7; lab dùng−7…7. Một code bị để dư là quyết định biểu diễn, không là chứng minh mọi engine phải làm vậy.

Affine z nói code nào biểu diễn0số thực. Với packed symmetric weights, phép cộng8 chỉ là offset encoding để nhét signed q vào nibble0…15. Nếu dùng code đó trực tiếp trong dot mà không trừ8, kết quả sai. Metadata format phải nêu rõ range, scale, offset, nibble order, group axis và padding.

Round-to-nearest-even áp vào `x/s` rồi cộng z trong code affine. Khi z nguyên nhưng lẻ, tự chuyển vị trí round qua z ở điểm tie có thể đổi kết quả của ties-to-even. Không lược bỏ thứ tự phép toán khi cần đối chiếu implementation.

## 5. Integer dot, bias và accumulator

Với scalar symmetric scales, dot giải mã là `sx sw Σqx qw`. Affine cho:

\[
s_xs_w\left[\sum q_xq_w-z_w\sum q_x-z_x\sum q_w+nz_xz_w\right].
\]

Các hạng correction giải thích vì sao zero-point thêm công việc và hợp đồng cho kernel. Nếu scale weight theo output row, mỗi output có sw riêng; nếu per-group, phải tích lũy các nhóm với scale khác nhau theo cách phù hợp.

Tổng nhiều tích cần accumulator đủ rộng; output có thể quay về float hoặc requantize với scale khác. Bias cũng cần quy ước phù hợp, không chỉ cast nó sang INT8 cùng weights. Lab chỉ tính tay một accumulator INT32 nhỏ; PackedLinear thực tế dequant và dùng FP32 matmul/bias, chưa có integer kernel tổng quát.

## 6. Group size, padding và effective bits

Một weight matrix m×n, group g theo input columns, cần `m×ceil(n/g)×g` slot trước tail nibble. Scale count là `m×ceil(n/g)`. Với4bit data, byte codes=`ceil(slot/2)`; với scaleFP32, thêm `4×scale_count`. BiasFP32 thêm4m.

Layer8→32với g32 cần1024slot cho256weights; mặc dù3/4slot là padding. Per-row n=8 chỉ cần256slot và32scale. Cấu trúc model nhỏ có thể làm format dành cho matmul lớn kém tiết kiệm hơn. Tensor có nhiều phần tử hơn không tự giảm padding nếu n không hợp alignment.

Effective bits/parameter cần nói đang chia cho parameter nào: toàn model hay chỉ phần quantized. Một con số4.5có thể gồm scale và phần FP16, nhưng chưa gồm file header/workspace/KV. So sánh định dạng cần cùng phạm vi đếm.

## 7. Persistent payload khác peak khi thực thi

PackedLinear giữ low-bit buffers giữa các forward nhưng tạo Ŵfloat32trong mỗi forward. Trong khoảng đó cả codes/scales và Ŵcùng tồn tại; khi thí nghiệm giữ thêm FP32 reference, tổng process còn lớn hơn. Layer-by-layer dequant không đồng nghĩa dequant toàn model cùng lúc, nhưng layer tạm vẫn cần được tính vào peak.

File save/load có serialization overhead và bản sao; optimizer/gradients không có ở quantized inference module của lab, nhưng đã có trong giai đoạn train FP32. Trong hệ thật còn allocator cache, fragmentation, attention scratch và framework objects. `tensor_bytes()` đếm state_dict trong implementation này, không đo RSS hoặc CUDA reserved.

Một fused kernel có thể giải mã theo tile để đổi trade-off. Đó là một implementation khác cần đo riêng. Không lấy tỷ lệ persistent payload của code minh họa rồi gán cho peak RAM/latency của một backend chưa chạy.

## 8. Calibration lineage và held-out data

Calibration có thể dùng train hoặc một tập riêng đại diện, nhưng không tune theo test rồi báo đó là kết quả giữ kín. Cần lưu model revision, preprocessing/feature order hoặc tokenizer/template, tập calibration và cách chọn row/range. Cùng indices mà giá trị dữ liệu đổi thì calibration cũng đổi.

Bundle lab hash cả values calibration, split/full data tensors, weights nguồn và payload; mean/std nằm trong state. Metadata được đối chiếu với expected identity khi load. Những hash này kiểm consistency với một manifest đã tin cậy, chưa chứng minh manifest được bên nào ký hoặc dữ liệu có được phép dùng.

Stress4×từ test là phép biến đổi định trước của cùng test. Không gộp nó với test như264quan sát độc lập, không tính CI giả dựa trên giả định hai tập độc lập. Muốn phát biểu về domain shift cần dữ liệu/source/time mới và protocol rõ hơn.

## 9. QAT và STE là cách chọn gradient xấp xỉ

Nếu `q̂(x)` là kết quả fake quant, biểu thức PyTorch `x + (q̂(x)-x).detach()` cho forward=q̂ và gradient theo x là1. Đây là identity STE; một recipe khác có thể chặn gradient ngoài clipping range. Hai recipe không tương đương và đều không là đạo hàm chính xác của round.

QAT còn cần optimizer, data, observer/range schedule, phần giữ precision cao và cuối cùng chuyển sang format/runtime đích. Một snippet gradient không đủ gọi là QAT training experiment. Bài38không so PTQ với một QAT checkpoint đã train; phần này là cầu nối để hiểu các recipe sau này.

Nếu fine-tune adapter với quantized base như bài35, cần phân biệt gradient qua input/layer với gradient cập nhật chính base. Quantized integer codes không tự trở thành tham số liên tục có thể tối ưu bằng Adam.

## 10. Những biến đổi tương đương chỉ tương đương trước quantization

Với D đường chéo dương, `(XD⁻¹)(WD)ᵀ=XWᵀ`. Có thể kiểm đại số bằng một ví dụ2channel. Sau rounding, `Q(XD⁻¹)` và `Q(WD)` có lỗi khác `Q(X)`/`Q(W)`, nên độ chính xác thay đổi dù phép float lý tưởng như nhau.

Điều này tạo trực giác cho các hướng dùng activation statistics. Abstract [SmoothQuant](https://arxiv.org/abs/2211.10438) mô tả chuyển khó khăn activation sang weights; abstract [AWQ](https://arxiv.org/abs/2306.00978) mô tả bảo vệ kênh weight theo activation. Bài không cài thuật toán chọn D hoặc tái lập kết quả hai nghiên cứu. [GPTQ](https://arxiv.org/abs/2210.17323) lại dùng thông tin xấp xỉ bậc hai; không gọi absmax rounding là GPTQ.

## 11. Quantized KV có sensitivity và axis riêng

Attention output là `softmax(QKᵀ/√dh)V`. Sai số K ảnh hưởng phân bố trọng số, rồi sai số V ảnh hưởng vector được trộn. Những phép tính tuyến tính trước softmax không đủ đo tác động sau softmax trong mọi input.

Diagnostic của lab dùng per-token/head group8cho cả K/V, không chọn axis theo từng loại tensor. Abstract [KIVI](https://arxiv.org/abs/2402.02750) cho thấy nghiên cứu lựa chọn axis khác cho keys/values; bài không cài phương pháp đó. Một runtime có thể giữ residual recent tokens ở precision cao, thêm metadata và chính sách chuyển nhóm; phải đọc recipe cụ thể.

Full autoregressive evaluation cần kiểm cache append, position/mask, nhiều layer/context length và chất lượng output như bài37. Một attention RMSE nhỏ không đủ xác nhận sinh chuỗi dài còn đạt yêu cầu. Cũng không suy dùng ít byte cache sẽ luôn giảm ITL vì có dequant/transfer cost.

## 12. Một quy trình nâng lab thành phép chọn triển khai

Bắt đầu bằng workload, model revision và yêu cầu metric/SLO. Khóa baseline, preprocessing và decoding. Kiểm format/packing bằng tensor nhỏ; thu calibration đại diện; so layer outputs, logits và metric công việc trên validation. Chỉ sau khi khóa cấu hình mới đánh giá test cuối theo protocol đã chọn.

Đo persistent/peak memory, load peak và latency/throughput trên chính runtime/kernel/device sẽ dùng. Có đường rollback cùng artifact gốc và identity để tái lập. Nếu output chất lượng không đạt, cân nhắc granularity, range, giữ layer nhạy precision cao hoặc QAT, thay vì chỉ tiếp tục giảm bit.

Lab hiện không có quyết định triển khai sản phẩm, quality gate production, GPU benchmark hoặc capacity load test. Các kết quả chỉ đủ xác nhận cơ chế PTQ/packing/calibration và một vài trade-off có thể quan sát trong mô hình tổng hợp nhỏ.
