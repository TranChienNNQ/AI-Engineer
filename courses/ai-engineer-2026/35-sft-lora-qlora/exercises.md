# Bài 35 — Bài tập

[Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán rồi mới chạy. GPU recipe có scope chưa chạy GPU; bài tập core đều làm được trên CPU.

## 1. Ba trục quyết định

Giải thích vì sao “chọn SFT hay LoRA?” chưa phải câu hỏi đúng. Lập ba cấu hình cùng SFT objective nhưng khác trainables/storage. Nêu khi nào prompt/rule/RAG vẫn là baseline nên thử cho hệ thống routing/tài liệu nội bộ.

## 2. Dữ liệu và leakage

Một ticket có ba paraphrase, cùng service và cùng answer. Nếu random split theo dòng, điều gì có thể sai? Nêu grouping/time/label/template metadata cần giữ. Chỉ ra giới hạn của split16/4/4 group trong fixture: vì sao chưa là unseen policy/event?

## 3. Shift và mask tính tay

Sequence=[BOS,USER,p1,p2,ASSISTANT,NET,EOS]. Viết input/label sau một lần shift và answer-only mask. Token đầu của answer nằm ở label index nào? Nếu prefix dài5 mà mask5 label đầu, mất supervision gì? Kiểm số supervised target của một record lab.

## 4. Gradient qua prompt và accumulation

Mask prompt loss có làm gradient prompt embedding luôn0 không? Giải thích đường phụ thuộc. Hai microbatch có sum loss2 và18, token counts2 và6: tính global token mean và mean-of-means. Viết code đối chiếu gradient accumulation với batch đầy đủ khi token count khác nhau.

## 5. Forward/backward LoRA

Dùng W₀/A/B/x/target trong bài chính, r1/alpha2. Tính Ax, base output, update, y, loss, gradA và gradB. W₀ frozen có ngăn gradient theo input không? Đối chiếu autograd.

## 6. Rank và parameter budget

Tính full/LoRA parameter cho matrix4096×4096,r8 và matrix2×3,r1. Với matrix d×d, tìm điều kiện r để adapter ít parameter hơn full matrix. Có thể suy overall model rank=r hoặc VRAM giảm cùng tỷ lệ không?

## 7. Initialization

So ba cách: A random/B0; A0/B random; A0/B0. Dự đoán initial update và factor nào nhận gradient đầu tiên. Chạy ca cả hai0 và giải thích vì sao optimizer step có thể không thay đổi nhánh. Alpha/r có tương đương LR trong mọi optimizer không?

## 8. NF4 và nibble packing

Hai code0 và2 được pack theo format bài thành byte bao nhiêu? Viết unpack kiểm lại. Với block absmax1, weight−.5 gần mức NF4 nào? NF4 có phải uniform INT4 không? Thử block toàn0 và block cuối có padding.

## 9. Frozen quantized base và autograd

Một frozen NF4Linear nằm giữa hai adapter. Nếu bọc forward nó trong no_grad, chuyện gì có thể xảy ra với adapter phía trước? Viết code kiểm grad theo input còn tồn tại và base không có trainable parameter. Khi B=0, quant branch giữ original W₀ hay W_hat?

## 10. Memory theo scope

Tính7B weights lý tưởng ởBF16/4-bit, theo byte và GiB. Tính scale overhead FP32 mỗi64 weight, rồi lý tưởng nested8-bit scale/second-level256. Liệt kê sáu thành phần chưa có trong con số weight payload. Vì sao LoRA resident state lớn hơn base vẫn có thể giảm training memory?

## 11. Target modules và kiểm trainables

Từ TinyLM, liệt kê năm adapter targets và tính1176 parameter khi r4. Giải thích tại sao không chỉ đưa một tên module từ model khác vào config rồi tin chắc đã gắn đúng. Phác sanity checks cho frozen state, optimizer param groups, first-step grads và tiny-batch fit.

## 12. Đọc kết quả thí nghiệm

Full/LoRA/packed NF4 đều24/24 task test; rule cũng24/24. Quant branch cóCE thấp nhất trong setup, nhưng old-text CE xấu nhất. Có thể kết luận QLoRA tốt hơn full SFT và không forgetting không? Nêu dữ kiện/giới hạn và thí nghiệm cần thêm.

## 13. Merge và re-quantization

Giải thích W₀+sBA so W_hat+sBA. Tại sao merge quant adapter vào original W₀ có thể không giữ output? Vì sao re-quantize merged weight cần eval lại? Trong core merged(), layer nào còn quantized? Nêu tolerance/metric cần kiểm ngoài forward parity.

## 14. Adapter checkpoint và nhầm base

Tạo adapter, save/load với base đúng rồi thay một base tensor .01 và thử load lại. Dự đoán kết quả. Checkpoint adapter cần metadata gì và có đủ để resume optimizer chính xác không? Hash có xác thực người phát hành không?

## 15. Thư viện và hardware

Chạy optional PEFT CPU demo/preflight nếu đã cài requirements-optional. Phân biệt bằng chứng: config tạo thành công, PEFT Linear train thành công, model load4bit thành công và GPU training có gradient/eval thành công. Tại sao cài được bitsandbytes wheel chưa đủ? Trên máy không CUDA, không yêu cầu giả lập kết quả GPU.

## 16. Thiết kế một vòng SFT có thể phát hành

Thiết kế pipeline cho trợ lý backend: mục tiêu/không tinh chỉnh gì, dataset/consent/split, template/mask, base/adapter/quantization, resource budget, checkpoint/eval/retention và release/rollback. Nêu ít nhất ba failure mode cần chặn trước khi deploy và kiểm khi PAD=EOS. Phân biệt smoke recipe2 update với pipeline được chấp nhận chất lượng.
