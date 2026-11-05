# Đề cương học AI Engineer

[Về mục lục](README.md)

Đối tượng: backend engineer; Deep Learning được học từ cơ chế nền tảng. Chưa có thời lượng học cố định nên lộ trình dùng tiêu chí năng lực thay cho cam kết số tuần. Đây là cách tổ chức kiến thức của bộ tài liệu, không phải một chuẩn nghề nghiệp duy nhất.

## Thứ tự và điều kiện học tiếp

| Phần | Các bài dự kiến | Sản phẩm / điều kiện học tiếp |
|---|---|---|
| 1. Cơ chế học và toán cần dùng | **Bài 1: mô hình, tham số, loss, huấn luyện**; bài 2: đạo hàm và gradient descent; bài 3: vector, ma trận và kích thước tensor; bài 4: xác suất, thống kê và bất định | Tự tính dự đoán, loss và một bước cập nhật; giải thích kích thước dữ liệu |
| 2. Dữ liệu và ML cơ bản | Bài 5: train/validation/test, leakage; bài 6: hồi quy và phân loại; bài 7: precision, recall, mất cân bằng; bài 8: cây, boosting và baseline | Bộ phân loại ticket có so sánh baseline và phân tích lỗi trên tập giữ riêng |
| 3. Deep Learning nền tảng | Bài 9: neuron, activation, mạng nhiều lớp; bài 10: chain rule và backpropagation; bài 11: PyTorch, autograd, DataLoader; bài 12: vòng train/eval, validation, checkpoint và theo dõi thí nghiệm; bài 13: optimizer, regularization, chẩn đoán | Tự huấn luyện, lưu/tải mạng nhỏ; giải thích đường cong học và lỗi thường gặp |
| 4. Kiến trúc và biểu diễn | Bài 14: CNN và ảnh; bài 15: dữ liệu chuỗi, ý tưởng RNN; bài 16: embedding; bài 17: attention; bài 18: Transformer | Xây bộ phân loại ảnh và attention nhỏ; giải thích luồng dữ liệu và masking |
| 5. LLM | Bài 19: tokenizer, mô hình ngôn ngữ; bài 20: tiền huấn luyện và tinh chỉnh; bài 21: decoding, context, inference; bài 22: API, đầu ra có cấu trúc và đánh giá | Ứng dụng LLM có kiểm tra đầu ra, bộ đánh giá và đo chi phí |
| 6. Dữ liệu riêng và truy xuất | Bài 23: tìm kiếm từ khóa và vector; bài 24: chia đoạn, embedding, reranking; bài 25: RAG và nguồn dẫn; bài 26: context engineering, phân quyền, prompt injection | Trợ lý tài liệu; đo riêng lỗi truy xuất và lỗi câu trả lời; thử tình huống thiếu nguồn |
| 7. Công cụ và agent | Bài 27: gọi công cụ và workflow; bài 28: vòng lặp agent, trạng thái và bộ nhớ; bài 29: tích hợp MCP; bài 30: đánh giá nhiều bước và giới hạn hành động | Trợ lý xử lý ticket trong môi trường thử, có lịch sử hành động và kiểm soát quyền |
| 8. Đưa vào vận hành | Bài 31: API phục vụ mô hình, hàng đợi, timeout; bài 32: tracing và theo dõi chất lượng; bài 33: phiên bản dữ liệu/mô hình, CI, rollback; bài 34: đo tải, độ trễ, chi phí | Sản phẩm triển khai được, xử lý lỗi và tái lập kết quả; kiến thức backend được dùng lại |
| 9. Nhánh nâng cao | A: SFT, LoRA/QLoRA, dữ liệu ưu tiên, RL; B: batching, KV cache, quantization, phục vụ phân tán; C: ảnh/âm thanh/văn bản; D: gợi ý, dự báo, bất thường | Chọn một nhánh theo mục tiêu; chứng minh cải tiến trên bài đo có kiểm soát |
| 10. Dự án tổng hợp | Xác định nhu cầu, baseline, dữ liệu, đánh giá, thử nghiệm, triển khai, báo cáo hạn chế | Người khác chạy lại được dự án và kiểm tra kết luận |

## Nội dung hiện có

- [Bài 1 — Mô hình học từ dữ liệu như thế nào?](01-mo-hinh-hoc-tu-du-lieu/README.md): đã biên soạn, kèm code và lời giải.
- [Bài 2 — Đạo hàm và gradient descent](02-dao-ham-gradient-descent/README.md): đã biên soạn; có ví dụ tính tay, code một/hai tham số, 8 bài tập, lời giải và 3 hình minh họa.
- [Bài 3 — Vector, ma trận và tensor](03-vector-ma-tran-tensor/README.md): đã biên soạn; có tính tay, code NumPy, 9 bài tập, lời giải và 4 hình minh họa.
- [Bài 4 — Xác suất, thống kê và bất định](04-xac-suat-thong-ke-bat-dinh/README.md): đã biên soạn; có tính tay, code NumPy, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 5 — Train/validation/test và rò rỉ dữ liệu](05-train-validation-test-ro-ri-du-lieu/README.md): đã biên soạn; có tính tay, code NumPy, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 6 — Hồi quy và phân loại](06-hoi-quy-phan-loai/README.md): đã biên soạn; có tính tay, code NumPy, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 7 — Precision, recall, confusion matrix và mất cân bằng dữ liệu](07-precision-recall-mat-can-bang/README.md): đã biên soạn; có tính tay, code NumPy, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 8 — Cây quyết định, boosting và baseline](08-cay-quyet-dinh-boosting-baseline/README.md): đã biên soạn; có tính tay, code NumPy thực sự fit mô hình, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 9 — Neuron, activation và mạng nhiều lớp](09-neuron-activation-mang-nhieu-lop/README.md): đã biên soạn; có forward NumPy, ví dụ tính tay, XOR đặt trọng số, thí nghiệm chỉ học head, 10 bài tập, lời giải và 5 hình minh họa; chưa backprop hidden.
- [Bài 10 — Chain rule, computational graph và backpropagation](10-chain-rule-computational-graph-backpropagation/README.md): đã biên soạn; có gradient qua nhánh và batch tính tay, backprop NumPy, gradient check, huấn luyện MLP end-to-end, 10 bài tập, lời giải và 5 hình minh họa.
- [Bài 11 — PyTorch, tensor, autograd và DataLoader](11-pytorch-tensor-autograd-dataloader/README.md): đã biên soạn, chạy PyTorch CPU; 10 bài tập, lời giải, 5 hình.
- [Bài 12 — Vòng train/eval, validation, checkpoint và theo dõi thí nghiệm](12-train-eval-validation-checkpoint/README.md): đã biên soạn, chạy PyTorch CPU; ví dụ tính tay, kiểm save/load và resume, 10 bài tập, lời giải, 5 hình.
- [Bài 13 — Optimizer, regularization và chẩn đoán quá trình học](13-optimizer-regularization-chan-doan/README.md): đã biên soạn, chạy PyTorch CPU; tính tay và kiểm SGD/momentum/Adam, L2/AdamW, clipping, bốn cấu hình regularization, 10 bài tập, lời giải và 5 hình.
- [Bài 14 — CNN và dữ liệu ảnh](14-cnn-du-lieu-anh/README.md): đã biên soạn, chạy CNN PyTorch CPU và đọc/ghi PNG; convolution/gradient tính tay, group split, preprocessing/inference, 10 bài tập, lời giải và 5 hình.
- [Bài 15 — Dữ liệu chuỗi và ý tưởng RNN](15-du-lieu-chuoi-rnn/README.md): đã biên soạn, chạy packed RNN PyTorch CPU; forward/BPTT tính tay, padding và window, baseline, 10 bài tập, lời giải và 5 hình.
- [Bài 16 — Embedding và biểu diễn](16-embedding-bieu-dien/README.md): đã biên soạn, chạy PyTorch CPU; lookup/gradient và geometry tính tay, vocabulary train-only, masked pooling, classifier học embedding, 10 bài tập, lời giải và 5 hình.
- [Bài 17 — Attention](17-attention/README.md): đã biên soạn chuyên sâu, chạy PyTorch CPU; QKV/gradient tính tay, mask/API/multi-head đối chiếu, recall model học attention, phụ lục toán/API, 14 bài tập, lời giải và 7 hình.
- [Bài 18 — Transformer](18-transformer/README.md): đã biên soạn chuyên sâu, chạy PyTorch CPU; block tính tay, pre/post-norm forward/backward đối chiếu, PE/mask/shift, encoder–decoder demo và decoder-only học sinh chuỗi, 16 bài tập, lời giải và 7 hình.
- [Bài 19 — Tokenizer và mô hình ngôn ngữ](19-tokenizer-mo-hinh-ngon-ngu/README.md): đã biên soạn, chạy CPU; ví dụ backend/Unicode, BPE tự xây, likelihood/PPL tính tay, Transformer LM trên văn bản, 16 bài tập, lời giải và 6 hình.
- [Bài 20 — Tiền huấn luyện và tinh chỉnh](20-tien-huan-luyen-tinh-chinh/README.md): đã biên soạn, chạy pretrain → SFT CPU thật, controls scratch/head-only, mask/gradient/LoRA tính tay, regression LM cũ; 16 bài tập, lời giải, phụ lục và 6 hình.
- [Bài 21 — Decoding, context và inference](21-decoding-context-inference/README.md): đã biên soạn, chạy CPU; sampling/beam tính tay, context/streaming, KV cache đúng offset đối chiếu full forward, nối checkpoint SFT bài 20; 16 bài tập, lời giải, phụ lục và 7 hình.
- [Bài 22 — API, đầu ra có cấu trúc và đánh giá](22-api-dau-ra-cau-truc-danh-gia/README.md): đã biên soạn, chạy FastAPI ASGI/HTTP localhost, JSON Schema/Pydantic strict, bounded retry/FakeClock và scorer 20 fixture; 16 bài tập, lời giải, phụ lục và 6 hình. Không gọi LLM thật.
- [Bài 23 — Tìm kiếm từ khóa và vector](23-tim-kiem-tu-khoa-vector/README.md): đã biên soạn, chạy NumPy CPU; TF-IDF/BM25/cosine tính tay, LSA 6 chiều, RRF, scope filter, save/load và đánh giá 18 query; 16 bài tập, lời giải, phụ lục và 7 hình. Chưa chạy neural embedding/ANN.
- [Bài 24 — Chunking, embedding và reranking](24-chunking-embedding-reranking/README.md): đã biên soạn, train bi-encoder/cross-encoder PyTorch CPU thật; evidence span theo nguồn, chunking ablation, contrastive/BCE, checkpoint chọn dev và bundle đối chiếu; 16 bài tập, lời giải, phụ lục và 7 hình. Chưa dùng model pretrained/ANN/RAG generation.
- [Bài 25 — RAG và nguồn dẫn](25-rag-nguon-dan/README.md): đã biên soạn, chạy retrieve/pack/extract/cite/validate/render CPU; snapshot theo ngày/quyền, annotation claim–evidence và 22 fixture; 16 bài tập, lời giải, phụ lục và 7 hình. Baseline extractive, chưa gọi LLM/NLI/API.
- [Bài 26 — Context engineering, phân quyền và prompt injection](26-context-engineering-phan-quyen-prompt-injection/README.md): đã biên soạn; chạy lọc quyền trước retrieval, memory lineage/cache revalidation, grant một lần và gateway CPU; 22 proposal fixture, 6 confidentiality check và residual riêng; 16 bài tập, lời giải, phụ lục và 7 hình. Chưa gọi LLM hoặc triển khai authentication thật.
- [Bài 27 — Gọi công cụ và workflow](27-goi-cong-cu-workflow/README.md): đã biên soạn; protocol/registry/correlation, state machine, retry/budget, review và write receipt/idempotency; 22 fault scenario, demo asyncio fan-out, 16 bài tập, lời giải, phụ lục và 6 hình. Chưa gọi LLM/API hoặc có durable workflow engine.
- [Bài 28 — Vòng lặp agent, trạng thái và bộ nhớ](28-vong-lap-agent-trang-thai-bo-nho/README.md): đã biên soạn; policy chọn action từ observation, stop/budget, latest SQLite checkpoint/events, CAS và read recovery; working summary, memory namespace/TTL/version/lineage/forget; 25 scenario, 16 bài tập, lời giải, phụ lục và 7 hình. Chưa dùng LLM, LangGraph hoặc distributed workflow engine.
- [Bài 29 — Tích hợp MCP](29-tich-hop-mcp/README.md): đã biên soạn; MCP SDK 2.3.0, protocol 2026-07-28/forced legacy, tools/resources/prompts, stdio child process và HTTP-ASGI; adapter alias/schema/catalog/source/budget, 33 kiểm tra, 16 bài tập, lời giải, phụ lục và 6 hình. Host Python cố định, chưa gọi LLM hoặc triển khai OAuth/remote server.
- Bài 30 trở đi: đề cương, chưa được biên soạn thành bài học. Tài liệu kỹ thuật và phiên bản sẽ được đối chiếu khi viết từng bài.

## Nguyên tắc tổ chức

Phần 1–3 cung cấp nền chung. Phần 4 nối nền với mô hình hiện đại; phần 5–8 đi theo hướng ứng dụng LLM và tận dụng kinh nghiệm backend. Có thể thực hành một API nhỏ sớm, nhưng vẫn cần quay lại cơ chế mô hình và đánh giá. Phần 9 là lựa chọn chuyên sâu; không cần hoàn thành mọi nhánh mới xây sản phẩm.

Mỗi phần cần bằng chứng từ bài làm hoặc sản phẩm, không chỉ việc đọc xong. Việc biên soạn tài liệu và tiến bộ của người học là hai trạng thái riêng. [Danh mục nguồn và phạm vi đã đọc](SOURCES.md).
