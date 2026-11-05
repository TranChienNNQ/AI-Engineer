# AI Engineer 2026 — Từ nền tảng đến hệ thống ứng dụng

Phiên bản: 0.29 · Cập nhật: 05/11/2026 — bổ sung bài 29 về tích hợp MCP.

Bộ tài liệu tiếng Việt dành cho người có nền lập trình backend, đã nghe các khái niệm Deep Learning nhưng chưa học sâu. Mục tiêu là xây, đánh giá và vận hành ứng dụng AI, đồng thời hiểu cơ chế mô hình đủ để tìm nguyên nhân lỗi và lựa chọn kỹ thuật phù hợp.

## Bắt đầu đọc

1. Xem [đề cương và thứ tự học](ROADMAP.md).
2. Đọc [bài 1 — Mô hình học từ dữ liệu như thế nào?](01-mo-hinh-hoc-tu-du-lieu/README.md).
3. Làm [bài tập](01-mo-hinh-hoc-tu-du-lieu/exercises.md), rồi đối chiếu [lời giải](01-mo-hinh-hoc-tu-du-lieu/solutions.md).
4. Xem [danh mục nguồn](SOURCES.md) khi cần đọc sâu hoặc kiểm tra phát biểu.

## Các bài đã biên soạn

| Bài | Nội dung | Tự thực hành |
|---|---|---|
| [1. Mô hình học từ dữ liệu](01-mo-hinh-hoc-tu-du-lieu/README.md) | Tham số, dự đoán, MSE, grid search | [Bài tập](01-mo-hinh-hoc-tu-du-lieu/exercises.md) · [Lời giải](01-mo-hinh-hoc-tu-du-lieu/solutions.md) |
| [2. Đạo hàm và gradient descent](02-dao-ham-gradient-descent/README.md) | Độ dốc, đạo hàm riêng, gradient, learning rate; 3 hình minh họa | [Bài tập](02-dao-ham-gradient-descent/exercises.md) · [Lời giải](02-dao-ham-gradient-descent/solutions.md) |
| [3. Vector, ma trận và tensor](03-vector-ma-tran-tensor/README.md) | Shape, phép nhân, broadcasting, gradient vector hóa, trục tensor; 4 hình minh họa | [Bài tập](03-vector-ma-tran-tensor/exercises.md) · [Lời giải](03-vector-ma-tran-tensor/solutions.md) |
| [4. Xác suất, thống kê và bất định](04-xac-suat-thong-ke-bat-dinh/README.md) | Phân bố, Bayes, lấy mẫu, khoảng tin cậy, calibration và entropy; 5 hình minh họa | [Bài tập](04-xac-suat-thong-ke-bat-dinh/exercises.md) · [Lời giải](04-xac-suat-thong-ke-bat-dinh/solutions.md) |
| [5. Train/validation/test và rò rỉ dữ liệu](05-train-validation-test-ro-ri-du-lieu/README.md) | Chọn mô hình, tiền xử lý, nhóm, thời gian và cross-validation; 5 hình minh họa | [Bài tập](05-train-validation-test-ro-ri-du-lieu/exercises.md) · [Lời giải](05-train-validation-test-ro-ri-du-lieu/solutions.md) |
| [6. Hồi quy và phân loại](06-hoi-quy-phan-loai/README.md) | Linear/logistic regression, loss, L2, ngưỡng, softmax; 5 hình minh họa | [Bài tập](06-hoi-quy-phan-loai/exercises.md) · [Lời giải](06-hoi-quy-phan-loai/solutions.md) |
| [7. Precision, recall và mất cân bằng](07-precision-recall-mat-can-bang/README.md) | Confusion matrix, F1, ngưỡng/chi phí, PR/ROC/AP, tỷ lệ lớp, macro/micro và class weight; 5 hình minh họa | [Bài tập](07-precision-recall-mat-can-bang/exercises.md) · [Lời giải](07-precision-recall-mat-can-bang/solutions.md) |
| [8. Cây quyết định, boosting và baseline](08-cay-quyet-dinh-boosting-baseline/README.md) | Split Gini/MSE, lá, greedy, hai vòng boosting tính tay, so sánh mô hình; 5 hình minh họa | [Bài tập](08-cay-quyet-dinh-boosting-baseline/exercises.md) · [Lời giải](08-cay-quyet-dinh-boosting-baseline/solutions.md) |
| [9. Neuron, activation và mạng nhiều lớp](09-neuron-activation-mang-nhieu-lop/README.md) | Neuron, ReLU/sigmoid/tanh/GELU, forward và shape, affine collapse, XOR, học head cố định; 5 hình minh họa | [Bài tập](09-neuron-activation-mang-nhieu-lop/exercises.md) · [Lời giải](09-neuron-activation-mang-nhieu-lop/solutions.md) |
| [10. Chain rule, computational graph và backpropagation](10-chain-rule-computational-graph-backpropagation/README.md) | Cộng gradient qua nhánh, backward ma trận, tính tay, gradient check, MLP end-to-end; 5 hình minh họa | [Bài tập](10-chain-rule-computational-graph-backpropagation/exercises.md) · [Lời giải](10-chain-rule-computational-graph-backpropagation/solutions.md) |
| [11. PyTorch, tensor, autograd và DataLoader](11-pytorch-tensor-autograd-dataloader/README.md) | Tensor, gradient đối chiếu bài 10, Dataset/batch, optimizer; 5 hình minh họa | [Bài tập](11-pytorch-tensor-autograd-dataloader/exercises.md) · [Lời giải](11-pytorch-tensor-autograd-dataloader/solutions.md) |
| [12. Vòng train/eval, validation và checkpoint](12-train-eval-validation-checkpoint/README.md) | Chọn best, early stopping, resume đủ trạng thái, log thí nghiệm; 5 hình minh họa | [Bài tập](12-train-eval-validation-checkpoint/exercises.md) · [Lời giải](12-train-eval-validation-checkpoint/solutions.md) |
| [13. Optimizer, regularization và chẩn đoán](13-optimizer-regularization-chan-doan/README.md) | SGD/momentum/Adam, L2/AdamW, Dropout, clipping và điều tra lỗi học; 5 hình minh họa | [Bài tập](13-optimizer-regularization-chan-doan/exercises.md) · [Lời giải](13-optimizer-regularization-chan-doan/solutions.md) |
| [14. CNN và dữ liệu ảnh](14-cnn-du-lieu-anh/README.md) | Pixel/channel, convolution, pooling, shape/receptive field, PNG Dataset và CNN; 5 hình minh họa | [Bài tập](14-cnn-du-lieu-anh/exercises.md) · [Lời giải](14-cnn-du-lieu-anh/solutions.md) |
| [15. Dữ liệu chuỗi và ý tưởng RNN](15-du-lieu-chuoi-rnn/README.md) | Hidden state/BPTT, padding/lengths, packed RNN, window và baseline; 5 hình minh họa | [Bài tập](15-du-lieu-chuoi-rnn/exercises.md) · [Lời giải](15-du-lieu-chuoi-rnn/solutions.md) |
| [16. Embedding và biểu diễn](16-embedding-bieu-dien/README.md) | One-hot/lookup, gradient, masked mean, cosine, vocabulary/OOV và task-trained embedding; 5 hình minh họa | [Bài tập](16-embedding-bieu-dien/exercises.md) · [Lời giải](16-embedding-bieu-dien/solutions.md) |
| [17. Attention](17-attention/README.md) | QKV, softmax/scaling, gradient, mask, self/cross, multi-head, phụ lục API; 7 hình và 14 bài tập | [Bài tập](17-attention/exercises.md) · [Lời giải](17-attention/solutions.md) |
| [18. Transformer](18-transformer/README.md) | Vị trí, residual, LayerNorm, FFN, encoder/decoder, shift/mask và sinh chuỗi; 7 hình, 16 bài tập, phụ lục | [Bài tập](18-transformer/exercises.md) · [Lời giải](18-transformer/solutions.md) |
| [19. Tokenizer và mô hình ngôn ngữ](19-tokenizer-mo-hinh-ngon-ngu/README.md) | Unicode/byte, BPE, special token, next-token LM, CE/PPL, ví dụ backend; 6 hình, 16 bài tập và phụ lục | [Bài tập](19-tokenizer-mo-hinh-ngon-ngu/exercises.md) · [Lời giải](19-tokenizer-mo-hinh-ngon-ngu/solutions.md) |
| [20. Tiền huấn luyện và tinh chỉnh](20-tien-huan-luyen-tinh-chinh/README.md) | Pretraining/CPT/SFT, answer mask, full/head-only, LoRA tính tay, transfer/forgetting; 6 hình, 16 bài tập và phụ lục | [Bài tập](20-tien-huan-luyen-tinh-chinh/exercises.md) · [Lời giải](20-tien-huan-luyen-tinh-chinh/solutions.md) |
| [21. Decoding, context và inference](21-decoding-context-inference/README.md) | Greedy/temperature/top-k/top-p/beam, budget, KV cache/offset mask, streaming và telemetry; 7 hình, 16 bài tập, phụ lục | [Bài tập](21-decoding-context-inference/exercises.md) · [Lời giải](21-decoding-context-inference/solutions.md) |
| [22. API, đầu ra có cấu trúc và đánh giá](22-api-dau-ra-cau-truc-danh-gia/README.md) | FastAPI, JSON Schema, validation/semantic, retry/deadline, 20 fault fixtures và eval; 6 hình, 16 bài tập, phụ lục | [Bài tập](22-api-dau-ra-cau-truc-danh-gia/exercises.md) · [Lời giải](22-api-dau-ra-cau-truc-danh-gia/solutions.md) |
| [23. Tìm kiếm từ khóa và vector](23-tim-kiem-tu-khoa-vector/README.md) | Analyzer/inverted index, TF-IDF/BM25, cosine/LSA, RRF, filter và retrieval eval; 7 hình, 16 bài tập, phụ lục | [Bài tập](23-tim-kiem-tu-khoa-vector/exercises.md) · [Lời giải](23-tim-kiem-tu-khoa-vector/solutions.md) |
| [24. Chunking, embedding và reranking](24-chunking-embedding-reranking/README.md) | Offset/overlap/budget, bi-encoder học contrastive, Transformer cross-encoder, evidence eval và bundle; 7 hình, 16 bài tập, phụ lục | [Bài tập](24-chunking-embedding-reranking/exercises.md) · [Lời giải](24-chunking-embedding-reranking/solutions.md) |
| [25. RAG và nguồn dẫn](25-rag-nguon-dan/README.md) | Evidence/claim/provenance, context budget, phiên bản, quote validation, multi-source và eval; 7 hình, 16 bài tập, phụ lục | [Bài tập](25-rag-nguon-dan/exercises.md) · [Lời giải](25-rag-nguon-dan/solutions.md) |
| [26. Context engineering, phân quyền và prompt injection](26-context-engineering-phan-quyen-prompt-injection/README.md) | Budget/authority, ACL, memory lineage, cache/revoke, grant/gateway và adversarial fixtures; 7 hình, 16 bài tập, phụ lục | [Bài tập](26-context-engineering-phan-quyen-prompt-injection/exercises.md) · [Lời giải](26-context-engineering-phan-quyen-prompt-injection/solutions.md) |
| [27. Gọi công cụ và workflow](27-goi-cong-cu-workflow/README.md) | Tool contract/correlation, chaining/routing/fan-out, retry/budget, review, idempotency và UNKNOWN; 6 hình, 16 bài tập, phụ lục | [Bài tập](27-goi-cong-cu-workflow/exercises.md) · [Lời giải](27-goi-cong-cu-workflow/solutions.md) |
| [28. Vòng lặp agent, trạng thái và bộ nhớ](28-vong-lap-agent-trang-thai-bo-nho/README.md) | Feedback loop, stop/budget, SQLite checkpoint/CAS/recovery, working summary và memory namespace/TTL/lineage; 7 hình, 16 bài tập, phụ lục | [Bài tập](28-vong-lap-agent-trang-thai-bo-nho/exercises.md) · [Lời giải](28-vong-lap-agent-trang-thai-bo-nho/solutions.md) |

| [29. Tích hợp MCP](29-tich-hop-mcp/README.md) | Host/client/server, tools/resources/prompts, JSON-RPC, stdio/HTTP-ASGI, version và policy adapter; 6 hình, 16 bài tập, phụ lục | [Bài tập](29-tich-hop-mcp/exercises.md) · [Lời giải](29-tich-hop-mcp/solutions.md) |

## Trạng thái thực tế

| Thành phần | Trạng thái |
|---|---|
| Đề cương từ nền tảng đến nâng cao | Đã lập; có thể điều chỉnh theo bài thực hành |
| Bài 1–29, bài tập và lời giải | Đã biên soạn; code đã chạy kiểm tra |
| Hình minh họa bài 2–29 | Đã tạo từ công thức/dữ liệu minh họa; có PNG, SVG và mã tái tạo |
| Bài 30 trở đi | Mới là đề cương, chưa có nội dung bài hoàn chỉnh |
| Mức hiểu của độc giả | Không được suy ra từ trạng thái tài liệu |

## Cách sử dụng

Đọc theo thứ tự khi gặp kiến thức mới. Mỗi bài có mục tiêu, giải thích, ví dụ, thực hành và tiêu chí tự đánh giá. Không cần trò chuyện với agent để xem lời giải. Dự đoán kết quả trước khi chạy, sau đó giải thích sự khác biệt giữa dự đoán và kết quả thực tế.

Markdown là bản nguồn. Có thể đọc trong trình xem hỗ trợ Markdown; công thức dùng cú pháp toán LaTeX, kèm diễn giải bằng lời. Code thực hành bài 1–2 chỉ cần Python 3 và thư viện chuẩn; bài 3–10 cần thêm NumPy; bài 11–21 cần PyTorch và NumPy; riêng bài 14 thêm Pillow để đọc/ghi ảnh. Bài 22 dùng FastAPI, Pydantic, jsonschema, httpx và Uvicorn; không cần PyTorch. Bài 23 chỉ cần NumPy cho tìm kiếm, không cần PyTorch; vector dày dùng LSA từ corpus, chưa dùng model embedding pretrained. Bài 24 dùng PyTorch và NumPy để học embedding/reranker nhỏ từ đầu. Bài 25 dùng jsonschema cho pipeline RAG với baseline trích xuất và fixture citation; không gọi LLM. Bài 26 dùng jsonschema cho context builder, policy và gateway trong bộ nhớ; proposal được lập trình sẵn, chưa là benchmark LLM. Bài 27 dùng jsonschema cho workflow ticket, fault/receipt trong RAM và demo asyncio đọc song song; chưa gọi model thật. Bài 28 dùng jsonschema và sqlite3 cho agent policy mô phỏng, checkpoint/memory và resume qua process; không gọi LLM. Bài 29 dùng MCP Python SDK 2.3.0, Pydantic và jsonschema cho server/client thật qua stdio và HTTP-ASGI trong process; host Python được lập trình sẵn, không gọi LLM. Không cần GPU hay tài khoản dịch vụ AI. Phiên bản môi trường đã kiểm tra được ghi trong từng bài. Hình được lưu sẵn; chỉ khi muốn tái tạo hình mới cần thư viện vẽ theo hướng dẫn riêng.

## Chia sẻ và cập nhật

Có thể chép nguyên thư mục này để chia sẻ; giữ cấu trúc để các liên kết tương đối và mã thực hành còn hoạt động. Bộ tài liệu không chứa hồ sơ tiến bộ cá nhân. Các nguồn ngoài giữ điều kiện sử dụng của tác giả tương ứng; đây là bài biên soạn với ví dụ riêng, không phải bản sao toàn văn các nguồn.

Khi thêm bài: cập nhật mục lục, nguồn và kết quả kiểm tra. Mốc “2026” xác định bối cảnh lộ trình, không có nghĩa mọi kỹ thuật trong đó mới xuất hiện năm 2026. Các phần về API, thư viện và mô hình cụ thể phải được kiểm tra lại khi viết. Không có cơ chế tự cập nhật nền.
