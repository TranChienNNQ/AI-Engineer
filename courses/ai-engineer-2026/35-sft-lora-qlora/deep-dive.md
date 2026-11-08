# Bài 35 — Đào sâu SFT, adapter và quantized base

[Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Objective, parameterization và numerical representation

SFT định nghĩa dữ liệu/target/loss. LoRA định nghĩa effective weight W₀+sBA với một tập trainables nhỏ. Quantization định nghĩa cách biểu diễn/lưu/tính toán base. Có thể thay một trục mà giữ hai trục khác: cùng SFT loss nhưng full/LoRA; cùng LoRA nhưng FP32/NF4 base; cùng base/adapter nhưng thay loss mask hoặc dataset.

Các so sánh cần chỉ ra những trục nào thay đổi. CPU experiment giữ cùng FP32 base ancestor và dataset nhưng quant branch dùng base approximation; LR của full và adapter khác nhau. Vì vậy không gọi đây là controlled ablation duy nhất về quantization hoặc ranking thuật toán.

## 2. Shift đúng một lần

Core TinyLM nhận inputs và trả logits, không tự shift labels trong forward. `example()` tự tạo x=seq[:-1],y=seq[1:] rồi mask prefix targets. Với m prefix token, label index m−1 mới là token answer đầu; mask y[:m−1].

GPU recipe giữ input_ids là full sequence và labels cùng vị trí, mask labels[:m]. Khi tính CE thủ công, dùng logits[:,:−1] so labels[:,1:]. Nếu chuyển sang AutoModelForCausalLM(labels=...) thì cần đọc contract internal shift thay vì tiếp tục shift hai lần. Hai cách layout có thể đúng nếu chỉ làm đúng một lần.

EOS/end-of-turn cần có supervision nếu là điều kiện dừng mong muốn. Tokenizer thật có thể có end-of-turn khác EOS chung; template quyết định sequence. Không suy EOS ID của toy tokenizer áp dụng cho HF model.

## 3. Mask loss không cắt context

Logits tại prompt positions có gradient trực tiếp0 khi labels IGNORE. Nhưng answer logits phụ thuộc attention qua prompt keys/values; gradient có thể trở về prompt embedding. Nếu embedding frozen thì không lưu parameter gradient cho nó, nhưng activations vẫn có thể cần để học adapter phía sau.

Một frozen dense layer giữa hai adapter vẫn phải truyền grad theo input. Bọc toàn model/base trong no_grad có thể làm adapter trước bị mất đường học. Với checkpointing và frozen embeddings, cần chuẩn bị graph/input-gradient theo framework; `prepare_model_for_kbit_training` trong recipe là bước tích hợp, không thay việc kiểm gradients thực.

## 4. Token weighting, padding và accumulation

Giả sử hai microbatch có token counts2/6. Backward `mean1/2 + mean2/2` cho mỗi nhóm ngang trọng số; backward `sum1/8 + sum2/8` đúng global token average. Sequence-average cũng có thể là objective có chủ đích, nhưng phải nói rõ thay vì vô tình thay objective theo cách chia batch.

Ở distributed setting, DDP reduction, world size, trainer loss normalization và số token giữa ranks có thể làm công thức cần điều chỉnh. Bài chỉ chứng minh single-process accumulation với model không dropout, không batch norm. Không lấy test ấy làm bằng chứng distributed equivalence.

Mask padding theo position/attention khi PAD=EOS; mask bằng ID sẽ xóa cả EOS thật. Truncation/packing có thể làm mất target hoặc nối hai conversation qua attention. Core từ chối oversize, chưa implement packing, multi-turn assistant masks hoặc long-context efficiency.

## 5. Rank của update và các cách tham số hóa tương đương

rank(BA)≤min(rankA,rankB)≤r. Tuy vậy W₀+BA có thể vẫn full rank. Với một ma trận r×r khả nghịch R, `(BR)(R⁻¹A)=BA`; A/B không có một biểu diễn duy nhất. Vì vậy không diễn giải từng chiều adapter như một “kỹ năng” độc lập đã được chứng minh.

Hiệu quả lưu trữ đạt khi r(n+m) nhỏ hơn mn. Với matrix vuông d×d, adapter r(2d); r≥d/2 thì không còn giảm số parameter cho riêng matrix đó. Số parameter không trực tiếp cho tốc độ: có thêm hai matmul và scale trong unmerged forward, còn dense base vẫn chạy.

Rank tăng mở rộng tập update có thể biểu diễn nhưng không đảm bảo validation tốt hơn. Data quality, target coverage, optimizer, alpha/dropout và compute budget còn ảnh hưởng. Bài dùng r4/alpha8, chưa rank sweep hoặc rsLoRA/DoRA/LoftQ/PiSSA.

## 6. Initialization và scaling

B=0,A random làm effective update0 nhưng cho B một hướng gradient phụ thuộc Ax. Nếu A cũng0 thì cả hai có thể không thoát khỏi zero update. Sau một SGD step đầu, A có thể giữ nguyên trong khi B đổi; các bước sau A mới nhận gradient khác0.

Alpha/r scale cả forward và gradient. Thay alpha không luôn tương đương thay LR, vì update của hai factors, optimizer moments, clipping và weight decay khác nhau. AdamW decay trên A/B còn tác động effective BA khác decay trực tiếp trên dense ΔW. Lab weight_decay0 để giảm số yếu tố cần so.

LoRA thường dùng dropout trên adapter input trong một số setup; merge parity chỉ có nghĩa với evaluation/dropout tắt hoặc điều kiện tương ứng. Code core/dropout0 giúp so deterministic logits.

## 7. Quantized base không phải original base

Quantize tạo W_hat=D(Q(W₀)). Khi B=0, quantized LoRA bắt đầu từ W_hat, không phải W₀. Adapter có thể học task và một phần ảnh hưởng quantization, nhưng không có bảo đảm sẽ khôi phục mọi lỗi lượng tử hóa.

NF4 codebook của bài lấy Appendix E, nhưng block/padding/packing/dequantize là local teaching format. Không dùng object ấy với bitsandbytes loader hoặc gọi nó là checkpoint tương thích NF4 của một runtime khác. Nó không có DQ, paging hoặc kernel fused; dequantize full matrix tạo temporary dense FP32 tensor.

Scale absmax có thể bị outlier chi phối: nhiều weight nhỏ bị dồn quanh0 và mất precision. Block nhỏ giảm phạm vi mỗi block nhưng tăng số scale/metadata. NF4 levels không cách đều, nên không mô tả nó như integer×một bước scale đều. Quantizer không có học parameter trong bài.

## 8. DQ, compute dtype và memory budget

Một FP32 scale/64 weight thêm.5 bit/weight. Second-level8-bit scale + one FP32/256 scale cho overhead8/64+32/(64×256); đây là arithmetic bỏ qua các metadata/alignment khác. Không suy mọi checkpoint đúng4.126953125 bit/weight hoặc thực tế vừa một GPU từ phép tính ấy.

Storage dtype, dequantization dtype, matmul dtype, adapter parameter/gradient dtype và optimizer-state dtype có thể khác nhau. BF16 compute không có nghĩa optimizer moments BF16; optimizer cụ thể có thể tự cast/upcast hoặc giữ FP32 states. Không gộp mọi dtype vào nhãn “4-bit training”.

Activation memory có thể tăng mạnh theo batch/sequence/attention implementation. Gradient checkpointing đổi memory lấy compute, paged optimizer xử lý một số memory pressure bằng paging. Không cơ chế nào tự bảo đảm không OOM hoặc không giảm throughput.

Core resident_state_bytes đếm state_dict tensor bytes, không optimizer/grad/activation/temporary/allocator. Global NF4 codebook64 byte không nằm trong từng model state. File .pt còn zip/metadata/format overhead, nên file size cũng khác resident tensor bytes.

## 9. Adapter identity và format

Adapter checkpoint core gồm A/B và metadata: base tensor hash, tokenizer fingerprint, template version, model config, dataset hash, kind/rank/alpha/targets, quantization format/codebook hash và frozen-state hash. Đây là inference artifact; loader không có optimizer/RNG để resume exact training.

Không có signature hoặc actor identity; hashes bảo vệ nhất quán so với metadata/base đã tin cậy, không chứng minh provenance hay sample quality. Codebook hash được giữ riêng vì constant ngoài state_dict cũng ảnh hưởng dequantization. Train/eval source/environment manifest rộng hơn có thể dùng cách bài33.

Không trộn adapter train trên baseA với baseB chỉ vì shape/module names trùng. Với HF model, base path có thể đổi nội dung; nên pin revision/artifact identity và tokenizer/template/quant settings. Core loader bảo thủ từ chối dataset/template mismatch để minh họa release contract; không phải mọi inference loader ngoài đời đều cần đúng dataset hash.

## 10. Merge, dequantize và re-quantize

Với eval LoRA FP32, merge W_eff=W₀+sBA. Thứ tự matmul/floating-point rounding có thể làm logits khác một lượng nhỏ. Core kiểm bằng tolerance, không yêu cầu bitwise equality cho merged logits. Adapter save/load cùng implementation lại được kiểm exact.

Với quantized base, W_hat+sBA mới là effective dense weight cần so. Merge vào original W₀ cho một model khác. Nếu sau đó quantize lại W_eff thì có thêm sai số Q₂; cần task/retention/e2e validation, không chỉ check matrix shape.

Core merged() chỉ chuyển các adapter target sang dense; k/o quantized không có adapter vẫn là NF4Linear. Không có export HF architecture, tokenizer bundles hoặc runtime serving adapter routing. Không nói merge chắc chắn giảm latency trước khi đo trên backend thật.

## 11. Retention, overfit và tiêu chí chấp nhận

Frozen base tensor không đổi nhưng effective logits vẫn đổi. Toy LoRA đạt task exact100% nhưng old-text CE tăng từ≈.933 lên≈9.269. Disable adapter có thể lấy lại FP32 base nếu mọi state liên quan còn nguyên; quantized base vẫn có approximation. Một adapter nhỏ chưa bảo đảm ít forgetting.

Đánh giá cần task metric, format, calibration/abstention khi phù hợp, safety/permissions, retention và latency/cost. Task routing fixture có rule100%, còn validation/test chung events; chưa chứng minh lợi ích tinh chỉnh hơn rule/prompt/RAG/LLM nền ngoài đời. Không dùng100% toy exact làm bằng chứng deploy-ready.

Chọn hyperparameters/checkpoint trên validation, giữ test để chấm sau lựa chọn và lưu kết quả thất bại. Nếu thay template/tokenizer/quantization/merge pipeline, chạy lại contract/eval trên bundle mới. Liên kết với CI/rollback bài33, đo tải/chi phí bài34.

## 12. Recipe GPU và giới hạn kiểm chứng

PEFT CPU demo dùng library thật trên Linear3→2, một SGD step, save/load safetensors/config và merge. Nó kiểm API/gradient/control flow nhỏ, chưa pretrained LM quality hoặc GPU kernels. BitsAndBytesConfig object chỉ là config.

gpu_recipe preflight đọc installed packages/CUDA/BF16. Trên máy biên soạn CUDA=False, train path dừng trước khi load model/write output. Cài được macOS wheel bitsandbytes không được diễn giải là đã chạy NF4 CUDA training. Các backend khác có support riêng; recipe chỉ chủ đích hỗ trợ một CUDA device/BF16 để giảm số tổ hợp chưa kiểm.

Train recipe dùng base local/trust_remote_code=False/local_files_only, hash base files, template prefix check, labels đúng vị trí và pad-by-length, load NF4+DQ, prepare checkpointing, LoRA all-linear và AdamW chuẩn. Chỉ smoke2 update/8 row, không validation selection hoặc claim chất lượng; GPU path chưa chạy trong phiên này. Nó không phải full pipeline để dùng production nguyên trạng.

Trên GPU thực cần xác minh load4bit thật, trainable names, gradient finite, frozen-state integrity, target-token coverage, memory peak, artifact reload và generation/eval bằng template thật. Preflight pass chưa chứng minh model vừa VRAM. Nếu lỗi template/kernels/hardware, giữ failure evidence rồi sửa đúng lớp; không gọi CPU imitation là QLoRA GPU thành công.
