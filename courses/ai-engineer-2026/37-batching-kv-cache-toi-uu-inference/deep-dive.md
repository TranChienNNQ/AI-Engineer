# Phụ lục bài 37 — Từ tensor đúng đến inference có thể đo

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Các suy luận và ví dụ số dưới đây do bài học biên soạn, cần đọc cùng phạm vi lab CPU. Kiến trúc attention/caching đối chiếu với [Transformers cache explanation](https://huggingface.co/docs/transformers/cache_explanation); quy ước toán và code cụ thể nằm trong bài chính.

## 1. Vì sao cache phải được xây theo từng layer?

Ở layer1, K/V được chiếu từ representation của token đầu vào layer1. Ở layer2, K/V được chiếu từ representation đã qua layer1, có chứa thông tin prefix. Hai bộ tensor thường khác nhau dù cùng token/vị trí. Cache một bộ ở đầu mạng rồi dùng mọi layer sẽ làm đổi hàm model.

Causal attention bảo đảm state token cũ không nhìn token tương lai. LayerNorm theo hidden dimension, FFN theo token và residual cũng không thêm phụ thuộc tương lai. Nhờ vậy có thể suy luận bằng quy nạp qua các layer: state cũ và K/V cũ vẫn đúng khi nối suffix. Nếu thay sang bidirectional attention hoặc phép chuẩn hóa trộn các vị trí theo cách phụ thuộc suffix, cần xem lại lập luận.

## 2. Ba loại “độ dài” trong cùng một request

**Số token đã phát ra** có thể lớn hơn số token output đã đưa lại vào model một đơn vị. Khi phát token thứN và dừng, token đó chưa cần K/V để dự đoán tokenN+1. Vì thế cache dàiP+N−1 trong vòng sinh của lab.

**Độ dài logic cache** là số vị trí thực đã ghi cho từng hàng. Nó loại PAD và quyết định position của lần append kế tiếp. **Capacity vật lý** là số slot đã cấp sẵn; `capacity−length` chưa là trạng thái hợp lệ. Model có thể đọc slice buffer lớn, nhưng mask phải loại mọi ô chưa hợp lệ.

Một buffer T=64 cho request dài3 không có nghĩa request có61token ngữ cảnh bằng0. Mask sai có thể làm các value0 ảnh hưởng softmax, vì key0 vẫn nhận score và chia bớt attention mass. Không thể chữa bằng cách chỉ đặt value PAD thành0.

## 3. Mask đúng có thể nhìn khác một tam giác ở góc tensor

Một chunk mới dàic sau prefixp có query position p,…,p+c−1. Keys chứa cả past lẫn chunk mới. Causal mask cần cho queryj nhìn0…p+j. Với p=3,c=2, ma trận là hai hàng11110 và11111.

Trong ragged batch, p là vector, không là một scalar lấy từ width lớn nhất. Kiểm cache/full bằng logits ở **mọi vị trí mới** sẽ bắt lỗi tốt hơn chỉ so một token argmax. Hai vector logits khác nhau vẫn có thể cùng argmax, nên output parity chưa đủ để kiểm một implementation cache.

Với SDPA đã đọc, boolean True là allowed; non-square `is_causal=True` căn phía trên trái. Code dùng `attn_mask` tường minh và `is_causal=False` để tránh phụ thuộc giả định vị trí. [PyTorch2.15 SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).

## 4. Chunked prefill khác sliding window

Chunked prefill giữ nguyên token và chia việc xử lý prefix thành các lần append. Nếu mask/position/cache đúng, logits cho token trong chunk sau tương đương chạy toàn prefix trong cùng model, trong tolerance số học.

Sliding-window attention thay tập key mà một query được nhìn theo kiến trúc. Tùy model/layer, cần giữ trạng thái và quy ước vị trí phù hợp; không tự cắt K/V đầu của một full-attention model rồi khẳng định output không đổi. Context truncation ở tầng ứng dụng lại thay input, cũng phải đánh giá nội dung bị mất.

Lab kiểm chunk đầu khác độ dài giữa hai hàng, sau đó append chunk tiếp; nó chưa kiểm một scheduler xen kẽ prefill và decode của nhiều phiên đang đến online.

## 5. GQA giảm persistent KV, chưa nói toàn bộ attention rẻ cùng tỷ lệ

Bốn query head và hai KV head nghĩa là mỗi nhóm hai query head dùng chung K/V. Persistent buffer chỉ cần hai KV head. Trong implementation sư phạm, `repeat_interleave` tạo tensor tạm bốn head để dùng phép nhân/SDPA đơn giản, nên footprint tạm không giảm cùng tỷ lệ.

Kernel phù hợp có thể dùng mapping head mà không lặp dữ liệu theo cách đó; lab không có kernel như vậy. Ba cấu hình Hkv=1/2/4 tạo model khác nhau từ random initialization. So parity cache/full **trong từng model** không chứng minh chất lượng giữa các model giống nhau. Nguồn [GQA](https://arxiv.org/abs/2305.13245) ở bài này chỉ đọc abstract, dùng để đối chiếu tên và cấu trúc, không tái lập uptraining.

## 6. Memory budget phải giữ chỗ cho token tương lai

Payload KV cho full-attention tăng tuyến tính theo tổng token đang giữ; request nhận vào chưa dùng hết số output tối đa. Nếu admission chỉ kiểm memory hiện tại mà nhận quá nhiều request có output budget dài, các bước sau có thể hết chỗ.

Một cách bảo thủ là dự trù prompt+output budget; đổi lại nhận ít request hơn. Cách linh hoạt hơn có paging, scheduling/preemption và policy khi thiếu block, nhưng cần quản lý deadline và công bằng. Dừng, recompute hay offload đều có chi phí và phải được phản ánh trong metric.

Công thức payload của bài không gồm weights, scratch space, temporary repeated heads, allocator overhead, block tables hoặc các bản sao do compaction. Vì vậy không lấy `VRAM / KV per request` làm số concurrent request chắc chắn chạy được. Dense cache lab cấp sẵn theo max prompt+budget, rồi giảm số hàng bằng copy khi request kết thúc.

## 7. Paging và prefix sharing có hai trách nhiệm riêng

Paging ánh xạ các block logic của một chuỗi sang vị trí vật lý. Prefix sharing cho nhiều chuỗi tham chiếu cùng trạng thái prefix khi hợp lệ. Nếu một nhánh ghi vào block đang được chia sẻ, cần cơ chế tách hoặc copy-on-write phù hợp; không được sửa trạng thái mà nhánh khác còn dùng.

Bài báo [PagedAttention](https://arxiv.org/pdf/2309.06180) mô tả block mapping và sharing; giáo trình không chép code hoặc benchmark của hệ đó. Ví dụ slot trong bài chỉ là tính ceil, chưa là implementation allocator. Giảm internal/external fragmentation theo một thiết kế không đồng nghĩa tổng memory waste bằng0.

Với prefix reuse giữa tenant, identity toán học của tensor và quyền sử dụng dữ liệu phải cùng hợp lệ. Model state đúng không đủ nếu prefix chứa tài liệu request mới không có quyền đọc. Tầng cache cần phối hợp hợp đồng của bài26, thay vì chỉ so một string cache key.

## 8. Vì sao batch nhỏ decode có thể bị giới hạn bởi đọc bộ nhớ?

Một phép nhân với một vector token dùng weights lớn nhưng ít tái sử dụng weights trong cùng lượt. Ghép nhiều token từ nhiều request cho phép một phần dữ liệu weight phục vụ nhiều phép tính. Khi batch tăng, arithmetic intensity có thể tăng, nhưng KV reads, compute và bộ nhớ cũng tăng.

Ví dụ14GB/700GB/s trong bài chính là cận dưới giả định chỉ cho weight transfer. Dạng tư duy roofline là thời gian không thể nhỏ hơn cả lượng byte/băng thông lẫn lượng phép tính/compute rate hữu dụng; thực tế còn overhead và khả năng overlap. Không cộng hoặc so các số peak khác đơn vị rồi coi đó là latency dự báo.

Đo batch throughput và per-request ITL riêng. Một batch tạo8token/20ms có aggregate400token/s, nhưng mỗi chuỗi vẫn có thể chỉ nhận50token/s. Con số khấu hao2.5ms/output token tổng hợp không phải ITL của một người dùng.

## 9. Dynamic batch wait, fairness và prefill/decode interference

Đợi thêm vài request có thể giúp ghép batch, nhưng người đến đầu phải chờ. Max batch size, max wait, max prefill tokens, tổng KV và deadline đều là ràng buộc có ích. Không có một ngưỡng tối ưu cho mọi phân bố prompt/output.

Lịch ưu tiên request ngắn có thể cải thiện mean latency trong một workload nhưng gây chờ lâu cho request dài. Lịch ưu tiên decode có thể giữ stream mượt nhưng làm TTFT của request mới xấu. Chunked prefill là một công cụ điều phối, chưa tự quyết định policy công bằng.

Mô phỏng tick trong [scheduling.py](scheduling.py) dùng mỗi token cùng chi phí và bỏ hẳn prefill, nên không dùng để chọn chunk size hay GPU batch limit. Nó chỉ làm rõ vì sao phải có cơ hội thay thành viên giữa các iteration. Nguồn [Continuous batching](https://huggingface.co/docs/transformers/continuous_batching) mô tả cơ chế/API hiện hành; bài không chạy manager/thread/distributed resources của thư viện đó.

## 10. Tối ưu kernel, compile và độ chính xác

Một attention backend có thể đổi thứ tự phép cộng và cách tính trung gian. FP32/FP16/BF16/quantized KV có đặc tính số học khác. So logits cần tolerance và lát cắt đủ rộng; so generation cần cùng stopping và sampling setup. Khi sampling, giữ seed chưa chắc bảo đảm chuỗi giống trên mọi engine do cách cấp RNG và thứ tự request khác nhau.

Static storage không tự làm graph tĩnh: code lab slice theo key length, tạo mask mới, dùng loop ghi từng hàng và copy khi compaction. `torch.compile`/CUDA graphs là chủ đề tiếp nối, không có kết quả hiệu năng trong bài. Nếu thử, tách compilation/cold start khỏi steady-state và kiểm bộ shape thực tế, thay vì chỉ một prompt cố định.

FlashAttention giảm I/O theo một tổ chức tính attention; KV cache tránh tính lại prefix; paging quản lý layout/bộ nhớ; batching/scheduling điều phối công việc. Các hướng có thể bổ sung nhau nhưng không thể cộng các hệ số speedup tùy ý. Chỉ đọc abstract [FlashAttention](https://arxiv.org/abs/2205.14135) ở bài này.

## 11. Thiết kế benchmark có thể phản biện

Khóa weights, token IDs, dtype, backend, thread count, prompt/output length, stopping và số request. Kiểm đầu ra trước. Warm-up mọi đường tính; xoay thứ tự đo để giảm bias theo thời gian. Giữ từng sample, báo statistic phù hợp và cửa sổ đo. Với GPU cần xác định synchronization/event timing; phép đo CPU hiện tại không làm việc đó.

Bảng lab đo makespan64output token trên bốn request. TTFT probe là một measurement khác, bắt đầu sau pack/cache allocation; không suy p95 API từ nó. Bảy repeat ít và workload nhỏ; min–max là range quan sát, chưa là confidence interval. Không viết “SDPA luôn nhanh hơn” từ bảng này.

Nếu có client/network, đo client-visible event như bài34, ghi queue wait/prefill/decode riêng như bài32. Chỉ khi workload và SLO hợp lệ mới bàn capacity hoặc cost. Token/s không có quality gate cũng có thể cải thiện bằng cách thay model/độ dài, nhưng đó không còn cùng bài toán.

## 12. Cache lifecycle và khả năng phục hồi

Mỗi cache trong lab gắn với request IDs theo thứ tự hàng và model object/parameter versions. Select hàng còn lại tạo storage độc lập, tránh viết tiếp làm đổi bản cache cũ. Đây là lựa chọn để học và kiểm đúng; peak memory tạm khi copy có thể cao hơn reserved_bytes của object cuối.

Sau EOS/budget, generator bỏ tham chiếu khi request kết thúc; không lưu KV vào checkpoint hoặc chia sẻ giữa phiên. Không hỗ trợ concurrency nhiều thread cùng ghi một cache, cache persisted/recovered hoặc distributed request migration. Nếu có exception giữa các layer vì lỗi runtime, người gọi nên bỏ cache thay vì giả định có rollback transaction; validation chỉ bảo vệ các lỗi đầu vào được kiểm trước khi ghi.

Bản nâng cấp production cần lifecycle rõ: tạo, append hợp lệ, kết thúc/cancel/deadline, release, và quan sát KV block usage. Gateway/auth/rate limit vẫn thuộc hệ phục vụ. Model inference nhanh hơn không thay được các hợp đồng backend ở bài31–34.
