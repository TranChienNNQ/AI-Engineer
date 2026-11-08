# Phụ lục bài 39 — Hiểu đúng các biên phân tán

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Các ví dụ số và protocol lab do giáo trình tự biên soạn. Nguồn chính cho thuật ngữ kiến trúc là [Ray Serve](https://docs.ray.io/en/latest/serve/architecture.html), [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) và phần layout của [PyTorch TP](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html). Bài không triển khai các framework đó.

## 1. DP inference không phải một bước training đồng bộ

Trong dense model inference, hai bản cùng revision nhận hai request riêng có thể tính độc lập. Không có gradient để đồng bộ. Tên data parallel nói về việc chia request/data; nó không buộc dùng wrapper DistributedDataParallel của training.

Model sharding lại đặt các phần của một lần forward trên nhiều rank. Lúc này một request phụ thuộc dữ liệu/collective của các rank trong group. Tăng số rank của một group và tăng số group độc lập là hai thay đổi khác nhau về memory, communication, failure domain và concurrency.

Một API endpoint có thể che cả hai cách bên dưới. Muốn suy throughput/latency hoặc rollout, cần biết replica logic thực sự gồm những thành viên nào, không chỉ đếm pod/process.

## 2. Chia đúng axis và đặt phi tuyến đúng vị trí

TP của lab chia hidden units thành các tập không giao nhau, nên ReLU của từng tập có thể tính riêng rồi cho output partial được sum ở lớp sau. Điều đó dựa trên shape và toán cụ thể, không phải mọi cách cắt ma trận đều tương đương.

Nếu chia input dimension và hai rank tạo partial preactivation cho **cùng một hidden unit**, cần cộng trước ReLU. Với partial−1và+1, `ReLU(−1+1)=0`, nhưng `ReLU(−1)+ReLU(1)=1`. Đặt ReLU ở mỗi partial rồi cộng là sai.

Bias cũng vậy: nếu output partial chưa chứa toàn bộ phép tổng, cần biết bias replicated được cộng ở đâu để tránh nhân lên theo số rank. Các layout trong [PyTorch TP](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) quyết định communication giữa các module; bài chỉ dùng một phân rã đơn giản của forward, không suy toàn bộ Transformer dùng cùng kế hoạch.

## 3. Pipeline bubble và latency một chuỗi

Forward pipeline balanced có warm-up và drain. Với m microbatch/p stage/t mỗi stage, makespan `(m+p−1)t` giả định các stage độc lập theo thời gian và không có communication cost. Tăng m làm tỷ lệ bubble giảm nhưng có thể tăng queue/latency cho từng request hoặc memory giữ activation.

Autoregressive generation có vòng phụ thuộc: cần output của token trước để chọn input token sau. Một sequence đơn không có một dãy token tương lai đã biết để tùy ý nhồi vào pipeline. Các request/sequence khác có thể cung cấp công việc độc lập để chồng lấp.

Microbatch không phải packet network, cũng chưa là một chunk prefill. Khi đọc lịch, xác định nó chia trục batch, token/prompt chunk hay bước thời gian; đừng đổi nghĩa giữa phép tính throughput và kết luận ITL. [PyTorch pipeline tutorial](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html) dùng partition/stage/schedule trong ngữ cảnh có thể khác serving forward-only.

## 4. Memory per-rank có replicated và sharded parts

Một tensor có thể shard, replicate hoặc nằm riêng trên một stage. Không thể lấy tổng weights/KV/activation rồi chia đồng loạt cho TP×PP. LayerNorm, embeddings, logits, residual layout, buffers của collective và metadata phụ thuộc thiết kế runtime/model.

KV còn phụ thuộc số KV head và layout như bài37. Nếu degree không chia được head count hoặc một phần state cần replicate, phép chia KV lý tưởng không áp nguyên được. Quantization ở bài38 lại thêm packing/scales/alignment và workspace. Hãy đọc báo cáo allocation/rank cụ thể, kiểm load peak và steady state riêng.

Ledger28GiB/TP4 trong bài chỉ xét90%sharded/10%replicated với giả định cố định. Số9.1GiB/rank không là phân bố của checkpoint/model có tên cụ thể; cũng chưa gồm PP trong cùng phép tính. Nếu muốn kết hợp hai chiều, cần lập ledger theo stage/rank thực tế.

## 5. Payload byte khác wire byte và process memory

Activation H của một hàng trong PP lab là4FP32value=16byte. Pipe còn serialize object, field names, request ID và protocol overhead; số byte qua OS không chỉ16. Tương tự,120byte weights không là RSS của một Python process đã import NumPy.

Nhiều process có startup/import/runtime overhead, copy khi truyền và parent còn giữ weights để dựng demo. Vì vậy lab không trình bày speedup hoặc giảm tổng RAM khi chia một model120byte. Nó xác nhận rằng payload gửi tới child chỉ có shard cần dùng và kết quả được ghép đúng.

Trên GPU/host khác, memory copy, staging buffers, transport, topology và overlap có thể chi phối. Một phép truyền lớn có thể bị bandwidth giới hạn; nhiều phép rất nhỏ có thể bị latency/launch giới hạn. Cần nhìn phân bố kích thước và đường critical path.

## 6. Collective là hợp đồng của cả group

Reduce/all-reduce/all-gather khác nhau về output layout. All-reduce sum cho cùng tổng trên mọi rank; reduce chỉ trả rank đích. Lab TP reduce về coordinator, không broadcast lại tổng tới hai child, nên gọi là all-reduce sẽ mô tả sai chương trình.

[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) yêu cầu các rank tham gia với count/dtype phù hợp. Việc worker còn sống nhưng bị kẹt ở một bước khác có thể gây chờ toàn group. Một timeout ở client không tự hủy các collective trong backend.

Mô hình ring của bài là công cụ tính α–β, chưa chọn thuật toán của một thư viện. Các engine có thể dùng thuật toán/transport khác theo shape/topology; không lấy một công thức ring làm dự báo chính xác mọi NCCL operation.

## 7. Placement phải cân đối liên kết và fault domain

Đặt các rank trao đổi thường xuyên trên liên kết nhanh có thể giảm critical-path communication. Đặt tất cả replica lên cùng một host lại khiến lỗi host ảnh hưởng nhiều bản cùng lúc. Cần phân biệt ưu tiên locality trong group với phân bố độc lập giữa các group.

[VLLM parallelism/scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) mô tả nhiều cách phối TP/PP và multi-node; đó là hướng đọc cấu hình, chưa một policy tối ưu cho mọi phần cứng. Placement còn phải thỏa memory, model/head constraints, network capacity, phiên bản phần mềm và lịch dùng chung cluster.

Bài không có GPU, scheduler tài nguyên hay fault-zone simulator. Không suy được chi phí thuê máy, NVLink/InfiniBand throughput hoặc topology tối ưu từ IPC CPU.

## 8. Admission phải theo phần việc thực còn tồn tại

Một request có trạng thái client terminal nhưng phần tính toán có thể còn ở worker. Lab tách `record.status` và `slot.busy`, giữ slot cho đến reply/death. Đây là cách tránh overbook trong hệ một slot/worker, không là cơ chế preempt mọi phép tính.

Trong engine có continuous batching, resource cần theo dõi có thể là active sequence, KV block, prefill token budget, running kernel và pending communication. Giải phóng một request trong router chưa bảo đảm tất cả tài nguyên đó được thu hồi ngay. Cancellation cần đi qua các tầng và có ack/quan sát phù hợp.

Không nên dùng client timeout để che overload rồi tiếp tục nhận vô hạn request mới. Những request đã hết hạn nhưng vẫn chạy có thể tạo tải ẩn. Metric goodput cần đi cùng số việc thực còn in-flight và tài nguyên giữ lại sau timeout/cancel.

## 9. Deadline propagation và retry amplification

Cùng host trong lab dùng cùng monotonic time base. Qua host, không coi raw monotonic là timestamp toàn cầu; [gRPC Deadlines](https://grpc.io/docs/guides/deadlines/) giải thích truyền budget đã trừ elapsed time để tránh clock skew.

Ví dụ budget ban đầu500ms, đã tiêu180ms, muốn dành20ms để trả kết quả: còn300ms cho bước downstream, không đặt lại500ms. Nếu hết budget thì không gửi thêm công việc chỉ vì worker khác rảnh. Retry còn tiêu compute và có thể đè lên attempt cũ chưa dừng.

Mỗi tầng retry độc lập có thể nhân số attempt xuống sâu. Cần một retry budget/correlation hợp lý và xử lý partial stream riêng. Lab không có retry tự động; `request_id` unique trong một history RAM và epoch fencing không phải một giao thức exactly-once bền vững khi coordinator chết.

## 10. Readiness, drain và rollout là các trạng thái khác nhau

Readiness nói worker/group đủ điều kiện nhận công việc mới; liveness chỉ là một tín hiệu còn hoạt động. Drain chủ động rút khỏi routing mới nhưng giữ việc cũ đến khi hoàn tất/budget dừng. Worker crash mất đột ngột lại cần kết thúc các attempt đang phụ thuộc nó.

Để thay model group, đảm bảo shard/revision/layout của các stage đồng nhất, warm-up/correctness gate qua trước khi đổi traffic. Giữ identity của request đang chạy để không trộn bản cũ/mới. Đồng thời dự trù memory khi warm phiên bản mới và phiên bản cũ còn xử lý stream.

Router code chỉ phục vụ revisionmlp-v1 cố định và tạo lại worker cùng revision khi restart. Nó chưa minh họa rollout V1→V2, canary, distributed registry hoặc atomic switch group; xem bài33cho nền tảng artifact/rollback. Hash của weights kiểm consistency với identity tin cậy, không tự xác thực tác giả hoặc cấp quyền network.

## 11. Prefill/decode disaggregation có giá chuyển state

Tách hai giai đoạn cho phép bố trí/điều phối tài nguyên riêng, nhưng decode cần state của prefix. [DistServe abstract](https://arxiv.org/abs/2401.09670) mô tả hướng tối ưu theo TTFT/TPOT và bandwidth; bài chỉ đọc abstract, chưa tái lập kết quả.

512MiB/8GiB/s=62.5ms chỉ là cận dưới của truyền đủ payload qua link giả định. Layout conversion giữa degreeTPkhác nhau, routing/ownership, allocation ở đích và cache identity đều có thể thêm cost. Nếu overlap được một phần transfer với compute, critical path có thể khác tổng cộng đơn giản; cần đo thực tế.

PP activation transfer không đồng nghĩa KV migration. H là output một stage cho microbatch đang đi qua; KV là lịch sử attention per-layer/head/token dùng qua nhiều bước decode. Không lấy kết quả PP MLP làm bằng chứng resume một stream LLM ở host mới.

## 12. Kiểm hệ phân tán theo lớp bằng chứng

**Toán:** so full với shard trên input cố định/ngẫu nhiên, dùng oracle khác cách tính, kiểm bias/phi tuyến/layout. **Process/protocol:** kiểm đúng process độc lập, request correlation, readiness/version/epoch, giới hạn admission, terminal outcome và cleanup. **Hệ thống thực:** còn network/transport failure, tail latency, workload thật, memory và quality/SLO.

Lab đã làm hai lớp đầu trong phạm vi nhỏ. Process IDs trong report là bằng chứng quan sát của mỗi lần chạy, không là golden value để so giữa hai lần. Timing30ms dùng tạo deadline thật có barrier, chưa là một latency benchmark. Stale epoch frame được tiêm trực tiếp vào handler để kiểm contract, không là mô phỏng network replay đầy đủ.

Để tiến tới một cluster, giữ nguyên cách tách bằng chứng: trước hết chứng minh tensor/identity đúng; sau đó fault injection ở cấp group/host/network; cuối cùng load test có kiểm TTFT/ITL/goodput/quality và rollback criteria. Chỉ khi đo đúng ràng buộc mới quyết định thêm replica, tăng degreeTP/PP hay đổi model/quantization.
