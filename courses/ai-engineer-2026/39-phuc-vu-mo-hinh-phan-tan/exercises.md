# Bài tập bài 39

[← Bài chính](README.md) · [Lời giải](solutions.md)

Các số thời gian/memory lớn là giả định sư phạm, không là đặc tả GPU đã đo. Hãy tính hoặc dự đoán trước khi chạy. Với snippet tạo process, giữ entry-point guard khi đặt trong script.

## 1. Chọn đúng bài toán phân tán

Model đã vừa một GPU nhưng queue dài; model không vừa một GPU; một host có thể hỏng. Với từng trường hợp, mô tả mục tiêu và vai trò replica/TP/PP. Vì sao thêm HTTP worker chưa chắc giải quyết được?

## 2. Rank và model group

D=2,T=4,P=2. Có bao nhiêu rank và replica logic? Viết mapping `((d×P)+p)×T+t` thành các rank theo group/stage. Nếu tăng D lên3 nhưng giữ T/P, thay đổi phần nào của bài toán?

## 3. TP tính tay

Với x, W1/W2 và bias trong bài, tính hidden, hai partial và output. Nếu mỗi rank cộng b2 rồi coordinator sum, output sai bao nhiêu? Nêu axis bị chia theo quy ước ma trận của lab.

## 4. Phi tuyến ở sai chỗ

Hai rank tạo partial preactivation−1và+1cho cùng một hidden unit do chia input dimension. So `ReLU(a+b)` với `ReLU(a)+ReLU(b)`. Vì sao cách này khác TP hidden-shard hợp lệ của lab?

## 5. Pipeline forward-only

Ba stage2ms và4microbatch: tính thời điểm finish từng stage, first latency, makespan và stage-slot utilization. Đổi stage times thành[2,4,2]ms thì sao? Có dùng công thức này cho một training iteration có backward hoặc một sequence autoregressive đơn tùy ý không?

## 6. Memory per-rank

Model weights28GiB,90%shard theoTP4/10%replicate. Tính weight/rank; budget16GiB/rank, other3GiB/rank thì còn bao nhiêu? Nêu ba loại bộ nhớ chưa chắc được chia như weights. Vì sao không lấy28/(TP×PP)cho mọi tensor?

## 7. Collective và communication

Phân biệt reduce/all-reduce/all-gather. Lab TP thực hiện cái nào? Với ring model p4,n16MiB,B8GiB/s,α10µs, tính time. Nếu bandwidth giảm còn1GiB/s thì sao? Liệt kê hai giả định khiến công thức chưa là dự báo NCCL thật.

## 8. Routing theo việc thay vì đếm request

Bốn job cost[1,9,1,9]đến cùng lúc, mỗi worker xử lý tuần tự, có hai worker. Tính loads round-robin và greedy chọn load thấp nhất. Policy cost-aware cần tín hiệu gì ở LLM và có thể sai ở đâu? Router lab có cài predictor đó không?

## 9. Timeout, cancel và slot vật lý

Chạy một worker hold, đợi STARTED, cancel client record. Trước khi release có gửi thêm việc vào worker được không? Kiểm status sau late response. So với timeout trong fixture: có phải compute tự dừng ngay khi caller ngừng chờ không?

## 10. Deadline qua nhiều host

Budget500ms, đã dùng180ms, dự trù20msđể gửi kết quả cuối. Còn bao nhiêu cho downstream? Vì sao không dùng một giá trị raw monotonic timestamp làm deadline chung cho hai máy? Retry sau timeout cần giữ những điều kiện nào?

## 11. Readiness, drain và revision

Process alive nhưng chưa load xong model có nên được route không? Mô tả drain một replica đang xử lý request. Vì sao stage0V1+stage1V2có cùng tensor shape vẫn không đủ để phục vụ? Worker epoch và model revision khác nhau thế nào?

## 12. Failure domain và availability

Giả định các thành phần độc lập, availability mỗi cái.99. Tính một group cần cả4rank và hai full replica mà chỉ cần một sống. Vì sao kết quả không áp nguyên vào hai replica cùng host/network/control plane? Lab crash test đã kiểm cấp nào?

## 13. KV transfer khi tách prefill/decode

KV512MiB, effective bandwidth8GiB/s. Tính transmission lower bound. Những cost/identity nào còn thiếu? Vì sao PP truyền một H16byte/microbatch trong lab chưa là KV migration/resume một stream LLM?

## 14. Đọc report

Chạy `all` với output mới. Giải thích weights payload120/120;56/56+bias8;80/40và max error0. Đọc10terminal outcome, hai late reply, một stale frame tổng hợp. Phân biệt63predicate đã chạy với số liệu chưa được đo như RSS/GPU speed/network partition.

## 15. Thiết kế kiểm trước deployment

Bạn muốn đưa một model lớn lên hai node. Đề xuất protocol kiểm tensor/layout/revision, readiness, workload/quality/SLO, memory/communication và lỗi một rank/một node. Nêu điều kiện rollback và vì sao một happy-path forward chưa đủ.

## 16. Tăng tài nguyên hay sửa nút thắt?

Một hệ có nhiều GPU nhưng TTFT/ITL vẫn xấu. Lập ba giả thuyết khác nhau liên quan routing/queue, communication/partition và prefill/decode/KV. Với mỗi giả thuyết, chọn phép đo để có thể bác bỏ nó; không mặc định mua thêm GPU là câu trả lời.
