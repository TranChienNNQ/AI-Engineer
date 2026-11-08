# Bài tập bài 40 — Kernel và tối ưu tính toán

[← Bài chính](README.md) · [Lời giải](solutions.md) · [Phụ lục](deep-dive.md)

Làm phần lập luận/tính tay trước khi chạy code. Phân biệt số giả định, ledger logic và số đo. Dùng thư mục bài 40 làm working directory cho Python.

## 1. Xác định lớp đang tối ưu

API có tokenizer, model graph, matmul library, GPU kernel và queue. Giải thích khác nhau giữa tăng batch, đổi FP32→INT4, fusion bias+ReLU và thêm replica. Một dòng `relu(x@W+b)` có luôn là một GPU kernel không?

## 2. Fusion tính tay

Với x=`[-1,0,1,2]`, a=2,b=1, tính ba bước và fused. Cho một phản ví dụ bác bỏ phép thay thành `a*ReLU(x)+b`. Native lab đếm ReLU comparison vào FLOP hay không?

## 3. Ledger byte và precision

Vector N=1.000.000 FP32, a/b scalar, buffers rời nhau. Tính logical traffic cho three-pass và fused, buffer scratch riêng và output bytes. Vì sao không được nói speedup chắc chắn bằng 3 hoặc DRAM traffic chắc chắn giảm 3 lần?

## 4. Tiled matmul không chia hết

Tính `A=[[1,2,3],[4,5,6]]`, `B=[[1,0],[0,1],[1,1]]` bằng hai khối trên trục K với tile2. Tính số tile product cho M=5,K=7,N=3,tile2. Nêu cách xử lý tail mà không đọc vượt buffer.

## 5. Reuse có tính output

Matmul vuông4×4 FP32: scalar operand loads512byte, tile2 operand loads256byte, cùng ghi C một lần64byte. Tính ratio tổng. Với BM=BN=BK=16, tính footprint tối giản của hai input tile và accumulator FP32, rồi giải thích vì sao chưa thể suy occupancy.

## 6. Roofline và đơn vị

Dùng F=2e6,D=8e6byte,P=100e12FLOP/s,B=1e12byte/s. Tính intensity, ridge, compute time, memory time và lower bound. Nếu bandwidth giả định giảm một nửa, điều gì đổi? Vì sao không lấy thời gian CPU ở bài chính đặt vào roofline GPU giả định này?

## 7. Sector và stride

32lane đọc FP32, base căn32byte, sector32byte. Đếm sector cho `index=l`, `l+1`, `8l`. Với ma trận row-major FP32[3,4], tính offset byte của phần tử[2,1]. Transpose view có luôn copy không?

## 8. Hợp đồng native kernel

Chạy ba tình huống: input stride2, output trùng input, output readonly. Dự đoán wrapper có từ chối không và kiểm lại. Giải thích quan hệ giữa kiểm alias và `restrict` trong C. Nếu cần hỗ trợ stride, có những hướng sửa nào?

## 9. Online attention tính tay

Scores=`[0,ln2,ln3]`, values=`[10,20,40]`. Tính m,l,z sau từng key và output cuối. Nếu quên rescale state cũ khi max tăng, với khối cuối ta được kết quả sai bao nhiêu? Viết recurrence đúng, không trộn z chưa chuẩn hóa với output đã chuẩn hóa.

## 10. Softmax từng khối và mask

Vì sao softmax riêng `[0,ln2]` và `[ln3]` rồi nối không bằng softmax toàn hàng? Viết một fixture có một hàng bị mask hoàn toàn, so dense và online. Trong lab `True` nghĩa gì, all-masked row trả gì và largest score tile có phải peak allocation không?

## 11. Numerical correctness

Dùng FP32 kiểm `(1e8 + -1e8) + 1` và `1e8 + (-1e8 + 1)`. Liên hệ với tile/reduction/FMA. Online NumPy FP64 đúng trong tolerance1e−12 có đủ kết luận kernel FP16 training đúng không? Cần kiểm thêm gì?

## 12. Đọc timing

Một report có 9 block timing, mỗi block200call rồi chia200. p90 của các số đó là gì? Có phải p90 latency request không? Lập bảng các khoản được tính/không tính trong benchmark đi kèm. Vì sao GPU enqueue wall time có thể chưa đo compute hoàn tất?

## 13. Amdahl và hòa vốn

Kernel chiếm40% thời gian request và nhanh3lần. Tính speedup end-to-end và trần nếu kernel miễn phí. Compile tốn2s, tiết kiệm0,1ms/call: sau bao nhiêu call hòa vốn và khi nào mới nhanh hơn? Giải thích tác động nhiều shape bị recompile.

## 14. Tự đọc report và lặp lại

Chạy modeall vào output mới, đọc raw samples, tự tính lại median/p10/p90 và ratio. Kiểm đủ ba N và80checks. Không assert timing phải giống bài chính. Tại sao ratio có thể lớn hơn traffic ratio3? Nêu bằng chứng cần thêm để kết luận kernel memory-bound trên phần cứng thật.

## 15. Kế hoạch GPU tiếp nối

Bạn cần fused softmax cho FP16 rows có N không chia hết tile. Viết hợp đồng, padding/mask, reference, miền test, cách đo cold/warm và fallback. Nêu khi nào tile lớn có thể làm chậm. Không cần GPU để làm bài lập luận; không ghi đã đo khi chưa chạy.

## 16. Review tối ưu trong API

Một PR nói “kernel nhanh3lần nên API p95 sẽ nhanh3lần”, dùng một shape, không log dtype/transfer, chỉ chạy một lượt chưa warm-up. Viết nhận xét và kế hoạch đánh giá đủ để chấp nhận hoặc bác bỏ. Kết nối bài32/34/38/39 và đề xuất một baseline thư viện phù hợp.
