# Phụ lục bài 40 — Semantics, bộ nhớ và đo tối ưu

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Bốn thứ dễ bị trộn vào một chữ “tối ưu”

**Đổi thuật toán** thay lượng công việc: sparse attention có thể không xét mọi cặp query–key. **Đổi schedule/layout** tính cùng công thức theo thứ tự và vị trí khác: tiling/online attention. **Đổi precision** thay miền số/rounding: FP32→BF16 hoặc quantization. **Đổi cách phục vụ** thay batch/routing/placement. Một hệ thống có thể kết hợp cả bốn, nhưng mỗi phép thay cần kiểm khác nhau.

Bỏ toàn bộ score matrix là thay schedule/lưu trữ; không tự bỏ số cặp query–key. Nén weights thay byte lưu trữ; nếu mỗi forward giải nén toàn bộ sang FP32 rồi gọi matmul cũ thì chưa có low-bit matmul. Baseline và kết luận phải khớp đúng lớp đã thay.

## 2. “Đúng” gồm những hợp đồng nào?

Một custom kernel cần ít nhất: operator semantics, shape và broadcast, dtype và accumulation, stride/alignment, mask, aliasing, device, numerical tolerance, và nếu training thì backward. Đôi khi còn cần determinism hoặc khả năng chạy concurrent/stream.

`native.bind` của bài chỉ nhận vector FP32 contiguous/aligned, buffers rời nhau, output writable, input hữu hạn và phép affine không overflow FP32. Nó không là operator tổng quát nhận mọi stride/dtype của NumPy. Không dùng raw C ABI với mảng ngắn hơn n: C không tự kiểm bounds. Python wrapper giữ reference đến buffers qua pointer arguments và closure để vùng nhớ còn sống khi call.

Đọc source thấy `restrict`: người gọi hứa các vùng đọc/ghi không alias theo hợp đồng của hàm. Vi phạm không chỉ làm chậm; compiler được phép dựa vào hứa đó, nên kết quả có thể không xác định. Copy input để sửa layout là một lựa chọn hợp lệ, nhưng phải công bố và đo khoản copy khi nó nằm trên critical path.

## 3. Source pass, instruction và memory transaction là ba lớp khác nhau

Ba loop ở source không đủ để suy số lần DRAM đọc. Một loop có thể vectorize; cache có thể giữ buffer; writeback có thể xảy ra sau lúc một đoạn code đã xong; prefetch làm thay đổi thời điểm đọc. Source-level byte ledger hữu ích để đề xuất tối ưu và so sánh mô hình, chưa thay hardware counter.

Trong C lab, helper `noinline` giữ các phase tách về mặt function implementation, không khóa toàn bộ compiler transformation hoặc cache. Fused có một local z; nó thường có thể nằm trong register, nhưng bài không đọc disassembly để xác nhận register allocation cụ thể. Không báo “đã giảm DRAM đúng 3 lần” từ kết quả này.

Với GPU, một kernel có thể còn phát sinh nhiều transaction do coalescing, cache miss hoặc register spilling. “Local memory” trong thuật ngữ CUDA không đồng nghĩa register: phần spill có thể nằm trong device memory. Việc lập trình viên viết biến cục bộ không bảo đảm dữ liệu luôn ở on-chip storage.

## 4. Một roofline phải nêu tầng nhớ và kiểu tính toán

Công thức `I=F/D`, trần `min(P,B*I)` đòi hỏi P và B đúng miền đang xét. Không dùng peak FP16 matrix throughput làm trần cho kernel scalar FP32 rồi kết luận utilization thấp. Không dùng số byte payload thấp bit cho kernel thực tế đã giải nén và đọc FP32 toàn bộ.

Có thể có hierarchical roofline: DRAM, L2, L1/shared memory. Một kernel có reuse tốt qua DRAM nhưng vẫn bottleneck ở tầng gần hơn hoặc instruction dependency. Vì vậy điểm nằm dưới đường trần chưa cho biết chính xác nguyên nhân: cần kết hợp timeline, counters và nguồn code.

Số `8µs` trong bài là lower bound cho P/B giả định với D=8MB. CPU lab không có thiết bị 100TFLOP/s/1TB/s đã đo. Không đặt số CPU µs vào đồ thị roofline giả định đó như một điểm “achieved GPU performance”.

## 5. Tiling dùng lại dữ liệu nhưng cũng tiêu tài nguyên

Với tile matmul `BM×BN`, nội tích đi theo các khối BK:

- A tile có BM×BK phần tử, B tile có BK×BN.
- Accumulator có BM×BN phần tử, thường đòi precision phù hợp.
- Double buffering/pipeline có thể cần nhiều bản A/B tile sống cùng lúc.

Ledger tối giản FP32 cho BM=BN=BK=16: A/B mỗi tile 1KiB, accumulator 1KiB, tổng 3KiB trước overhead. Đó không phải occupancy calculator: layout, registers per thread, shared memory allocation và padding do compiler quyết định. Tăng BM/BN lên 32 có accumulator 4KiB; không được giả định vẫn giữ nguyên số block resident hoặc latency.

Triton/CPU/BLAS có thể dùng nhiều tầng tile: tile ở cache/shared memory, microtile ở registers, rồi instruction-level tile. Một tile trong Python không tương đương một thread block hay instruction matrix tile. Lab chỉ minh họa phép cộng các tile product, không autotune kernel vendor.

## 6. Chứng minh recurrence online attention

Sau khi đọc tập key J, định nghĩa `m=max(s_j)`, `l=sum exp(s_j-m)`, `z=sum exp(s_j-m)*v_j`. Gọi m' là max sau khi thêm khối mới. Mỗi số hạng cũ có:

\[
e^{s_j-m'}=e^{s_j-m}e^{m-m'}.
\]

Do đó cùng hệ số α rescale được cả l và z cũ. Khối mới tính trực tiếp theo m'. Cộng hai phần cho đúng định nghĩa trên tập hợp mở rộng; quy nạp từ khối đầu đến khối cuối cho `z/l=softmax(s)V`. Không cần materialize phân phối cuối đầy đủ nếu chỉ cần output.

Code giữ **z chưa chuẩn hóa**. Một implementation khác giữ output đã chia l sau mỗi bước có recurrence khác và phải nhân lại l khi kết hợp. Không trộn hai công thức. Việc trừ max làm exponent không dương, hạn chế overflow exp; nó không bảo vệ mọi input cực lớn khỏi overflow ngay trong matmul. Lab kiểm score hữu hạn sau matmul và không tuyên bố xử lý mọi FP64 hữu hạn mà không overflow intermediate.

Dòng hoàn toàn bị mask không có probability distribution chuẩn hóa. Lab chọn output 0 và giữ l=0; đây là policy rõ để tránh NaN, cần thống nhất với caller. Mask `True` là **được phép chú ý** trong bài; không suy cùng convention cho mọi API.

## 7. Largest tile chưa là peak memory

Ở fixture, lớn nhất một score tile là 3×4=12 FP64 value, 96 byte. Nhưng code còn có s, p, temporary do `where/exp`, accumulator z, m/l, output, input và có thể full boolean mask do caller giữ. Cộng 96 byte rồi gọi là “peak memory 96 byte” là sai.

Để chứng minh peak allocation cần công cụ và cách đo phù hợp allocator/backend. Để chứng minh HBM IO cần mô hình hoặc counter ở đúng cấp. Chỉ số `max_score_elements` là instrumentation cấu trúc thuật toán để xác nhận không tạo **một score tile** lớn hơn bound đã nêu; không theo dõi mọi allocation do NumPy thực hiện.

## 8. Precision, reduction order và reproducibility

Floating point không kết hợp chính xác: với FP32, `(1e8 + -1e8) + 1` có thể bằng 1 trong khi `1e8 + (-1e8 + 1)` bằng 0. Tiling/split-K/tree reduction đổi thứ tự cộng; FMA thay một lần rounding cho mul+add; fast-math có thể cho phép biến đổi thêm. Numerical parity cần rtol/atol phù hợp dữ liệu và precision, không đòi bitwise ở mọi backend.

C lab tắt FP contraction để so cùng hai lần rounding với NumPy FP32; các fixture exact trong môi trường đã kiểm. Điều đó không chứng minh approximation exp, FP16 accumulation hay Tensor Core cho cùng kết quả. Online NumPy dùng FP64 và tolerance1e−12, chưa là chuẩn tolerance cho LLM.

Training cần kiểm gradient: finite differences/gradcheck ở shape nhỏ, backward mask và accumulation. Một kernel fused ReLU có điểm không khả vi tại 0; kiểm gradient phải nêu convention và tránh diễn giải finite differences tại kink như một lỗi chắc chắn.

## 9. Đọc benchmark report

Mỗi `samples_ns` chứa 9 số **thời gian trung bình mỗi call trong một block lặp**, không là 9 latency request và không là từng native instruction. p10/median/p90 tính từ 9 block averages bằng quantile tuyến tính của NumPy. Không gọi p90 này là p90 API latency hoặc confidence interval.

Thứ tự xen kẽ giảm một dạng bias drift nhưng chưa ngẫu nhiên hóa hoàn toàn hoặc kiểm soát nhiệt độ/tần số. N=256 dễ bị Python/ctypes dispatch chi phối; N lớn có nhiều compute/data movement hơn nhưng repeated buffers vẫn cache-warm. Có thể thêm noop native call để ước lượng overhead, nhưng không được tự trừ một hằng số nhiễu rồi gán phần còn lại là kernel device time.

So sánh cả hai trong C tránh đồng nhất lợi ích “bỏ Python loop” với lợi ích fusion. Tuy nhiên native three-pass của bài là baseline phục vụ cơ chế, không đại diện thư viện tối ưu nhất. Với sản phẩm, cần baseline NumPy/PyTorch/compiler/library đúng workload.

## 10. Kernel benchmark và end-to-end benchmark

Một GPU event interval có thể đo thời gian device trên một stream; host wall time có thể bao gồm dispatch, queue/sync hoặc transfer. Hai số khác nhau không tự chứng minh một phép đo sai: có thể chúng đo hai đối tượng khác nhau. Cần nêu boundaries, streams và sync. Transfer host↔device có thể nhỏ nếu data luôn resident, nhưng rất lớn nếu mỗi request phải copy nhiều; đo theo ứng dụng.

Service benchmark cần arrival pattern, batch, sequence distribution, cache state, concurrency và SLO. Amdahl trong bài giả định các phần cộng tuần tự; overlap hoặc queue khiến công thức đơn giản chưa đủ dự báo. Khi benchmark native C xong, chưa chứng minh endpoint AI nhanh hơn: phải tích hợp đúng rồi đo lại.

## 11. Chọn ranh giới specialization

Kernel chỉ cho contiguous FP32 vector có ít nhánh và dễ kiểm. Kernel tổng quát cho stride/broadcast/dtype tùy ý cần nhiều logic hoặc nhiều biến thể. Thêm biến thể làm tăng compile cache, kiểm thử và cơ hội dispatch sai. Một dispatch key có thể gồm device architecture, dtype, shape class, stride/layout, mask và compile flags.

Autotuning chọn config trên một tập workload đã định. Giữ riêng workload đánh giá và ghi budget tuning/cold cost. Không dùng mỗi shape một lần đo rồi chọn lượt nhanh nhất vì đó có thể là nhiễu. Có fallback đúng cho shape ngoài miền; fallback cũng cần quan sát được trong trace để phát hiện traffic đang đi đường chậm.

## 12. Hồ sơ đủ để người khác review một tối ưu

Lưu code/flags/compiler/library/device; input distribution và seed; shape/dtype/layout/mask; baseline và correctness metric; warm-up/raw timing/quantile/đơn vị; timed region; cold-start; memory metric định nghĩa rõ; request-level effect và hạn chế. Với model cần thêm chất lượng/retention, với distributed cần communication/failure behavior.

Bộ lab chỉ lưu report JSON khi chạy, còn source và hình minh họa ở bộ chia sẻ. Không lấy report runtime làm “test golden” cho timing: correctness có thể tái lập, wall time không có cam kết bằng nhau. [Danh mục nguồn và phạm vi đã đọc](../SOURCES.md).
