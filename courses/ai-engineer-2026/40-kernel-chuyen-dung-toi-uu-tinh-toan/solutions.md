# Lời giải bài 40 — Kernel và tối ưu tính toán

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

Đáp án kết hợp lập luận và code kiểm độc lập. Các số hardware chưa đo được luôn giữ nhãn giả định.

## 1. Xác định lớp đang tối ưu

Tăng batch đổi tổ chức workload và có thể đổi shape kernel; INT4 đổi representation/precision và cần kernel phù hợp; fusion gộp tính toán để giảm intermediate/launch; replica đổi cách phân phối request và tài nguyên. Một dòng Python có thể gọi nhiều kernels, một library kernel hoặc compiled fused graph. Không suy kernel count từ số dòng source; kiểm runtime/profile.

## 2. Fusion tính tay

Mul cho `[-2,0,2,4]`, add cho `[-1,1,3,5]`, ReLU cho `[0,1,3,5]`. Fused giữ cùng công thức mỗi phần tử nên cùng kết quả. `a*ReLU(x)+b` cho x=−1 kết quả1, trái với0. Ledger đếm2FLOP/phần tử cho mul/add; ReLU comparison được loại khỏi số FLOP đã định nghĩa, không có nghĩa so sánh không tốn compute.

## 3. Ledger byte và precision

Ba pass:3×(4Nread+4Nwrite)=24MB. Fused:4Nread+4Nwrite=8MB. Hai scratch t/u riêng chiếm8MB; y chiếm4MB, x chiếm4MB. Fused chỉ cần x/y theo hợp đồng, không hai scratch; đếm buffer capacity khác với cộng read/write traffic.

```python
n=1_000_000
separate=3*(4*n+4*n)
fused=4*n+4*n
assert separate==24_000_000 and fused==8_000_000
assert separate/fused==3
print({'scratch_bytes':2*4*n,'output_bytes':4*n})
```

Các buffer có thể nằm trong cache, compiler có thể vectorize khác, và overhead dispatch không giảm theo byte. Cần counter để kết luận traffic qua một tầng nhớ và cần timing để kết luận tốc độ. Không đếm capacity8MB thành traffic8MB: hai đại lượng có ý nghĩa khác nhau.

## 4. Tiled matmul không chia hết

K=0…1 cho `[[1,2],[4,5]]`, K=2 cho `[[3,3],[6,6]]`; tổng `[[4,5],[10,11]]`. M5/K7/N3,tile2 có3×4×2=24tile product. NumPy slice cắt tại boundary; GPU có thể pad0 và mask các load/store không hợp lệ. Không đổi memory contract chỉ vì output cuối có shape đúng.

```python
import math
import numpy as np
from algorithms import tiled_matmul
c,_=tiled_matmul([[1,2,3],[4,5,6]],[[1,0],[0,1],[1,1]],2)
assert np.array_equal(c,[[4,5],[10,11]])
a=np.arange(35).reshape(5,7);b=np.arange(21).reshape(7,3)
y,s=tiled_matmul(a,b,2)
assert s['tile_products']==math.ceil(5/2)*math.ceil(7/2)*math.ceil(3/2)==24
assert np.allclose(y,a@b)
```

## 5. Reuse có tính output

Tổng scalar576byte, tile320byte, ratio576/320=1,8. Operand ratio512/256=2 không phải ratio toàn bộ traffic. Tile16: A16×16×4=1024byte, B1024byte, accumulator1024byte, tổng3072byte=3KiB. Chưa kể double buffering, alignment, register allocation hay overhead, nên chưa biết số block resident/occupancy. Trên CPU đây cũng không là phép đo cache capacity tiêu thụ chính xác.

## 6. Roofline và đơn vị

I=0,25FLOP/byte; ridge100FLOP/byte. Compute2e−8s=0,02µs, memory8e−6s=8µs, lower bound8µs. B giảm một nửa thì memory16µs, ridge200 và trần tại cùng I giảm từ250 xuống125GFLOP/s. Compute bound riêng không đổi.

```python
from arithmetic import roofline
r=roofline(2e6,8e6,100e12,1e12)
slow=roofline(2e6,8e6,100e12,.5e12)
assert r['intensity']==.25 and r['ridge']==100
assert abs(r['compute_s']-2e-8)<1e-20
assert abs(r['memory_s']-8e-6)<1e-18
assert abs(slow['lower_bound_s']-16e-6)<1e-18
```

Số đo CPU chứa ctypes/loop và cache của CPU; P/B giả định không thuộc thiết bị đó, D chưa là counter. Ghép các số khác hệ đo không tạo thành utilization hoặc roofline analysis hợp lệ.

## 7. Sector và stride

Sector index=`floor(4*index/32)`. Với l=0…31: l cho sector0…3, l+1 cho0…4, 8l cho0…31. Offset[2,1]=4×(2×4+1)=36byte. Transpose có thể là view đổi stride cùng storage, chưa copy; khi một kernel bắt buộc contiguous, caller có thể phải tạo bản contiguous và trả chi phí đó.

```python
import numpy as np
from arithmetic import sectors
assert [sectors(1),sectors(1,1),sectors(8)]==[4,5,32]
a=np.arange(12,dtype=np.float32).reshape(3,4)
assert 2*a.strides[0]+a.strides[1]==36
assert np.shares_memory(a,a.T)
assert not a.T.flags.c_contiguous
```

## 8. Hợp đồng native kernel

Cả ba tình huống bị từ chối. Hợp đồng buffers rời nhau làm lời hứa `restrict` hợp lệ; nếu raw pointer alias mà C vẫn dùng restrict thì behavior có thể không xác định. Hỗ trợ stride bằng cách viết index stride và kiểm bounds, hoặc copy vào contiguous rồi copy ra đúng semantics. Với copy path, phải đo cả phần copy nếu nằm trong workload sản phẩm.

```python
import tempfile
import numpy as np
from native import Native
with tempfile.TemporaryDirectory() as build:
    native=Native(build)
    x=np.arange(4,dtype=np.float32);y=np.empty_like(x)
    ro=y.copy();ro.setflags(write=False)
    cases=[lambda:native.bind(np.arange(8,dtype=np.float32)[::2],y,1,0),
           lambda:native.bind(x,x,1,0),lambda:native.bind(x,ro,1,0)]
    for case in cases:
        try:case()
        except ValueError:pass
        else:raise AssertionError('contract not enforced')
```

## 9. Online attention tính tay

Các trạng thái đúng: `(0,1,10)`; `(ln2,3/2,25)`; `(ln3,2,170/3)`. Output85/3≈28,3333. Nếu ở bước cuối giữ nguyên l/z cũ rồi thêm key3 với trọng số1, ta được l=2,5,z=65,output26; thấp hơn kết quả đúng7/3≈2,3333.

```python
import math
import numpy as np
from algorithms import online_attention
q=np.array([[1.]]);k=np.array([[0.],[math.log(2)],[math.log(3)]])
v=np.array([[10.],[20.],[40.]])
y,_=online_attention(q,k,v,query_tile=1,key_tile=1)
assert abs(y.item()-85/3)<1e-12
wrong=(25+40)/(1.5+1)
assert wrong==26 and abs((85/3-wrong)-7/3)<1e-12
```

Recurrence với z chưa chuẩn hóa: `m'=max(m,max(t))`, `α=exp(m-m')`, `p=exp(t-m')`, `l'=αl+sum(p)`, `z'=αz+pV`. Với output đã chuẩn hóa o=z/l, phải phục hồi l*o khi gộp. Đó là lý do không copy recurrence giữa hai cách giữ state mà bỏ hệ số l.

## 10. Softmax từng khối và mask

Khối đầu cho `[1/3,2/3]`, khối sau cho `[1]`, nối có tổng2. Toàn hàng đúng là `[1/6,2/6,3/6]`. Mỗi khối đã dùng denominator khác nhau; cần thống kê và rescale để kết hợp.

```python
import numpy as np
from algorithms import online_attention,dense_attention
q=np.array([[1.],[2.]])
k=np.array([[0.],[1.],[2.]])
v=np.array([[10.],[20.],[40.]])
mask=np.array([[False,False,False],[True,False,True]])
y,stats=online_attention(q,k,v,mask,query_tile=1,key_tile=2)
assert np.array_equal(y[0],np.zeros(1))
assert np.allclose(y,dense_attention(q,k,v,mask),rtol=1e-12,atol=1e-12)
assert stats['max_score_elements']==2
```

True là được phép. Hàng mask hết trả0 theo convention lab; không là phân phối xác suất. Chỉ số2 là largest single score tile, chưa gồm p/temporaries/mask/input/output/allocator; không bằng peak memory.

## 11. Numerical correctness

```python
import numpy as np
a,b,c=np.float32(1e8),np.float32(-1e8),np.float32(1)
left=np.float32(np.float32(a+b)+c)
right=np.float32(a+np.float32(b+c))
assert left==1 and right==0
print(left,right)
```

Khác thứ tự cộng có thể khác rounding; FMA có thể khác với mul rồi add. Vì vậy phải chọn reference, tolerance và worst-case/error metric theo dtype/use case. FP64 NumPy forward không chứng minh GPU FP16 forward hoặc backward đúng. Kiểm thêm dtype/accumulation, mask/broadcast/stride, gradient/frozen weights, numerical stability, model metric và các shape sử dụng thật. Không dùng bitwise equality như chuẩn duy nhất mọi backend, cũng không nới tolerance tùy ý để che lỗi.

## 12. Đọc timing

p90 là quantile của9block averages theo cách nội suy đã công bố. Nó không là quantile request latency, không là confidence interval. Timed region gồm Python loop+ctypes dispatch+C compute; không gồm input generation/validation, allocation, compile/load hoặc correctness. Compiler subprocess được ghi riêng. Buffers warm, native single-thread, không pin core/frequency.

GPU enqueue có thể trả về trước khi device hoàn tất, nên host clock chỉ bao quanh enqueue sẽ bỏ phần compute đang chạy sau đó. Event trên stream và sync đúng boundary giúp đo đối tượng mong muốn; latency end-to-end và device interval có thể là hai metric khác nhau hợp lệ. Không lấy sync benchmark làm bằng chứng production overlap giữ nguyên.

## 13. Amdahl và hòa vốn

Speedup=`1/(.6+.4/3)=15/11≈1,364`; trần khi kernel bằng0 là5/3≈1,667. Hòa vốn2/.0001=20.000call, call thứ20.001 mới làm tổng nhanh hơn theo mô hình này. Chưa tính cache compilation, warm-up hay chi phí mới khác.

```python
from arithmetic import amdahl
assert abs(amdahl(.4,3)-15/11)<1e-12
assert abs(1/(1-.4)-5/3)<1e-12
compile_s,saved_s=2.,.0001
assert 20_000*saved_s==compile_s
assert 20_001*saved_s>compile_s
```

Nhiều shape bị recompile làm tăng cold cost và có thể không có đủ call tái sử dụng mỗi biến thể. Tách histogram shape và cold/warm hit/miss; tránh chỉ báo steady-state trên shape phổ biến rồi bỏ cold tail.

## 14. Tự đọc report và lặp lại

Sau `lab.py --mode all --out /tmp/lesson40-my-run`, mở report, kiểm environment/compiler/source hash/flags, `checks.passed==checks.checks==80`, ba row đúng N. Với mỗi mode, tự dùng quantile tuyến tính trên9samples_ns để tính p10/median/p90 và ratio hai median. Đơn vịns chia1000 để raµs. Không lấy median của9ratio nếu report định nghĩa ratio hai median.

Ratio>3 có thể do cache, phase/dispatch overhead hoặc khác vectorization; cũng có thể do nhiễu. Bài chỉ quan sát được ratio trong lượt đo, chưa xác định nguyên nhân. Để chứng minh memory bottleneck, cần counter/timeline đúng tầng nhớ và xem đạt bandwidth nào, compute/instruction giới hạn nào, thay workload/size và repeat có kiểm soát. Logical bytes/time là effective rate theo ledger, không tự là measured DRAM bandwidth.

## 15. Kế hoạch GPU tiếp nối

Ví dụ hợp đồng: input matrix row-major FP16, softmax theo cột, output cùng shape, accumulation FP32, finite input, rows fit resource budget; không backward ở bước đầu. Tile pad tới miền block hợp lệ, load ngoài N dùng−∞, store có mask. Giữ reference stable softmax FP32 rồi cast đúng output; test N1/3/257/1003, nhiều row, cực trị, tails và rejection/fallback cho stride hoặc shape ngoài miền.

Tách compile/warm-up, GPU-event hoặc synchronized wall time theo metric đã chọn, repeat và lưu samples. Cùng device/dtype/input/semantics, có baseline library/compiler. Tile lớn có thể tăng register/shared-memory pressure, giảm resident blocks hoặc spill. Nếu rows quá dài, dùng reduction nhiều bước hoặc fallback; không ép kernel một-row vào workload ngoài hợp đồng. Backward và mixed precision training là bước kiểm tiếp riêng. Chưa chạy GPU thì báo kế hoạch/chưa đo.

## 16. Review tối ưu trong API

Nhận xét: số3lần chỉ mô tả kernel ở một shape trong một phép đo chưa đủ tin cậy, không chứng minh p95 API. Cần baseline NumPy/PyTorch/vendor GEMM hoặc compiled operator phù hợp, cùng dtype/layout/quality; correctness ở shape và mask khác nhau; warm-up/repeat/raw samples/cold-start; công bố copy/transfer/sync; sau đó chạy endpoint với workload và concurrency đã định.

Bài32 giúp trace tách queue/tokenize/model/communication; bài34 xác định arrival pattern, percentile và goodput/SLO; bài38 yêu cầu chất lượng và kernel support khi đổi precision; bài39 cho thấy local compute nhanh có thể chuyển bottleneck sang communication. Dùng profile để tính trần Amdahl sơ bộ rồi kiểm end-to-end. Chấp nhận khi evidence cho thấy lợi ích ở mục tiêu sản phẩm với giới hạn rõ; không chỉ vì một headline kernel speedup.
