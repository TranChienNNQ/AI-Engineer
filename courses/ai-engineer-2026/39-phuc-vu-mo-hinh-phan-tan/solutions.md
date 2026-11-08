# Lời giải bài 39

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

Mười snippet Python độc lập chạy từ thư mục bài này, chỉ cần core requirements. Các câu thiết kế có nhiều lời giải hợp lý nếu chỉ rõ giả định và phép đo kiểm chứng.

## 1. Chọn đúng bài toán phân tán

Queue dài khi model đã fit gợi ý thử replica/batching/routing và kiểm bottleneck. Model không fit cần giảm footprint hoặc chia một replica logic qua nhiều worker/device. Fault tolerance cần các bản/group độc lập, detection và xử lý công việc đang dở. HTTP worker chỉ là tiến trình phục vụ API; có thể cùng tranh một model/GPU/queue mà không thêm năng lực tính.

## 2. Rank và model group

Có16rank,hai model group/replica logic,mỗi group8rank. Stage0group0là0–3,stage1là4–7; group1là8–11và12–15. Tăng D lên3cho24rankvà3group; chưa làm nhỏ shard hoặc nhanh hơn một request trong group hiện tại.

```python
D,T,P=2,4,2
mesh={d:{p:[((d*P)+p)*T+t for t in range(T)] for p in range(P)} for d in range(D)}
assert mesh[0][0]==[0,1,2,3] and mesh[0][1]==[4,5,6,7]
assert mesh[1][1]==[12,13,14,15]
flat=[r for stages in mesh.values() for ranks in stages.values() for r in ranks]
assert sorted(flat)==list(range(16))
print(mesh)
```

## 3. TP tính tay

Hidden[1,1,0,3],partial[0,3]và[3,−6],output[3.5,−3.5]. Nếu cộng bias trên cả hai partial rồi sum, output[4,−4], lệch thêm đúng một b2. Lab chia cột W1/hidden và hàng W2theo quy ước W có shape input×output.

```python
import numpy as np
from model import weights,tp_parts,forward
w=weights();x=np.array([[1,2,0,-1]],dtype=np.float32)
p=[np.maximum(x@v['w1']+v['b1'],0)@v['w2'] for v in tp_parts(w)]
good=p[0]+p[1]+w['b2'];bad=(p[0]+w['b2'])+(p[1]+w['b2'])
assert np.array_equal(good,forward(x,w))
assert bad.tolist()==[[4.,-4.]]
assert np.array_equal(bad-good,w['b2'][None])
print(good.tolist(),bad.tolist())
```

## 4. Phi tuyến ở sai chỗ

Cùng một unit cần sum partial trước ReLU:ReLU0=0. ReLU riêng cho0+1=1là sai. Hidden-shard trong lab là hai tập unit khác nhau, nên ReLU riêng hợp lệ trước phép nhân W2và reduce output. Phải dựa vào phân rã toán chứ không chỉ tên TP.

```python
import numpy as np
a,b=-1.,1.
correct=np.maximum(a+b,0)
wrong=np.maximum(a,0)+np.maximum(b,0)
assert correct==0 and wrong==1
print(correct,wrong)
```

## 5. Pipeline forward-only

Balanced finish rows theo microbatch:[2,4,6],[4,6,8],[6,8,10],[8,10,12]ms. First6/makespan12/util2/3. Uneven first8/makespan20/util8/15. Bỏ communication/overhead, không có backward; autoregressive sequence còn token dependency nên không tự áp như batch độc lập.

```python
from arithmetic import pipeline
a=pipeline([2,2,2],4);b=pipeline([2,4,2],4)
assert a['finish']==[[2,4,6],[4,6,8],[6,8,10],[8,10,12]]
assert a['makespan']==12 and a['stage_slot_utilization']==2/3
assert b['first_latency']==8 and b['makespan']==20
assert abs(b['stage_slot_utilization']-8/15)<1e-12
assert pipeline([2,2,2],1)['makespan']==6
print(a['finish'],b['finish'])
```

## 6. Memory per-rank

Weights6.3sharded+2.8replicated=9.1GiB, khác7GiB. Sau other3GiB,còn3.9GiBtheo ledger. KV,activation,collective buffers,embedding/norm hoặc quantization metadata cần layout riêng. PP còn tùy stage/layercost,nên không chia mọi byte cho TP×PP.

```python
weights_gib=28;fraction=.9;tp=4
per_rank=weights_gib*fraction/tp+weights_gib*(1-fraction)
remaining=16-3-per_rank
assert abs(per_rank-9.1)<1e-12 and abs(remaining-3.9)<1e-12
assert per_rank>weights_gib/tp
print(per_rank,remaining)
```

## 7. Collective và communication

Reduce trả tổng về rank đích;all-reduce trả cùng tổng cho mọi rank;all-gather ghép shards. Lab reduce về coordinator. Ring model cho2.9896875msở8GiB/s; ở1GiB/scho23.4975ms. Chưa tính contention,latency khác theo topology,algorithm selection,kernel/serialization hoặc overlap.

```python
from arithmetic import ring_allreduce
a=ring_allreduce(4,16*1024**2,8*1024**3,10e-6)
b=ring_allreduce(4,16*1024**2,1*1024**3,10e-6)
assert abs(a*1000-2.9896875)<1e-12
assert abs(b*1000-23.4975)<1e-12
print(a*1000,b*1000)
```

## 8. Routing theo việc thay vì đếm request

Round-robin loads[2,18];greedy estimated work[11,9]. LLM cần ước lượng prompt/output work,active sequences,KV/chính sách batch; output length và interference khó biết chính xác. Router lab chỉ chọn slot trống eligible đầu tiên, không dự đoán độ dài hoặc cài cost-aware queue.

```python
costs=[1,9,1,9]
rr=[sum(costs[i::2]) for i in range(2)]
greedy=[0,0];assignment=[]
for cost in costs:
    i=min(range(2),key=lambda i:(greedy[i],i))
    greedy[i]+=cost;assignment.append(i)
assert rr==[2,18] and greedy==[11,9]
assert max(rr)==18 and max(greedy)==11
print(rr,greedy,assignment)
```

## 9. Timeout, cancel và slot vật lý

Client record có thể terminal nhưng worker còn hold/compute. Vì vậy submit vào slot đó bịOVERLOADEDcho đến release/reply/death. Kết quả đến muộn không được đổiCANCELLED/TIMEOUTthànhOK. Fixture không chứng minh kernel được preempt; nó chứng minh coordinator không coi caller bỏ chờ là tài nguyên đã thu hồi.

```python
from router import Router

def main():
    with Router(replicas=1) as r:
        job=r.submit('held',[[1,2,0,-1]],hold=True)
        r.until(lambda:job['started']);r.cancel('held')
        assert job['status']=='CANCELLED' and r.slots[0].busy=='held'
        try:r.submit('next',[[1,2,0,-1]])
        except RuntimeError as e:assert str(e)=='OVERLOADED'
        else:raise AssertionError('physical slot overbooked')
        r.release(0);r.until(lambda:r.slots[0].busy is None)
        assert job['status']=='CANCELLED' and job['publications']==1 and r.counters['late']==1
        print('cancel kept slot until reply')

if __name__=='__main__':
    main()
```

## 10. Deadline qua nhiều host

Còn500−180−20=300ms. Raw monotonic origin khác host không dùng như đồng hồ toàn cục; cần deadline propagation/remaining timeout theo hợp đồng transport. Retry cần budget còn lại,identity input/model/attempt, giới hạn số lần và cách kết thúc attempt cũ/partial stream. Lab không tự retry hoặc bảo đảm exactly-once qua coordinator crash.

## 11. Readiness, drain và revision

Chưa load/warm-up/đúngrevision thì chưa ready dù process alive. Drain loại khỏi admission mới, đợi việc cũ hoàn tất hoặc xử lý theo budget. StageV1/V2cùng shape có thể cùng “type” nhưng weights khác hàm, không được trộn. Epoch đánh dấu lần worker/group được tạo lại;revision đánh dấu model/artifact. Restart cùng model có epoch mới nhưng revision có thể giữ nguyên như lab.

## 12. Failure domain và availability

Group cần cả4rank:.99⁴≈.96059601. Hai full replica độc lập chỉ cần một sống:1−.01²=.9999. Independence không đúng khi dùng chung host/power/network/control plane. Lab kiểm process fail-stop ở một full-model replica cục bộ và routing sang replica còn sống, chưa lỗi host hoặc group TP/PP thực bị partition.

```python
from math import isclose
a=.99
group=a**4
replicas=1-(1-a)**2
assert isclose(group,.96059601) and isclose(replicas,.9999)
assert replicas>a>group
print(group,replicas)
```

## 13. KV transfer khi tách prefill/decode

Cận dưới62.5ms. Chưa serialization/handshake/layout conversion/allocation/queue/contention; chưa tính khả năng overlap. Cần cùng prefix/model/positions/KVdtype/layout và quyền truy cập. H16bytecủaPPMLPkhông là KV per-layer/head/token để dùng qua nhiều decode step.

```python
kv_bytes=512*1024**2
bandwidth_bytes_s=8*1024**3
seconds=kv_bytes/bandwidth_bytes_s
assert seconds==.0625
assert 4*4==16  # H của một microbatch một hàng, không phải KV512MiB
print(seconds*1000,'ms, hypothetical lower bound')
```

## 14. Đọc report

Fullweights120byte,replica child giữ240bytecộng;TP56+56+8bias=120;PP80+40=120. Parent vẫn giữ model nguồn;không làRSS. Maxerror0trên số nhỏ/happy path,random oraclecó tolerance khác. Snapshot10requestcó6OKvàbốn outcome lỗi/cancel;late2,stale1tiêm tổng hợp.63predicategồm32math/shapevà31process/lifecycle;không là63loại networkfault,không cóGPUbenchmark.

## 15. Thiết kế kiểm trước deployment

Khóa checkpoint/tokenizer/template/quant format/layout;kiểm group đủrank và readiness/correctness trên input nhỏ vớibaseline. Tạo workload prompt/output đại diện,quality gate,slices vàTTFT/ITL/goodputSLO. Đo per-rank memory/loadpeak/collective wait/transfer và lỗi mất mộtrank/node;theo dõiattempt/cancel/cleanup. Rollback khiquality/correctness hoặcSLO vượt tiêu chí đã định, giữ identity phù hợp với request đang stream. Happy path không bộc lộ deadlock,overload,revision mix hoặc correlated failure.

## 16. Tăng tài nguyên hay sửa nút thắt?

Giả thuyết1:queue/routing lệch tải;đo queuework/token,admission wait,cache affinity vàslot thực còn chạy. Giả thuyết2:TPcommunication/stage imbalance;đo collective/recv wait trên critical path,stage service time,link bandwidthvàpayload. Giả thuyết3:prefill dài/KV pressure làm decode giật;đo TTFT/ITLtheo length,scheduler events,KV usage/transfer/recompute.

Mỗi thay đổi cầnso cùng workload/quality/SLO;thêm replica có thể giúp1nhưng không sửa mộtcollective đang chậm hoặc mộtgroup thiếu memory đúnglayout. Những phép đo này là đề xuất tiếp nối, chưa là kết quả lab đã thực hiện.
