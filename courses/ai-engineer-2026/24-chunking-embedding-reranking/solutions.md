# Bài 24 — Lời giải

[← Bài chính](README.md) · [Đề bài](exercises.md)

Các snippet chạy từ thư mục bài với môi trường `requirements.txt`. Các snippet dùng bundle cần chạy `lab.py --mode train --out /tmp/lesson24-run-01` trước; dùng thư mục mới nếu tên đó đã có. Số làm tròn chỉ để trình bày, không thay cho kiểm số thực có tolerance.

## 1. Đơn vị truy xuất

Cả trang có thể bị nén thành một vector chứa cả auth hết hạn và thiếu quyền, hoặc bước sau lấy sai phần của trang. Từng câu có thể mất tiền đề “token hợp lệ” hoặc đối tượng mà “quyền này” đang chỉ tới. Có thể chia theo section, thêm title/heading có kiểm budget, giữ vị trí nguồn và khi cần lấy thêm đoạn lân cận phù hợp quyền.

SourceHit chỉ xác nhận một đoạn từ đúng trang. Câu hỏi về 403 mà lấy đoạn 401 cùng trang vẫn không đủ bằng chứng trả lời. Chọn chunk phải căn theo nhu cầu, không chỉ theo file format.

## 2. Cửa sổ

L13: `[0,6),[4,10),[8,13)`, tổng17, tỷ lệ17/13≈1.3077. L10: `[0,6),[4,10)`, tổng12. Nếu sinh thêm `[8,10)`, đó là đuôi đã được phủ và không cần cho thuật toán này.

Chunk đầu phủ C, mỗi chunk tiếp theo tiến S=C−O; số bước bổ sung là trần của phần chưa phủ chia S. L=0 cho 0. O=C phải reject vì S=0. Overlap chỉ nên nằm từ 0 đến C−1.

```python
from retrieval import windows
assert windows(13,6,2) == [(0,6),(4,10),(8,13)]
assert sum(b-a for a,b in windows(13,6,2)) == 17
assert windows(10,6,2) == [(0,6),(4,10)]
try:
    windows(13,6,6)
except ValueError:
    print('Rejected zero step')
else:
    raise AssertionError('Invalid overlap was accepted')
```

## 3. Span

Không overlap: `[5,9)` đi qua biên6, không chunk nào chứa trọn. O=2: `[4,10)` chứa span. Span dài hơn C không thể nằm trọn trong một chunk C token, nhưng hệ thống khác vẫn có thể ghép nhiều chunk; lab chưa đo việc đó.

Offset là ký tự Python trong source NFC, không phải byte/token. Kiểm mọi span và chunk:

```python
from data import dataset
from retrieval import chunk_sources
sources, queries = dataset()
byid = {s['id']:s for s in sources}
for q in queries:
    e = q['evidence']
    if e:
        assert byid[e['source_id']]['text'][e['start']:e['end']] == e['quote']
for strategy,size,overlap in [('fixed',24,0),('fixed',24,8),('section',48,0)]:
    for c in chunk_sources(sources,strategy,size,overlap):
        assert c['text'] == byid[c['source_id']]['text'][c['start']:c['end']]
print('All canonical spans and chunk slices match')
```

## 4. Budget

Body budget99; body110 vượt11 token. Embedding/reranker/generator có thể dùng tokenizer, prefix, special token và limit khác nhau; phải đếm đúng đầu vào thực của từng bước. Query dài hơn cũng làm body budget cặp giảm.

Trong API tokenizer Hugging Face đã đọc, stride là số token overlap: nếu C=6,stride=2 thì bước tiến4. Công cụ khác có thể dùng stride để chỉ bước tiến. Không suy nghĩa chỉ từ tên biến.

## 5. Masked mean: normalize có thể che một số lỗi

Mean đúng `[1,1]`, normalize `[.707107,.707107]`. Nếu mask tử số đúng nhưng chia3, mean=`[2/3,2/3]`; normalize ngay lập tức vẫn cùng hướng. Nếu cộng cả PAD rồi chia3, mean=`[34,34]`, vẫn cùng hướng trong ví dụ đặc biệt này vì PAD `[100,100]` cùng hướng với tổng thật. Không vì vậy mà bỏ mask là đúng: đổi PAD thành `[100,0]` sẽ làm mean/hướng lệch, và projection/bias/tanh sau mean có thể phá tính bất biến theo scale.

Key padding mask chỉ ngăn token khác dùng PAD như key/value trong attention; output ở PAD query có thể vẫn khác 0. Pooling toàn bộ vị trí vẫn cần mask. Đây là invariant kiến trúc, không dựa vào việc embedding PAD ban đầu là 0.

```python
import torch
from torch.nn.functional import normalize
h = torch.tensor([[[2.,0.],[0.,2.],[100.,100.]]])
mask = torch.tensor([[1.,1.,0.]]).unsqueeze(-1)
mean = (h*mask).sum(1)/mask.sum(1)
torch.testing.assert_close(mean, torch.tensor([[1.,1.]]))
torch.testing.assert_close(normalize(mean,dim=-1), torch.tensor([[2**-.5,2**-.5]]))
h[0,2] = torch.tensor([100.,0.])
assert not torch.allclose(normalize(h.mean(1),dim=-1),normalize(mean,dim=-1))
print('PAD masking matters')
```

## 6. Loss và gradient

Logits `[4,0,−3]`, probabilities≈`[.981135,.017970,.000895]`, CE≈.019045. Gradient theo score `(p−y)/.2`≈`[-.094324,.089851,.004473]`. Trừ gradient làm positive tăng và negative giảm theo score nếu coi score là biến độc lập; cập nhật trọng số encoder còn đi qua các đạo hàm khác.

Triplet loss=.05. Lab train bằng CE trên cosine/temperature, không phải triplet. Kiểm autograd độc lập:

```python
import torch
s = torch.tensor([.8,0.,-.6], dtype=torch.float64, requires_grad=True)
loss = torch.nn.functional.cross_entropy((s/.2).unsqueeze(0),torch.tensor([0]))
loss.backward()
torch.testing.assert_close(loss,torch.tensor(.01904500787925745,dtype=torch.float64))
torch.testing.assert_close(s.grad,torch.tensor([-.094323987828413,.08985059034406032,.004473397484352668],dtype=torch.float64))
print(float(loss.detach()), s.grad)
```

## 7. Nhãn và negative

Hai query cùng positive không cần hai cột positive khác ID. Nếu duplicate cùng chunk thành hai cột rồi mỗi hàng chỉ nhận đường chéo, loss sẽ coi một bản sao đúng là negative. Lab dùng25 cột chunk duy nhất,36 target là column index tương ứng; cột target có thể lặp giữa các hàng.

Loss đa positive≈.094344, single positive≈.407606 với logits đã cho. Đổi tập positive làm đổi mục tiêu, không là một phép cải thiện model không tốn công. Hard negative có thể là false negative khi có câu trả lời tương đương chưa được gán nhãn, bản mới/cũ đều còn hợp lệ, hoặc query mơ hồ cho phép nhiều nhu cầu.

## 8. Giới hạn hai tầng

CandidateHit@3=1 trước/sau vì B trong candidate. EvidenceHit@1 trước0, sau1. RR@3 trước1/2, sau1. Nếu không có B hoặc chunk nào khác chứa span trong candidate, mọi hoán vị đều không có span: Hit/RR đều0.

Tập top-k sau rerank là tập con candidate. Nếu top-k có một chunk đúng, candidate chắc chắn đã có chunk đó; suy ra bất đẳng thức Hit@k≤CandidateHit@M theo từng query và theo trung bình cùng mẫu số. Chèn gold vào eval hoặc đổi tiêu chí nhãn sẽ phá điều kiện so sánh.

## 9. BCE và hiệu chỉnh

A label 0: `softplus(−.2)≈.5981388694`. B label 1: `softplus(−1.4)≈.2204174099`. Logit là số thực không bị chặn, nên1.4 hợp lệ. Sigmoid(1.4)≈.8022 là phép chuyển thang, chưa là bằng chứng calibration.

Với p=.2,w=5, r=`wp/(wp+1−p)=1/1.8=5/9≈.5556`. Weighted BCE đổi nghiệm lý tưởng; mining còn đổi phân bố cặp train, nên không chọn ngưỡng serving chỉ từ .5 hay công thức lý tưởng này.

## 10. Đọc ablation

| Cấu hình | Chunk | Token body | Coverage | Test SourceHit@3 |
|---|---:|---:|---:|---:|
| fixed24 | 40 | 874 | .5556 | 1 |
| overlap24_8 | 54 | 1,234 | .7222 | 1 |
| section48 | 25 | 874 | 1 | 1 |

Đúng parent không nói đoạn lấy về có trọn span. Bộ nhỏ có heading thuận lợi, nhãn chỉ một canonical span, các cấu hình đổi cả size và prefix heading, query nhiều term khớp trực tiếp, chưa có nguồn/intent mới. Những giới hạn đó ngăn kết luận section luôn thắng. Dữ liệu thực có section rất dài, bảng/code/parser lỗi có thể cho kết quả khác.

## 11. Tensor và thứ tự từ

Query `[36,Tq]` và doc `[25,Td]` → vectors `[36,32]`, `[25,32]` → scores `[36,25]` → CE target `[36]`. Cross-encoder nhận IDs/segment `[B,Tpair]`, attention hai chiều, pooling `[B,32]`, logits `[B]`, labels `[B]`; không dùng causal mask.

Bi-encoder word-mean không có position/contextual attention nên hai input cùng term counts có cùng vector trước/sau projection, kể cả đổi thứ tự. Kiểm chứng này là giới hạn đã biết của kiến trúc, không phải đánh giá hiểu ngôn ngữ tổng quát:

```python
import torch
from retrieval import BiEncoder
model = BiEncoder(vocab_size=10,dim=32).eval()
with torch.inference_mode():
    a = model(torch.tensor([[4,5,6,0]]))
    b = model(torch.tensor([[6,4,5,0]]))
torch.testing.assert_close(a,b)
print('Permutation invariant word-mean encoder')
```

## 12. Chẩn đoán và lựa chọn baseline

Run mặc định chọn bi epoch 40 (dev MRR≈.6574) và cross epoch 10 (≈.4630). Epoch cuối80/30 có train loss thấp hơn ban đầu nhưng không tự thay checkpoint đã chọn; đọc `best_epoch` và history, không đoán theo dòng log cuối.

Query test-06-1 (invalidation): hybrid chưa có span trong top-3, nhưng có trong top-6; rerank đưa lên hạng3. Query test-07-1 (connection pool): hybrid có span ở hạng1, rerank làm nó rơi khỏi top-3. Đây là cùng bộ candidate, nên lỗi ở thứ hạng cuối, không thể đổ cho thiếu chunk đúng trong candidate.

BM25 dev/test MRR≈.9352/1; hybrid≈.7778/.8889; rerank≈.4630/.3148. Không có cơ sở chọn reranker này để tăng chất lượng. Nó học được một số tín hiệu nhưng chưa vượt baseline. Các khả năng cần điều tra là ít dữ liệu, kiến trúc/mục tiêu, negative/nhãn; bài chưa làm thí nghiệm cô lập để kết luận nguyên nhân duy nhất.

## 13. Bundle và cache

Snippet tạo hai lần load, so tất cả query/scores, rồi sửa cache trong một bản sao tạm. Không sửa bundle gốc:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import numpy as np
import torch
from data import dataset
from retrieval import Pipeline

torch.set_num_threads(1)
path = Path('/tmp/lesson24-run-01/bundle')
a, b = Pipeline.load(path), Pipeline.load(path)
_, queries = dataset()
for q in queries:
    ids = a.retrieve(q['text'],'hybrid')
    assert ids == b.retrieve(q['text'],'hybrid')
    assert a.rerank(q['text'],ids) == b.rerank(q['text'],ids)
    np.testing.assert_allclose(a.pair_scores(q['text'],ids),b.pair_scores(q['text'],ids),atol=1e-6)
with TemporaryDirectory() as tmp:
    copy = Path(tmp)/'bundle'
    shutil.copytree(path,copy)
    vectors = np.load(copy/'vectors.npy',allow_pickle=False)
    vectors[0,0] += .5
    np.save(copy/'vectors.npy',vectors,allow_pickle=False)
    try:
        Pipeline.load(copy)
    except AssertionError:
        print('Detected mismatched vector cache')
    else:
        raise AssertionError('Corrupted vector cache accepted')
print('Query/rerank round-trips match')
```

Đổi reranker riêng không nhất thiết đổi vector bi-encoder, nhưng cần đổi rerank cache/version/eval; bundle demo gom hai model cùng file nên hash snapshot cũng đổi. Bundle chỉ đủ inference, thiếu optimizer/RNG/cursor cho resume chính xác. Hash kiểm nhầm file trong luồng tin cậy, chưa là chữ ký chống giả mạo.

## 14. Dữ liệu

Vocabulary từ public chunk + query train; restricted S10 không đi vào snapshot public. Miner dùng retriever được đóng băng sau chọn dev, nhưng chỉ dùng query/nhãn train để tạo cặp. Positive train bị miss sẽ được thêm chỉ vào tập pair train; eval không thêm gold. Run mặc định forced_train_positives=0.

Dev/test dùng để chọn checkpoint/đánh giá, không dùng trực tiếp gradient. Các split khác query text nhưng chung intent/source/span, nên chưa kiểm intent mới; nguồn đích ở corpus không tự là leakage. Chỉnh pipeline theo lỗi test rồi gọi test ấy là unseen mới là báo cáo sai giao thức. Cần test mới phù hợp tác vụ muốn suy rộng.

## 15. No-answer và context

Toàn OOV bị gate trước retrieval, là policy của model word-level, không là khả năng tự biết của neural model. “Cache trên sao Hỏa” có từ quen nên vẫn trả candidates; reranker phải xếp một danh sách dù không có câu đáp ứng. Lab chưa có abstention classifier hay ngưỡng đã chọn bằng nhãn no-answer.

Parent–child tìm child rồi mở rộng parent; overlap lặp một phần text trước encode từng chunk; late chunking contextualize token trên văn bản dài rồi mới pool theo span. Vector của chunk độc lập đã mất tương tác với token ngoài chunk, nên cộng/mean các vector ấy không khôi phục attention đã không diễn ra.

## 16. Sản phẩm đạt yêu cầu

Người khác cần chạy được corpus/query, hiểu canonical span/metric/split, đối chiếu được top-M/top-k, token overhead và các lỗi cụ thể. Thay chunker khi coverage thấp; thay retriever/M khi evidence có trong index nhưng candidate miss; điều tra rerank khi candidate hit nhưng final fail. Với câu nhiều bằng chứng, mở rộng nhãn/metric trước khi kết luận hệ thống đầy đủ.

Nếu ít nhãn, một báo cáo chunking + BM25 tái lập được và chỉ rõ giới hạn hữu ích hơn việc thêm neural model chưa có bằng chứng cải thiện. Nếu tiếp tục thử sau khi xem test, ghi vòng thử mới và giữ tập đánh giá độc lập cho báo cáo sau cùng.
