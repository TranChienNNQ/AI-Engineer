# Lời giải bài 41 — AI đa phương thức

[← Bài tập](exercises.md) · [Bài chính](README.md) · [Phụ lục](deep-dive.md)

## 1. Thiết kế từ câu hỏi sản phẩm

Đọc mã lỗi có thể dùng OCR/field extraction; tìm tiếng máy tương tự cần audio representation phù hợp; trả lời về nội dung người dùng nói có thể đi qua ASR→LLM. OCR mất một phần layout/visual context, transcript mất non-speech/prosody và lỗi ASR truyền sang bước sau. Nếu câu hỏi đòi thông tin đó, cần giữ raw/feature và thiết kế task tương ứng. Nhiều input không buộc nhiều output: ảnh+audio có thể chỉ cho một câu trả lời text.

## 2. Tokenizer và representation văn bản

Hai lần đỏ, một lần dọc cho `[1/3,0,0,0,2/3]`. BoW giữ counts đã chuẩn hóa, không giữ vị trí. `đỏ dọc` và `dọc đỏ` có cùng vector. Tokenizer pretrained có vocabulary/special token/template/normalization riêng; đổi nó có thể làm encoder nhận sai ID/representation. Toàn OOV bị từ chối trong lab.

```python
import numpy as np
from preprocess import words,bow
v=['dọc','lam','lục','ngang','đỏ']
assert words('ĐỎ DỌC')==['đỏ','dọc']
assert np.allclose(bow('đỏ đỏ dọc',v),[1/3,0,0,0,2/3])
assert np.array_equal(bow('đỏ dọc',v),bow('dọc đỏ',v))
```

## 3. Patch và thứ tự phần tử

RGB4×4/P2 cho4patch12value; patch đầu gồm pixel0,1,4,5 theo HWC. RGB16×16/P4 cho16patch48value=768feature. Global mean có thể giữ màu nhưng không giữ hướng: vạch dọc và bản transpose của nó có cùng mean kênh.

```python
import numpy as np
from media import patchify
im=np.repeat(np.arange(16).reshape(4,4,1),3,axis=2)
p=patchify(im,2)
assert p.shape==(4,12)
assert p[0].tolist()==[0,0,0,1,1,1,4,4,4,5,5,5]
vertical=np.zeros((16,16,3));vertical[:,::2,0]=1
horizontal=vertical.transpose(1,0,2)
assert np.array_equal(vertical.mean(axis=(0,1)),horizontal.mean(axis=(0,1)))
```

## 4. Tính thời lượng và sample rate

800/8000=0,1s. Nếu diễn giải cùng samples ở16000Hz thì0,05s và tần số phát tương ứng tăng đôi. Resampling đúng phải tạo samples phù hợp timeline và xử lý alias khi cần; không chỉ thay metadata.

Dùng `read_wav('assets/red-vertical.wav')` nhận800float sample theo metadata đã kiểm. Wrapper yêu cầu mono, sample width2byte, rate8000 và PCM không nén, frame budget cho phép; rate mismatch bị từ chối. `lab.py --mode check` có WAV roundtrip/metadata/rate/budget tests. Hàm đọc chuẩn hóa bằng32767 theo hợp đồng writer riêng, không áp dtype này cho mọi WAV.

## 5. STFT và DFT tính tay

Frames=`1+floor((800−128)/64)=11`; bins65; Δf62,5Hz. Frame cuối bắt đầu640/kết thúc768, bỏ32sample. DFT x có `[0,−2i,0,+2i]`; rFFT giữ ba giá trị không âm tần số, power`[0,4,0]`. Ở8Hz, bin1=2Hz; ở8kHz, bin1=2kHz.

```python
import numpy as np
from media import stft_power
x=np.array([0.,1.,0.,-1.])
assert np.allclose(np.fft.rfft(x),[0,-2j,0])
power,freq=stft_power(np.zeros(800))
assert power.shape==(11,65) and freq[1]==62.5
assert 800-(640+128)==32
assert np.all(np.log1p(power)==0)
```

Hann/no-Hann là hai phép tính khác, nên không lấy DFT hand làm một frame STFT đã nhân window. `log1p` không thực hiện mel filterbank và không dùng công thức dB/reference. Độ phân giải tần số/thời gian và padding cần khớp processor của model thực.

## 6. Cosine và symmetric contrastive loss

Các vector đều unit. Cosine `[[1,0],[.6,.8]]`, logitsτ.5 cho `[[2,0],[1.2,1.6]]`. Row losses≈.126928/.513015; column losses≈.371101/.183901. Trung bình bốn loss≈.298736. Tính riêng row mean≈.319972 là chưa symmetric.

```python
import numpy as np
u=np.array([[1.,0.],[.6,.8]])
v=np.eye(2);scores=u@v.T/.5
row=np.log(np.exp(scores).sum(axis=1))-np.diag(scores)
col=np.log(np.exp(scores).sum(axis=0))-np.diag(scores)
loss=(row.mean()+col.mean())/2
assert np.allclose(row,[.126928011,.513015252])
assert abs(loss-.29873616756976046)<1e-12
```

Dimension chỉ là shape; hệ tọa độ/ngữ nghĩa cần alignment được học. Cosine là similarity, không xác suất. Softmax có denominator theo candidate set; thay gallery có thể đổi probability dù positive score giữ nguyên. Không tự dùng nó như confidence đã calibration.

## 7. Positive lặp

Hai row lớp0 có numerator2e và denominator2e+1. Row lớp1 có numeratore và denominatore+2. Vì ma trận đối xứng, loss hai chiều bằng nhau:

\[
\frac{2\ln((2e+1)/(2e))+\ln((e+2)/e)}3\approx0,29638.
\]

Diagonal CE thay numerator row0/1 thành e, cho≈.758478: nó coi positive khác ID là negative. Toàn batch cùng label thì numerator=denominator, loss0 và không có tín hiệu phân biệt lớp.

```python
import math
import torch
from model import positive_set_loss
x=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=torch.float64)
labels=torch.tensor([0,0,1])
expected=(2*math.log((2*math.e+1)/(2*math.e))+math.log((math.e+2)/math.e))/3
assert abs(float(positive_set_loss(x,x,labels,1))-expected)<1e-12
assert abs(float(positive_set_loss(x,x,torch.zeros(3,dtype=torch.long),1)))<1e-12
```

Positive-set mass không buộc mọi positive có probability đều nhau. Nó khác mean log-probability trên positive; tên objective và supervision phải ghi rõ. Bộ dữ liệu lab dùng label màu/hướng làm positive mask, nên đây là supervised learning.

## 8. Fusion và nguồn thiếu

Candidate0=`.75*.9+.25*.2=.725`; candidate1=`.75*.4+.25*.8=.5`, chọn0. Ảnh missing chỉ giữ audio, chọn1. Khi cả hai thiếu, không có evidence để tính score nên từ chối trong API lab.

```python
import numpy as np
from metrics import late_fuse
image=np.array([[.9,.4]]);audio=np.array([[.2,.8]])
assert np.allclose(late_fuse(image,audio,.75),[[.725,.5]])
assert late_fuse(None,audio).argmax()==1
try:late_fuse()
except ValueError:pass
else:raise AssertionError('missing modalities accepted')
```

Zero image/audio vẫn có shape hợp lệ; encoder có bias và normalizer nên tạo một embedding và closed-set argmax. None là trạng thái không dùng nhánh. Nếu nguồn chất lượng thấp, cần policy riêng; code hiện tại không học quality gate hoặc abstention threshold.

## 9. Pairing và timestamp

640/8000=.08s; cộng clip offset2,5 cho2,58s. Đây là frame-start time, không tự là timestamp phát hiện một từ hoặc một âm cụ thể. Interval đầu frame dài128/8000=.016s theo quy ước, còn annotation sự kiện cần kiểm riêng.

Tensor shape không chứa ý nghĩa event/time. Giữ source/recording/session/event ID, timestamp/timebase/offset, sample rate, image region, file/preprocess/model version, quyền truy cập và mapping derivative→raw source. Ảnh trước/audio sau có thể là context khác nhau hợp lệ nhưng không nên gộp như cùng hiện trạng mà không nói rõ.

## 10. Split và lineage

Mỗi lớp train16/validation4/test4group, nhân6 cho96/24/24record. Các modality của cùng group luôn cùng split. Caption/concept giống train; test chỉ giữ biến thể instance mới từ cùng generator, không chứng minh unseen-class hoặc zero-shot.

```python
from data import dataset
from preprocess import fit
s=dataset();ids={k:{r['id'] for r in v} for k,v in s.items()}
assert {k:len(v) for k,v in s.items()}=={'train':96,'validation':24,'test':24}
assert not(ids['train']&ids['test']) and not(ids['train']&ids['validation'])
prep=fit(s['train'])
assert set(prep['fit_ids'])==ids['train']
```

Crop/chunk rồi split từng file có thể đưa các bản gần trùng của cùng recording/cảnh sang nhiều tập, làm score lạc quan. Split theo đơn vị độc lập trước augmentation/chunk; lưu group lineage để kiểm. Mean/std/vocab fit train; không refit lên test chỉ vì format media khác.

## 11. Metric retrieval

Top1candidate index1 có label1, nên class-match đúng. Pair-ID metric có thể yêu cầu một recording cụ thể, nhưng dữ liệu hai caption cùng lớp không chứa đủ thông tin để phân biệt record ID. Không đổi metric rồi phạt positive tương đương mà không xem nhiệm vụ.

Chọn ngẫu nhiên độc lập trên gallery cân bằng có expected accuracy1/6; random encoder không phải bộ chọn ngẫu nhiên độc lập uniform từng query, và số mẫu24 còn nhỏ. Vì vậy random-init có thể0/24hoặc2/24 mà không mâu thuẫn expectation. Rule biết generator đạt24/24 cho biết nhiệm vụ đơn giản; không chứng minh neural model vượt baseline chỉ vì hơn random.

## 12. Gradient và save/load

Bundle gồm spec, state, state hash, preprocess/vocab/train-fit IDs/stats, preprocess hash và data hash. `load_bundle` kiểm spec/digest/key/shape/dtype/finite trước dùng model. Kiểmgradient yêu cầu tất cả parameter có gradient hữu hạn và từng encoder có tổng norm gradient dương; check update xác nhận state thật đổi. Inference save/load cho embedding exact ở fixture này.

```python
import tempfile
from pathlib import Path
from data import dataset,SPEC,fingerprint
from preprocess import fit
from model import Align
from training import save_bundle,load_bundle
s=dataset();prep=fit(s['train']);model=Align(len(prep['vocab']))
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'model.pt';save_bundle(path,model,prep,fingerprint(s))
    restored,_,_=load_bundle(path)
    try:load_bundle(path,dict(SPEC,sample_rate=16000))
    except ValueError:pass
    else:raise AssertionError('wrong spec accepted')
```

Digest chứa trong bundle không chứng minh nguồn đáng tin: người sửa có thể tạo digest khác. Bundle không lưu optimizer/RNG/sampler nên không exact resume training. `weights_only=True` là lựa chọn loader, không tuyên bố một artifact bất kỳ từ bên ngoài luôn an toàn hay phù hợp ứng dụng.

## 13. Đọc thí nghiệm và fault fixture

Runall kiểm120updates, best epoch120 theo validation loss≈.020487, test loss≈.0265635, sáu hướng24/24. Rule image/audio cũng24/24; unimodal và matched fusion cùng accuracy100%, chưa có bằng chứng fusion cải thiện accuracy. Muốn đo lợi ích bổ sung cần dataset mà các nguồn có thông tin bổ sung, noise/missing và metric phù hợp.

Faultroll4 làm audio khác lớp ở24row, fusion chỉ3/24; black/silent mỗi loại4/24. Fixture cố ý chỉ đo lỗi ghép trong một generator, không đại diện mọi noise/corruption hoặc conflict tự nhiên. Không tune w trên test, không báo100% thành khả năng hiểu mọi ảnh/tiếng nói. Trước khi dùng ngoài miền cần data/evaluation mới.

## 14. Context và chi phí

Tổng40+16+11+3=70, dense pairs4900. Image tăng4lần thì64patch, tổng118, pairs13924, ratio13924/4900≈2,84163. Không tăng16lần vì các token khác giữ nguyên và đang tính trên tổng sequence.

Con số là ví dụ ghép sequence giả định. Processor/model có thể pool/downsample/chunk hoặc dùng cross-attention; token billing của API có quy tắc khác. Giữ nhãn giả định, đo context/latency/memory theo hệ thống đang dùng thay vì lấy patch/frame count thô làm hóa đơn.

## 15. Từ retrieval sang generation

VLM: ảnh→vision encoder→connector/projector→visual token kết hợp text token→conditional language decoder→answer. Cần objective và dữ liệu image-conditioned instruction/answer; projector không tự hiểu hình vì output dimension bằng word embedding. Lab chỉ học embedding similarity, không decoder/generation.

ASR pipeline: audio→speech recognizer→transcript+time/quality metadata→LLM. Nó có thể mất tiếng động máy hoặc cách nói mỉa mai/âm sắc nếu chỉ giữ text. Encoder audio trực tiếp cũng không tự bảo đảm giữ đúng thông tin đó; cần training/task/evaluation. Tone trong assets do công thức sin tạo, không là model-generated speech.

## 16. Kế hoạch đánh giá sản phẩm

Ví dụ: trả mã lỗi từ screenshot, tóm tắt lời người dùng và nêu đoạn audio chứa tiếng bíp. Annotation gồm field đúng, transcript/time intervals và vùng ảnh hỗ trợ claim. Split theo event/session/device/document trước crop/chunk; cần tập mới cho thiết bị/ngôn ngữ/noise ngoài train. Baseline có thể OCR+rule, ASR→text retrieval, unimodal và multimodal fusion.

Đo field accuracy, ASR WER/CER khi có annotation phù hợp, retrieval relevance/Recall@k, grounded claim accuracy và abstention; thêm slice ảnh mờ/chữ nhỏ/noise/missing/contradiction. Swap/ablation kiểm phụ thuộc nguồn, đồng thời kiểm region/time thực sự support claim. Đo decode/preprocess/encode/fusion/generation và request goodput/latency/cost cùng workload. Source permissions, preprocess/version/lineage theo bài25–26/33; không coi schema đúng hoặc citation tồn tại là claim đã đúng.
