# Lời giải — Embedding và biểu diễn

[Về bài học](README.md) · [Về đề bài](exercises.md)

## Bài 1

Không: ID là khóa danh mục, thứ tự đánh số không tạo quan hệ nghĩa. Hai one-hot khác nhau có dot0,cosine0,norm mỗi vector1,L2=√2. Token có thể là word/subword/character hoặc ID đối tượng. Tokenizer lower+whitespace không bỏ dấu phẩy, nên `login,` khác `login`; trong vocabulary lab nó có thể thành UNK.

## Bài 2

Lookup=[[1,0],[0,2],[1,0],[0,0]]. One-hot ID3=[0,0,0,1,0], nhân E ra[0,2]. Input IDs(7,6),one-hot(7,6,100),embedding(7,6,8). Bảng có100×8=800 parameter. Linear100→8 không bias có800 weight, cùng số; ưu điểm lookup ở đây là tránh dựng one-hot dense và nhân như phép tổng quát. Shape weight Linear là(8,100), bảng E là(100,8).

## Bài 3

Tổng[2,2],3 token thật nên masked mean=[2/3,2/3]; naive=[.5,.5]. Thêm3 PAD, masked vẫn[2/3,2/3], naive thành[2/7,2/7]. padding_idx không biết bạn muốn chia cho token thật hay toàn chiều tensor; cần mask và denominator. UNK là token thật nên vẫn được tính, dù vector0 trong lab. All-PAD/text rỗng bị chặn; không tự dự đoán từ một mẫu không có nội dung.

## Bài 4

L=4/9,gradient p=[2/3,2/3]. Mỗi occurrence thật có gradient[2/9,2/9]. Hàng2 có hai occurrence nên[4/9,4/9]; hàng3=[2/9,2/9]. Hàng2 mới=[43/45,−2/45],hàng3=[−1/45,89/45],p mới=[17/27,17/27],loss≈.396433470508. ID nguyên không nhận gradient để đổi giá trị; embedding weight nhận gradient. Hàng vắng có thể vẫn bị optimizer state/decay/objective khác tác động; kết luận không đổi chỉ được kiểm trong các điều kiện cụ thể. PAD gradient qua lookup bị chặn không tự là quy tắc bảo vệ trước mọi phép sửa weight bên ngoài.

## Bài 5

Dot a=1,b=10;cos a=1,b=1/√2≈.707106781;L2 a=0,b=√181≈13.453624047. Dot ưu tiên b,cosine ưu tiên a. Sau normalize,b̂=[1/√2,1/√2],q̂=q;d²=2−√2≈.585786438=2−2cos. Không dùng đẳng thức này cho vector chưa normalize. Zero vector không có hướng nên cosine toán học không xác định; epsilon trong API chỉ xử lý chia0 về mặt số học, không tạo nghĩa cho vector0.

## Bài 6

Static lookup của cùng ID trong cùng trạng thái model giống nhau dù ngữ cảnh khác. Static khác freeze: bảng static vẫn có thể học trong train. Contextual output phụ thuộc ngữ cảnh encoder cho phép đọc. Mean pooling cùng token multiset cho cùng vector nên hai câu ví dụ không phân biệt được. Vector sentence/retrieval cần cách encode/objective và đánh giá phù hợp, không tự có chỉ nhờ lấy mean. Lab dùng embedding theo nhãn topic, chưa đánh giá retrieval hoặc huấn luyện contextual encoder.

## Bài 7

Vocab={<pad>:0,<unk>:1,error:2,invoice:3,login:4}; validation IDs=[1,4]. Mapping không khớp sẽ lấy vector token khác, lỗi ngữ nghĩa dù đúng tensor shape. Khi toàn OOV, pooled0 và logits=head.bias; softmax vẫn trả một phân bố nhưng không chứng minh hiểu nội dung. Cần giữ tokenizer/mapping/special IDs cùng model, theo dõi OOV và cân nhắc dữ liệu phục vụ phù hợp.

## Bài 8

Q trực giao có QQᵀ=I, nên(pQ)(WQ)ᵀ=pQQᵀWᵀ=pWᵀ. Với bias giữ nguyên thì logits giữ nguyên; cosine cũng giữ. Tọa độ riêng có thể thay mà chức năng không đổi nên không tự gán mỗi chiều một nhãn ngữ nghĩa. Gần cosine chỉ thể hiện geometry đã học theo objective/corpus; `issue` có thể mang tín hiệu tác vụ, không đồng nghĩa `login`. Projection2D có thể làm méo khoảng cách; kiểm trong đủ số chiều thay vì chỉ nhìn hình.

## Bài 9

Mỗi template27 sample, các tập108/54/54. E có84 số,head4×3+3=15,tổng99. Batch18 train108 có6 update/epoch,60 epoch cho360. Rule đã biết keyword cũng giải được validation100%, nên không chứng minh cần model phức tạp hoặc embedding vượt rule. Chưa là benchmark ticket thật. Một head trực tiếp trên bag-of-token21 chiều có21×3+3=66 parameter, còn ít hơn99 trong ví dụ này; số chiều thấp của vector không tự bảo đảm tổng model nhỏ hơn.

## Bài 10

Báo cáo mẫu: Corpus synthetic216 ticket thuộc auth/billing/network được sinh từ8 template,9 keyword và3 service. Split theo template4/2/2 cho108/54/54 mẫu; keyword cố ý đã có ở train. Tokenizer lower+whitespace fit vocabulary train-only21 ID gồm PAD0,UNK1. Tỷ lệ OOV occurrence của validation≈55.56%,test≈63.64%,chủ yếu từ nền template giữ riêng. Model Embedding21×4→masked mean→Linear4×3 có99 parameter,Adam lr=.03,batch18,seed2026,60 epoch tương ứng360 update. Chọn epoch60 theo validation CE≈.004514543;test sau lựa chọn CE≈.012118222,accuracy100%,confusion diagonal18 mỗi lớp. Majority validation≈33.33%,keyword rule100%. Hàng login có cosine gần token/password và cả issue nên classification tốt chưa chứng minh representation là ngữ nghĩa tổng quát hoặc phù hợp retrieval. Đây là một seed/dữ liệu dễ,không đại diện ticket thật hay OOV ở keyword quan trọng.

Inference cần code/architecture,weight,vocab,tokenizer version,special IDs,class mapping và chính sách đầu vào. best.pt chưa có optimizer/RNG/epoch-state đầy đủ để resume. PAD và UNK không nên gộp vì PAD cần bỏ khỏi pooling,còn UNK là một token thật; cùng giá trị vector không tạo cùng vai trò.

## Thực hành mở rộng — Code đối chiếu

Từ thư mục bài16,dùng môi trường đã cài. Output trong thư mục tạm; không chấm test để lựa chọn:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import torch
from lab import (make_records,build_vocab,encode,TicketClassifier,TicketDataset,
                 loader,evaluate,train,Config,setup,masked_mean)

# Sai phân số trên bảng toy, chỉ hai hàng không phải PAD.
E = torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,2.],[1.,1.]],dtype=torch.float64)
ids = torch.tensor([[2,3,2,0]])
def loss(table):
    return (.5*masked_mean(table[ids],ids).square().sum()).item()
for row,answer in [(2,4/9),(3,2/9)]:
    for column in range(2):
        high=E.clone();low=E.clone();high[row,column]+=1e-6;low[row,column]-=1e-6
        assert abs((loss(high)-loss(low))/2e-6-answer)<1e-9

setup();records=make_records();vocab=build_vocab(records)
model=TicketClassifier(len(vocab));model.eval()
ids=encode('login invoice login',vocab)[None]
with torch.no_grad():
    reference=model(ids)
    padded=torch.nn.functional.pad(ids,(0,5),value=0)
    torch.testing.assert_close(model(padded),reference,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model(ids.flip(1)),reference,atol=1e-6,rtol=1e-6)
    # Hoán vị hai ID được dùng và cùng hoán vị hàng; PAD0 được giữ nguyên.
    permutation=torch.arange(len(vocab))
    i,j=vocab['login'],vocab['invoice'];permutation[i]=j;permutation[j]=i
    assert not torch.allclose(model(permutation[ids]),reference,atol=1e-6,rtol=1e-6)
    changed=copy.deepcopy(model)
    changed.embedding.weight[permutation]=model.embedding.weight.clone()
    torch.testing.assert_close(changed(permutation[ids]),reference,atol=1e-6,rtol=1e-6)

validation=TicketDataset(records,'validation',vocab)
a=evaluate(model,loader(validation,20));b=evaluate(model,loader(validation,54))
assert abs(a['loss']-b['loss'])<1e-6 and a['confusion']==b['confusion']
with TemporaryDirectory() as directory:
    report=train(Path(directory)/'run',Config(epochs=2,batch_size=20))
    assert report['global_step']==12 and report['prediction_roundtrip']
print('Extension checks passed')
```

Train108/batch20 có6 batch kích thước20,20,20,20,20,8,mỗi epoch6 update,hai epoch12 update. Validation54 chia20,20,14. Dung sai float32 chấp nhận khác biệt nhỏ do thứ tự cộng; count/confusion vẫn phải nhất quán. Hoán vị đầy đủ ID+hàng không đổi chức năng nhưng phải lưu mapping mới nếu dùng tiếp.
