# Bài tập — Neuron, activation và mạng nhiều lớp

[Về bài học](README.md) · [Lời giải](solutions.md)

Tính tay trước rồi chạy code từ thư mục bài 9 với NumPy. Mọi tham số/dữ liệu là ví dụ giảng dạy. Mỗi mẫu một hàng, W=(input, output), bias 1-D; ngưỡng nhãn binary là p≥0.5. Những phần ghi mở rộng có thể quay lại sau.

## Bài 1. Một neuron và bốn loại giá trị

Cho x=(2, −1), w=(0.5, −2), b=−0.5.

1. Tính z, activation identity/ReLU/sigmoid/tanh.
2. Nếu đây là head binary, giá trị nào là logit, xác suất và nhãn ở ngưỡng 0.5?
3. Nếu đây là hidden neuron dùng ReLU, có được gọi a=2.5 là xác suất 250% không?
4. Thay input bằng (0, 0), giữ tham số: bias ảnh hưởng ra sao? Thay b=0 thì sao?

## Bài 2. Đọc activation và độ dốc

Với z=`[-2,-1,0,1,2]`:

1. Tính ReLU và Leaky ReLU α=0.1 bằng tay.
2. Tính sigmoid(0), tanh(0), đạo hàm của hai hàm ở 0; so với sigmoid'(10) xấp xỉ bằng máy.
3. Vì sao ReLU tuyến tính từng đoạn nhưng không phải một hàm affine trên toàn trục? Nó có đạo hàm cổ điển ở 0 không?
4. Một neuron ra ReLU=0 cho một input có chắc đã “chết” với mọi input không?
5. **Mở rộng:**  biết Φ(−1)≈0.158655, Φ(1)≈0.841345, tính GELU(−1), GELU(1). Vì sao output không phải xác suất?

```python
import numpy as np
from lab import activate, sigmoid
z=np.array([-2.,-1.,0.,1.,2.])
for name in ('relu','leaky_relu','sigmoid','tanh','gelu'):
    print(name,activate(z,name))
p=float(sigmoid(10.))
print('sigmoid derivative at 10:',p*(1-p))
```

## Bài 3. Shape và số tham số

Mạng dense 3→4→2→1, có ReLU ở hai hidden, output identity, mỗi lớp có bias. Batch n=5.

1. Ghi shape từng W, b, Z, A và output cuối.
2. Đếm tổng tham số. Nếu batch tăng từ 5 lên 50 thì số tham số là bao nhiêu?
3. Nếu bỏ tất cả bias, còn bao nhiêu tham số?
4. Một input là vector 3 phần tử; chuyển thành shape nào để gọi `forward` trong lab?
5. Nếu chuyển W của lab sang weight `torch.nn.Linear`, shape lưu weight từng lớp sẽ khác thế nào? Chỉ đối chiếu quy ước, chưa cần cài PyTorch.

## Bài 4. Forward một batch

Dùng X, W¹, b¹, W², b² trong `hand_example()` của lab.

1. Tính từng ô Z¹, H sau ReLU, Z² cuối.
2. Tính sigmoid, nhãn và BCE với y=`[1,0]`.
3. Đếm tham số và giải thích vì sao bias lớp 1 được dùng chung qua hai hàng.
4. Chỉ lấy hàng thứ hai để chạy forward riêng: kết quả có đổi không? Trong mạng chỉ gồm dense và activation từng phần tử này, vì sao một mẫu không cần nhìn mẫu còn lại?

```python
from lab import hand_example, forward, bce_logits
import numpy as np
X,layers=hand_example()
logits,trace=forward(X,layers)
print(trace[0]['Z'],trace[0]['A'],logits,sep='\n')
print(bce_logits(np.array([1.,0.]),logits[:,0]))
print(forward(X[1:2],layers)[0])
```

## Bài 5. Gộp affine và vai trò phi tuyến

1. Thay activation hidden của `hand_example` bằng identity; tính W_eq, b_eq và output hai hàng.
2. So với có ReLU, hàng nào đổi? Vì sao hàng kia không đổi?
3. Một chuỗi 10 lớp affine rồi sigmoid cuối có tự giải được XOR trên feature gốc không? Giải thích bằng phép gộp.
4. **Mở rộng:**  kiến trúc affine 2→1→2 cho W_eq=W¹W². Vì sao không thể có W_eq là ma trận đơn vị 2×2? Không cần dùng thuật ngữ rank nếu chưa quen.

```python
from lab import hand_example, collapse_affine, dense, forward
X,layers=hand_example()
affine=[(W,b,'identity') for W,b,_ in layers]
W,b=collapse_affine(affine)
print(W,b,dense(X,W,b),sep='\n')
print(forward(X,layers)[0])
```

## Bài 6. XOR với hai ReLU

Cho h₁=ReLU(x₁−x₂), h₂=ReLU(x₂−x₁), z=4h₁+4h₂−2.

1. Tính H, z, p, nhãn cho bốn góc XOR.
2. Viết W¹, b¹, W², b² theo quy ước hàng của lab; đếm tham số.
3. Hai điểm nào bị đưa về cùng H? Việc đó có gây mất thông tin cần thiết để phân loại bốn góc không?
4. Trong không gian hidden, biên z=0 là gì? Trong X, nó là những đường nào?
5. Với x=(0.6, 0.4), tính p/nhãn. Có được khẳng định đó là nhãn XOR “thật” chỉ từ bảng bốn góc không?
6. Code có huấn luyện các weight đặt tay này không?

Chạy `python3 lab.py --mode xor` để tự đối chiếu.

## Bài 7. Head theo nhiệm vụ và lỗi sigmoid hai lần

1. Đề xuất raw output/biến đổi/loss cho: dự đoán latency; một nhóm routing trong 3 nhóm; nhiều chủ đề ticket đồng thời; ưu tiên có/không.
2. Vì sao dùng softmax cho nhiều nhãn đồng thời có thể áp đặt sai giả định?
3. Với logit z=0, tính sigmoid(z), rồi sigmoid(sigmoid(z)). Chúng có cùng nghĩa không?
4. Logits một mẫu là `[0,ln2,ln3]`, tính softmax. Thêm1000 vào mọi logit có đổi xác suất lý thuyết không?
5. Với head binary giữ shape `(n,1)` và nhãn `(n,)`, phép trừ trực tiếp có thể tạo shape nào? Lab lấy 1-D bằng thao tác gì?

## Bài 8. Học head trên feature cố định

Chạy `python3 lab.py --mode readout`.

1. Tham số nào được fit? Tham số nào giữ nguyên? Hai model dùng feature gì?
2. Với head trên H XOR, khởi tạo w=(0, 0), b=0, λ=0.01: tính BCE, objective, dw, db đầu tiên. Cập nhật một bước lr=0.2.
3. Vì sao head logistic trên raw X khởi đầu w=b=0 không di chuyển trong ví dụ đối xứng? Có nghĩa mọi bài logistic đều phải tránh khởi tạo 0 không?
4. Đối chiếu BCE và objective sau 3000 bước. Vì sao không gọi objective có penalty là BCE?
5. Accuracy train 100% trên bốn hàng có đủ báo mạng đã học biểu diễn XOR end-to-end hoặc dự đoán đúng dữ liệu nhiễu chưa?

```python
from lab import xor_data, xor_features, readout_objective
import numpy as np
X,y=xor_data()
H=xor_features(X)
loss,obj,dw,db=readout_objective(H,y,np.zeros(2),0.,l2=.01)
print(loss,obj,dw,db)
print(-.2*dw,-.2*db)
```

## Bài 9. Forward, gradient và cập nhật — cầu nối bài 10

1. Xếp đúng thứ tự: cập nhật tham số; tính dự đoán; tính loss; tính gradient. Sau cập nhật cần chạy bước nào để biết dự đoán mới?
2. Nếu chỉ cập nhật W², b² thì có căn cứ gọi W¹, b¹ đã học không? Một trace chứa Z, A có tự là autograd không?
3. Hai hidden neuron có weight/bias và kết nối đối xứng giống hệt nhau; gradient descent có thể giữ chúng giống nhau thế nào? Phân biệt điều này với việc khởi tạo head logistic bằng 0 trên feature đã cố định.
4. Chọn hidden width, activation, số bước bằng test accuracy qua hàng chục lần thử có còn giữ test độc lập không?

## Bài 10. Rà lỗi và viết báo cáo

Một người viết: “Mạng 2→2→1 có 9 tham số nên batch 100 phải có 900 tham số; thêm 5 lớp linear là đủ học mọi phi tuyến; hidden sigmoid là xác suất của từng nhãn; XOR 100% trên bốn hàng chứng minh tốt hơn cây và production sẵn sàng.”

1. Sửa từng lập luận bằng giải thích cụ thể.
2. Thử các input sai shape/bias và NaN với `dense`; vì sao báo lỗi sớm hữu ích hơn để broadcast tùy ý?
3. Viết báo cáo 5–8 câu cho thí nghiệm readout: dữ liệu, phần cố định/được fit, mục tiêu, cấu hình, kết quả và giới hạn. Nêu rõ việc cần làm trước khi kết luận chất lượng ngoài mẫu.

```python
import numpy as np
from lab import dense
X=np.ones((2,2)); W=np.ones((2,3))
bad_cases=[(X,W,np.zeros((2,1))), (np.ones(2),W,np.zeros(3)),
           (np.array([[np.nan,1.]]),W,np.zeros(3))]
for inputs in bad_cases:
    try:
        dense(*inputs)
    except ValueError as error:
        print(type(error).__name__,str(error))
```

## Tiêu chí tự đánh giá

Làm được forward và đếm tham số; phân biệt hidden activation với head probability; giải thích affine collapse và vai trò biểu diễn; biết chính xác code đã huấn luyện gì. Phần GELU/rank/symmetry là mở rộng. [Đối chiếu lời giải](solutions.md).
