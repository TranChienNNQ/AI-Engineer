# Bài tập — Chain rule, graph và backpropagation

[Về bài học](README.md) · [Lời giải](solutions.md)

Tất cả dữ liệu là giả định. Mỗi mẫu một hàng; nhãn binary là vector(n, ), logit là(n, 1). Mặc định mean BCE, ReLU hidden, L2=0 trừ khi ghi khác. Tính tay trước rồi chạy code từ thư mục bài 10 với NumPy.

## Bài 1. Một chuỗi ba bước

Cho u=2x−3, v=u³, L=v tại x=2.

1. Tính u, v, L và các đạo hàm cục bộ.
2. Tính dL/dx bằng chain rule rồi đối chiếu đạo hàm trực tiếp của (2x−3)³.
3. Đổi x=1: đạo hàm dL/dx có âm chỉ vì u âm không? Tính để kiểm tra.
4. Giải thích vì sao chỉ viết du/dx=2 chưa đủ gọi đó là gradient của L.

## Bài 2. Cộng gradient từ các nhánh

u=wx, a=u+b, v=a², r=3a, L=v+r. Khởi đầu x=2, w=0.5, b=1.

1. Vẽ graph và điền giá trị từng node.
2. Tính hai đóng góp về a rồi dw, db, dx.
3. Nếu chỉ giữ nhánh a² hoặc chỉ giữ nhánh 3a thì dw sai thành bao nhiêu?
4. Đổi x=1, w=1, b=−1: a=0 có khiến toàn bộ dL/da=0 không? Tính dw, db, dx ở điểm mới.
5. Vì sao dùng `a*a` phải xét cả hai vị trí dùng a khi tính ngược?

```python
from lab import scalar_example
print(scalar_example())
print(scalar_example(x=1.,w=1.,b=-1.))
```

## Bài 3. Local derivative không thay upstream gradient

Phép toán z=ab+c có a=−2, b=3, c=1. Giả sử phần còn lại của graph đã trả về ∂L/∂z=4.

1. Tính z và ∂L/∂a, ∂L/∂b, ∂L/∂c.
2. Nếu upstream thay thành −0.5, các gradient mới là gì?
3. Với output ReLU(z), khi z=−5 và upstream từ phía sau bằng 4, gradient về z qua đường ấy là bao nhiêu?
4. Khi ReLU output bằng 0, có được bỏ luôn việc cộng gradient từ một nhánh khác dùng chung z không?

## Bài 4. Backward một affine batch

```text
X = [[1,2],[3,4]]
W = [[1,-1],[2,0]]
b = [0.5,-0.5]
G = dL/dZ = [[1,2],[-1,3]]
```

1. Tính Z=XW+b.
2. Tính dW, db, dX; ghi shape từng kết quả.
3. Tự tính một ô dW bằng tổng theo từng mẫu để kiểm tra matmul.
4. Vì sao bias gradient phải sum theo hàng? Nếu G đã chứa hệ số mean loss thì có chia 2 lần nữa không?

```python
import numpy as np
X=np.array([[1.,2.],[3.,4.]])
W=np.array([[1.,-1.],[2.,0.]])
b=np.array([.5,-.5])
G=np.array([[1.,2.],[-1.,3.]])
print(X@W+b)
print(X.T@G,G.sum(axis=0),G@W.T,sep='\n')
```

## Bài 5. Backprop mạng bài 9

Dùng `hand_example()` trong lab. Đặt q=σ(−1.5).

1. Ôn forward Z₁, H, Z₂ và BCE với y=[1, 0].
2. Tính G₂, dW₂, db₂, G_H.
3. Tính ReLU mask, G₁, dW₁, db₁ và dX.
4. Vì sao db₂=0 chưa chứng minh output bias không học ở mọi batch?
5. Vì sao một ô G₁=0 ở hàng2, cột1 không làm mọi gradient của hidden neuron1 bằng 0?

```python
from lab import hand_example, loss_and_gradients
X,y,params=hand_example()
loss,obj,grads,trace=loss_and_gradients(X,y,params)
print(loss)
for name in ('dZ2','dH','dZ1','dX'):
    print(name,trace[name])
for name,gradient in grads.items():
    print(name,gradient)
```

## Bài 6. Một bước và nhánh regularization

1. Dùng gradient bài5, cập nhật đồng thời mọi parameter với lr=0.1. Tính hoặc chạy lại forward để đối chiếu loss mới.
2. Nếu cập nhật W₂ trước rồi dùng W₂ mới tính G_H, đang lấy gradient của cùng lần forward cũ không? Giải thích.
3. Giữ parameter **ban đầu**, thêm λ=0.1 phạt W₁, W₂. Tính penalty và objective; gradient mỗi W được cộng thêm gì? Gradient bias có đổi không?
4. ∂J/∂X có nhận trực tiếp hạng λX không? Vì sao?

```python
from lab import hand_example, loss_and_gradients, sgd_step
X,y,p=hand_example()
loss,_,g,_=loss_and_gradients(X,y,p)
new=sgd_step(p,g,.1)
print(loss,loss_and_gradients(X,y,new)[0])
print(loss_and_gradients(X,y,p,l2=.1)[:2])
```

## Bài 7. Mean, sum và microbatch không đều

1. Với hai mẫu tính tay, λ=0: tính sum BCE và quan hệ gradient sum/mean.
2. Với λ=0.1 giữ nguyên cho hai objective, sum objective có bằng 2×mean objective không? Tính để kiểm tra.
3. Viết quan hệ giữa G_sum và G_mean cho một weight W khi λ giữ nguyên; gradient bias thì sao?
4. Chia batch4 mẫu thành microbatch kích thước1 và3, có mean data loss lần lượt L₁, L₃. Mean data loss toàn batch là gì? Khi cùng parameter và λ=0, gradient toàn batch cần kết hợp hai gradient mean thế nào?
5. Phân biệt cộng đường trong một graph với vô tình giữ gradient của bước tối ưu trước.

Kiểm tra `python3 lab.py --mode reduction`.

## Bài 8. Kiểm gradient và kink — mở rộng

1. Mạng 2→2→1 có 9 parameter scalar. Sai phân trung tâm cần bao nhiêu lần đánh giá objective cho một epsilon, ngoài backward analytic?
2. Chạy check với L2=0.01, đọc cả max_abs và relative. Vì sao b₂ ReLU có relative khác thường nhưng vẫn có thể đúng?
3. Tính sai phân trung tâm f(x)=max(0, x) tại x=0 với ε>0. So với quy ước ReLU'(0)=0: đây có tự chứng minh backward sai không?
4. Nếu flag `crossed_relu_kink=True`, cần xem gì trước khi kết luận?
5. Vì sao kiểm accuracy bằng perturb parameter không là kiểm gradient BCE? Vì sao epsilon cực nhỏ không luôn tốt hơn?

```python
from lab import hand_example, gradient_check
X,y,p=hand_example()
for activation in ('relu','tanh'):
    for row in gradient_check(X,y,p,activation=activation,l2=.01):
        print(activation,row['name'],row['max_abs'],row['relative'],row['crossed_relu_kink'])
```

## Bài 9. End-to-end và khởi tạo toàn0

Chạy `python3 lab.py --mode train`.

1. Mạng có bao nhiêu parameter? Phần nào được fit khác bài 9?
2. Nêu seed, activation, learning rate, L2, số bước và kết quả train cuối. Có validation/test không?
3. Vì sao norm thay đổi của W₁, b₁ là bằng chứng chúng đã được cập nhật, nhưng chưa là bằng chứng biểu diễn tốt ngoài mẫu?
4. Khởi tạo toàn bộ parameter bằng 0 trên XOR cân bằng và dùng tanh. Tính H, P, G₂, dW₂, G_H, dW₁, db₁, db₂ ở bước đầu. Gradient có thể đứng yên ở điểm này dù có parameter khác làm loss thấp hơn không?

```python
import numpy as np
from lab import xor_data, init_params, loss_and_gradients
X,y=xor_data()
zero={key:np.zeros_like(value) for key,value in init_params().items()}
loss,obj,grads,trace=loss_and_gradients(X,y,zero,activation='tanh',l2=.001)
print(loss,obj)
for name,gradient in grads.items():
    print(name,gradient)
```

## Bài 10. Debug và báo cáo kết quả

Một ghi chú nói: “backprop tự chọn learning rate, gradient phải bằng loss, mọi backward cần xóa các đường cộng cũ ngay lập tức; mạng XOR 100% nên đủ bằng chứng production.”

1. Sửa từng lập luận, chỉ rõ gradient, optimizer và đánh giá có vai trò gì.
2. Autograd có phải thử mọi parameter±ε không? Lab này đã dùng autograd chưa?
3. Code tính dX có nghĩa phải cập nhật dữ liệu X không?
4. Viết báo cáo 5–8 câu cho train XOR: cấu hình, phần được fit, train loss/objective, kết quả và giới hạn; nêu bước tiếp theo để đánh giá trên dữ liệu thật.

## Tiêu chí tự đánh giá

Tính được gradient qua nhánh, affine batch và hidden activation; cập nhật sau khi backward hoàn tất; kiểm đúng objective và reduction; không biến điểm train thành kết luận ngoài mẫu. Gradient check ở kink và tích lũy microbatch là phần mở rộng. [Đối chiếu lời giải](solutions.md).
