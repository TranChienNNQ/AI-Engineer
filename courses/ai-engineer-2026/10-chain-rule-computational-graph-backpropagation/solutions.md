# Lời giải — Chain rule, graph và backpropagation

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các số liệu được đối chiếu với code. Gradient viết dưới đây là đạo hàm của loss/objective đã nêu, không phải giá trị parameter mới nếu chưa áp bước cập nhật.

## Bài 1

Tại x=2: u=1, v=L=1. dL/dv=1, dv/du=3u²=3, du/dx=2, nên dL/dx=6. Đạo hàm trực tiếp của (2x−3)³ là 6(2x−3)², cũng cho 6.

Tại x=1, u=−1, L=−1 nhưng dL/dx=6. Dấu giá trị hàm/trung gian không tự quyết định dấu đạo hàm. du/dx chỉ mô tả đoạn x→u; gradient của L còn cần độ nhạy của các phần sau u.

## Bài 2

u=1, a=2, v=4, r=6, L=10. Nhánh bình phương góp 4, nhánh 3a góp 3, tổng da=7. Vì a=wx+b: dw=7×2=14, db=7, dx=7×0.5=3.5.

Chỉ nhánh a² cho dw=4×2=8; chỉ nhánh 3a cho dw=3×2=6. Không giữ được cả hai đường thì không còn gradient của L=v+r.

Ở x=1, w=1, b=−1: u=1, a=0, v=r=L=0. Nhánh bình phương góp 0 nhưng nhánh tuyến tính góp 3, nên da=3, dw=db=dx=3. `a*a` có hai vị trí đầu vào cùng phụ thuộc a, mỗi vị trí góp a vào dv/da, tổng 2a.

## Bài 3

z=(−2)×3+1=−5. Upstream g=4 cho da=4×3=12, db=4×(−2)=−8, dc=4. Nếu g=−0.5 thì da=−1.5, db=1, dc=−0.5.

Qua ReLU tại z=−5, local slope=0 nên upstream 4 bị chặn: gradient về z từ **đường ReLU ấy** là 0. Nếu z còn đi qua nhánh khác, vẫn phải cộng đóng góp của nhánh khác. Chặn một đường không cho phép xóa gradient toàn node.

## Bài 4

Z=`[[5.5,-1.5],[11.5,-3.5]]`.

$$
dW=X^TG=\begin{bmatrix}-2&11\\-2&16\end{bmatrix},\quad
db=[0,5],\quad
dX=GW^T=\begin{bmatrix}-1&2\\-4&-2\end{bmatrix}
$$

dW có shape(2, 2), db=(2, ), dX=(2, 2). Ví dụ dW₁₂=1×2+3×3=11. Bias₂ xuất hiện ở cả hai hàng, mỗi hàng góp2 và3 nên tổng 5. Nếu G đã có hệ số mean loss thì không chia 2 nữa; các phép cộng đường vẫn phải cộng như định nghĩa.

## Bài 5

Z₁=`[[1,1],[-1.5,2]]`, H=`[[1,1],[0,2]]`, Z₂=`[[1.5],[-1.5]]`. Với y=[1, 0], mean BCE=$\ln(1+e^{-1.5})\approx0.201413278$. Đặt q≈0.182425524.

$$
G_2=\begin{bmatrix}-q/2\\q/2\end{bmatrix},\quad
dW_2=\begin{bmatrix}-q/2\\q/2\end{bmatrix},\quad db_2=[0]
$$

$$
G_H=\begin{bmatrix}-q&q/2\\q&-q/2\end{bmatrix},\quad
\text{mask}=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
G_1=\begin{bmatrix}-q&q/2\\0&-q/2\end{bmatrix}
$$

$$
dW_1=\begin{bmatrix}-q&q\\-2q&q/2\end{bmatrix},\quad db_1=[-q,0],\quad
dX=\begin{bmatrix}-3q/2&0\\q/2&-q/2\end{bmatrix}
$$

db₂=0 vì hai mẫu góp−q/2 và+q/2, không phải do b₂ không nằm trong graph. Mask chặn hàng2, hidden1 nhưng hàng1 vẫn cho G₁₁=−q; vì W₁ dùng chung qua batch nên các weight nối tới neuron1 vẫn có gradient.

## Bài 6

Sau cập nhật lr=0.1, L2=0:

```text
W1 = [[1.018242552,-1.018242552],[0.536485105,0.990878724]]
b1 = [-0.981757448,0]
W2 = [[2.009121276],[-1.009121276]]
b2 = [0.5]
```

Forward lại cho logits≈`[1.756726896,-1.527447026]`, mean BCE≈0.177846597. Gradient được tính tại một trạng thái parameter cụ thể. Dùng W₂ mới với cache cũ cho G_H sẽ trộn hai trạng thái, không còn đúng backward của lần forward ban đầu.

Tại parameter ban đầu, tổng W₁²=3.25, tổng W₂²=5, tổng8.25. λ=0.1 cho penalty=0.1/2×8.25=0.4125, objective≈0.613913278. Gradient W₁ cộng`[[0.1,-0.1],[0.05,0.1]]`; W₂ cộng`[[0.2],[-0.1]]`. Bias không có penalty nên gradient bias không đổi tại cùng parameter/dữ liệu.

Penalty này chỉ phụ thuộc W₁, W₂, không phụ thuộc trực tiếp X nên không có hạng λX trong dX. Khi parameter được cập nhật rồi forward mới, dX có thể đổi gián tiếp vì mạng đã đổi; đó là một vấn đề khác với đạo hàm trực tiếp tại một trạng thái cố định.

## Bài 7

Với λ=0, n=2, sum BCE≈0.402826556, gấp2 mean BCE≈0.201413278. Mọi gradient sum cũng gấp2 gradient mean, kể cả bias.

Với λ=0.1: J_mean≈0.201413278+0.4125=0.613913278; J_sum≈0.402826556+0.4125=0.815326556. Hai lần J_mean≈1.227826556, không bằng J_sum.

Cho một weight, G_mean=G_data_mean+λW, G_sum=2G_data_mean+λW=2G_mean−λW. Với bias không phạt, G_sum=2G_mean. Không coi quy tắc mean/sum là độc lập với quy ước penalty.

Microbatch kích thước1 và3 có mean data loss toàn bộ là (L₁+3L₃)/4. Khi cùng parameter và λ=0, gradient mean toàn bộ là (g₁+3g₃)/4; không phải(g₁+g₃)/2. Để mô phỏng một bước batch lớn, không cập nhật parameter giữa hai microbatch rồi coi như gradient cùng điểm.

Cộng đường trong một graph là bắt buộc theo chain rule; giữ gradient cũ qua nhiều bước chỉ hợp lệ khi chủ đích và đúng cách chuẩn hóa. Xóa các đóng góp của nhánh trước **trong cùng backward** sẽ sai, còn bỏ quên gradient bước trước có thể làm bước mới dùng sai mục tiêu.

## Bài 8

Có9 parameter scalar nên cần 18 đánh giá objective cho sai phân trung tâm ở một epsilon, cộng lần backward analytic để so. Lab không dùng numeric gradient trong training 5000 bước.

Ví dụ ReLU b₂ có analytic gradient khoảng10⁻¹⁷ và numeric gradient có thể0. Sai số tuyệt đối vẫn khoảng10⁻¹⁷; chia cho denominator có floor1e−12 có thể cho relative≈10⁻⁵. Do vậy nhìn relative một mình sẽ dễ kết luận sai. Các tensor có gradient khác 0 trong ví dụ có sai số nhỏ theo cả hai cách đo.

Với f=ReLU tại 0, `(f(ε)−f(−ε))/(2ε)=ε/(2ε)=0.5`. Hàm không khả vi cổ điển ở0; so với quy ước slope 0 không chứng minh implementation sai. Khi mask thay giữa parameter−ε và parameter+ε, phép thử đang đi qua kink; cần xem vị trí và các nhánh, chuyển fixture/epsilon hoặc dùng tanh mượt để kiểm các công thức khác trước.

Accuracy là hàm sau ngưỡng, thường hằng trong một lân cận rồi nhảy; không phải objective BCE liên tục đang lấy gradient. ε quá nhỏ có thể làm hai giá trị objective gần nhau bị mất độ chính xác khi trừ; không có bảo đảm cứ giảm ε là kiểm tốt hơn. Check qua một điểm cũng không chứng minh đúng ở mọi miền.

## Bài 9

Mạng 2→4→1 có2×4+4+4×1+1=17 parameter. Cả W₁, b₁, W₂, b₂ được fit, khác bài 9 chỉ fit head trên feature cố định.

Seed 2026, tanh hidden, 5000 bước full-batch, lr=1, λ=0.001 phạt cả hai W. Train BCE đầu≈0.697609930, cuối≈0.009790833; objective cuối≈0.050311237, accuracy train1. Norm thay đổi W₁, b₁ lần lượt≈4.903403, 3.575171, nên có bằng chứng lớp ẩn đã được cập nhật. Chỉ đo trên bốn hàng train, chưa có validation/test hoặc bằng chứng đối với dữ liệu ngoài mẫu/nhiễu.

Nếu mọi parameter0, tanh hidden cho H=0, Z₂=0, P=0.5. G₂ theo thứ tự XOR bằng`[0.125,-0.125,-0.125,0.125]` dạng cột. dW₂=HᵀG₂=0, db₂=sum G₂=0. W₂=0 nên G_H=0, G₁=0, dW₁=db₁=0; L2 cũng có gradient0 vì W=0. Điểm này đứng yên với gradient descent dù một mạng khác có objective thấp hơn như lần chạy ngẫu nhiên đã cho. Gradient0 không bảo đảm tối ưu toàn cục cho MLP phi lồi.

## Bài 10

Backprop tính gradient của objective theo các biến, không tự chọn learning rate. Loss là một scalar; gradient theo W₁ là một ma trận cùng shape W₁ và thường không có giá trị bằng loss. Phải cộng đủ mọi đường về cùng node trong một backward; reset bộ đệm giữa các bước là việc khác. Điểm train XOR 100% chưa phải bằng chứng production.

Autograd áp các quy tắc đạo hàm cục bộ vào graph đã ghi, không cần thử từng parameter±ε như sai phân. Lab dùng công thức NumPy viết tay, chưa dùng engine autograd. Có thể tính dX để kiểm tra độ nhạy nhưng training thông thường vẫn giữ dữ liệu X cố định và chỉ cập nhật parameter.

Một báo cáo mẫu 7 câu:

> Đây là thí nghiệm end-to-end trên bảng XOR bốn hàng train, chưa có validation/test. Mạng 2→4→1 dùng tanh hidden và một logit đầu ra, với tổng 17 parameter. Khởi tạo ngẫu nhiên seed 2026 và bias0, rồi cập nhật W₁, b₁, W₂, b₂ bằng backprop NumPy và gradient descent full-batch. Chạy5000 bước, lr=1, mean BCE cộng L2=0.001/2 nhân tổng bình phương hai W, không phạt bias. Kết quả train BCE≈0.009790833, objective≈0.050311237, accuracy 1; norm thay đổi W₁, b₁ khác 0 cho thấy hidden đã được cập nhật. Đây là kết quả của một cấu hình cố định, không bảo đảm mọi khởi tạo hội tụ hoặc mô hình làm tốt trên dữ liệu nhiễu. Trước kết luận cho bài toán thực tế cần dữ liệu đại diện, baseline và giao thức train/validation/test nằm ngoài quá trình chọn cấu hình.

## Nếu còn vướng

- Vẽ các đường đi rồi nhân theo từng đường, cộng tại biến dùng chung.
- Đặt shape bên cạnh G₂, G_H, G₁ và từng parameter gradient trước khi sửa code.
- Tách forward/loss, backward và bước optimizer; không thay tham số ở giữa backward.
- Kiểm cùng objective bằng sai phân ở mạng nhỏ, tránh kink và đọc cả sai số tuyệt đối.
- Khi báo kết quả, nói rõ phần nào đã được fit và điểm số là train hay giữ riêng.
