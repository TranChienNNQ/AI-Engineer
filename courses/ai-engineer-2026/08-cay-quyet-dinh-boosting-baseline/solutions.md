# Lời giải — Cây quyết định, boosting và baseline

[Về bài học](README.md) · [Về đề bài](exercises.md)

Các số liệu đã được đối chiếu bằng code. Mọi dữ liệu tự tạo phục vụ giảng dạy; giải được một ví dụ hoặc có train error thấp chưa chứng minh chất lượng ngoài mẫu.

## Bài 1

Đạo hàm MSE theo hằng c là $2(c-\bar y)$. Mean y=10/4=2.5, SSE=11,MSE=11/4=2.75. Đây là tối ưu trong họ dự đoán hằng theo MSE.

Với c=2, MAE=(1+1+1+3)/4=1.5. Với c=2.5, MAE=(1.5+1.5+0.5+2.5)/4=1.5. Do số mẫu chẵn, mọi c trong đoạn hai giá trị giữa [1,3] đều tối ưu tổng sai số tuyệt đối. NumPy chọn trung điểm 2 làm median ở đây; median tối ưu không nhất thiết duy nhất.

Baseline xác suất cho train 98/2 là p=0.02; baseline nhãn majority là 0. Không fit tỷ lệ từ test. Baseline rất yếu chỉ chứng minh vượt một mốc thấp, chưa đủ cho quyết định thay mô hình tuyến tính, rule hoặc hệ thống hiện có vốn có thể rẻ/ổn định hơn.

## Bài 2

Gốc p=3/6=0.5,Gini=0.5.

| t | Gini trái | Gini phải | Điểm có trọng số | Gain |
|---:|---:|---:|---:|---:|
| 1.5 | 0 | 12/25=0.48 | 0.4 | 0.1 |
| 2.5 | 0 | 3/8=0.375 | 0.25 | 0.25 |
| 3.5 | 4/9 | 4/9 | 4/9 | 1/18 |
| 4.5 | 3/8=0.375 | 0 | 0.25 | 0.25 |
| 5.5 | 12/25=0.48 | 0 | 0.4 | 0.1 |

Lab chọn 2.5 do thứ tự thử; lá trái p=0,lá phải p=0.75. Dự đoán p=`[0,0,0.75,0.75,0.75,0.75]`; nhãn `[0,0,1,1,1,1]`. Trên train có TP=3,FP=1,FN=0,TN=2,accuracy=5/6,F1=6/7. Không xem đây là điểm test.

Trung bình không trọng số ở 2.5 cho `(0+3/8)/2=3/16=0.1875`, khác 1/4 vì hai lá có 2 và 4 mẫu. Hai mẫu có cùng x luôn đi chung phía với mọi điều kiện theo x đó; không được dùng thứ tự hàng để tách chúng.

## Bài 3

| x | Lá | p | Nhãn ở 0.5 |
|---:|---|---:|---:|
| 2 | Trái | 0 | 0 |
| 2.5 | Trái do `<=` | 0 | 0 |
| 2.6 | Phải | 0.75 | 1 |
| 10 | Phải | 0.75 | 1 |

Tăng ngưỡng xác suất lên 0.8 biến nhãn tại p=0.75 thành 0; không đổi cấu trúc hoặc p của cây. Nó là thay quy tắc quyết định, cần chọn bằng development đúng giao thức.

`min_leaf=3` chỉ cho split t=3.5, với 3 mẫu mỗi bên. Gain=1/18>0 nên hợp lệ dù không tốt bằng split không bị ràng buộc. `min_leaf=4` không có split hợp lệ trên 6 mẫu vì cần ít nhất 8. `max_depth=0` cho một lá p=0.5. Lá một positive p=1 chỉ phản ánh nhãn train trong vùng, không đủ bằng chứng chắc chắn ở mọi mẫu tương lai.

## Bài 4

Mean=2.5,SSE=11,MSE=2.75. Tại các ngưỡng:

| t | Mean trái/phải | SSE |
|---:|---|---:|
| 1.5 | 1 và 3 | 0+(4+0+4)=8 |
| 2.5 | 1 và 4 | 0+0+1+1=2 |
| 3.5 | 5/3 và 5 | 4/9+4/9+16/9=8/3 |

Chọn 2.5, dự đoán tại x=`[0,2.5,3,10]` là `[1,1,4,4]`. Depth=2,min_leaf=1 có thể chia nhánh phải tiếp ở 3.5, cho dự đoán train `[1,1,3,5]`, MSE=0. Nó chưa nói gì về nhiễu hoặc vùng ngoài dữ liệu đã có.

Mean của một tập con không trọng số nằm giữa min/max của tập con, cũng nằm giữa min/max train; mỗi dự đoán của cây thuộc một mean như vậy. Tổng cây fit residual không phải một mean duy nhất của y_train: các đóng góp có dấu và được cộng nhiều vòng, nên không suy nguyên bảo đảm min/max ấy cho ensemble tổng.

## Bài 5

F₀=2.5,r₁=`[-1.5,-1.5,0.5,2.5]`. Chọn split 2.5, lá h₁=−1.5 và +1.5.

Với η=1/2, F₁=`[7/4,7/4,13/4,13/4]`, MSE=17/16=1.0625. r₂=y−F₁=`[-3/4,-3/4,-1/4,7/4]`. Dùng r₁ lại sẽ fit hướng sửa lỗi của mô hình cũ, không phải âm gradient tại dự đoán hiện tại.

SSE khi fit r₂ ở t=1.5,2.5,3.5 lần lượt là 7/2,2,1/6. Vì vậy chọn 3.5, h₂ có lá trái −7/12,lá phải 7/4. F₂=`[35/24,35/24,71/24,33/8]`. Sai số F₂−y=`[11/24,11/24,−1/24,−7/8]`, nên MSE:

$$
\frac14\left(\frac{121+121+1}{576}+\frac{49}{64}\right)
=\frac{19}{64}=0.296875.
$$

Với dấu sai F₁=F₀−0.5h₁=`[3.25,3.25,1.75,1.75]`, MSE=`(2.25²+2.25²+1.25²+3.25²)/4=89/16=5.5625`, cao hơn ban đầu. Nếu định nghĩa error=F−y thì trừ cây error mới đúng; phải giữ nhất quán cả hai dấu.

## Bài 6

Stump residual vòng đầu vẫn là `[-1.5,-1.5,1.5,1.5]` vì được fit từ cùng F₀.

| η | Dự đoán sau một vòng | Train MSE |
|---:|---|---:|
| 0.25 | [2.125,2.125,2.875,2.875] | 113/64=1.765625 |
| 0.5 | [1.75,1.75,3.25,3.25] | 17/16=1.0625 |
| 1 | [1,1,4,4] | 1/2=0.5 |

Với vùng lá cố định, gọi residual r_i và mean residual của lá là a. Khi cộng ηa, tổng bình phương trong lá biến đổi thành:

$$
\sum_i(r_i-\eta a)^2
=\sum_i r_i^2-n_{leaf}a^2(2\eta-\eta^2).
$$

Vì 0<η≤1 cho 2η−η²>0, train SSE không tăng. Nếu a=0 thì không đổi. Đây là lập luận cho squared error với giá trị lá mean residual và số học lý tưởng; nó không kiểm soát error trên mẫu mới chưa được dùng fit. η=1 tốt nhất trong ba lựa chọn **một vòng trên train này**, chưa chứng minh tốt nhất khi nhiều vòng hoặc có nhiễu.

Test loss dùng chọn checkpoint làm test tham gia tuning. Lab chỉ so M trong lưới, chưa có patience/early-stopping checkpoint tự động; validation vẫn cần được giữ khỏi việc fit trọng số/giá trị lá.

## Bài 7

Có 10 ứng viên: mean, linear, 3 cây và 5 mức số vòng boosting. Mọi ứng viên dùng cùng một feature x, train 32 mẫu và validation 21 mẫu, không dùng test 8 mẫu để chọn.

Linear thắng validation với MSE≈5.105988; train MSE≈6.594010. Boost200 train≈1.697643 nhưng validation≈5.754190, kém linear trong cấu hình này. Train error thấp có thể phản ánh fit nhiễu hoặc độ linh hoạt, không tự là tổng quát hóa tốt hơn.

Tiêu chí validation MSE được định trước; đổi metric chỉ để họ mô hình mình thích thắng làm thay câu hỏi sau khi xem dữ liệu. Nếu có mục tiêu nghiệp vụ thật khác cần thiết kế lại có căn cứ và đánh giá độc lập, không gọi việc đổi mục tiêu ấy là bằng chứng thắng cùng phép đo.

Code chấm mô hình chọn cộng hai baseline cố định, bỏ bản trùng. Vì chọn linear, test chỉ có linear MSE≈6.019883 và mean≈88.979560; không có test của các cây/boosting. Điều này giữ test ngoài việc tìm ứng viên thắng.

Giới hạn: dữ liệu synthetic một feature; train/validation/test có cách tạo nhiễu khác nhau; tập nhỏ và lưới hạn chế; chưa có group/time, nhãn trễ, outlier thực tế hoặc đo chi phí phục vụ. Các điểm số không là benchmark đại diện production.

## Bài 8

Gốc XOR có p=1/2,Gini=1/2. Chia theo x₁≤0.5 cho mỗi nhánh một 0,một1 nên Gini con=1/2,điểm sau chia=1/2,gain=0. Chia x₂ cũng vậy.

Lab dừng khi không có gain dương, nên root không split và p=0.5 cho mọi mẫu. Tăng max_depth không bỏ điều kiện dừng ở gốc.

Quy tắc đặt tay: nếu x₁≤0.5 thì đoán 1 khi x₂>0.5, ngược lại đoán 1 khi x₂≤0.5. Hai tầng hỏi phân tách đúng bốn điểm. Không có mâu thuẫn vì khả năng biểu diễn của họ mô hình và thuật toán tìm mô hình là hai vấn đề khác nhau. Thư viện có thể cho phép split gain bằng0 hoặc dùng điều kiện khác, nên không suy mọi implementation dừng giống mini-lab.

## Bài 9

Bagging/random forest điển hình fit các cây trên bootstrap train và random forest còn chọn tập feature ứng viên tại nút; các cây không fit residual tuần tự. Khi suy luận chúng thường được gộp bằng trung bình/bỏ phiếu theo thiết kế. Gradient boosting fit cây mới theo âm gradient phụ thuộc mô hình tích lũy, rồi cộng đóng góp theo hệ số; những vòng sau phụ thuộc vòng trước.

Năm cây hoàn toàn giống nhau cho cùng dự đoán, lấy mean vẫn y như một cây; không có tính đa dạng để giảm phần lỗi không tương quan giữa các cây.

Binary log loss cho âm gradient y−p=`[-0.2,0.3]`. Cây hồi quy cần xấp xỉ các giá trị liên tục ấy; tổng F là logit rồi mới sigmoid lấy p. Việc tối ưu giá trị lá còn có thể tùy loss/thuật toán.

`SquaredBoost` có thể chạy với y=0/1 vì đó cũng là số, nhưng đang tối ưu squared error theo y−F, không phải logistic log loss theo y−sigmoid(F). Chạy được không chứng minh đúng mô hình xác suất mong muốn. Đây cũng không phải AdaBoost được triển khai bằng cách đổi tên lớp.

## Bài 10

Cây ít cần chuẩn hóa không có nghĩa mọi input hợp lệ: mini-lab từ chối NaN/inf; mã nominal 0/1/2 có thể áp đặt thứ tự không đúng nghĩa. Train MSE=0 chưa phải validation/test tốt. Lá hằng không tự ngoại suy xu hướng tới tải gấp10. Root thể hiện split hữu ích theo mục tiêu fit trong dữ liệu, không chứng minh quan hệ nhân quả.

Imputer/encoder có tham số phải fit từ train; trong CV phải fit lại chỉ trên train của từng fold. Validation/test chỉ transform theo cấu hình fit được; nếu encoder dùng thông tin nhãn cần kiểm soát thêm leakage thích hợp.

Đổi x thành100x chuyển split2.5 thành250, input tương ứng vẫn đi cùng lá. Tại grid`[0,2.5,3,10]`, hai mô hình cho cùng`[1,1,4,4]`. Phát biểu dành cho đổi đơn vị affine tăng và số học lý tưởng; sai số làm tròn ở biên hoặc phép biến đổi phi tuyến tổng quát không được tự bảo đảm ngoài mẫu.

Một báo cáo mẫu 7 câu:

> Đây là thí nghiệm synthetic hồi quy latency một feature, chưa phải dữ liệu production. Mean train và hồi quy tuyến tính có intercept là hai baseline được fit từ train. Dùng chung train 32 mẫu, validation 21 mẫu để so 10 ứng viên gồm baseline, cây và boosting stump theo lưới cố định. Chọn linear bởi validation MSE thấp nhất≈5.105988, giữ mô hình fit từ train và không refit train+validation. Sau chốt, test 8 mẫu cho linear MSE≈6.019883, baseline mean≈88.979560. Boost 200 có train MSE thấp hơn linear nhưng không được chọn bằng validation, nên không quét test của nó để đảo kết luận. Dữ liệu nhỏ được dựng với mức nhiễu khác nhau giữa các phần và lưới hẹp, nên chưa chứng minh họ mô hình nào tốt nhất khi vận hành.

## Nếu còn vướng

- Nhầm split và ngưỡng xác suất: viết riêng hai điều kiện x≤t và p≥τ.
- Nhầm Gini với accuracy: tính p trong từng nút rồi mới tính độ pha trộn.
- Nhầm boosting: lập bảng F cũ → residual → h → F mới, giữ rõ dấu và learning rate.
- Nhầm baseline: chỉ rõ cách fit và metric mà baseline tối ưu.
- Nhầm chất lượng: xem validation và giao thức chọn trước, không chọn bằng train hoặc quét test.
