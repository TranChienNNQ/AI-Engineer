# Bài tập — Dữ liệu chuỗi và ý tưởng RNN

[Về bài học](README.md) · [Lời giải](solutions.md)

Tính tay hoặc dự đoán trước khi chạy. Các số giả định chỉ phục vụ kiểm cơ chế.

## Bài 1. Trục và thứ tự

Bốn chuỗi, mỗi chuỗi6 bước, mỗi bước3 feature: shape batch_first là gì? Một sample, một x_t trong batch có shape gì? Phân biệt shuffle sample và shuffle các bước trong sample. Vì sao mean không phân biệt[1,−1,2] và[2,−1,1]?

## Bài 2. Forward ba bước

Với x=[1,−1,2], w_x=.5,w_h=.25,b=0,h0=0, tính a_t và h_t. Đảo chuỗi cho cùng final state không? Ở bước3 có tạo thêm weight mới không? h0=0 có nghĩa model không thể học từ bước1 không?

## Bài 3. Shape API và parameter

RNN input_size=3,hidden_size=5,num_layers=1,batch_first=True,một hướng, có bias. Với batch4,T=6, ghi output/h_n/weight shapes và số parameter. Head5→2 thêm bao nhiêu? Vì sao PyTorch hai bias không giống khái niệm một bias khi xét update cả hai?

## Bài 4. BPTT tính tay

Dùng forward bài2 và L=½(h3−.5)². Tính δ3,δ2,δ1, dL/dw_x,dL/dw_h,dL/db. Vì sao gradient weight phải cộng qua thời gian? Nếu có loss trực tiếp tại h2, recursion gradient cần thay đổi gì?

## Bài 5. Mất/bùng gradient

Với h_t=a h_{t−1}, tính dh20/dh0 khi a=.5,1,1.5. Công thức này có phải gradient của mọi tanh RNN không? Clipping giải quyết được tín hiệu mất dần không? Detach hidden khác reset hidden thế nào?

## Bài 6. Padding tính tay

Cell ReLU h_t=ReLU(x_t+h_prev+.5), h0=0. A=[1],B=[2,3,4], right pad0. Tính state mỗi bước, hidden cuối tensor và hidden thật cuối. Thay padding A bằng99 có ảnh hưởng packed state khi lengths đúng không? Vì sao không đếm phần tử khác0 để lấy length?

## Bài 7. Mask và denominator

Loss token thật A=[1,4],B=[9], cùng pad thành tensor2×3. Tính mean trên6 ô sau gán padding0, mean trên3 token thật và mean per-sequence. Ba đại lượng có cùng mục tiêu không? Lab tính CE theo số token hay số chuỗi?

## Bài 8. Window và thông tin tương lai

Dãy[10,12,11,13,15], lookback3,horizon1 tạo input/target nào? Một người random split các window chồng lấn rồi tính scaler trên toàn timeline: cần sửa gì? Validation dùng lịch sử đã quan sát ở đoạn train có luôn là leakage không?

## Bài 9. Thiết kế thực nghiệm

96 group,mỗi group2 chuỗi đảo nhau, split64/16/16 group cho bao nhiêu sample? Model input1,hidden8,head2 có bao nhiêu parameter? Batch16,30 epoch tạo bao nhiêu update? Vì sao endpoint baseline đạt100% khiến kết luận “cần RNN” không được hỗ trợ?

## Bài 10. State, checkpoint và báo cáo

Chạy mode all, đọc data.json, preprocess, metrics và report. Viết6–10 câu nêu bài toán, split, độ dài, chuẩn hóa, cách dùng packed state, cấu hình, chọn checkpoint, test và baseline. Nếu carry hidden từ batch trước trong bài này thì sai gì? best.pt đã đủ resume chưa?

## Thực hành mở rộng

1. So packed hidden với chạy từng chuỗi riêng ở batch chưa sắp lengths; kiểm thứ tự mẫu không bị tráo.
2. Kiểm thêm padding cuối hoặc đổi riêng vùng padding không làm logits đổi trong dung sai; không sửa bước thật.
3. Chạy Config(epochs=2,batch_size=19) trong output mới, dự đoán số update và batch cuối. So loss validation khi batch_size=19 và32.
4. Tính gradient w_x,w_h,b của scalar_reference bằng sai phân trung tâm để đối chiếu manual BPTT. Giữ cùng dữ liệu và cùng target khi dịch parameter.
