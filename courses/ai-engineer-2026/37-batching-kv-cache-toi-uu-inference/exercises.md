# Bài tập bài 37

[← Bài chính](README.md) · [Lời giải](solutions.md)

Các con số hệ thống dưới đây là giả định, trừ bảng benchmark đã ghi rõ trong bài. Dự đoán trước khi chạy code. Dùng byte, KiB/MiB/GiB và GB đúng đơn vị.

## 1. Token đầu tiên đến từ đâu?

Prompt có4token, muốn sinh3token. Vẽ ba lượt forward, nêu input của từng lượt khi có cache và token đầu ra tương ứng. Khi kết thúc, cache dài bao nhiêu? KV cache có phải response cache hoặc memory hội thoại không?

## 2. Đếm công việc bằng tay

Với P=4,N=3, tính số vị trí logic qua model khi recompute và khi cache. Tính số cặp causal query–key hợp lệ mỗi head/layer. Vì sao hai tỷ số này không phải hệ số speedup wall time?

## 3. Một query nhìn ba key

dh=1, query1, keys `[1,0,1]`, values `[2,4,8]`. Tính scores, softmax và output. Hai key/value đầu đã có trong cache thì cần tính lại phần nào? Vì sao không lưu chỉ Q cũ?

## 4. Bẫy causal non-square

Cache có3token, chunk mới có2token. Viết mask2×5 đúng. Với query dài1/keys dài3, tại sao dùng trực tiếp SDPA `is_causal=True` có thể sai trong tình huống decode ở cuối prefix? Viết một ví dụ số cho thấy khác biệt.

## 5. Dự toán KV

L=32,Hq=32,dh=128,T=4096,b=2. Tính payload KV một request cho MHA Hkv=32 và GQA Hkv=8. Tám request GQA cần bao nhiêu GiB? Những phần bộ nhớ nào chưa tính? Weight4bit có tự làm KV4bit không?

## 6. Padding và lấy logit cuối

Lengths `[2,9,3,8]`, batch size2. Tính số ô dense theo thứ tự đến và sau bucketing; efficiency theo số ô token là bao nhiêu? Với right-padded batch lengths `[3,5]`, cần gather logit ở đâu và position decode tiếp theo là gì?

## 7. Kiểm logits trước argmax

Chạy model lab trên hai prompt khác độ dài. So logits valid của từng hàng với chạy riêng. Sau đó so generation single/batch và cache/full. Vì sao chỉ so argmax có thể bỏ sót lỗi?

## 8. Request kết thúc sớm

Dùng ba prompt của demo. Cưỡng bức request0 phát EOS ở bước1, request1 ở bước3, request2 dùng hết budget4. Dự đoán active IDs từng bước, outputs, stop reason và số vị trí logic có cache. Nêu rõ đây có phải bằng chứng model biết dừng hay không.

## 9. Lịch batching tính tay

Hai slot; A đến0 cần4token, B đến0 cần1, C đến1 cần2, D đến2 cần1. Mỗi tick tạo1token/request active, bỏ qua prefill. Tính completion time, TTFT của C/D và active-slot fraction cho static cohort/continuous. Giả định nào làm ví dụ không thể dùng làm GPU benchmark?

## 10. Cấp phát khác sử dụng

Lab L=2,Hkv=2,dh=8,b=4, lengths `[3,5]`, capacity16/hàng. Tính used/reserved KV bytes. Với page4token và requests lengths5/9 cần bao nhiêu page, slot trống? “Near-zero waste” có nghĩa mọi byte hệ thống đều hữu ích không?

## 11. Đọc throughput đúng nghĩa

Weights14GB, bandwidth giả định700GB/s. Bỏ mọi chi phí khác. Tính lower bound step time. Nếu một step phục vụ8request và mỗi request nhận1token, aggregate token/s và ITL giả định là bao nhiêu? Vì sao không gọi2.5ms là ITL? Batching có thể làm TTFT tăng bằng cách nào?

## 12. Cache identity và compaction

Cache có hai hàng A/B. B kết thúc, còn A; hoặc cần đổi thứ tự B/A. Phải làm gì với tensor và request IDs? Kiểm code từ chối owner/row/weight mismatch. Khi người gọi đổi prompt nhưng vẫn dùng cùng request ID, guard có tự biết mọi trường hợp đó không?

## 13. Prefix reuse và cập nhật model

Hai request cùng chứa chuỗi “timeout”, một request ở sau system prompt khác. Có dùng chung KV của riêng từ đó được không? Liệt kê điều kiện prefix reuse; thay temperature có bắt buộc thay KV của prefix y hệt không? Sau DPO cập nhật weights thì sao? Cache chia sẻ giữa tenant cần điều kiện nào ngoài toán học?

## 14. Thiết kế và kiểm benchmark

Chạy `bench` với output mới và≥3repeat. Tính lại token/s từ median_ms. Giải thích bảng là thời gian của một request hay cả4request. Vì sao cấm EOS trong benchmark fixed work nhưng vẫn phải kiểm EOS ở bài kiểm lifecycle? Local token probe loại những chi phí nào?

## 15. Chọn hướng tối ưu theo bottleneck

Bạn thấy TTFT tăng khi nhận một prompt rất dài trong lúc nhiều request đang decode; memory KV gần giới hạn. Đề xuất hai thay đổi để thử và metric cần kiểm lại. Phân biệt KV cache, FlashAttention, paged KV, compile và weight quantization. Những hướng nào lab đã thật sự chạy?

## 16. Đọc kết quả có giới hạn

Demo cached31/36 và full164/204 là những số gì? Bảng CPU có đủ để chọn batch size cho model7B trên GPU không? Thiết kế một thí nghiệm tiếp theo có workload, quality/correctness gate, điểm đo latency, memory budget và điều kiện dừng rõ ràng.
