# Bài tập bài 38

[← Bài chính](README.md) · [Lời giải](solutions.md)

Dự đoán trước khi chạy. Lab dùng narrow symmetric range, khác range signed đầy đủ của một số format khác. Các giả định bộ nhớ phải được ghi khi tính.

## 1. Storage, compute và đối tượng quantization

Một checkpoint ghi “4-bit”. Nêu ít nhất bốn câu hỏi cần trả lời trước khi ước lượng RAM/VRAM và tốc độ. Phân biệt weight-only, activation quantization, KV quantization và cast FP32→BF16.

## 2. Symmetric quantization tính tay

Cho weights `[−3.5,−1.2,0,.7,2.4,3.5]`, narrow INT4 q∈[−7,7]. Tính scale, code, reconstruction, MSE và max error. Với tensor toàn0, vì sao phải quy định trường hợp riêng? Khi nào error≤s/2 không còn đúng?

## 3. Affine và thứ tự round

UINT4 range0…15, float range[−1,2]. Tính scale/zero-point; quantize `[−1,−.35,0,.65,2]`. Với x/s=.5 và zero-point1, so `round(x/s)+z` và `round(x/s+z)` theo ties-to-even. Có thể tự đổi thứ tự không?

## 4. Packing và padding

Signed code`[-7,-2,0,1,5,7]` dùng offset8, code đầu ở nibble thấp. Tính ba byte. Nếu có ba unsigned code[1,6,8], cần mấy byte và giải mã lấy mấy code? Vì sao biết byte mà thiếu shape/count/group vẫn chưa đủ?

## 5. Dot integer và zero-point

x=`[1,−2,3]`,w=`[.5,.25,−1]`,sx=.5,sw=.25. Tính qx/qw, accumulatorINT32 và dot giải scale. Nếu có zero-point khác0, viết biểu thức cần tính; vì sao không dùng unsigned nibble trực tiếp?

## 6. Granularity và byte metadata

Ma trận W có shape2×5. Dùng4bit, scaleFP32, group3theo cột từng hàng. Tính slot sau padding, codesbyte, số scale và tổngbyte chưa bias/header. So với một scale/hàng. Vì sao g32của MLP lab bằng W8-rowvề footprint?

## 7. Weight MSE và metric công việc

Cho Ŵ=W+ΔW. Viết output error của một Linear khi x giữ nguyên và dùng Cauchy–Schwarz để đặt bound từng output. Vì sao input lớn hoặc logit margin nhỏ làm weight error nhỏ chưa đủ? Cùng accuracy có chứng minh cùng prediction không?

## 8. Calibration shift

StaticINT8 scale1/127, input[−4,0,4]. Tính reconstruction và số phần tử vượt range. So dynamic per-row. Trong lab, calibration lấy ở đâu, test/stress có độc lập nhau không và vì sao cần lưu value hash thay vì chỉ indices?

## 9. PTQ, QAT và STE

Viết pipeline PTQ của lab. Giải thích vì sao round khó dùng gradient thông thường. Với `x+(qhat−x).detach()`, forward và gradient theo x là gì? Bài có train QAT hoặc chạy INT4 matmul thật chưa?

## 10. Memory ledger7B

98%parameter4bit, group64scale2byte,2%FP16; N=7tỷ, giả sử không padding. Tính payload tổng, effective bits/parameter. Budget16GiB, other2GiB, KV512MiB/request: tính trần payload cho weightsFP16 và mixed4. Vì sao chưa được dùng con số này làm admission capacity?

## 11. Đọc bảng chất lượng

Chạy `all` với output mới. Giải thích W4-g8 có126/132đúng giống FP32 nhưng agreement130/132. So W4-row/g8/g32 về byte và RMSE. Giải thích static accuracy124→106và clip6→657khi stress4×; điều gì chưa được chứng minh về dynamic quantization?

## 12. Artifact và identity

Liệt kê metadata cần lưu để tái dựng/cân nhắc lại một model quantized. Thử load rồi cố ý đổi bit/group/feature order hoặc payload để kiểm guard. Hash có tự xác thực tác giả/chữ ký không? File size có bằng tensor payload không?

## 13. Persistent memory và peak memory

Trong PackedLinear của lab, lúc forward những tensor nào có thể cùng tồn tại? Nếu process còn giữ FP32reference thì byte quantized module có phải tổngRAMkhông? Đề xuất hai tối ưu khác ngoài giảm bit, kèm cost/trade-off cần đo.

## 14. K/V lượng tử hóa

Chạy modeattention. Vì sao giảm weights chưa giảm KV? Với K/V shape[2,2,17,8], tính payloadFP32 và packed4bit kèm scaleFP32/group8. Lỗi K và V đi vào attention khác nhau thế nào? Lab có phải KIVI hoặc KVengine autoregressive không?

## 15. Gọi đúng tên phương pháp

Phân biệt absmax round-to-nearest, GPTQ, AWQ, SmoothQuant và NF4. Với D đường chéo dương, chứng minh `(XD⁻¹)(WD)ᵀ=XWᵀ`. Vì sao đẳng thức float chưa bảo đảm hai vế quantized giống nhau?

## 16. Thiết kế phép chọn triển khai

Trợ lý RAG phải vừa memory budget và đạt quality/SLO. Đề xuất protocol có calibration đại diện, validation/test, memory measurement, workload/context, kernel/device và rollback. Nêu ít nhất ba kết luận không được rút ra từ lab MLP867parameter này.
