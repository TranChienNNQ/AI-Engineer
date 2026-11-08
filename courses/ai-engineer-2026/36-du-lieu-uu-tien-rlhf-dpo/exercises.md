# Bài tập bài 36

[← Bài chính](README.md) · [Lời giải tách riêng](solutions.md)

Dùng log tự nhiên. Ví dụ backend và số liệu đều giả định, trừ bảng kết quả được sinh từ lab. Hãy tính hoặc dự đoán trước khi chạy code.

## 1. Viết một cặp ưu tiên có thể đánh giá

Prompt: “API 5xx tăng sau deploy.” Viết hai câu trả lời không dài quá ba câu mỗi câu trả lời. Chọn một câu và nêu hai tiêu chí quyết định. Nêu trường hợp cần abstain thay vì ép chọn.

## 2. SFT và dữ liệu ưu tiên khác gì?

Biến cặp ở câu 1 thành một mẫu SFT. Điều gì trong phép so sánh đã mất? Chosen luôn là câu hoàn hảo và rejected luôn sai có đúng không?

## 3. Reward model tính tay

Cho `r+=ln3`, `r−=0`. Tính xác suất chosen thắng, loss một phiếu thuận và đạo hàm theo hai reward. Thêm 100 vào cả hai thì kết quả đổi gì?

## 4. Bất đồng ba thuận một ngược

Với `p=σ(w)`, viết loss trung bình của 3 phiếu thuận/1 phiếu ngược. Tìm w tối ưu. Nếu tất cả bốn phiếu đều thuận và không regularization, có nghiệm tối ưu hữu hạn không?

## 5. Tính KL và mục tiêu

Reference `[.5,.5]`, policy `[.75,.25]`, reward `[ln3,0]`, β=1. Tính expected reward, KL(policy||ref), J và so với J của reference. Một log-ratio mẫu có thể âm không?

## 6. Suy ra nghiệm đóng

Dùng Lagrange hoặc đẳng thức KL để suy ra `p* ∝ q exp(r/β)`. Nêu hai điều kiện của công thức. Giữ reward, tăng β đến rất lớn thì p* tiến về đâu?

## 7. Tính một cặp DPO

Reference A/B/C=`[.6,.3,.1]`, policy=`[.7,.2,.1]`, nhãn A>B, β=.5. Tính raw logp gap, reference logp gap, normalized margin m, z và loss. Nêu vì sao không chỉ dùng raw gap.

## 8. Loss ban đầu ln2 nhưng có học không?

Policy bắt đầu bằng reference. Tính m, loss và `dL/dm`. Chỉ ra lỗi nếu code cho `reference=policy` rồi bật optimizer trên policy. Phân biệt freeze, eval và no_grad.

## 9. Kiểm gradient bằng sai phân số

Với policy logps `[-.4,-1.2]`, reference logps `[-.6,-1.0]`, β=.7, tính hai đạo hàm bằng PyTorch; kiểm bằng finite difference bước10⁻⁶. Hàm sai phân dùng NumPy `logaddexp` độc lập với hàm loss PyTorch.

## 10. Shift, mask và EOS

Viết input/target của `BOS USER route timeout ASSISTANT NET CHECK EOS`. Những logit nào vào completion score? Với logit mọi từ bằng 0 và vocab V=29, score completion NET CHECK EOS bằng bao nhiêu? Vì sao đổi logit prompt/PAD không đổi loss trực tiếp nhưng đổi prompt token có thể đổi score?

## 11. Sum và mean có thể đảo thứ tự không?

A có logps `[-.1,-.1]`, B có `[-.08,-.08,-.08]`. Tính sum/mean và thứ tự. Đưa ra hai kiểm tra khi chosen và rejected khác độ dài, có EOS và PAD.

## 12. Preference tốt hơn nhưng chosen ít có khả năng hơn

Reference `[.4,.3,.3]`, policy `[.2,.1,.7]`, nhãn A>B, β=1. Tính DPO loss trước/sau. Xác suất chosen thay đổi thế nào? Cặp này đã kiểm soát C chưa?

## 13. Chẩn đoán bandit reward hacking

Chạy mode bandit. Giải thích RM học gì từ hai hành động đầu và vì sao ngoại suy sai sang hành động thứ ba. Tính utility từ policy β=.5. Tăng β đến5 có chứng minh policy đã an toàn không?

## 14. Chia dữ liệu và cache

Một incident có 5 prompt, mỗi prompt có 4 ứng viên. Tạo mọi cặp cho ra bao nhiêu cặp? Vì sao không random split từng cặp? Liệt kê các thành phần phải khớp để dùng lại reference score cache. Trường hợp tie/both bad có nên tự gán chosen theo vị trí A không?

## 15. Đọc report TinyLM và thiết kế đánh giá

Chạy `all` với output mới. Giải thích SFT có 0 relative win/24 tie nhưng vẫn đúng route12/12; DPO có24 relative win và exact quality12/12 nghĩa là gì, chưa chứng minh gì? Đề xuất một baseline bổ sung và ba lát cắt ngoài dữ liệu lab. Kiểm load checkpoint: thay β/tokenizer/reference identity thì cần làm gì?

## 16. PPO, DPO và hệ thống thực thi

Tính PPO clipped term với ε=.2 cho `(ρ=1.5,A=2)` và `(ρ=.5,A=−2)`. Phân biệt πold với πref, clipping với hard probability constraint. Trợ lý đạt preference accuracy cao đề xuất chạy lệnh restart production: những trách nhiệm nào vẫn thuộc hệ thống? Lab đã chạy PPO thật chưa?
