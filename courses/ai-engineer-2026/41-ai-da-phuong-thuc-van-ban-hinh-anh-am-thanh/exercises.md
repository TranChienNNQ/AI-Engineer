# Bài tập bài 41 — AI đa phương thức

[← Bài chính](README.md) · [Lời giải](solutions.md) · [Phụ lục](deep-dive.md)

Làm tính tay/lập luận trước khi chạy code. Các snippet trong lời giải dùng thư mục bài41 làm working directory.

## 1. Thiết kế từ câu hỏi sản phẩm

Một trợ lý nhận text, screenshot và audio ghi âm. Phân biệt việc đọc mã lỗi, tìm tiếng thiết bị tương tự và trả lời về những gì người dùng đã nói. Khi nào OCR→text hoặc ASR→text có thể đủ, khi nào chúng làm mất thông tin cần thiết? Nhận nhiều input có buộc sinh nhiều loại output không?

## 2. Tokenizer và representation văn bản

Dùng vocabulary `['dọc','lam','lục','ngang','đỏ']`, tính BoW chuẩn hóa cho `đỏ đỏ dọc`. Kiểm NFC/lowercase cho chuỗi decomposed `ĐỎ DỌC`. Vì sao BoW không hiểu thứ tự và không thể thay tokenizer của model pretrained? Toàn OOV được xử lý thế nào?

## 3. Patch và thứ tự phần tử

Ảnh4×4RGB, mỗi channel cùng dãy0…15. P2: tính số patch, shape và patch đầu. Với RGB16×16/P4, tính số patch/dim và tổng feature. Cho biết global mean màu có đủ phân biệt vạch ngang/dọc không.

## 4. Tính thời lượng và sample rate

800sample ở8kHz dài bao lâu? Đọc cùng samples thành16kHz thì sao? Tại sao đổi header không là resample đúng? Viết phép kiểm WAV example của bài, chỉ ra channel/sample width/rate được chấp nhận.

## 5. STFT và DFT tính tay

Tính frame count, bins, bin spacing và tail bị bỏ cho S800/F128/H64/f_s8000,no-pad. Với x=`[0,1,0,−1]`, tính rFFT/power không Hann. Nếu f_s=8Hz thì bin1 là bao nhiêuHz? `log1p(power)` có phải log-mel/dB không?

## 6. Cosine và symmetric contrastive loss

Cho image vectors `[1,0]`, `[.6,.8]`, text vectors `[1,0]`, `[0,1]`, τ=.5. Tính ma trận cosine/logits và hai row CE, rồi tính symmetric loss cả row/column. Vì sao cùng d chưa là cùng không gian ngữ nghĩa và cosine không là confidence?

## 7. Positive lặp

Ba unit vectors `[1,0]`, `[1,0]`, `[0,1]`, labels`[0,0,1]`, τ1, cùng một set ở hai modality. Tính positive-set mass loss. So với diagonal CE: hai vector cùng lớp có bị coi là negative không? Nếu toàn batch cùng label thì objective của lab cho gì và học được gì?

## 8. Fusion và nguồn thiếu

Ảnh scores`[.9,.4]`, audio`[.2,.8]`, w=.75: tính fused và candidate được chọn. Khi ảnh missing thì kết quả nào? Viết code kiểm cả hai nguồn missing bị từ chối. Vì sao zero image/zero waveform không tương đương None?

## 9. Pairing và timestamp

Frame bắt đầu sample640 ở8kHz, clip bắt đầu2,5s trong recording: timestamp tuyệt đối là gì? Nếu ảnh chụp trước sự kiện còn audio ghi sau, tensor shape đúng có đủ bảo đảm ghép đúng không? Nêu metadata và provenance cần giữ.

## 10. Split và lineage

Dataset6lớp×24group/lớp, mỗi group3phương thức, index0–15train/16–19validation/20–23test. Tính record count; kiểm fit_ids và split disjoint trong code. Tại sao không gọi test là unseen-class/zero-shot? Nếu tạo nhiều crop/audio chunk trước khi chia ngẫu nhiên từng file thì có rủi ro gì?

## 11. Metric retrieval

Gallery24candidate có4positive cùng lớp mỗi query. Giải thích class top1 khác pair-ID recall. Cho query label1, candidate labels`[0,1,1]`, scores`[.1,.9,.8]`: đúng hay sai? Vì sao random-init baseline không bắt buộc đúng4/24 và cần rule baseline?

## 12. Gradient và save/load

Chạycheck, tìm các kiểm gradient đi tới ba encoder và save/load exact. Inference bundle giữ những gì? Sửa sample_rate mong đợi để load phải từ chối. Vì sao digest không là xác thực, và bundle chưa đủ exact training resume?

## 13. Đọc thí nghiệm và fault fixture

Chạyall vào output mới. Đối chiếu sáu hướng retrieval, rule baseline, missing/black/silent/mispaired. Tại sao fusion đúng cặp không chứng minh lợi ích hơn unimodal trong dataset này? Vì sao mispaired3/24 không đo khả năng chống mọi corruption ngoài đời?

## 14. Context và chi phí

Giả định ghép40text+16image+11audio+3special: tổng và số dense attention position pairs? Nếu image patches tăng4lần, tổng và tỷ lệ pair count đổi thế nào? Vì sao đây chưa là số API billing hoặc số token của mọi model?

## 15. Từ retrieval sang generation

Vẽ bằng lời một VLM trả lời câu hỏi về ảnh và một pipeline ASR→LLM. Projector có vai trò gì, cần objective gì, phần nào không có trong lab? Nêu ví dụ transcript làm mất thông tin mà audio encoder có thể cần giữ. Tone tổng hợp trong assets có phải model-generated speech không?

## 16. Kế hoạch đánh giá sản phẩm

Thiết kế evaluation cho trợ lý ticket đa phương thức: nhiệm vụ/annotation, split theo nguồn/sự kiện, baseline, metric, slice lỗi, missing/conflict và latency/cost. Chỉ rõ một claim cần image region/audio interval hỗ trợ và cách kiểm model có dùng nguồn thay vì chỉ đoán theo text prior.
