# Phụ lục bài 41 — Representation, alignment và bằng chứng

[← Bài chính](README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

## 1. Shared embedding space không có nghĩa mọi thông tin phải giống nhau

Một ảnh và câu mô tả có thể khớp về vật thể nhưng câu không mô tả màu nền, góc chụp hoặc chữ nhỏ. Audio của cùng cảnh có thể chứa tiếng ngoài khung hình. Alignment chỉ tối ưu quan hệ/objective đã định; không hứa vector pooled giữ toàn bộ raw information.

Cùng dimension chưa đủ: nếu một encoder quay toàn bộ hệ tọa độ bằng một ma trận trực giao còn encoder kia không quay, cosine nội bộ mỗi modality giữ nguyên nhưng cross-modal cosine có thể đổi. Training phải học hệ tọa độ tương thích, và processor/checkpoint khác version có thể phá tương thích đó.

Không nên xếp text/image/audio vào một trục “thông minh hơn”. Chọn representation dựa trên thông tin tác vụ cần giữ. Transcript có thể đủ cho tìm câu đã nói, chưa đủ cho xác định tiếng động nền. OCR text có thể đủ cho search mã lỗi, chưa đủ cho giải thích bố cục dashboard.

## 2. Một-vector retrieval và token-level reasoning

Retrieval encoder thường tạo vector pooled để lập chỉ mục nhanh. Model nhận token-level image/audio features có thể giữ chi tiết hơn cho câu hỏi phụ thuộc region/time, nhưng số token lớn hơn và cần objective/task phù hợp. Một điểm cosine cao giữa image và caption không cung cấp bounding box tự động.

Muốn trả “chữ này nằm ở góc phải”, cần representation/decoder và evaluation có quan hệ không gian. Muốn nói “tiếng bíp xuất hiện ởgiây2,5”, cần timeline/annotation đúng độ phân giải. Không lấy attention heatmap bất kỳ làm bằng chứng xác nhận định vị đúng; cần ground-truth hoặc kiểm định tác vụ.

## 3. Patchify giữ gì, mất gì?

Patchify không overlap trong lab chỉ sắp xếp lại phần tử; có thể unpatchify hoàn toàn nếu còn shape/order. Mất thông tin thường xảy ra khi resize/crop, projection nén, pooling hoặc precision thay đổi. Patch size lớn làm số token nhỏ hơn nhưng mỗi token phải tổng hợp vùng lớn hơn; không suy patch lớn luôn tốt.

Với RGB16×16/P4:16patch×48value=768value, bằng raw pixel count. Sau projection patch→d, số feature có thể tăng/giảm; đó không là byte savings tự động. Lab flatten patch sequence trước MLP nên giữ vị trí bằng thứ tự feature cố định; chưa có học positional embedding/translation invariance như các kiến trúc khác.

## 4. Audio sample, frame và token là ba đơn vị khác nhau

800sample ở8kHz là0,1s. Với frame128/hop64/no-pad có11frame; không có nghĩa clip chứa11audio token trong mọi model. Một encoder convolution có thể downsample; codec có thể có nhiều codebook; một pipeline có thể chunk dài rồi pool. Không chuyển thẳng frame count thành LLM context/billing mà bỏ processor/model.

STFT trong bài nhân Hann đối xứng; librosa/torch hoặc model khác có thể dùng periodic window, center padding, FFT size khác hoặc normalization khác. Shape giống nhau chưa bảo đảm giá trị giống nhau. Power chưa là mel: mel cần thêm filterbank theo trục tần số; `log1p` chưa là dB vì log base/reference khác. Không có bước inverse-STFT nên lab chưa đánh giá tái tạo waveform.

## 5. Sample rate, Nyquist và aliasing

Một sin liên tục có tần số f lấy mẫu tại f_s tạo sequence `sin(2πfn/f_s+φ)`. Vì thêm bội nguyên f_s không đổi các sample phase, các tần số khác nhau có thể cho sequence trùng/khó phân biệt nếu không giới hạn băng thông. Với tín hiệu band-limited lý tưởng, điều kiện lấy mẫu cần xét Nyquist; tín hiệu thực cần anti-alias filter và thiết kế phù hợp.

Ví dụ f_s=8000, tone7000Hz có phase ở sample n tương đương−1000Hz modulo8000, nên các sample không biểu diễn một tone7000Hz độc lập như mong muốn. Lab chỉ tạo300/600/1200Hz và biến thiên±1%, thấp hơn Nyquist4000Hz. Không resample bằng cách đổi header, cũng không gọi `x[::2]` là downsampling có anti-alias đúng cho mọi tín hiệu.

## 6. Diagonal CE và hai loại multi-positive objective

Diagonal CE giả định mỗi row có một target index. Nếu hai candidate là hai caption hợp lệ cùng nội dung, phạt một caption chỉ vì khác index tạo false negative. Cách sửa phụ thuộc mục tiêu dữ liệu:

**Positive-set mass** của lab tối đa tổng probability trên tập đúng. Nếu một positive đã nhận gần hết mass, loss có thể nhỏ dù các positive khác có score thấp. **Mean positive log-probability** trung bình `−log p_j` trên các positive buộc phân bố chú ý tới từng positive nhiều hơn. Hai objective không bằng nhau và có thể ưu tiên khác nhau; bài không nhận loại thứ nhất là bản tái hiện nguyên gốc CLIP hoặc supervised contrastive loss chuẩn mọi nơi.

Khi toàn batch cùng lớp, positive-set là toàn candidate set, numerator bằng denominator, loss0. Không còn negative để học phân biệt. Lab full-batch gồm cả sáu lớp, label mask dùng giám sát có sẵn. Bài không lấy những record cùng màu nhưng khác hướng làm positive: cần giữ granularity đúng nhiệm vụ.

## 7. Temperature, cosine và confidence

L2normalize giúp cosine có thang giới hạn; chiaτ nhỏ làm distribution sau softmax sắc hơn và có thể tăng độ lớn gradient, không làm raw cosine thành xác suất đúng. Nếu candidate set thay đổi, denominator/softmax probability đổi dù query và positive similarity không đổi.

Một closed-set classifier luôn có argmax, kể cả input im lặng hoặc ảnh đen. Trong fixture, cả24input giống nhau cho cùng embedding/output, class-balanced test nên constant prediction đúng4/24. Con số đó không là detector “không có thông tin”. Muốn abstain/OOD cần policy và calibration/evaluation riêng; lab chỉ xử lý modality missing rõ ràng bằng None.

## 8. Alignment qua cầu text chưa bảo đảm mọi cặp đều tốt

Nếu ảnh và audio được học với text rất chung như “sự kiện”, chúng có thể cùng gần text nhưng không giữ chi tiết để phân biệt với nhau. Shared-space training qua một modality làm cầu có thể hữu ích, nhưng kết quả phụ thuộc coverage và objective; không dùng tính bắc cầu của ngôn ngữ tự nhiên như một proof về cosine.

Lab thêm trực tiếp I–A objective ngoài I–T/A–T. Không có ablation bỏ I–A rồi khẳng định nó cần thiết hoặc vô ích. Muốn nghiên cứu vai trò đó, cố định split/seed/budget, chạy baseline bỏ cặp theo kế hoạch và so các hướng retrieval; không chỉ xem loss tổng.

## 9. Missing, corrupt và contradictory là các trạng thái khác nhau

**Missing:** không có file/feature nhánh đó, có thể route sang nhánh còn lại. **Corrupt:** file không decode/shape sai hoặc tín hiệu không dùng được; cần xử lý trước hoặc đánh dấu chất lượng. **Contradictory:** mỗi file hợp lệ nhưng nội dung/sự kiện không khớp, hoặc thật sự hai nguồn có quan điểm khác nhau.

Đưa zero waveform vào encoder vẫn cho một embedding qua bias/normalizer, không tương đương bỏ nhánh. Ghép audio của lớp khác vẫn giữ tensor shape hợp lệ, nên schema chỉ kiểm shape không phát hiện được semantic mismatch. Artifact ID, event/time và provenance giúp phát hiện lỗi ghép; score disagreement có thể là tín hiệu, chưa là proof nguồn nào sai.

Equal-weight late fusion giả định score có thể trộn theo trọng số cố định. Trong bài, mismatched fixture dùng roll4 trên test sorted theo lớp, làm24cặp khác lớp; kết quả3/24 không là adversarial robustness đo trên đời thực. Không chọn w tốt nhất trên chính fixture rồi trình bày như tổng quát.

## 10. Split và metric theo đơn vị ngữ nghĩa

Record/group là một cảnh tổng hợp với ba phương thức. Group không cắt qua train/validation/test; nhưng tất cả dùng cùng sáu concept, cùng caption template, cùng generator và rule. Đó là held-out instance variation, chưa held-out class/source/domain. Text caption còn lặp nguyên văn giữa split một cách chủ đích; metric không kiểm khả năng hiểu câu diễn đạt mới.

Metric top1 là **class-match**: gallery24record có4positive/lớp. Một query trả candidate khác ID nhưng cùng lớp vẫn đúng. Đây không là pair-ID recall; muốn pair-ID retrieval cần nội dung đủ để phân biệt instance và annotation phù hợp. Đổi metric mà không đổi dữ liệu có thể phạt các đáp án tương đương.

Random chance cho label cân bằng là1/6 theo kỳ vọng của chọn candidate độc lập ngẫu nhiên; random initialization không có nghĩa mỗi lượt nhất thiết đạt4/24. Rule baseline biết codebook cũng24/24, nên không diễn giải neural score100% thành vượt phương pháp đơn giản. Dataset được tạo để học cơ chế.

## 11. Generation cần thêm gì?

Encoder retrieval cung cấp vector similarity; để sinh answer token cần conditional decoder và objective. Visual/audio connector có thể biến feature thành chuỗi mà LLM sử dụng, nhưng cần cách tổ chức mask/position/context nhất quán và training phù hợp. Mất grounding trong training có thể khiến output trôi theo prior ngôn ngữ dù input media thay đổi.

Sinh ảnh/audio lại cần representation đầu ra và decoder phù hợp, chẳng hạn hệ thống latent/codec hoặc quá trình denoising/autoregressive tùy kiến trúc. Tone trong lab là công thức sin ở generator, không kết quả model sinh âm. Không có speech recognition, image caption decoder hoặc text-to-speech trong code.

## 12. Bộ kiểm đủ để nghiên cứu tiếp

Một experiment có thể mở rộng theo ba hướng riêng: dữ liệu thật có annotation/source consent; model pretrained với processor đúng version; hoặc generation/grounding task. Mỗi hướng cần baseline và evaluation mới. Đừng thay encoder lớn rồi giữ metric sáu lớp để kết luận model biết mọi ảnh/âm.

Lưu version dataset/split/preprocess/model, metadata region/time, raw metric theo slice và trường hợp thất bại. Inference bundle của bài giữ state/preprocess/spec/data hash và kiểm digest/shape/dtype/finite. Digest không xác thực người gửi, nên chưa là cơ chế an toàn cho artifact tùy ý từ bên ngoài. Không có optimizer/sampler/RNG state để exact resume, cũng không có load test, streaming ASR hoặc multimodal agent gateway.

Kết nối bài25–26: claim cần evidence đúng nguồn và quyền; ảnh/audio chứa nội dung cũng đi qua trust boundary. Kết nối bài37–40: nhiều token và preprocessing nặng ảnh hưởng memory/latency, cần đo end-to-end thay vì chỉ encoder FLOP. [Nguồn và phạm vi đã đọc](../SOURCES.md).
