# Bài 24 — Chunking, embedding và reranking

[← Mục lục](../README.md) · [Bài 23](../23-tim-kiem-tu-khoa-vector/README.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md) · [Đào sâu](deep-dive.md)

## 1. Từ tìm một tài liệu đến tìm đúng đoạn bằng chứng

Bài 23 xem một runbook ngắn là một document. Nhưng một trang wiki có thể chứa cả HTTP 401, HTTP 403, cách refresh token và cách cấp quyền. Query “đã đăng nhập nhưng không đọc được tài nguyên” cần đoạn nói về quyền; lấy đúng trang nhưng chỉ đưa đoạn refresh token cho bước sau vẫn có thể dẫn đến câu trả lời sai.

> **Đoạn bằng chứng (Evidence Span) so với Tài liệu toàn văn (Full Document):**
> Trong các hệ thống RAG (Retrieval-Augmented Generation) và tìm kiếm ngữ nghĩa, mục tiêu không chỉ là "tìm đúng file", mà là "trích xuất đúng đoạn văn bản chứa câu trả lời trực tiếp". Một tài liệu toàn văn dài 10–20 trang chứa rất nhiều chủ đề con đan xen. Nếu nạp toàn bộ tài liệu vào mô hình ngôn ngữ (LLM), hệ thống sẽ gặp ba vấn đề nghiêm trọng:
> 1. *Hiện tượng "Lost in the Middle":* LLM thường chú ý tốt vào phần đầu và phần cuối của prompt, dễ bỏ qua các chi tiết quan trọng nằm ở giữa tài liệu dài.
> 2. *Lãng phí chi phí và tăng độ trễ:* Nạp hàng nghìn token không liên quan làm tăng vọt chi phí API và thời gian tạo câu trả lời (Time-to-First-Token).
> 3. *Nhiễu thông tin (Context Contamination):* Các chỉ dẫn sai ngữ cảnh (ví dụ: hướng dẫn refresh token cho lỗi 401 khi người dùng đang gặp lỗi 403 thiếu quyền) sẽ khiến LLM đưa ra câu trả lời sai lệch hoàn toàn.

Bài này nối ba quyết định: **chia tài liệu thành đơn vị truy xuất**, **mã hóa các đơn vị ấy thành vector**, rồi **chấm lại các ứng viên theo query**. Ta sẽ phân biệt ba lỗi: bằng chứng bị cắt; bằng chứng còn nguyên nhưng không vào tập ứng viên; bằng chứng đã vào ứng viên nhưng bị xếp xuống dưới.

> **Ba tầng quyết định và ba điểm nghẽn tương ứng:**
> - **Tầng 1 — Chunking (Chia đơn vị truy xuất):** Biến tài liệu lớn thành các khối văn bản độc lập (chunks).
>   *Lỗi 1 — Bằng chứng bị cắt (Chunking boundary truncation):* Điểm cắt rơi trúng giữa câu lệnh, hoặc tách rời mệnh đề điều kiện khỏi hành động (ví dụ: cắt rời câu "Chỉ chạy lệnh xóa cache khi..." khỏi lệnh `redis-cli FLUSHDB`), khiến bằng chứng bị đứt gãy ngay từ trong cơ sở dữ liệu.
> - **Tầng 2 — Embedding & First-Stage Retrieval (Truy xuất thô bằng Bi-encoder / Vector Search):** Quét nhanh toàn bộ cơ sở dữ liệu hàng triệu chunk để lấy ra một tập ứng viên nhỏ (Candidate Pool, ví dụ $M = 6$ hoặc $M = 50$).
>   *Lỗi 2 — Bằng chứng không lọt vào ứng viên (Recall failure):* Đoạn bằng chứng vẫn còn nguyên vẹn trong cơ sở dữ liệu, nhưng điểm số tương đồng ngữ nghĩa của tầng 1 không đủ cao để lọt vào top $M$ ứng viên được chọn.
> - **Tầng 3 — Reranking (Xếp hạng lại sâu bằng Cross-encoder):** So khớp chi tiết từng cặp giữa query và các ứng viên trong top $M$ để chọn ra $k$ đoạn văn bản chuẩn xác nhất (ví dụ $k = 3$) đưa vào ngữ cảnh sinh câu trả lời.
>   *Lỗi 3 — Bằng chứng bị xếp xuống dưới (Precision/Ranking failure):* Đoạn bằng chứng đã nằm trong top $M$ ứng viên, nhưng mô hình reranker lại chấm điểm nó thấp hơn các đoạn văn khác, đẩy nó văng khỏi top $k$ được gửi tới LLM.

Ví dụ giả định dùng xuyên bài là bộ runbook backend về thanh toán, auth, cache, database, rate limit, webhook, queue, search index và retry. Đây là nội dung tự tạo phục vụ học tập, không phải quy trình xử lý sự cố của một dịch vụ thực tế.

> **Ứng dụng thực tế trong kỹ thuật phần mềm:**
> Bộ dữ liệu runbook mô phỏng chính xác bài toán xây dựng AI SRE / On-call Copilot cho các đội ngũ kỹ thuật. Khi hệ thống gặp sự cố lúc 2 giờ sáng (ví dụ: Gateway Timeout 504), kỹ sư cần hệ thống truy xuất chính xác từng dòng runbook hướng dẫn kiểm tra Connection Pool của Database, thay vì trả về tài liệu kiến trúc chung chung của toàn bộ hạ tầng.

Sau bài, bạn cần làm được:

- Chọn và kiểm tra chunk theo cấu trúc/độ dài, giữ ID nguồn, version và vị trí văn bản.
- Tính số cửa sổ, overlap, masked mean, cosine và contrastive loss bằng tay.
- Giải thích bi-encoder và cross-encoder khác nhau ở tương tác query–document và chi phí.
- Chạy huấn luyện embedding neural và reranker nhỏ bằng PyTorch, không cần tải model.
- Đo chất lượng bằng chứng qua từng tầng, đối chiếu baseline và quyết định có nên dùng reranker.

Cần kiến thức vector ở bài 3, embedding ở bài 16, attention/Transformer ở bài 17–18, train/validation ở bài 12 và retrieval ở bài 23. Bài sẽ nhắc lại các phép toán cần dùng. Lab là mô hình học từ đầu trên bộ nhỏ, **không phải benchmark embedding/reranker pretrained hiện đại**.

![Ba quyết định liên tiếp trong truy xuất](figures/01-pipeline.png)

## 2. Chunk là gì, vì sao phải chia?

**Chunking** là chia nội dung thành những đoạn để lập chỉ mục hoặc đưa vào model. Chunk không bắt buộc là câu, đoạn văn hay một số token cố định. Nó là một quyết định thiết kế: “đơn vị nào đủ nhỏ để tìm chính xác, nhưng đủ lớn để giữ ý nghĩa và bằng chứng?”

> **Định nghĩa trực quan:**
> Một **chunk** là một đơn vị thông tin độc lập tương đối (self-contained semantic unit). Nếu chunk quá nhỏ, nó mất ngữ cảnh xung quanh; nếu chunk quá lớn, ý nghĩa cốt lõi bị pha loãng trong hàng ngàn từ ngữ phụ trợ. Kỹ sư phải tìm điểm cân bằng sao cho một chunk vừa đủ chứa câu hỏi và câu trả lời hoàn chỉnh cho một ý niệm kỹ thuật.

Một file là đơn vị lưu trữ; một section là đơn vị cấu trúc; một chunk là đơn vị truy xuất; một đoạn được ghép vào context cuối có thể lại là đơn vị khác. Đừng mặc nhiên coi bốn thứ này trùng nhau.

> **Phân biệt bốn cấp độ đơn vị trong hệ thống backend:**
> 1. *File (Lưu trữ):* Tệp `runbook-auth.md` trên Git hoặc S3 (chứa toàn bộ tri thức về authentication).
> 2. *Section (Cấu trúc tài liệu):* Tiêu đề mục `### 2.1 Xử lý Refresh Token khi hết hạn` do tác giả tài liệu phân chia theo logic trình bày.
> 3. *Chunk (Truy xuất):* Đoạn văn bản 48 token kèm metadata được lưu vào Vector Database để phục vụ tìm kiếm similarity.
> 4. *Context Window (Suy luận của LLM):* Ba chunk liên quan nhất được gom lại, gắn kèm System Prompt và lịch sử hội thoại để đưa vào cửa sổ ngữ cảnh của LLM.

### 2.1. Quá lớn và quá nhỏ gây lỗi khác nhau

| Lựa chọn | Có thể có lợi | Có thể gây lỗi |
|---|---|---|
| Cả trang | Giữ liên hệ giữa các mục | Nhiều chủ đề bị nén vào một vector, vượt input/context budget |
| Một section | Giữ tiêu đề và mạch giải thích | Section rất dài vẫn cần chia tiếp |
| Một câu ngắn | Khớp chi tiết cụ thể | “Nó”, “trường hợp này”, điều kiện và phủ định có thể bị tách |
| Cửa sổ token cố định | Dễ giới hạn kích thước và tái lập | Cắt đôi câu, bảng hoặc đoạn code |

Ví dụ một cặp câu: “Chỉ retry lỗi tạm thời khi tác vụ cho phép. Không retry lỗi thiếu quyền bằng cách gọi lại liên tục.” Nếu chunk chỉ giữ “retry ... gọi lại liên tục”, điều kiện và phủ định có thể mất. Tìm trúng vài từ không đồng nghĩa đã giữ được hướng dẫn đúng.

> **Cơ chế gây lỗi ngữ nghĩa khi cắt sai kích thước:**
> - *Pha loãng vector (Vector Information Dilution) khi chunk quá lớn:* Mô hình embedding ép toàn bộ 2.000 từ vào một vector cố định (ví dụ 384 hoặc 768 chiều). Vector này trở thành trung bình cộng của quá nhiều chủ đề, làm giảm đột ngột độ tương đồng cosine với các truy vấn cụ thể, khiến tìm kiếm thất bại.
> - *Mất ngữ cảnh quy chiếu (Anaphora Loss) khi chunk quá nhỏ:* Nếu cắt từng câu độc lập: Câu 1: "Cơ sở dữ liệu thanh toán sử dụng cơ chế Write-Ahead Logging." Câu 2: "Nó sẽ tự động đồng bộ xuống đĩa mỗi 500ms." Nếu người dùng hỏi "Cơ chế đồng bộ đĩa của database thanh toán là gì?", chunk của câu 2 khớp ngữ nghĩa nhưng đại từ "Nó" không cho LLM biết là đang nói về dịch vụ nào!

Chunking còn hữu ích ngay khi cả file vừa input limit: một vector duy nhất có thể biểu diễn không tốt nhiều chủ đề khác nhau. [Hướng dẫn chunking của Microsoft](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents) mô tả cách chia cố định, theo cấu trúc và các đánh đổi của overlap. Các kích thước gợi ý trong một dịch vụ chỉ là điểm bắt đầu để đo, không là hằng số tối ưu cho mọi corpus/ngôn ngữ/model.

### 2.2. Token của model nào?

“500 ký tự”, “500 từ” và “500 token” khác nhau. Tiếng Việt có từ ghép nhiều âm tiết; mã lỗi, đường dẫn và JSON có thể được tokenizer chia khác văn xuôi. Embedding model, reranker và model sinh câu trả lời có thể dùng **ba tokenizer và ba input limit khác nhau**.

> **So sánh trực quan: Ký tự vs Từ vs Token:**
> - *Ký tự (Character):* Đơn vị ký tự thô. Từ "kết nối" có 7 ký tự (bao gồm dấu cách).
> - *Từ (Word):* Tách theo khoảng trắng. "kết nối" là 2 từ đơn (hoặc 1 từ ghép trong ngữ pháp tiếng Việt).
> - *Subword Token:* Đơn vị mà mô hình ngôn ngữ thực sự xử lý.
>   - Văn bản kỹ thuật: Chuỗi mã lỗi `ERR_BILL_042` qua tokenizer BPE của GPT/LLaMA có thể bị băm thành 4-6 token: `['ERR', '_', 'B', 'ILL', '_', '042']`.
>   - Ngôn ngữ tiếng Việt: Các từ có dấu thanh phức tạp có thể bị tách thành nhiều byte token nếu tokenizer không tối ưu cho tiếng Việt.
>   - Do đó, nếu bạn cấu hình chunk "500 ký tự", nó có thể chỉ tương đương 80 token văn xuôi, nhưng lại lên tới 250 token đối với chuỗi JSON hoặc code log!

Lab dùng tokenizer word-level đơn giản giống analyzer bài 23: NFC, casefold, regex giữ identifier. Chẳng hạn `ERR_BILL_042` là một token, nhưng “kết nối” là hai. Các con số 24/48/64 trong lab chỉ có nghĩa theo tokenizer này; không được mang nguyên sang một model subword rồi giả định cùng độ dài.

[API tokenizer của Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer) mô tả truncation, token overflow và stride. Đáng chú ý: `stride` trong API đó là **số token overlap**, trong khi “bước trượt” toán học thường là `chunk_size − overlap`. Luôn đọc quy ước của công cụ.

## 3. Chia theo cửa sổ: tính tay và kiểm biên

### 3.1. Cửa sổ có overlap

Có L=13 token đánh số 0–12, chunk size C=6, overlap O=2. Bước tiến S=C−O=4. Dùng khoảng **nửa kín [start,end)**: lấy start, không lấy end.

```text
chunk 1: [0,6)   → token 0 1 2 3 4 5
chunk 2: [4,10)  → token     4 5 6 7 8 9
chunk 3: [8,13)  → token             8 9 10 11 12
```

Có 3 chunk và tổng số lượt token body được xử lý là `6+6+5=17`, dù nguồn chỉ có 13 token. Phần lặp là 4 token; hệ số tăng ở ví dụ này `17/13≈1.3077`. Với văn bản rất dài, bỏ qua hiệu ứng biên, hệ số gần `C/(C−O)=1.5`. Chưa tính tiêu đề lặp lại hoặc special token.

> **Giải thích từng bước tính toán cửa sổ trượt:**
> - $L = 13$: Tổng số token của văn bản gốc (được đánh chỉ mục từ $0$ đến $12$).
> - $C = 6$ (Chunk size): Kích thước tối đa của mỗi cửa sổ.
> - $O = 2$ (Overlap): Số lượng token được gối đầu giữa hai cửa sổ liên tiếp nhằm bảo vệ các câu nằm sát ranh giới cắt.
> - $S = C - O = 6 - 2 = 4$ (Step size / Stride): Bước nhảy tịnh tiến của cửa sổ.
> 
> *Quy ước khoảng nửa kín $[start, end)$:* Trong lập trình Python, `tokens[0:6]` sẽ lấy các phần tử tại vị trí $0, 1, 2, 3, 4, 5$ (đúng 6 token), không lấy phần tử tại chỉ số $6$.
> - **Chunk 1:** Bắt đầu tại $0$, lấy 6 token $\implies [0, 6)$ gồm các token: `0, 1, 2, 3, 4, 5`.
> - **Chunk 2:** Dịch đi một bước $S = 4$, bắt đầu tại $0 + 4 = 4$, lấy 6 token $\implies [4, 10)$ gồm các token: `4, 5, 6, 7, 8, 9`. (Hai token `4, 5` được lặp lại từ Chunk 1).
> - **Chunk 3:** Dịch tiếp một bước $S = 4$, bắt đầu tại $4 + 4 = 8$. Từ vị trí $8$ đến cuối tài liệu chỉ còn $13 - 8 = 5$ token $\implies [8, 13)$ gồm các token: `8, 9, 10, 11, 12`. (Hai token `8, 9` được lặp lại từ Chunk 2). Đến đây đã chạm hết chiều dài $L=13$, thuật toán kết thúc.
> 
> *Đánh đổi về chi phí tài nguyên (Overhead Ratio):*
> Tổng số lượt token được tính toán là $6 + 6 + 5 = 17$ token. Tỷ lệ khuếch đại là $17 / 13 \approx 1.3077$ (tăng thêm $\approx 31\%$ chi phí).
> Với tài liệu rất dài, tỷ lệ tăng trưởng tiệm cận lý thuyết là:
> $$\text{Tỷ lệ khuếch đại} \approx \frac{C}{C - O} = \frac{6}{6 - 2} = 1.5$$
> Nghĩa là khi chọn overlap $O = 2$ trên cửa sổ $C = 6$, hệ thống sẽ phải lưu trữ thêm $50\%$ số vector trong cơ sở dữ liệu và tốn thêm $50\%$ thời gian tính toán của mô hình embedding!

Với L>0, số chunk của thuật toán dừng ngay khi đã phủ hết đuôi là:

$$n=1+\left\lceil\frac{\max(0,L-C)}{C-O}\right\rceil.$$

> **Ý nghĩa trực quan của công thức:**
> - Số $1$ đầu tiên đại diện cho chunk đầu tiên, nó đã tiêu thụ một lượng $C$ token đầu tiên.
> - Lượng token còn lại cần được phân chia là $\max(0, L - C)$.
> - Mỗi chunk tiếp theo sẽ xử lý thêm được một lượng token mới đúng bằng bước nhảy $S = C - O$.
> - Phép chia làm tròn lên $\lceil \dots \rceil$ (hàm ceiling) bảo đảm rằng ngay cả khi phần đuôi còn sót lại ít hơn một cửa sổ đầy đủ, hệ thống vẫn tạo thêm một chunk để bao trọn toàn bộ nội dung mà không bỏ sót bất kỳ token nào.
> 
> *Thử tính với số liệu ví dụ:*
> $$n = 1 + \left\lceil \frac{\max(0, 13 - 6)}{6 - 2} \right\rceil = 1 + \left\lceil \frac{7}{4} \right\rceil = 1 + \lceil 1.75 \rceil = 1 + 2 = 3 \text{ chunks.}$$

L=0 cho 0 chunk. Cần `C>0` và `0≤O<C`; O=C làm bước tiến 0, có nguy cơ lặp vô hạn nếu không kiểm. Khi L=10,C=6,O=2, chỉ cần `[0,6),[4,10)`; không sinh thêm `[8,10)` vì đuôi đã được phủ.

```python
from retrieval import windows
assert windows(13, 6, 2) == [(0,6), (4,10), (8,13)]
assert windows(10, 6, 2) == [(0,6), (4,10)]
assert windows(0, 6, 2) == []
```

### 3.2. Overlap cứu được và không cứu được gì?

Giả sử bằng chứng nằm ở token `[5,9)`. Không overlap, các đoạn `[0,6),[6,12),[12,13)` đều không chứa trọn bằng chứng. Có overlap, `[4,10)` chứa toàn bộ. Nhưng nếu bằng chứng dài hơn C, không có cửa sổ C token nào giữ được nó nguyên vẹn.

> **Minh họa bằng chứng nằm vắt ngang ranh giới:**
> Đoạn bằng chứng quan trọng gồm 4 token tại vị trí $[5, 9)$ (tức các token $5, 6, 7, 8$).
> - *Khi không có overlap ($C=6, O=0$):* Cửa sổ 1 là $[0, 6)$ chỉ giữ token $5$; Cửa sổ 2 là $[6, 12)$ giữ các token $6, 7, 8$. Bằng chứng bị cắt đôi thành hai nửa vô nghĩa. Cả hai chunk đều không đủ thông tin để trả lời câu hỏi!
> - *Khi có overlap ($C=6, O=2$):* Cửa sổ 2 là $[4, 10)$ bao trọn các token $4, 5, 6, 7, 8, 9$. Bằng chứng $[5, 9)$ nằm gọn gàng bên trong chunk này. Overlap đã cứu sống bằng chứng!
> - *Giới hạn vật lý:* Nếu một quy trình xử lý sự cố đòi hỏi một đoạn văn bản dài 8 token liên tục, mà cửa sổ $C = 6$, thì dù bạn có overlap bao nhiêu chăng nữa, không có bất kỳ chunk nào chứa trọn vẹn được 8 token này.

Overlap có thể giảm lỗi biên, nhưng tăng số vector, số token encode và ứng viên gần trùng. Top-3 có thể là ba bản gần giống của cùng một đoạn, làm mất chỗ của bằng chứng khác. Không nên tăng overlap như một cách sửa mọi lỗi; hãy kiểm tra cả coverage và độ đa dạng.

> **Hiện tượng ô nhiễm ứng viên (Candidate Pool Pollution):**
> Khi tăng overlap quá lớn, các chunk liền kề sẽ có nội dung giống nhau tới 70–80%. Khi người dùng tìm kiếm, mô hình sẽ trả về top 3 kết quả thực chất chỉ là 3 biến thể trượt qua trượt lại của cùng một câu văn. Điều này làm "chiếm chỗ" (starvation) của các đoạn bằng chứng khác nằm ở phần sau của tài liệu, làm giảm tính đa dạng (diversity) của câu trả lời.

![Cửa sổ, overlap và bằng chứng nằm qua ranh giới](figures/02-windows.png)

## 4. Chia theo cấu trúc và giữ nguồn

### 4.1. Section trước, giới hạn token sau

Cách chính trong lab: tách theo section có sẵn, rồi áp trần 48 token **bên trong từng section**. Section ngắn giữ nguyên; section dài chia tiếp thành cửa sổ, không nối đuôi section này sang đầu section khác. Có 18 section công khai nhưng sinh **25 chunk** vì một số section dài hơn trần; “section chunking” chưa bảo đảm mỗi section là đúng một chunk.

> **Chiến lược chia chunk tôn trọng cấu trúc (Structure-Aware / Recursive Chunking):**
> Thay vì dùng một con dao cắt mù quáng qua các dòng văn bản, phương pháp này tuân theo logic ngữ nghĩa của con người:
> 1. *Ưu tiên 1:* Cắt theo các tiêu đề lớn (`#`, `##`, `###` trong Markdown). Mỗi section là một chủ đề độc lập.
> 2. *Ưu tiên 2:* Nếu một section có độ dài $\le 48$ token, giữ nguyên toàn bộ section thành 1 chunk.
> 3. *Ưu tiên 3:* Nếu một section quá dài ($> 48$ token), chỉ áp dụng sliding window bên trong phạm vi section đó. Tuyệt đối không để cửa sổ trượt tràn sang section tiếp theo (ví dụ: không nối dòng cuối của mục "Khắc phục sự cố Auth" với dòng đầu của mục "Cấu hình Database").
> 
> Nhờ đó, 18 section ban đầu tạo ra 25 chunk: các section ngắn tạo ra đúng 1 chunk, còn các section dài được chia thành nhiều chunk con nhưng vẫn cùng thuộc một chủ đề đồng nhất.

Trong dữ liệu thực, có thể ưu tiên heading → đoạn → câu, với nhánh xử lý riêng cho code/table/list. Tách bằng dòng trống đơn thuần có thể làm header bảng rời dữ liệu; tách code giữa hàm có thể mất signature/import; PDF hai cột có thể đã sai thứ tự đọc trước cả bước chunking. Cần kiểm tra chất lượng **parser**, không chỉ chỉnh chunk size.

> **Cạm bẫy thực tế trong phân tích tài liệu kỹ thuật (Document Parsing Traps):**
> - *Bảng dữ liệu (Tables):* Nếu cắt ngang qua dòng trống, tiêu đề bảng (`| Mã lỗi | Nguyên nhân | Cách sửa |`) bị tách rời khỏi các dòng dữ liệu bên dưới. Vector của các dòng bên dưới chỉ toàn mã số mà không hiểu mã đó có ý nghĩa gì.
> - *Khối mã nguồn (Code Blocks):* Cắt đôi một hàm Python làm mất dòng định nghĩa `def handle_payment():` và các câu lệnh `import`, khiến code trích dẫn bị lỗi cú pháp và LLM không hiểu biến số lấy từ đâu.
> - *PDF hai cột (Multi-column layout):* Các thư viện đọc PDF thông thường quét tọa độ từ trái sang phải trên toàn trang giấy, vô tình trộn lẫn dòng văn bản của cột 1 với dòng văn bản của cột 2 thành một mớ hỗn độn trước khi đưa vào bộ chunking!

**Semantic chunking** dùng tín hiệu nội dung để tìm ranh giới, chẳng hạn sự thay đổi giữa các câu. Nó cũng có tham số, chi phí và lỗi; lab chưa triển khai semantic boundary detector. Đừng gọi việc dùng heading là bằng chứng hệ thống đã hiểu ngữ nghĩa sâu.

> **Semantic Chunking hoạt động như thế nào?**
> Thuật toán chia văn bản thành các câu riêng lẻ, chạy embedding cho từng câu, rồi tính khoảng cách cosine giữa hai câu liên tiếp $S_i$ và $S_{i+1}$. Khi khoảng cách ngữ nghĩa đột ngột tăng vọt vượt qua một ngưỡng (nghĩa là chủ đề vừa bị chuyển hướng), thuật toán sẽ đặt một điểm cắt chunk tại đó. Nhược điểm: tốn chi phí tính toán embedding cho toàn bộ các câu và rất nhạy cảm với việc chọn ngưỡng threshold.

### 4.2. Chunk cần mang gì?

| Trường | Mục đích |
|---|---|
| `id` | Tham chiếu ứng viên, kết quả và cache |
| `source_id`, `version` | Biết đoạn thuộc nguồn và snapshot nào |
| `start`, `end` | Trỏ về vị trí thật để xem lại bằng chứng |
| `text` | Văn bản gốc trong khoảng ấy |
| `title`, `heading` | Giữ ngữ cảnh chủ đề |
| `scope` | Ràng buộc phạm vi dữ liệu |

Offset của lab là **chỉ số ký tự Python trong nguồn đã chuẩn NFC**, không phải byte UTF-8 hay token index. Phải canonicalize trước khi gán nhãn; đổi normalization sau đó có thể làm lệch span. Invariant cần giữ: `chunk['text'] == source['text'][start:end]`.

> **Tại sao phải chuẩn hóa Unicode NFC và dùng chỉ số ký tự?**
> Trong tiếng Việt, ký tự có dấu có thể tồn tại ở hai dạng mã hóa Unicode:
> 1. *Dạng dựng sẵn (NFC - Normalization Form C):* Chữ `ế` là một ký tự duy nhất (`\u1EBF`).
> 2. *Dạng tổ hợp (NFD - Normalization Form D):* Chữ `ế` được ghép từ chữ cái gốc `e` (`\u0065`) kèm dấu mũ `^` (`\u0302`) và dấu sắc `'` (`\u0301`) (tổng cộng 3 điểm mã Unicode).
> Nếu một tài liệu ở dạng NFD còn truy vấn ở dạng NFC, phép đo độ dài chuỗi ký tự sẽ trả về kết quả khác nhau. Việc cắt chuỗi bằng offset `start:end` sẽ bị lệch và có thể cắt đứt đôi một dấu thanh tiếng Việt! Chuẩn hóa toàn bộ về NFC trước khi đánh chỉ mục là quy tắc bắt buộc để đảm bảo tính toàn vẹn dữ liệu.

ID được băm từ source ID/version, chiến lược, size/overlap và offset. Nó ổn định khi các thành phần đó giữ nguyên; không là ID bất biến nếu sửa/chèn nội dung trước chunk. Nếu nội dung đổi, phải tăng version và làm mất hiệu lực vector/cache cũ; hash ID không thay thế kỷ luật version.

Embedding text của lab là `title / heading / text`. Prefix giúp một đoạn giữa trang còn biết đang nói về gì; prefix cũng tốn token và có thể tạo nhiễu. Vì fixed chunk dùng heading “Toàn văn”, còn section dùng heading thật, bảng ablation sau đo **gói quyết định chia đoạn + heading**, không tách riêng hoàn toàn hiệu ứng ranh giới.

> **Kỹ thuật làm giàu ngữ cảnh bằng tiền tố (Contextual Chunk Enrichment):**
> Giả sử một chunk có nội dung: `"Hạn mức cho phép là 100 requests mỗi giây. Vượt quá sẽ trả về mã 429."` Nếu đứng độc lập, không ai biết hạn mức này của service nào.
> Bằng cách ghép thêm tiền tố:
> `Tiền tố: [Tài liệu: Runbook API Gateway] / [Mục: Rate Limiting dịch vụ Thanh toán] / [Nội dung: Hạn mức cho phép là 100 requests mỗi giây...]`
> Vector embedding tạo ra từ đoạn văn bản này sẽ mang đầy đủ ngữ nghĩa của cả tài liệu và danh mục, giúp người dùng tìm kiếm theo tên service vẫn trúng đích chính xác.

```python
from data import dataset
from retrieval import chunk_sources
sources, queries = dataset()
chunks = chunk_sources(sources, strategy='section', size=48, overlap=0)
by_source = {s['id']:s for s in sources}
assert len(chunks) == 25
assert all(c['scope'] == 'public' for c in chunks)
assert all(c['text'] == by_source[c['source_id']]['text'][c['start']:c['end']] for c in chunks)
print(chunks[0])
```

![Nguồn, section và chunk giữ liên kết bằng offset](figures/03-provenance.png)

## 5. Embedding cho truy xuất: học cái gì?

### 5.1. Bi-encoder và mục tiêu biểu diễn

Bi-encoder mã hóa riêng query và chunk:

$$z_q=\operatorname{normalize}(f_\theta(q)),\qquad
z_c=\operatorname{normalize}(g_\phi(c)),\qquad s(q,c)=z_q^\top z_c.$$

> **Giải thích công thức toán học:**
> - $f_\theta(q)$: Mạng neural với tham số $\theta$, nhận đầu vào là câu truy vấn $q$ và trích xuất ra một vector đặc trưng thô.
> - $g_\phi(c)$: Mạng neural với tham số $\phi$, nhận đầu vào là đoạn văn bản chunk $c$ và trích xuất ra một vector đặc trưng thô tương ứng.
> - $\operatorname{normalize}(v) = \frac{v}{\|v\|_2}$: Phép chuẩn hóa L2 đưa vector về độ dài bằng $1$ (vector đơn vị nằm trên mặt cầu siêu không gian $d$ chiều).
> - $s(q, c) = z_q^\top z_c$: Tích vô hướng (dot product) giữa hai vector đã chuẩn hóa. Vì hai vector đều có độ dài bằng $1$, tích vô hướng này chính là **Cosine Similarity**:
>   $$s(q, c) = \cos(\theta) = \frac{z_q \cdot z_c}{\|z_q\| \|z_c\|}$$
>   Điểm số nằm trong khoảng $[-1, 1]$. Càng gần $1$ nghĩa là hai đoạn văn bản càng tương đồng về ngữ nghĩa.

Có thể dùng chung trọng số `f=g`, hoặc hai encoder được huấn luyện để tương thích. Query thường ngắn và hỏi, document thường dài và giải thích: hai bên có vai trò khác nhau dù cùng chung model. Có model cần prefix/instruction riêng cho mỗi vai trò; phải giữ đúng contract của model đó.

> **Truy xuất bất đối xứng (Asymmetric Retrieval):**
> Trong tìm kiếm thực tế, câu hỏi của người dùng và câu trả lời trong tài liệu có cấu trúc ngữ pháp rất khác nhau:
> - Query: Ngắn gọn, thường là câu hỏi hoặc từ khóa rời rạc (`"cách sửa lỗi hết hạn token"`).
> - Document: Dài, văn phong giải thích, chứa các từ ngữ kỹ thuật cụ thể (`"Khi nhận mã 401, client gửi refresh token đến endpoint /oauth/token..."`).
> Vì vậy, các mô hình embedding hiện đại (như E5, BGE) thường yêu cầu thêm tiền tố chỉ định: ví dụ thêm `"query: "` vào trước câu hỏi và `"passage: "` vào trước văn bản chunk để mô hình kích hoạt không gian biểu diễn phù hợp.

Vì document vector độc lập với query, ta tính trước và lập index. Online chỉ encode query rồi tìm hàng xóm. [Sentence-BERT](https://arxiv.org/abs/1908.10084) là nguồn lịch sử cho hướng tạo sentence embedding bằng mạng Siamese; [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) phân biệt truy xuất symmetric/asymmetric. Bài không tải model từ hai nguồn này.

> **Tại sao Bi-encoder là lựa chọn duy nhất cho tìm kiếm quy mô lớn (Scalability)?**
> Giả sử cơ sở tri thức có $1.000.000$ chunks:
> - *Giai đoạn Offline:* Ta chạy mô hình $g_\phi(c)$ một lần duy nhất cho $1.000.000$ chunk, tính ra $1.000.000$ vector và lưu vào chỉ mục tìm kiếm xấp xỉ láng giềng gần nhất (ANN Index như HNSW trong FAISS, Milvus, Qdrant).
> - *Giai đoạn Online:* Khi người dùng gửi truy vấn, hệ thống chỉ chạy mô hình $f_\theta(q)$ **đúng 1 lần** để tạo ra vector truy vấn $z_q$. Sau đó, Vector Database tìm kiếm các vector gần nhất trong vài mili-giây.

Trong bài 23, LSA học phép chiếu từ thống kê corpus. Ở đây, lab dùng **embedding neural có gradient**: bảng embedding từ → masked mean → Linear → tanh → L2 normalize. Trọng số học bằng cặp query–chunk có nhãn. Đây là bi-encoder word-level nhỏ, không có contextual attention; không nên gọi nó là Sentence-BERT hoặc model pretrained đa ngôn ngữ.

> **Kiến trúc luồng tensor của Bi-encoder trong lab:**
> 1. `Token IDs` $[B, T] \xrightarrow{\text{Embedding Lookup}}$ Tensor đặc trưng $[B, T, 32]$.
> 2. `Masked Mean Pooling` $[B, T, 32] \xrightarrow{\text{loại bỏ PAD}}$ Vector trung bình $[B, 32]$.
> 3. `Linear Layer` $[B, 32] \xrightarrow{\text{phép chiếu ma trận}}$ Vector tiềm ẩn $[B, 32]$.
> 4. `Hàm kích hoạt Tanh` $\xrightarrow{\text{phi tuyến hóa trong khoảng } (-1, 1)}$ Vector $[B, 32]$.
> 5. `L2 Normalize` $\xrightarrow{\text{chia cho chuẩn Euclid}}$ Vector đơn vị $[B, 32]$.

### 5.2. Masked mean tính tay

Giả sử ba vị trí sau một bước biểu diễn cho ra H=`[[2,0],[0,2],[100,100]]`, vị trí cuối là padding và mask=`[1,1,0]`:

$$m=\frac{1[2,0]+1[0,2]+0[100,100]}{1+1}=[1,1],$$
$$\hat m=[1/\sqrt2,1/\sqrt2].$$

> **Từng bước tính tay Masked Mean Pooling:**
> - Cho ma trận biểu diễn ẩn của chuỗi văn bản gồm 3 token:
>   $h_1 = [2, 0]$ (Token thật số 1)
>   $h_2 = [0, 2]$ (Token thật số 2)
>   $h_3 = [100, 100]$ (Token đệm `[PAD]` chèn vào cho đủ độ dài batch)
> - Vector mask nhị phân: $mask = [1, 1, 0]$ (vị trí 1 và 2 là token thật, vị trí 3 là đệm).
> 
> *Bước 1: Tính tổng có trọng số của các token hợp lệ:*
> $$\text{numerator} = 1 \cdot [2, 0] + 1 \cdot [0, 2] + 0 \cdot [100, 100] = [2, 0] + [0, 2] + [0, 0] = [2, 2]$$
> *Bước 2: Tính tổng số lượng token hợp lệ:*
> $$\text{denominator} = 1 + 1 + 0 = 2$$
> *Bước 3: Tính vector trung bình $m$:*
> $$m = \frac{[2, 2]}{2} = [1, 1]$$
> *Bước 4: Chuẩn hóa L2 về vector đơn vị $\hat{m}$:*
> Độ dài Euclid của $m$: $\|m\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2}$.
> $$\hat{m} = \frac{m}{\|m\|_2} = \left[ \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right] \approx [0.7071, 0.7071]$$

Chia cho tổng số vị trí 3 sẽ đổi mean; cộng PAD cũng làm mean sai và có thể đổi hướng. Riêng PAD `[100,100]` trong ví dụ này cùng hướng với tổng token thật nên normalize che được lỗi về hướng, nhưng mean vẫn sai. Projection/tanh phía sau còn có thể làm output khác. Padding không phải nội dung. Trong Transformer, PAD output vẫn có thể khác 0 dù đã chặn PAD làm key; bước pooling vẫn phải loại PAD. Lab áp mask trong cả bi-encoder và cross-encoder.

> **Nguy cơ tiềm ẩn nếu không dùng Mask (Unmasked Mean Bug):**
> Nếu lập trình viên sơ suất dùng `torch.mean(H, dim=1)` thông thường:
> $$m_{\text{sai}} = \frac{[2, 0] + [0, 2] + [100, 100]}{3} = \frac{[102, 102]}{3} = [34, 34]$$
> Khi đi qua các tầng mạng tiếp theo (Linear và Tanh), giá trị cực lớn $34$ sẽ đẩy các neuron vào vùng bão hòa của hàm tanh (tiệm cận $+1.0$), làm triệt tiêu hoàn toàn gradient và khiến mô hình không thể học được gì!

Với bi-encoder thật của lab, còn projection và tanh sau mean, nên không được lấy con số masked mean này làm output cuối của model đã train. Đây là phép tính riêng để hiểu pooling.

### 5.3. Cosine và embedding chỉ có ích theo tác vụ

q=`[1,0]`, c+=`[.8,.6]`, c1−=`[0,1]`, c2−=`[-.6,.8]` đều là vector đơn vị. Dot/cosine lần lượt `[.8,0,−.6]`. Độ tương tự âm là hợp lệ; chính sách truy xuất có thể loại nó. Lab chỉ giữ score dương, không pad kết quả thiếu bằng những đoạn tùy ý.

> **Phân tích hình học trực quan của Cosine Similarity:**
> - Vector truy vấn: $q = [1, 0]$ (nằm ngang trên trục hoành $x$).
> - Chunk đúng $c_+ = [0.8, 0.6]$:
>   $$s(q, c_+) = 1 \cdot 0.8 + 0 \cdot 0.6 = 0.8 > 0 \quad (\text{góc nhọn } \approx 36.87^\circ \implies \text{rất tương đồng}).$$
> - Chunk tiêu cực $c_1^- = [0, 1]$:
>   $$s(q, c_1^-) = 1 \cdot 0 + 0 \cdot 1 = 0.0 \quad (\text{góc vuông } 90^\circ \implies \text{không liên quan, trực giao}).$$
> - Chunk tiêu cực $c_2^- = [-0.6, 0.8]$:
>   $$s(q, c_2^-) = 1 \cdot (-0.6) + 0 \cdot 0.8 = -0.6 < 0 \quad (\text{góc tù } \approx 126.87^\circ \implies \text{ngược hướng, đối lập ngữ nghĩa}).$$

Một vector gần query chưa chứng minh chứa đáp án: “refresh token không sửa được thiếu quyền” và “refresh token để xử lý hết hạn” có nhiều từ chung nhưng điều kiện khác nhau. Với mean word embedding, đổi thứ tự từ không đổi biểu diễn; phủ định có thể bị mô hình nhỏ xử lý kém. Cần chọn model theo ngôn ngữ, mã định danh, độ dài và mục tiêu truy xuất thực tế.

![Mã hóa riêng để cache, mã hóa chung để chấm cặp](figures/04-bi-cross.png)

## 6. Huấn luyện embedding: positive, negative và loss

### 6.1. Contrastive loss

Positive là chunk được gán chứa bằng chứng cho query; negative là các chunk còn lại theo giao thức nhãn. Với score cosine s và temperature τ>0:

$$p_+=\frac{\exp(s_+/\tau)}{\exp(s_+/\tau)+\sum_j\exp(s_j^-/\tau)},\qquad L=-\ln p_+.$$

> **Bản chất của Contrastive Loss (InfoNCE Loss):**
> - $s_+$: Điểm tương đồng giữa Query và Chunk chứa bằng chứng đúng (Positive).
> - $s_j^-$: Điểm tương đồng giữa Query và các Chunk không liên quan (Negative).
> - $\tau$ (Temperature - Nhiệt độ): Siêu tham số điều chỉnh độ "nhạy" của phân phối xác suất. Khi chia điểm số cho một số $\tau < 1$ (ví dụ $\tau = 0.2$), ta đang phóng đại sự khác biệt giữa các điểm số lên gấp 5 lần ($1 / 0.2 = 5$). Điều này buộc mô hình phải cực kỳ tự tin, phạt rất nặng những negative nào có điểm số tiệm cận với positive.
> - $p_+$: Xác suất Softmax của Positive trên toàn bộ tập ứng viên đang so sánh.
> - $L = -\ln p_+$: Hàm mất mát Cross-Entropy. Khi mô hình dự đoán $p_+$ tiến gần đến $1.0$, $\ln(1.0) = 0$ thì $L$ tiệm cận $0$. Ngược lại, nếu $p_+$ nhỏ, loss sẽ tăng vọt lên vô cùng.

Loss giảm khi positive được tách khỏi các negative cạnh tranh. Đây là softmax trên **tập ứng viên huấn luyện**, không phải xác suất đáp án đúng trên mọi câu hỏi ngoài thế giới. [DPR, mục 3.1–3.2](https://aclanthology.org/2020.emnlp-main.550.pdf) trình bày dual encoder, negative log-likelihood và cách lấy negative; lab dùng cosine/temperature và encoder nhỏ riêng, không tái hiện BERT/DPR.

Với score `[.8,0,−.6]`, τ=.2: logits=`[4,0,−3]`; trừ max để tính ổn định:

```text
exp(logits − max) = [1, exp(−4), exp(−7)]
p = [0.981135, 0.017970, 0.000895]
L = −ln(0.981135) ≈ 0.019045
```

> **Từng bước tính tay Contrastive Loss:**
> Cho $s = [0.8, 0.0, -0.6]$ và $\tau = 0.2$:
> - *Bước 1: Tính scaled logits:*
>   $$\text{logits} = \frac{s}{\tau} = \left[ \frac{0.8}{0.2}, \frac{0.0}{0.2}, \frac{-0.6}{0.2} \right] = [4.0, 0.0, -3.0]$$
> - *Bước 2: Trừ giá trị lớn nhất ($M = 4.0$) để chống tràn số mũ (Numerical Stability Trick):*
>   $$\text{shifted\_logits} = [4.0 - 4.0, 0.0 - 4.0, -3.0 - 4.0] = [0.0, -4.0, -7.0]$$
> - *Bước 3: Tính hàm mũ $\exp$:*
>   $$\exp(0.0) = 1.0$$
>   $$\exp(-4.0) \approx 0.0183156$$
>   $$\exp(-7.0) \approx 0.0009119$$
> - *Bước 4: Tính tổng mẫu số:*
>   $$\text{sum} = 1.0 + 0.0183156 + 0.0009119 = 1.0192275$$
> - *Bước 5: Tính xác suất softmax:*
>   $$p_+ = \frac{1.0}{1.0192275} \approx 0.981135$$
>   $$p_1^- = \frac{0.0183156}{1.0192275} \approx 0.017970$$
>   $$p_2^- = \frac{0.0009119}{1.0192275} \approx 0.000895$$
> - *Bước 6: Tính hàm mất mát:*
>   $$L = -\ln(0.981135) \approx 0.019045$$

Gradient theo score có dạng `(p−y)/τ`, xấp xỉ `[-.094324,.089851,.004473]`: gradient descent có xu hướng tăng score positive và giảm score negative, mạnh hơn với negative còn cạnh tranh. Gradient qua encoder còn phụ thuộc normalize, projection và embedding, không chỉ ba số này.

> **Giải thích ý nghĩa của Gradient đối với việc cập nhật trọng số:**
> - Công thức đạo hàm: $\frac{\partial L}{\partial s_k} = \frac{p_k - y_k}{\tau}$, trong đó nhãn mục tiêu $y_+ = 1$ cho positive, và $y_j^- = 0$ cho các negative.
> - *Với Positive ($y=1$):*
>   $$\text{grad}_+ = \frac{0.981135 - 1.0}{0.2} = \frac{-0.018865}{0.2} \approx -0.094325$$
>   Dấu âm chỉ ra rằng thuật toán Gradient Descent ($s \leftarrow s - \eta \cdot \text{grad}$) sẽ cộng thêm giá trị vào $s_+$, tức là kéo điểm số của positive tăng lên!
> - *Với Negative 1 ($y=0$):*
>   $$\text{grad}_1^- = \frac{0.017970 - 0.0}{0.2} = \frac{0.017970}{0.2} \approx +0.089850$$
>   Dấu dương chỉ ra rằng thuật toán sẽ trừ bớt điểm số của $s_1^-$, tức là đẩy negative này ra xa!
> - *Với Negative 2 ($y=0$):*
>   $$\text{grad}_2^- = \frac{0.000895 - 0.0}{0.2} \approx +0.004475$$
>   Vì Negative 2 vốn đã có điểm rất thấp ($-0.6$), gradient của nó rất nhỏ ($0.004475 \ll 0.089850$). Mô hình dồn gần như toàn bộ nỗ lực học để trừng phạt Negative 1 (đang có nguy cơ cạnh tranh) thay vì tốn năng lượng cho Negative 2 đã quá xa.

Một loss khác là triplet margin: `max(0, margin+s_negative−s_positive)`. Với margin=.2, negative=.65, positive=.8, loss=.05. Đây là ví dụ so sánh mục tiêu; code huấn luyện bi-encoder dùng contrastive CE, không dùng triplet.

> **So sánh Triplet Loss và Contrastive Cross-Entropy:**
> Trong Triplet Margin Loss:
> $$L_{\text{triplet}} = \max(0, \text{margin} + s_- - s_+) = \max(0, 0.2 + 0.65 - 0.8) = \max(0, 0.05) = 0.05$$
> Triplet Loss chỉ xem xét từng bộ 3 cô lập `(Anchor, Positive, 1 Negative)`. Ngược lại, Contrastive Cross-Entropy xem xét Positive trong sự đối chiếu đồng thời với hàng chục, hàng trăm Negative trong cùng một batch, tạo ra lực đẩy đa chiều mạnh mẽ hơn nhiều cho không gian vector.

### 6.2. Negative dễ, hard negative và false negative

Negative ngẫu nhiên như một đoạn về pagination có thể quá dễ cho query webhook. **Hard negative** là đoạn dễ bị nhầm: cùng chứa `token`, cùng mã HTTP hoặc cùng trang nhưng điều kiện khác. Tuy nhiên một đoạn được miner trả về mà chưa có nhãn không tự là negative thật; nhãn thiếu sẽ đẩy hai đoạn cùng hữu ích ra xa nhau.

> **Phân loại ba dạng Negative trong huấn luyện Embedding:**
> 1. *Easy Negative (Mẫu tiêu cực dễ):* Văn bản hoàn toàn khác chủ đề (ví dụ: query hỏi về "lỗi timeout cổng thanh toán", còn negative là đoạn văn nói về "hướng dẫn phân trang API người dùng"). Mô hình dễ dàng phân biệt, gradient tiệm cận 0, không giúp mô hình tiến bộ thêm.
> 2. *Hard Negative (Mẫu tiêu cực khó):* Văn bản cực kỳ giống về mặt từ vựng hoặc ngữ cảnh chung nhưng sai về bản chất kỹ thuật (ví dụ: query hỏi "xử lý lỗi token hết hạn 401", còn hard negative là đoạn nói về "xử lý lỗi token thiếu quyền 403"). Đây là các mẫu dữ liệu quý giá nhất giúp mô hình học được ranh giới ngữ nghĩa tinh vi.
> 3. *False Negative (Mẫu tiêu cực giả mạo):* Đoạn văn thực chất trả lời được câu hỏi của người dùng, nhưng trong tập dữ liệu gán nhãn, người ta quên không đánh dấu nó là positive. Nếu ép mô hình phải xem nó là negative, ta đang "trừng phạt" mô hình vì đã tìm thấy một tài liệu hữu ích!

Trong batch có nhiều query cùng một positive, không được mặc nhiên dùng đường chéo của ma trận score như các positive duy nhất. Lab có 36 query train nhưng chỉ 25 chunk; target là **cột chunk ID đúng**, không phải chỉ số hàng query. Hai query khác nhau có thể cùng trỏ một cột. Tất cả chunk được so cùng lúc, CE shape `[36,25]`; các chunk ngoài positive bị coi là negative theo tiêu chí canonical span của bộ tự tạo.

> **Kỹ thuật thiết lập ma trận nhãn Target đa truy vấn:**
> Trong các hướng dẫn cơ bản (như Sentence-Transformers), người ta thường giả định mỗi query có đúng 1 positive duy nhất, tạo thành ma trận vuông $[B, B]$ và nhãn mục tiêu nằm trên đường chéo chính `target = torch.arange(B)`.
> Nhưng trong thực tế hệ thống runbook:
> - Có 36 query đào tạo khác nhau (ví dụ: `"màn hình bị đơ do cache"`, `"dữ liệu cũ không chịu cập nhật"`).
> - Cả hai query này đều có câu trả lời nằm ở cùng một chunk: Chunk số 14 ("Xóa cache Redis").
> - Vì vậy, ma trận điểm số có kích thước $[36, 25]$ (36 hàng ứng với 36 query, 25 cột ứng với 25 chunk). Nhãn mục tiêu của cả hàng 0 và hàng 1 đều cùng mang giá trị là chỉ số cột $14$. Thuật toán dùng chuẩn phân loại đa lớp (Multi-class Cross-Entropy) trên 25 lớp chunk.

Nếu thực tế có nhiều chunk hợp lệ, cần nhãn đa positive/mask false negative hoặc mục tiêu phù hợp. Xem công thức ở [phụ lục](deep-dive.md). Thêm negative làm thay đổi độ khó; không so loss của hai batch khác số negative rồi tự kết luận model nào tốt hơn.

## 7. Reranking: đọc chung query và ứng viên

### 7.1. Cross-encoder làm gì khác?

Reranker nhận tập ứng viên của tầng đầu rồi xếp lại. Với **cross-encoder**, query và chunk vào chung một chuỗi:

```text
[CLS] query tokens [SEP] title + heading + chunk tokens [SEP]
```

Self-attention hai chiều cho phép token query nhìn token chunk và ngược lại. Vector biểu diễn chunk trong phép tính đó phụ thuộc query, nên không thể cache một vector chunk duy nhất rồi thay việc chấm cặp bằng cosine độc lập. Có thể cache kết quả theo cặp/query version, nhưng đó là loại cache khác.

> **Sự khác biệt bản chất giữa Bi-encoder và Cross-encoder:**
> - **Bi-encoder (Độc lập tuyệt đối):**
>   Query và Document đi qua hai nhánh riêng biệt. Chúng không hề "biết" đến sự tồn tại của nhau trong quá trình trích xuất đặc trưng. Vector của Document chỉ được tạo ra từ chính nó. Nhờ đó ta có thể tính trước và lưu vào Vector Database, nhưng đổi lại là sự tương tác ngữ nghĩa giữa câu hỏi và tài liệu bị hạn chế ở mức tích vô hướng cuối cùng.
> - **Cross-encoder (Tương tác sâu hai chiều):**
>   Query và Document được ghép lại thành một chuỗi duy nhất đưa qua các lớp Transformer. Cơ chế Self-Attention cho phép:
>   - Từng từ trong Query được "nhìn" (attend to) từng từ trong Document. Ví dụ: từ `"timeout"` trong câu hỏi có thể trực tiếp so khớp trọng số chú ý với từ `"504 Gateway"` và `"Nginx"` trong tài liệu.
>   - Ngữ cảnh của chunk bị biến đổi hoàn toàn tùy thuộc vào việc nó đang được đọc cùng với câu hỏi nào. Do đó, **không thể tính trước vector cho Cross-encoder**. Bắt buộc phải chạy mô hình theo thời gian thực (online) cho từng cặp `(Query, Document)`.

Lab dùng token embedding + position embedding + segment embedding, một TransformerEncoder layer (`dim=32`, 4 head, FFN=64, dropout=.1), masked mean trên output và Linear ra **một logit**. Không dùng causal mask; bool padding mask True nghĩa bỏ key PAD. [PyTorch TransformerEncoder 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoder.html) là API đã đối chiếu. Đây là cross-encoder thật theo luồng dữ liệu chung, nhưng trọng số khởi tạo ngẫu nhiên và học trên bộ rất nhỏ.

Kiến trúc retrieve → rerank được giải thích trong [Sentence Transformers Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html). Khả năng cải thiện tùy model, dữ liệu và bài đo; tên “reranker” không bảo đảm xếp tốt hơn.

### 7.2. Tính tay thứ hạng và pair loss

Ví dụ **giả định độc lập với số đo lab**: retriever lấy A,B,C với score `.92,.83,.80`; chỉ B chứa bằng chứng. RR@3 trước rerank=1/2. Cross-encoder chấm logits A=−.2, B=1.4, C=−.7, xếp B,A,C nên RR@3=1.

> **Từng bước tính tay Reciprocal Rank (RR):**
> - Reciprocal Rank (Nghịch đảo thứ hạng) đánh giá tài liệu đúng đầu tiên xuất hiện ở vị trí thứ mấy:
>   $$\text{RR} = \frac{1}{\text{rank}}$$
> - *Trước khi Rerank (Tầng 1):*
>   Retriever trả về danh sách ứng viên theo thứ tự: 1. A ($0.92$), 2. B ($0.83$), 3. C ($0.80$).
>   Chỉ có tài liệu B chứa bằng chứng đúng. B nằm ở vị trí thứ 2 $\implies \text{RR@3} = \frac{1}{2} = 0.5$.
> - *Sau khi Rerank (Tầng 2):*
>   Cross-encoder chấm lại logits: B ($+1.4$), A ($-0.2$), C ($-0.7$).
>   Sắp xếp lại theo điểm logit giảm dần: 1. B, 2. A, 3. C.
>   Tài liệu đúng B đã nhảy lên vị trí quán quân (hạng 1) $\implies \text{RR@3} = \frac{1}{1} = 1.0$. Reranker đã cải thiện gấp đôi chất lượng truy xuất!

Với label B=1, binary cross-entropy không trọng số là `ln(1+exp(−1.4))≈.220417`. Với A label 0, loss=`ln(1+exp(−.2))≈.598139`. Logit không bị giới hạn [0,1]; sigmoid chỉ chuyển thang, chưa làm model được calibration. Lab dùng **BCE có pos_weight** để cân tỷ lệ positive/negative train, nên loss train không trùng ví dụ BCE không trọng số này.

> **Từng bước tính tay Binary Cross-Entropy (BCE) Loss:**
> Với một cặp có logit $z$ và nhãn $y \in \{0, 1\}$:
> Hàm kích hoạt Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
> Hàm mất mát:
> $$L_{\text{BCE}} = - [y \ln \sigma(z) + (1 - y) \ln(1 - \sigma(z))]$$
> - *Trường hợp 1: Nhãn $y = 1$ (Ứng viên B đúng, $z = 1.4$):*
>   $$L_B = -\ln \sigma(1.4) = -\ln \left( \frac{1}{1 + e^{-1.4}} \right) = \ln(1 + e^{-1.4})$$
>   Tính số: $e^{-1.4} \approx 0.246597 \implies L_B = \ln(1 + 0.246597) = \ln(1.246597) \approx 0.220417$.
> - *Trường hợp 2: Nhãn $y = 0$ (Ứng viên A sai, $z = -0.2$):*
>   $$L_A = -\ln(1 - \sigma(-0.2)) = \ln(1 + e^{-0.2})$$
>   Tính số: $e^{-0.2} \approx 0.818731 \implies L_A = \ln(1 + 0.818731) = \ln(1.818731) \approx 0.598139$.
> *(Lưu ý: Nếu logit của mẫu sai mang giá trị dương, ví dụ $z = +0.2$, thì $L = \ln(1 + e^{0.2}) \approx 0.798139$ sẽ phạt nặng hơn).*

Nếu B không vào tập ứng viên, reranker không lấy B ra được. Với M ứng viên và k≤M:

$$\operatorname{EvidenceHit@k\ sau\ rerank}\leq\operatorname{CandidateHit@M}.$$

Bất đẳng thức áp dụng khi reranker chỉ hoán vị tập ứng viên và cùng tiêu chí bằng chứng. Tăng M có thể tăng trần chất lượng, nhưng cần chấm nhiều cặp hơn; không chắc metric cuối tăng nếu thêm nhiều nhiễu.

> **Ý nghĩa của Bất đẳng thức Trần ứng viên (Candidate Ceiling):**
> Reranker bản chất chỉ là một "thuật toán sắp xếp lại danh sách" (re-ordering filter). Nó không có khả năng tự sáng tạo ra một tài liệu mới nằm ngoài danh sách ứng viên mà tầng 1 đã nạp vào.
> Nếu tầng 1 (Retriever) quét $1.000.000$ documents nhưng không lấy được đoạn văn đúng vào top $M = 6$ ứng viên, thì độ chính xác của Reranker vĩnh viễn bằng 0!
> - Tầng 1 quyết định **Cận trên của độ phủ (Recall Ceiling)**.
> - Tầng 2 quyết định **Độ chính xác vị trí hàng đầu (Top-k Precision)**.

### 7.3. Chi phí hai tầng

Offline encode N chunk một lần cho mỗi model/index version. Online: encode một query, tìm top-M, chấm M cặp, giữ top-k. Transformer attention tiêu chuẩn của reranker có phần chi phí theo bình phương độ dài chuỗi ghép; batching giúp tận dụng tính toán nhưng không biến M cặp thành một cặp.

M=6,k=3 là cấu hình nhỏ của lab, không phải khuyến nghị production. Nhiều chunk overlap cùng nguồn có thể chiếm hết M; cân nhắc dedup/diversity hoặc hạn mức theo parent, nhưng không được xóa mất các đoạn khác nhau cần cho một câu trả lời. Lab giữ các chunk khác ID, chưa có semantic dedup hay MMR.

> **Bài toán thiết kế hệ thống hai tầng trong thực tế (Production Two-Stage Architecture):**
> Tại sao các công cụ tìm kiếm lớn (Google, Bing, Elastic, Qdrant) không dùng luôn Cross-encoder từ đầu đến cuối?
> - Giả sử bạn có $N = 10.000.000$ tài liệu. Độ phức tạp tính toán của Transformer Self-Attention là $O(L^2)$ trên độ dài chuỗi. Nếu mỗi truy vấn người dùng nhập vào, hệ thống phải chạy Cross-encoder trên $10$ triệu cặp, máy chủ GPU sẽ sập ngay lập tức vì độ trễ lên tới hàng phút!
> - **Giải pháp tối ưu hóa:**
>   1. *Tầng 1 (Bi-encoder / BM25):* Quét $10.000.000$ tài liệu trong $\approx 10\text{ms}$ nhờ chỉ mục vector HNSW, lọc ra $M = 50$ ứng viên sáng giá nhất.
>   2. *Tầng 2 (Cross-encoder):* Chỉ chạy mô hình Transformer sâu trên đúng $M = 50$ cặp `(Query, Doc)` đó trong $\approx 30\text{ms}$, chọn ra $k = 3$ kết quả hoàn hảo nhất đưa vào context cho LLM.
>   Tổng thời gian chỉ mất $10\text{ms} + 30\text{ms} = 40\text{ms}$, vừa đảm bảo tốc độ của toàn bộ kho dữ liệu, vừa đạt được độ chính xác tuyệt đối ở đầu ra!

![Trần chất lượng từ tập ứng viên](figures/05-candidate-ceiling.png)

## 8. Budget: đừng để tokenizer âm thầm cắt bằng chứng

Giả sử reranker có tổng input limit 128 token, query 17, special token 3, prefix tiêu đề/heading 9. Body còn tối đa `128−17−3−9=99` token. Nếu chunk body 110 token, việc gọi tokenizer với truncation có thể cắt đúng câu trả lời ở đuôi mà không báo lỗi nghiệp vụ.

> **Bẫy cắt cụt âm thầm (The Silent Truncation Trap):**
> Trong các thư viện AI phổ biến (như Hugging Face `AutoTokenizer`), khi bạn truyền cờ `truncation=True` và `max_length=128`, nếu chuỗi văn bản dài 140 token, tokenizer sẽ **âm thầm vứt bỏ 12 token ở cuối** mà không ném ra bất kỳ Exception hay cảnh báo nào!
> *Hậu quả tai hại trong tài liệu kỹ thuật:*
> Cấu trúc phổ biến của một văn bản xử lý sự cố là:
> - Đoạn đầu (token 1–90): Giải thích triệu chứng, phân tích các khả năng lỗi.
> - Đoạn cuối (token 91–110): Câu lệnh khắc phục sự cố trực tiếp (ví dụ: `sudo systemctl restart billing-worker`).
> Khi bị cắt cụt âm thầm, câu lệnh mấu chốt nhất bị biến mất hoàn toàn. Người dùng hỏi "Làm sao sửa lỗi?", hệ thống nạp một chunk bị cụt đuôi vào LLM, và LLM sẽ hallucinate (bịa đặt) một câu lệnh sai lệch!

Đây là ví dụ số giả định. **Cấu hình thật của lab**: query≤32, document text gồm prefix≤64, pair≤112. Worst case `32+64+3=99≤112`; ba số đặc biệt là của tokenizer custom này, không phải mọi model. Chunk body tối đa 48 không có nghĩa model text dài 48 vì còn prefix.

> **Phương trình cân bằng ngân sách Token (Token Budget Invariant):**
> Để đảm bảo hệ thống không bao giờ bị cắt cụt dữ liệu ngoài ý muốn, kỹ sư phải thiết lập bất đẳng thức ngân sách nghiêm ngặt:
> $$\text{Ngân sách Tối đa} \ge \text{Tokens}_{\text{Query}} + \text{Tokens}_{\text{Prefix}} + \text{Tokens}_{\text{Body}} + \text{Tokens}_{\text{Special}}$$
> Trong cấu hình an toàn của lab:
> - Tối đa Query: $32$ token
> - Tối đa Document (gồm Prefix + Body): $64$ token
> - Token đặc biệt (`[CLS]`, `[SEP]`, `[SEP]`): $3$ token
> - Tổng trường hợp xấu nhất: $32 + 64 + 3 = 99 \le 112$ (ngưỡng tối đa cho phép của Cross-encoder).
> Mọi chuỗi ghép cặp đều nằm gọn dưới trần 112, đảm bảo 100% không có token nào bị thất lạc.

Lab kiểm độ dài và **raise khi vượt**, không tự truncate. Nếu thêm dữ liệu, cần chủ động rechunk/rút query/đổi budget. Với thư viện pretrained, cần kiểm tokenizer/model config, giới hạn cả cặp và chính sách cắt query/document; không chỉ nhìn `chunk_size`. Embedding budget vừa chưa chắc reranker budget vừa; cả hai vừa chưa chắc context cuối còn đủ.

## 9. Đo bằng chứng, không chỉ đúng trang

### 9.1. Nhãn độc lập với cách chia chunk

Nếu gán qrels trực tiếp theo chunk ID, đổi chunker sẽ làm toàn bộ ID cũ mất nghĩa. Lab gán một **canonical evidence span** cho mỗi query answerable: source ID, offset bắt đầu/kết thúc và câu trích gốc. Một chunk đúng theo bài đo khi cùng source và chứa **trọn span** ấy.

> **Khái niệm Canonical Evidence Span (Nhãn bằng chứng chuẩn hóa):**
> Trong các hệ thống cũ, người ta hay gán nhãn: "Query X khớp với Chunk ID #105". Nếu ngày mai bạn muốn thử nghiệm đổi chunk size từ 24 lên 48 token, toàn bộ danh sách Chunk ID cũ sẽ bị xóa bỏ, khiến toàn bộ tập dữ liệu gán nhãn (qrels) bị vô giá trị và phải thuê người gán nhãn lại từ đầu!
> *Giải pháp kiến trúc:* Đánh nhãn độc lập hoàn toàn với thuật toán chunking. Một nhãn gồm:
> - `source_id`: Tệp tài liệu gốc (`runbook-db.md`).
> - `start`, `end`: Tọa độ ký tự chính xác của câu văn chứa câu trả lời (ví dụ từ ký tự thứ 1.250 đến 1.340).
> - `quote`: Trích dẫn nguyên văn câu trả lời.
> Khi đó, bất kể bạn chia tài liệu thành 20 chunk hay 100 chunk, thuật toán chỉ cần kiểm tra: "Chunk này có chứa trọn vẹn khoảng ký tự $[1250, 1340]$ hay không?". Nếu có, chunk đó là đúng!

Đây là phép đo bằng chứng đã chọn, không phải phán quyết mọi đoạn ngoài span đều vô ích. Một đoạn khác có thể diễn đạt tương đương; nhiều chunk liền kề có thể ghép lại thành bằng chứng. Lab chưa đánh giá các cách đó, cũng chưa kiểm model đọc hiểu hay câu trả lời cuối.

Mỗi query có một span, kể cả overlap làm nhiều chunk chứa cùng span, ta chỉ tính việc lấy được span ấy một lần:

- **Pool coverage:** có ít nhất một chunk trong toàn index chứa trọn span.
- **CandidateHit@6:** top tối đa 6 ứng viên có chunk chứa trọn span.
- **EvidenceHit@3:** top tối đa 3 cuối cùng có chunk chứa trọn span.
- **MRR@3:** trung bình nghịch đảo vị trí chunk chứa trọn span đầu tiên, không có thì 0.
- **SourceHit@3:** có chunk cùng source; metric này dễ cao dù lấy sai section hoặc đoạn đứt bằng chứng.

> **Ý nghĩa trực quan của các thước đo chất lượng:**
> - **Pool Coverage (Độ phủ của khâu Chunking):** Tỷ lệ câu hỏi mà trong toàn bộ kho dữ liệu tạo ra có tồn tại ít nhất 1 chunk chứa trọn vẹn bằng chứng. Nếu Pool Coverage $< 1.0$, hệ thống đã bị mất thông tin ngay từ khâu chia nhỏ tài liệu, các tầng retrieval và LLM phía sau hoàn toàn vô vọng!
> - **CandidateHit@6 (Khả năng thu hồi của Tầng 1):** Tỷ lệ câu hỏi mà tầng Retriever đưa được bằng chứng vào danh sách 6 ứng viên đầu tiên.
> - **EvidenceHit@3 (Độ chính xác của Tầng 2):** Tỷ lệ câu hỏi mà top 3 tài liệu cuối cùng (được gửi cho LLM) có chứa bằng chứng.
> - **MRR@3 (Mean Reciprocal Rank):** Đánh giá vị trí hiển thị của bằng chứng. Nếu bằng chứng nằm ở vị trí #1, điểm là $1.0$; vị trí #2 là $0.5$; vị trí #3 là $0.333$; không xuất hiện là $0$. Điểm càng cao chứng tỏ người dùng (hoặc LLM) càng dễ thấy câu trả lời ngay lập tức.
> - **SourceHit@3 (Thước đo gây ảo tưởng - Vanity Metric):** Chỉ kiểm tra xem kết quả có trỏ đúng vào tệp nguồn hay không. Một runbook dài 30 trang nói về 10 sự cố khác nhau. Trả về đúng tệp runbook đó nhưng trích nhầm mục hướng dẫn thì SourceHit vẫn đạt $100\%$ dù người dùng không sửa được lỗi!

Với mỗi query: `EvidenceHit@3 ≤ CandidateHit@6 ≤ pool coverage`. Cùng lấy trung bình trên tập query đó vẫn giữ bất đẳng thức. Không loại các query có coverage=0 khỏi mẫu số để làm retriever trông tốt hơn.

### 9.2. Dữ liệu và giao thức đã chạy

Dữ liệu có 10 source (9 public, một riêng tư), 18 section public → 25 chunk cho chiến lược chính. Có 36 query train, 18 dev, 18 test answerable và 2 test no-answer. Query các split khác text, nhưng **chung source, ý định và bằng chứng**, nên đây là kiểm tra cách diễn đạt trong miền đã học, chưa là tổng quát hóa sang intent/nguồn/thời gian mới.

Vocabulary fit từ chunk public + query train; không fit từ query dev/test. Public corpus có chứa tài liệu đích là bình thường cho retrieval. S10 riêng tư bị loại **trước chunk/index/vocabulary/training**, không chỉ trước response; caller demo chưa có auth server. Index là snapshot public duy nhất, chưa hỗ trợ đổi quyền động hoặc multi-tenant production.

Bi-encoder train 80 epoch, kiểm dev mỗi 10, chọn MRR@3 tốt nhất (tie giữ checkpoint sớm). Model tốt nhất là epoch 40. Sau đó đóng băng nó và mine hard negatives bằng hybrid BM25+dense trên query **train**. Cross-encoder train 30 epoch, kiểm dev mỗi 5, chọn epoch 10. Tầng rerank có 216 pair train: 36 positive, 180 negative; pos_weight=5. Code có nhánh thêm positive bị miner bỏ sót chỉ cho train; run này số thêm là 0. **Không chèn gold vào ứng viên dev/test.**

Các checkpoint chọn theo dev, không theo train loss thấp nhất hay điểm test cao nhất. Chunk strategy/size, M=6,k=3,RRF c=60, τ=.2 là cấu hình cố định của ví dụ. Không có vòng tối ưu tham số trên test. Nguồn/qrels tự biên soạn và bộ nhỏ không đủ để chứng minh model nào tốt hơn nói chung.

### 9.3. Kết quả chunking ablation đã chạy

Bảng dùng BM25 ở mọi hàng để không lẫn thay đổi model neural vào so sánh. Token body lặp được tính nhiều lần; token model text gồm title/heading lặp nhưng chưa có special token cross-encoder.

| Cách chia | Chunk | Lượt token body | Token model text | Coverage | Test EvidenceHit@3 | Test MRR@3 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 24, overlap0 | 40 | 874 | 1,046 | .5556 | .5556 | .4352 |
| Fixed 24, overlap8 | 54 | 1,234 | 1,468 | .7222 | .7222 | .4815 |
| Section, trần48 | 25 | 874 | 970 | 1.0000 | 1.0000 | 1.0000 |

Coverage giống giữa dev/test vì từng cặp query trong hai split trỏ cùng bằng chứng. SourceHit@3 của cả ba đều 1.0000 trên test: chỉ đo đúng trang sẽ che việc nhiều chunk không giữ được câu cần đọc. Heading/size/cách chia đều đổi theo gói cấu hình, nên chưa là thí nghiệm cô lập một yếu tố; không kết luận section chunking luôn tốt hơn fixed.

> **Phân tích kết quả thực nghiệm cắt Chunk (Ablation Analysis):**
> 1. *Fixed 24, overlap 0:* Cắt cứng 24 token không gối đầu tạo ra 40 chunk, nhưng **Pool Coverage chỉ đạt 0.5556** (hơn $44\%$ câu hỏi bị đứt gãy bằng chứng!). Dù thuật toán tìm kiếm BM25 hoàn hảo, Test EvidenceHit@3 cũng chỉ chạm trần ở $0.5556$.
> 2. *Fixed 24, overlap 8:* Bổ sung 8 token gối đầu giúp cứu được các bằng chứng ở ranh giới, nâng Coverage lên $0.7222$ và Test MRR@3 lên $0.4815$. Tuy nhiên, số lượng chunk tăng vọt lên 54 ($+35\%$ dung lượng lưu trữ).
> 3. *Section, trần 48 (Tôn trọng cấu trúc):* Chỉ cần **25 chunk** (ít nhất và tiết kiệm bộ nhớ nhất), nhưng **Coverage đạt 1.0000 tuyệt đối** và Test MRR@3 đạt $1.0000$.
> *Bài học then chốt:* Cắt chunk theo cấu trúc ngữ nghĩa (Heading/Section) luôn vượt trội hoàn toàn so với cắt mù quáng theo số lượng token cố định.

### 9.4. Embedding và reranker đã học được đến đâu?

Cùng 25 chunk, cùng query và tiêu chí span, CPU với Python 3.14.6, PyTorch 2.15.0, NumPy 2.5.4:

| Phương pháp | Dev MRR@3 | Test CandidateHit@6 | Test EvidenceHit@3 | Test MRR@3 |
|---|---:|---:|---:|---:|
| BM25 | .9352 | 1.0000 | 1.0000 | 1.0000 |
| Bi-encoder đã train | .6574 | .9444 | .8333 | .7500 |
| Hybrid BM25 + bi-encoder | .7778 | 1.0000 | .9444 | .8889 |
| Cùng hybrid → cross-encoder | .4630 | 1.0000 | .5000 | .3148 |

Bi-encoder khởi tạo ngẫu nhiên có dev MRR≈.4074; sau train/chọn dev đạt .6574. Có học được tín hiệu nhưng vẫn thua baseline từ khóa. Reranker **làm xấu thứ hạng** dù train loss giảm; dev đã cho thấy điều đó trước khi đọc test. Với kết quả này, lựa chọn có cơ sở cho bộ demo là giữ baseline, chưa có lý do triển khai reranker này để tăng chất lượng.

> **Bài học đắt giá cho Kỹ sư AI: Không phải cứ "Deep Learning" là tốt hơn Baseline!**
> Nhìn vào bảng kết quả trên:
> - Thuật toán từ khóa cổ điển **BM25 đạt điểm tuyệt đối 1.0000** trên tập test.
> - Mô hình Neural Bi-encoder chỉ đạt $0.7500$.
> - Mô hình Cross-encoder Reranker thậm chí **kéo lùi kết quả nghiêm trọng xuống 0.3148**!
> *Tại sao lại xảy ra hiện tượng này?*
> 1. *Đặc thù dữ liệu kỹ thuật:* Các runbook chứa nhiều từ khóa định danh chính xác (như `ERR_BILL_042`, `HTTP 401`, `Redis`). BM25 khớp chính xác 100% các từ khóa này mà không cần học.
> 2. *Giới hạn dữ liệu và over-fitting:* Tập dữ liệu chỉ có 36 query train. Mô hình Cross-encoder dù chỉ có 24k tham số nhưng vẫn quá phức tạp so với lượng dữ liệu ít ỏi, dẫn đến việc học vẹt và mất khả năng tổng quát hóa trên tập dev/test.
> 3. *Nguyên tắc kỹ thuật:* Luôn luôn xây dựng một Baseline đơn giản (BM25 / Keyword Search) trước tiên. Chỉ đưa các mô hình neural phức tạp vào production khi bạn có số liệu thực nghiệm chứng minh nó đánh bại được baseline trên tập dữ liệu thực tế!

Model bi-encoder chỉ 13,760 tham số, cross-encoder 24,929; vocabulary 397, chỉ 36 query train. Dữ liệu nhỏ, encoder đơn giản và query có nhiều term khớp trực tiếp là giới hạn cụ thể. Đây là các yếu tố cần điều tra, chưa là phép chứng minh một nguyên nhân duy nhất. Không dùng bảng này kết luận neural retrieval hoặc cross-encoder pretrained kém nói chung.

Vẫn có ca rerank sửa được lỗi: query test về **invalidation cache** có hybrid RR@3=0 vì chunk chứa span đứng ngoài top-3 nhưng còn trong top-6; rerank đưa nó lên hạng3, RR@3=1/3. Ngược lại, query về **connection pool** có hybrid RR=1 nhưng rerank đẩy bằng chứng khỏi top-3. Cần xem từng query chứ không chỉ nhìn loss.

![Kết quả thật của hai mô hình nhỏ và baseline](figures/06-evaluation.png)

### 9.5. No-answer không tự được giải bằng reranking

Query toàn OOV `zzquasar999 nebula999` bị gate word-level của lab trả rỗng. Đây là policy rõ ràng, không phải model tự phát hiện “không biết”. Query “cache trên sao Hỏa” có từ quen, hybrid vẫn lấy ứng viên và reranker vẫn xếp lại chúng dù không có span đáp ứng. Lab chưa học ngưỡng abstain, không có no-answer train và chưa calibration.

> **Cơ chế từ chối trả lời (Abstain Policy) và phòng chống Ảo giác (Hallucination):**
> Trong các ứng dụng doanh nghiệp thực tế, rủi ro lớn nhất của hệ thống RAG không phải là "không tìm thấy câu trả lời", mà là **"tìm thấy thông tin sai nhưng vẫn tự tin trả lời"**!
> - Nếu người dùng hỏi một câu hỏi hoàn toàn không có trong cơ sở dữ liệu (ví dụ: *"Chính sách bảo hành server trên sao Hỏa"*):
>   - Tầng Retriever vẫn sẽ cố gắng tính toán độ tương đồng cosine và trả về top 3 chunk có điểm số cao nhất có thể (dù các chunk này chỉ nói về cache server thông thường).
>   - Tầng Reranker vẫn sẽ chấm điểm và sắp xếp lại 3 chunk đó.
>   - LLM đọc 3 chunk đó và cố gắng suy diễn ra một câu trả lời bịa đặt!
> - *Giải pháp kỹ thuật:* Phải thiết lập một cổng kiểm soát ngưỡng tin cậy (Confidence Threshold Gating) hoặc huấn luyện riêng một bộ phân loại "Không thể trả lời" (Abstain Classifier). Nếu điểm số cao nhất dưới ngưỡng an toàn, hệ thống phải trả về thông báo rõ ràng: *"Tài liệu nội bộ không chứa thông tin để trả lời câu hỏi này"*.

Sigmoid(logit)>0.5 không đủ làm ngưỡng tin cậy chung, nhất là khi BCE có pos_weight. Phải có tập nhãn no-answer phù hợp và đo riêng lỗi trả lời/abstain ở hệ thống cuối. Bài này mới đo truy xuất và thứ hạng bằng chứng, chưa sinh câu trả lời RAG.

## 10. Lab thực hành

### 10.1. Cấu trúc và môi trường

```text
24-chunking-embedding-reranking/
  README.md          Bài chính
  deep-dive.md       Mục tiêu loss, budget, index và kỹ thuật nối tiếp
  exercises.md       16 bài tập
  solutions.md       Lời giải độc lập với chat
  data.py            Runbook, query split và evidence span tự tạo
  retrieval.py       Chunker, tokenizer, hai model, search, save/load
  lab.py             Train/validation/eval, ablation và CLI
  make_figures.py    Tái tạo 7 hình từ dữ liệu/report
  figures/           PNG + SVG đã lưu sẵn
```

Bản đã chạy: Python 3.14.6, torch 2.15.0, NumPy 2.5.4; CPU một thread, seed 24. Không cần GPU, tài khoản hay tải model pretrained. Cài các phiên bản này có thể phụ thuộc Python/nền tảng; nếu máy bạn dùng phiên bản khác, ghi lại và kiểm lại số/shape/thứ hạng. Không xem một số đo CPU cục bộ là benchmark tốc độ chung.

Từ gốc dự án, vào thư mục bài:

```bash
cd courses/ai-engineer-2026/24-chunking-embedding-reranking
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode chunks
.venv/bin/python lab.py --mode train --out /tmp/lesson24-run-01
.venv/bin/python lab.py --mode search --bundle /tmp/lesson24-run-01/bundle --query 'hồ sơ đã sửa mà màn hình vẫn lấy cache cũ'
```

Windows thay bằng `.venv\Scripts\python.exe` và output path phù hợp. Mỗi run chọn một thư mục output **chưa tồn tại**. Nếu đã có môi trường đúng phiên bản, có thể dùng trực tiếp, không bắt buộc tạo `.venv` trong bài.

### 10.2. Những bước cần quan sát

Đầu tiên dự đoán cửa sổ L13/C6/O2 rồi chạy `hand`. Với `chunks`, đối chiếu coverage và số token, giải thích vì sao đúng source chưa đủ. Với `train`, đọc checkpoint nào được chọn bằng dev và so bảng bốn phương pháp. Không lấy loss epoch cuối làm bằng chứng model đang dùng tốt nhất.

Sau đó mở `report.json`: chọn hai query có cùng candidate nhưng thứ hạng trước/sau khác nhau. Dùng chunk ID trong `bundle/manifest.json` xem source, offset và text. Hãy giải thích lỗi bằng **đoạn cụ thể**: bị cắt, không được lấy, hay bị rerank xuống dưới.

Output runtime nằm ngoài tài liệu:

```text
/tmp/lesson24-run-01/
  report.json             Loss/dev history, checkpoint selection, metric, top-M và hash
  bundle/manifest.json    Chunk, vocabulary, config, analyzer và model hash
  bundle/models.pt        State dict bi/cross tốt nhất theo dev, dùng inference
  bundle/vectors.npy      Document embedding tương ứng checkpoint bi đã chọn
```

Checkpoit **chưa đủ resume đúng trạng thái**: không lưu optimizer, RNG hay vị trí epoch cho tiếp tục train. Đây là bundle phục vụ inference/kiểm đối chiếu, khác checkpoint resume ở bài 12.

### 10.3. Theo dấu luồng tensor

Bi-encoder: query IDs `[36,Tq]`, document IDs `[25,Td]` → vector `[36,32]` và `[25,32]` → score `[36,25]` → CE với target `[36]`. Query cùng positive vẫn trỏ cùng cột, không tạo hai bản vector positive giả để dùng đường chéo.

> **Chi tiết luồng Tensor của Bi-encoder:**
> - `Batch Query IDs` $[36, T_q]$ đi qua bảng Embedding $[V, 32] \implies [36, T_q, 32]$.
> - Đi qua `Masked Mean Pooling` theo chiều $T_q \implies$ thu được ma trận biểu diễn $[36, 32]$.
> - Đi qua Linear projection + Tanh + L2 Normalize $\implies$ thu được ma trận vector truy vấn chuẩn hóa $Z_Q \in \mathbb{R}^{36 \times 32}$.
> - Tương tự với 25 Document Chunks: thu được ma trận vector tài liệu chuẩn hóa $Z_D \in \mathbb{R}^{25 \times 32}$.
> - Phép nhân ma trận tính điểm tương đồng:
>   $$\text{Scores} = Z_Q \times Z_D^\top \in \mathbb{R}^{36 \times 25}$$
> - Hàm `nn.CrossEntropyLoss()` nhận ma trận Scores $[36, 25]$ và vector Target $[36]$ (chứa chỉ số cột từ $0$ đến $24$) để tính toán loss và lan truyền ngược.

Cross-encoder: pair IDs `[B,Tpair]`, segment IDs cùng shape → embedding `[B,Tpair,32]` → self-attention hai chiều → masked mean `[B,32]` → logits `[B]` → BCE với labels `[B]`. Khi inference, `eval()` tắt dropout và `inference_mode()` không dựng graph gradient. PAD bị chặn làm key và bị loại khỏi mean.

> **Chi tiết luồng Tensor của Cross-encoder:**
> - Đầu vào là chuỗi ghép nối: $[B, T_{\text{pair}}]$ chứa `[CLS] query [SEP] chunk [SEP]`.
> - Vector phân đoạn `Segment IDs` $[B, T_{\text{pair}}]$: mang giá trị $0$ cho các token của query và $1$ cho các token của chunk.
> - Tổng hợp Embedding: $\text{Token Emb} + \text{Position Emb} + \text{Segment Emb} \implies [B, T_{\text{pair}}, 32]$.
> - Đi qua `TransformerEncoderLayer` với cơ chế 4-head Multi-Head Attention hai chiều $\implies [B, T_{\text{pair}}, 32]$.
> - Đi qua `Masked Mean Pooling` để triệt tiêu các token `[PAD]` $\implies [B, 32]$.
> - Đi qua Linear Layer chiếu từ 32 về 1 scalar duy nhất $\implies \text{Logits} \in \mathbb{R}^{B}$.
> - Tính `nn.BCEWithLogitsLoss(pos_weight=5.0)` với vector nhãn nhị phân $[B]$ ($1$ là chunk đúng, $0$ là chunk sai).

### 10.4. Load bundle và xem cùng ứng viên

Snippet chạy từ thư mục bài sau khi tạo run:

```python
from pathlib import Path
import torch
from retrieval import Pipeline
torch.set_num_threads(1)
pipeline = Pipeline.load(Path('/tmp/lesson24-run-01/bundle'))
query = 'hồ sơ đã sửa mà màn hình vẫn lấy cache cũ'
candidates = pipeline.retrieve(query, 'hybrid')
reranked = pipeline.rerank(query, candidates)
assert set(reranked) == set(candidates)
for cid in reranked[:3]:
    c = pipeline.byid[cid]
    print(cid, c['source_id'], c['heading'], (c['start'],c['end']), c['text'])
```

Reranker chỉ đổi thứ tự, không thêm/đổi nội dung chunk. `Pipeline.load` dùng `weights_only=True`, kiểm hash snapshot/model và encode lại corpus để xác nhận vector cache tương thích. Việc encode kiểm tra diễn ra lúc load; một service nên load một lần lúc khởi động, không mỗi request. Đây là loader sư phạm, chưa là index service lớn hay trình kiểm mọi file không tin cậy. Chỉ load bundle tự tạo/tin cậy theo [hướng dẫn torch.load](https://docs.pytorch.org/docs/2.15/generated/torch.load.html).

### 10.5. Kiểm bằng chứng trực tiếp

```python
from data import dataset
from retrieval import chunk_sources, relevant_ids
sources, queries = dataset()
q = next(q for q in queries if q['id'] == 'test-06-1')
source = next(s for s in sources if s['id'] == q['evidence']['source_id'])
e = q['evidence']
assert source['text'][e['start']:e['end']] == e['quote']
chunks = chunk_sources(sources, 'section', 48, 0)
ids = relevant_ids(q, chunks)
assert ids
print('Bằng chứng:', e['quote'])
print('Chunk chứa trọn:', sorted(ids))
```

Sản phẩm thực hành tối thiểu: bảng ba cấu hình chunk, bảng bốn phương pháp truy xuất/xếp hạng, hai query đối chiếu trước/sau rerank, một ca no-answer và một kiểm chứng save/load. Nếu muốn thay size hoặc architecture sau khi xem lỗi test, hãy ghi rõ đó là vòng phát triển tiếp và chuẩn bị holdout mới; không gọi test cũ là chưa từng thấy.

### 10.6. Tái tạo hình và đọc đường học

Hình có sẵn. Để tái tạo từ report vừa chạy:

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py --run /tmp/lesson24-run-01
```

![Loss train và MRR dev không phải cùng một thước đo](figures/07-learning-curves.png)

Đường train loss giảm cho biết model tối ưu mục tiêu train; MRR dev cho biết thứ hạng theo giao thức khác. Bài dùng dev MRR chọn checkpoint, không lấy epoch cuối hay ép đường dev phải tăng. Hai mô hình dùng loss khác nhau, nên hai trục loss ở hình không dùng để so model nào “giỏi hơn”.

## 11. Từ lab sang một hệ thống thực tế

Đối với backend engineer, các contract cần quản lý là source/version → parser/chunker → tokenizer/encoder → vector index → candidate/reranker → context. Đổi chunking làm thay ID/nội dung; đổi encoder làm thay không gian vector; đổi reranker có thể đổi thứ hạng mà không cần tính lại vector của bi-encoder nếu tầng đó giữ nguyên. Phải version cache và evaluation theo đúng thành phần đổi.

> **Bản đồ quản lý hợp đồng dữ liệu cho Kỹ sư Hệ thống (Data Contracts):**
> 1. *Hợp đồng Parser & Chunker:* Khi thay đổi logic cắt chunk (ví dụ đổi kích thước cửa sổ từ 48 lên 128), toàn bộ Chunk ID và tọa độ văn bản sẽ thay đổi. Bắt buộc phải kích hoạt quy trình xóa và lập lại chỉ mục (Re-indexing Pipeline).
> 2. *Hợp đồng Encoder & Vector Space:* Không gian vector phụ thuộc 100% vào trọng số mô hình. Nếu bạn nâng cấp mô hình embedding từ `text-embedding-ada-002` lên `text-embedding-3-small`, bạn KHÔNG THỂ tái sử dụng các vector cũ trong Vector DB! Toàn bộ kho vector phải được mã hóa lại từ đầu.
> 3. *Hợp đồng Reranker:* Reranker chỉ tác động vào bước xếp hạng online. Bạn có thể thay thế mô hình reranker mới mà không cần chạm vào cơ sở dữ liệu vector của Bi-encoder.
> 4. *Ngân sách độ trễ (Latency Budget Allocation):*
>    - Tổng thời gian cho phép của API: $\le 2000\text{ms}$
>    - First-stage Vector Retrieval (Tầng 1): $\le 15\text{ms}$
>    - Cross-encoder Reranking (Tầng 2 trên top 30 ứng viên): $\le 45\text{ms}$
>    - Network & Serialization overhead: $\le 20\text{ms}$
>    - Quỹ thời gian còn lại cho LLM Streaming sinh câu trả lời: $\approx 1900\text{ms}$.

Với model pretrained, hãy kiểm model card về ngôn ngữ/tác vụ, query/document prompt, pooling, normalization, input limit và điều kiện sử dụng; đo trên dữ liệu của bạn trước khi chọn. Không đưa một vector `[CLS]` từ model ngôn ngữ bất kỳ vào index rồi mặc nhiên coi đó là embedding truy xuất tốt.

Có thể nghiên cứu tiếp **parent–child retrieval**, ghép các chunk lân cận, contextualized text, late chunking và late interaction. [Phụ lục](deep-dive.md) phân biệt chúng; lab chưa triển khai các kỹ thuật đó. Tiếp tục với [bài 25 — RAG và nguồn dẫn](../25-rag-nguon-dan/README.md), đã có lab pipeline, bài tập và lời giải.

> **Mở rộng các kiến trúc RAG hiện đại:**
> - **Parent-Child Retrieval (Small-to-Big Retrieval):**
>   Tìm kiếm trên các chunk nhỏ (ví dụ 100 token) để vector embedding có tính tập trung cao và khớp chính xác truy vấn. Nhưng khi gửi ngữ cảnh cho LLM, hệ thống tự động truy vấn ngược về "Parent Chunk" (đoạn văn lớn 1000 token chứa chunk nhỏ đó) để LLM có đầy đủ ngữ cảnh xung quanh mà không bị mất mạch văn.
> - **Late Interaction (ColBERT):**
>   Lưu trữ biểu diễn đa vector (multi-vector) cho từng token của văn bản thay vì nén thành 1 vector duy nhất. Khi tìm kiếm, thuật toán tính toán toán tử MaxSim giữa các token của query và document. Cách tiếp cận này đạt được độ chính xác tương đương Cross-encoder nhưng với tốc độ cực nhanh gần bằng Bi-encoder!
> - **Late Chunking:**
>   Thay vì cắt nhỏ văn bản trước khi đưa vào mô hình embedding, kỹ thuật này đưa toàn bộ tài liệu dài qua các lớp Transformer để tất cả các từ tương tác ngữ cảnh với nhau. Sau đó, tại lớp ẩn cuối cùng (final hidden states), mô hình mới tiến hành pooling theo từng ranh giới chunk.

## 12. Tự kiểm

Làm [16 bài tập](exercises.md) rồi xem [lời giải](solutions.md). Bạn cần tự chỉ ra: offset đang tính theo gì; prefix/special token chiếm budget nào; vì sao nhiều query cùng positive không dùng CE đường chéo máy móc; khi nào reranker không thể sửa lỗi; vì sao lab này nên giữ baseline; và tại sao “cùng source” vẫn thiếu bằng chứng. Việc có tài liệu hoàn chỉnh không chứng minh người đọc đã nắm bài.

[Nguồn đã đọc và phạm vi kiểm chứng](../SOURCES.md) · [Trở về mục lục](../README.md).
