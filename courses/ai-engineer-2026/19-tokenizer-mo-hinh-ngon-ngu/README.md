# Bài 19 — Tokenizer và mô hình ngôn ngữ: từ văn bản đến dự đoán token

[Mục lục](../README.md) · [Bài 18 — Transformer](../18-transformer/README.md) · [Đào sâu và ứng dụng](deep-dive.md) · [Bài tập](exercises.md) · [Lời giải](solutions.md)

Biên soạn ngày 04/11/2026. Ví dụ văn bản và dữ liệu trong bài do tác giả tự tạo theo tình huống backend; không dùng log khách hàng. Code chạy CPU, không tải pretrained model hoặc gọi dịch vụ AI.

## 1. Mục tiêu và cầu nối với bài 18

Bài 18 đã có Transformer nhận chuỗi ID và dự đoán ID tiếp theo. *(Giải thích tường minh: Trong mạng nơ-ron, "ID" là các số nguyên không âm $0, 1, 2, \ldots$ dùng làm chỉ mục định danh cho các đơn vị dữ liệu rời rạc. Bản thân mạng Transformer chỉ là một hàm toán học gồm các phép nhân ma trận số thực, nên nó không thể trực tiếp đọc các ký tự chữ cái mà bắt buộc phải nhận đầu vào là các vector số tương ứng với các ID này).* Nhưng văn bản thật không phải một dãy A–D với vocabulary cho trước: có tiếng Việt, dấu cách, xuống dòng, emoji, URL, JSON, tên biến và lỗi chính tả. *(Định nghĩa từ vựng - Vocabulary: Là tập hợp hữu hạn tất cả các từ, mảnh ký tự hoặc ký hiệu hợp lệ mà mô hình được phép nhìn thấy và sinh ra. Trong thực tế phát triển phần mềm, một dòng log backend như `{"status": 504, "msg": "gateway_timeout"}` chứa dấu ngoặc nhọn, dấu hai chấm, dấu gạch dưới, khoảng trắng và số nguyên; không có một danh sách từ vựng đơn giản nào chứa sẵn mọi tổ hợp ký tự này nếu không có cơ chế chia nhỏ linh hoạt).* Bài này xây cầu nối giữa văn bản và mô hình ấy. *(Định nghĩa Tokenizer: Tokenizer chính là "bộ phiên dịch hai chiều" giữa con người và mô hình nơ-ron: chiều xuôi (encode) cắt nhỏ chuỗi văn bản tự nhiên thành các mảnh và ánh xạ thành chuỗi token ID để đưa vào mô hình; chiều ngược (decode) ghép các token ID do mô hình sinh ra trở lại thành văn bản con người đọc được).*

Sau bài này, bạn cần tự làm được:

- Phân biệt từ, Unicode code point, byte, token, token ID và embedding. *(Nắm vững ranh giới giữa đơn vị ngôn ngữ học của con người, đơn vị lưu trữ nhị phân của máy tính và biểu diễn vector trong không gian toán học).*
- Giải thích pipeline tokenizer; tính tay hai vòng BPE và chạy tokenizer có encode/decode. *(Hiểu bản chất của thuật toán ghép cặp byte - Byte-Pair Encoding - để nén từ vựng mà không bao giờ gặp lỗi từ lạ).*
- Tạo input/label cho causal language model; tính xác suất chuỗi, CE và perplexity. *(Hiểu trực giác toán học đằng sau kỹ thuật dịch chuyển token - token shift, hàm mất mát Cross-Entropy và chỉ số đo lường độ bối rối - Perplexity).*
- Huấn luyện LM nhỏ trên văn bản, đánh giá trên split giữ riêng và đọc output với giới hạn phù hợp. *(Áp dụng quy trình chuẩn chia train/validation/test để kiểm chứng mô hình không học vẹt trên dữ liệu log backend).*
- Nhận ra lỗi tokenizer/model không khớp, normalization làm đổi dữ liệu, tràn context và so metric sai đơn vị. *(Những cạm bẫy thực tế mà các kỹ sư phần mềm thường gặp khi tích hợp LLM vào hệ thống backend).*

Nên học ba lượt: phần 2–7 cho văn bản và tokenizer; phần 8–11 cho mục tiêu học; phần 12–15 cùng lab cho thực hành. Cần hiểu embedding bài 16, softmax bài 6 và causal Transformer bài 18. Không cần nhớ hết công thức attention để bắt đầu đọc ví dụ văn bản dưới đây.

## 2. Ví dụ thực tế: hệ thống nhận một ticket ra sao?

Một người viết:

```text
API /orders trả 504 sau 30s. Xin kiểm tra giúp tôi 🙂
```

Con người nhận ra URL, mã lỗi và ý nhờ hỗ trợ. Nhưng input của embedding phải là các số nguyên. *(Giải thích tường minh: Não bộ con người xử lý câu trên bằng ngữ nghĩa và kinh nghiệm sống — ta hiểu ngay `/orders` là endpoint xử lý đơn hàng, `504` là lỗi Gateway Timeout, và biểu tượng `🙂` thể hiện thái độ lịch sự. Ngược lại, máy tính và các lớp embedding trong Deep Learning chỉ có thể thực hiện phép toán đại số tuyến tính giữa các số thực. Bạn không thể lấy ma trận trọng số nhân trực tiếp với chữ "A" hay chữ "P", mà bắt buộc phải gán cho mỗi mảnh chữ một mã định danh bằng số nguyên duy nhất, ví dụ `API` $\to$ ID 259, ` /` $\to$ ID 260).* Tokenizer chia văn bản thành các đơn vị và ánh xạ chúng sang ID; Transformer đọc embedding của các ID đó cùng thông tin vị trí. Head dự đoán một phân phối trên vocabulary; decoder của tokenizer ghép token sinh được về văn bản.

*(Định nghĩa các thành phần cốt lõi trong quy trình:
1. **Token (Mảnh văn bản):** Đơn vị xử lý cơ bản nhỏ nhất của văn bản sau khi chia cắt (có thể là một từ trọn vẹn như "Xin", một phần của từ như "order", một ký tự đặc biệt như "/", hoặc một byte dữ liệu thô).
2. **Token ID:** Con số nguyên không âm đại diện cho token đó trong bảng từ điển (vocabulary index), dùng làm chỉ mục dòng để tra cứu vector trong bảng embedding.
3. **Embedding:** Vector số thực nhiều chiều (ví dụ 32 chiều trong bài này) biểu diễn tọa độ ngữ nghĩa của token trong không gian toán học, giúp các token có liên quan ngữ nghĩa nằm gần nhau.
4. **Output Head:** Lớp tuyến tính (Linear layer) ở tầng cuối cùng của mô hình, nhận vector đặc trưng từ Transformer và chiếu lên toàn bộ từ điển để tạo ra các điểm số thô.
5. **Logits:** Điểm số thô (chưa chuẩn hóa thành xác suất) cho từng token trong từ điển. Token nào có logit càng lớn thì mô hình đánh giá khả năng nó xuất hiện kế tiếp càng cao.
6. **Tokenizer Decoder:** Bộ giải mã nhận dãy token ID do mô hình chọn và ráp nối các byte tương ứng để tái tạo lại chuỗi ký tự hiển thị ban đầu).*

```text
văn bản → tokenizer.encode → IDs → embedding + vị trí → Transformer
                                                → logits → chọn ID → tokenizer.decode
```

Tokenizer không tự xác nhận HTTP 504 có đúng hay dịch vụ có lỗi thật không. *(Ví dụ thực tế: Tokenizer chỉ là một bộ tiền xử lý chuỗi (string parser) thuần túy dựa trên quy tắc thống kê. Nếu người dùng nhập vào một dòng log giả mạo hoặc một đoạn mã tấn công SQL Injection như `DROP TABLE orders;`, tokenizer vẫn cắt nhỏ thành các mảnh token và chuyển thành số nguyên mà hoàn toàn không có nhận thức logic về mặt nghiệp vụ).* Model cũng không truy cập dashboard chỉ vì trong ticket có tên API. Muốn kết luận tình trạng hệ thống cần log/metric hoặc công cụ bên ngoài; đó là lớp ứng dụng khác. *(Ứng dụng thực tế: Một lỗi nhận thức phổ biến của kỹ sư mới là kỳ vọng mô hình ngôn ngữ "tự biết" server backend có lỗi thật hay không. Bản chất mô hình chỉ là bộ máy tính toán xác suất từ kế tiếp dựa trên văn bản. Muốn mô hình xác minh sự cố thực tế, kiến trúc backend của bạn phải cung cấp cho mô hình cơ chế gọi công cụ (Tool Calling / Function Calling) — ví dụ kết nối tới Prometheus hoặc Grafana API để lấy metric thật, sau đó đưa dữ liệu đó vào prompt để mô hình tóm tắt).*

Ví dụ các mảnh `API`, ` /`, `orders`, ` trả`, ` 504` chỉ là **minh họa cách có thể chia**, không phải output của một tokenizer thương mại cụ thể. Muốn biết token/ID chính xác phải chạy đúng tokenizer và đúng phiên bản; bài này cung cấp tokenizer riêng để kiểm được mọi ví dụ có số. *(Lưu ý về dấu cách: Hãy chú ý dấu cách thường được gắn liền vào đầu của từ tiếp theo (như ` /`, ` trả`, ` 504`). Kỹ thuật này giúp phân biệt rõ giữa một từ đứng độc lập có khoảng trắng phía trước với một chuỗi ký tự nằm giữa từ, đảm bảo quá trình giải mã (decode) tái tạo lại chính xác từng khoảng trắng ban đầu mà không làm mất định dạng văn bản).*

![Luồng văn bản, ID và mô hình](figures/01-pipeline.png)

## 3. Token không đồng nghĩa với từ hoặc ký tự

| Đơn vị | Ví dụ | Ý nghĩa |
|---|---|---|
| Từ theo ngôn ngữ | “máy học”, “cơ sở dữ liệu” | Cách phân định từ phụ thuộc ngôn ngữ; khoảng trắng không luôn là ranh giới từ hoàn chỉnh |
| Code point Unicode | `é` dạng dựng sẵn là U+00E9 | Đơn vị của `len(str)` trong Python, không luôn bằng ký tự nhìn thấy |
| Byte UTF-8 | `é` thành hai byte 195,169 | Đơn vị lưu trữ/biểu diễn, không phải ngữ nghĩa |
| Token | Có thể là từ, mảnh từ, dấu cách hoặc mảnh byte | Do tokenizer cụ thể quy định |
| Token ID | Số nguyên ánh xạ tới một mục vocabulary | Không phải độ lớn ý nghĩa; ID 100 không “gấp đôi” ID 50 |
| Embedding | Vector học được của ID | Được model cập nhật bằng gradient, không phải chính thuật toán chia token |

*(Giải thích chi tiết các đơn vị cho người mới:
- **Từ theo ngôn ngữ (Word):** Trong tiếng Anh, ranh giới từ thường được phân tách rõ ràng bằng dấu cách (`database` là 1 từ). Nhưng trong tiếng Việt, "máy học" là một từ ghép gồm hai âm tiết (tiếng) ngăn cách bởi dấu cách; còn trong tiếng Nhật hay tiếng Trung thì hoàn toàn không có dấu cách giữa các từ. Vì vậy, coi "mỗi từ là một token" là cách tiếp cận không thể áp dụng cho mọi ngôn ngữ.
- **Code point Unicode (Điểm mã Unicode):** Là một con số quy chuẩn quốc tế đại diện cho một ký tự hoặc biểu tượng trừu tượng (ký hiệu dạng `U+XXXX`). Hàm `len("chuỗi")` trong Python chỉ đếm số lượng code point, không phản ánh số byte bộ nhớ hay số token.
- **Byte UTF-8:** Đơn vị lưu trữ nhị phân thực tế trên ổ cứng hoặc đường truyền mạng (mỗi byte từ 0 đến 255). Ký tự ASCII chuẩn chỉ tốn 1 byte, nhưng tiếng Việt có dấu tốn 2 đến 3 byte, và emoji tốn 4 byte.
- **Token:** Đơn vị phân đoạn do thuật toán tokenizer đặt ra nhằm tối ưu hóa việc học của mạng nơ-ron. Một token có thể dài hơn một từ (ví dụ cụm từ quen thuộc), hoặc ngắn hơn một từ (một mảnh từ ghép), hoặc chỉ là một byte đơn lẻ.
- **Token ID:** Chỉ đơn thuần là số thứ tự dòng trong bảng từ vựng (chỉ mục ma trận). Token ID 200 hoàn toàn không mang ý nghĩa "lớn hơn gấp đôi" Token ID 100 về mặt ngữ nghĩa toán học.
- **Embedding:** Vector số học trong không gian nhiều chiều. Chỉ khi token ID được tra cứu qua bảng embedding, nó mới trở thành các con số thực mang thông tin ngữ nghĩa mà mô hình có thể tính toán đạo hàm).*

`é` và `e` cộng dấu sắc U+0301 trông tương tự, nhưng một chuỗi có 1 code point, chuỗi kia có 2. Emoji gia đình có thể gồm nhiều code point và ký tự nối. Vì vậy đếm `len(text)` không tự cho số token, số glyph hiển thị hay số byte.

*(Định nghĩa và giải thích:
- **Composed (Dạng dựng sẵn - Precomposed):** Ký tự được biểu diễn bằng một điểm mã Unicode duy nhất. Ví dụ chữ `é` dựng sẵn có mã `U+00E9` (1 code point, khi mã hóa UTF-8 tốn 2 byte `0xC3 0xA9`).
- **Decomposed (Dạng tổ hợp / phân rã):** Ký tự được tạo thành bằng cách ghép chữ cái gốc `e` (`U+0065`) với ký hiệu dấu sắc rời `combining acute accent` (`U+0301`) (tổng cộng 2 code point, khi mã hóa UTF-8 tốn 3 byte `0x65 0xCC 0x81`).
- **Glyph (Hình chữ hiển thị):** Hình ảnh thực tế hiển thị trên màn hình mà mắt người nhìn thấy. Cả hai chuỗi trên đều vẽ ra cùng một glyph chữ `é`, nhưng trong bộ nhớ máy tính chúng là hai mảng byte hoàn toàn khác nhau!).*

```python
import unicodedata
composed = 'é'
decomposed = 'e\u0301'
assert composed != decomposed
assert unicodedata.normalize('NFC', decomposed) == composed
print(len(composed), len(composed.encode('utf-8')))    # 1 2
print(len(decomposed), len(decomposed.encode('utf-8')))  # 2 3
```

NFC ghép các dạng tương đương chuẩn tắc khi có thể; NFKC còn xử lý tương đương compatibility, có thể đổi cách biểu diễn cần giữ. API normalization và khác biệt các dạng được mô tả trong [Python 3.14 unicodedata](https://docs.python.org/3.14/library/unicodedata.html). Lab **không normalize** để giữ roundtrip nguyên bản; không dùng NFKC như bước “làm sạch” mặc định cho mọi dữ liệu.

*(Trực giác và phân biệt NFC vs NFKC:
- **NFC (Normalization Form C):** Ghép các tổ hợp ký tự rời rạc thành dạng dựng sẵn chuẩn tắc (Canonical Composition). Ví dụ chuyển chữ `e` kèm dấu sắc rời thành một ký tự `é` duy nhất. Đây là chuẩn an toàn nhất để so sánh chuỗi văn bản.
- **NFKC (Normalization Form KC):** Chuẩn hóa tương thích (Compatibility). Dạng này biến đổi rất mạnh tay: nó biến ký tự phân số `½` thành chuỗi 3 ký tự `1/2`, biến ký tự số mũ `x²` thành `x2`, biến các ký tự hoa mỹ thuật trong vòng tròn thành chữ cái thường. Trong các hệ thống xử lý mã nguồn hoặc toán học, nếu bạn tự động áp dụng NFKC, bạn có thể vô tình làm sai lệch biểu thức toán học hoặc phá hỏng cú pháp code).*

### Vì sao backend engineer cần quan tâm?

- JSON `{"price": 10}` và `{"price":10}` có thể mang cùng ý nghĩa ứng dụng nhưng là chuỗi khác, có thể có token khác. *(Ví dụ thực tế: Trong hệ thống microservices, hai payload JSON này khi đưa vào parser JSON đều cho ra cùng một dictionary. Nhưng với tokenizer của LLM, chuỗi thứ nhất có token `" 10"` (có dấu cách trước số), chuỗi thứ hai có token `"10"` (không dấu cách). Hai token này có ID khác nhau, khiến vector embedding khác nhau và làm hỏng cơ chế Prompt Caching của các hệ thống LLM Gateway).*
- Trong Python/YAML, indent có ý nghĩa; `split()` rồi nối bằng một dấu cách có thể phá chương trình. *(Ví dụ thực tế: Cấu trúc khối lệnh của Python phụ thuộc hoàn toàn vào thụt lề (indentation). Nếu lập trình viên áp dụng một hàm tiền xử lý ngây thơ như `text.split()` rồi nối lại bằng một dấu cách đơn lẻ, toàn bộ 4 dấu cách `    ` hoặc tab `\t` ở đầu dòng sẽ bị xóa sạch, biến đoạn code hợp lệ thành lỗi cú pháp `IndentationError` ngay lập tức).*
- `UserID`, `user_id`, `userId` là những identifier khác nhau. Lowercase không phải phép biến đổi vô hại. *(Ví dụ thực tế: Trong lập trình phần mềm, `UserID` thường là tên Class hoặc kiểu dữ liệu, `userId` là tên thuộc tính đối tượng (camelCase), còn `user_id` là tên cột trong cơ sở dữ liệu (snake_case). Nếu tiền xử lý hạ chữ thường toàn bộ (lowercase) theo thói quen cũ của bài toán phân loại văn bản truyền thống, mô hình AI sẽ không thể viết code đúng quy ước đặt tên của dự án).*
- Chuỗi `123456789` có thể bị chia nhiều mảnh; không giả định model nhìn nó như một số nguyên duy nhất. *(Ví dụ thực tế: Tokenizer thường không lưu riêng từng số lớn vào từ điển mà chẻ số thành từng cụm nhỏ, ví dụ `123`, `456`, `789`. Mô hình ngôn ngữ nhìn thấy ba token rời rạc chứ không hề có biểu diễn số học về giá trị "một trăm hai mươi ba triệu bốn trăm năm mươi sáu nghìn bảy trăm tám mươi chín". Đây là lý do cốt lõi vì sao các LLM rất hay tính toán sai các phép nhân chia số lớn nếu không có công cụ máy tính (calculator tool) hỗ trợ).*
- Tokenizer hỗ trợ mọi byte không đồng nghĩa model hiểu mọi ngôn ngữ hoặc xử lý emoji tốt. *(Ví dụ thực tế: Việc tokenizer có đủ 256 byte cơ sở giúp hệ thống không bao giờ bị văng lỗi crash khi gặp chuỗi ký tự lạ hoặc emoji mới. Tuy nhiên, việc mô hình chia được byte không có nghĩa là nó hiểu ngữ nghĩa của ký tự đó. Nếu mô hình chưa từng được huấn luyện sâu trên văn bản tiếng Việt hoặc emoji đó, nó chỉ nhìn thấy một chuỗi các byte rời rạc và rất dễ sinh ra văn bản ngô nghê hoặc lỗi encoding).*

![Code point và UTF-8 của hai dạng é](figures/02-unicode.png)

## 4. Các cách chia token và đánh đổi

| Cách | Trực giác | Lợi ích | Giới hạn |
|---|---|---|---|
| Theo khoảng trắng/từ | Tách “api timeout” thành hai mục | Dễ hiểu, chuỗi ngắn trên từ quen | Từ mới cần UNK hoặc cơ chế dự phòng; dấu câu/space cần xử lý |
| Theo code point | Tách từng phần tử của chuỗi Python | Vocabulary nhỏ hơn từ, dễ tự viết | Chuỗi dài; ký tự chưa thấy có thể OOV nếu alphabet chỉ fit từ train |
| Theo byte | UTF-8 → 256 đơn vị byte cơ sở | Bao phủ mọi chuỗi Unicode hợp lệ mà không cần UNK | Tiếng Việt/emoji có thể tốn nhiều token; một token riêng có thể chưa decode được |
| Subword | Ghép các mảnh hay gặp thành đơn vị lớn hơn | Cân bằng độ dài và vocabulary | Phụ thuộc dữ liệu/thuật toán; không đảm bảo theo hình vị hay ngữ nghĩa |

*(Giải thích chi tiết 4 cách tiếp cận:
1. **Tách theo khoảng trắng/từ (Word-level Tokenization):**
   - *Cơ chế:* Cứ thấy dấu cách là tách thành một từ riêng.
   - *Hạn chế:* Bảng từ vựng sẽ phình to khổng lồ (hàng triệu từ). Mỗi khi người dùng gõ từ mới, thuật ngữ chuyên ngành (như `Kubernetes`), hoặc gõ sai chính tả (như `timeouut`), hệ thống không có từ này trong từ điển và bắt buộc phải thay bằng token `[UNK]` (Unknown - không xác định). Khi gặp `[UNK]`, mô hình hoàn toàn "mù" trước nội dung của từ đó.
2. **Tách theo ký tự / code point (Character-level Tokenization):**
   - *Cơ chế:* Tách rời từng ký tự một (ví dụ `"api"` $\to$ `["a", "p", "i"]`).
   - *Hạn chế:* Từ vựng rất nhỏ (vài nghìn ký tự), không lo thiếu từ. Nhưng chuỗi token sẽ cực kỳ dài. Một đoạn văn bản 1,000 từ có thể bị kéo dài thành 6,000 token ký tự. Vì độ phức tạp tính toán của Transformer tăng theo bình phương độ dài chuỗi ($O(T^2)$), chuỗi quá dài sẽ làm bộ nhớ GPU cạn kiệt và thời gian phản hồi tăng vọt.
3. **Tách theo byte (Byte-level Tokenization):**
   - *Cơ chế:* Mọi chuỗi văn bản đều được chuyển thành mảng byte UTF-8, sử dụng 256 giá trị byte cơ sở (từ `0x00` đến `0xFF`).
   - *Ưu điểm:* Độ phủ 100% mọi dữ liệu nhị phân trên máy tính, hoàn toàn loại bỏ token `[UNK]`.
   - *Hạn chế:* Các ký tự tiếng Việt hoặc emoji tốn từ 2 đến 4 byte, khiến chuỗi đầu vào vẫn bị dài nếu không được nén tiếp.
4. **Tách theo từ con (Subword Tokenization - BPE, WordPiece, Unigram):**
   - *Cơ chế:* Là giải pháp dung hòa hoàn hảo giữa cấp độ từ và cấp độ byte. Các từ phổ biến xuất hiện thường xuyên (như "order", "xin", "chào") được giữ nguyên thành một token duy nhất. Các từ hiếm hoặc từ mới được tự động chẻ thành các mảnh nhỏ quen thuộc (ví dụ từ `"microservices"` được chẻ thành `"micro"` + `"services"`). Đây là thuật toán nền tảng của mọi mô hình ngôn ngữ lớn hiện đại như GPT-4, Claude, Gemini và Llama).*

Vocabulary càng lớn không tự làm model tốt hơn: embedding và output head có thể lớn hơn; đơn vị hiếm khó được học đủ. Vocabulary nhỏ có thể làm T dài hơn, tăng tải attention và dùng nhiều context hơn. Cần đo trên loại văn bản thật của ứng dụng.

*(Phân tích đánh đổi toán học và tài nguyên hệ thống:
1. **Kích thước ma trận tham số:** Ma trận Embedding ở đầu vào và ma trận Linear Head ở đầu ra đều có kích thước phụ thuộc vào kích thước từ vựng $V$. Nếu chiều ẩn là $D = 4096$, tăng kích thước từ vựng từ $V = 32,000$ lên $V = 256,000$ (tăng thêm 224,000 token) sẽ khiến mô hình tốn thêm $2 \times (224,000 \times 4096) \approx 1.83 \text{ tỷ tham số}$ chỉ để lưu bảng từ điển!
2. **Vấn đề từ vựng đuôi dài (Long-tail problem):** Những token hiếm chỉ xuất hiện 1-2 lần trong toàn bộ tập dữ liệu huấn luyện sẽ nhận được rất ít gradient cập nhật, khiến vector biểu diễn của chúng gần như là nhiễu ngẫu nhiên, gây lãng phí bộ nhớ vô ích.
3. **Độ dài ngữ cảnh $T$ và chi phí vận hành:** Nếu chọn từ vựng quá nhỏ, một đoạn log dài 500 từ có thể bị băm thành 2,500 token. Với context window giới hạn (ví dụ 4,096 token), bạn chỉ nhồi được rất ít dữ liệu vào prompt. Đồng thời, vì chi phí gọi API của các dịch vụ LLM thương mại tính tiền theo số lượng token, một tokenizer chia vụn sẽ trực tiếp làm tăng chi phí hóa đơn hàng tháng của doanh nghiệp).*

BPE là một cách học subword bằng ghép cặp thường gặp. Ngoài BPE còn có thuật toán khác; bài này không triển khai hay coi mọi tokenizer subword đều là BPE. [Hugging Face, phần BPE training/tokenization](https://huggingface.co/learn/llm-course/chapter6/5).

## 5. Pipeline tokenizer: bốn việc riêng

1. **Normalization:** có đổi dạng Unicode, chữ hoa/thường hay không? Lab chọn không đổi. *(Định nghĩa: Giai đoạn tiền xử lý nhằm chuẩn hóa chuỗi về một dạng đồng nhất. Tuy nhiên trong kỹ thuật backend và sinh mã nguồn, việc giữ nguyên từng khoảng trắng, xuống dòng và ký tự gốc là tối quan trọng để không làm hỏng cú pháp).*
2. **Pre-tokenization:** tạo ranh giới trước thuật toán chính. Lab tách các run whitespace và non-whitespace, giữ nguyên cả hai; không dùng `str.split()` làm mất space. *(Định nghĩa: Phân tách văn bản thành các cụm thô ban đầu dựa trên quy tắc đơn giản (như tách cụm khoảng trắng liên tiếp và cụm ký tự liên tiếp). Việc này ngăn không cho thuật toán gộp nhầm các ký tự qua ranh giới từ, ví dụ không bao giờ gộp chữ cuối của từ này với chữ đầu của từ kia thành một token kỳ quặc).*
3. **Mô hình tokenization:** chuyển từng run thành byte ID, áp các merge BPE đã học theo thứ tự. *(Định nghĩa: Trọng tâm thuật toán subword — biến các cụm thô thành các dãy ID số nguyên dựa trên danh sách quy tắc gộp - merge table - đã học).*
4. **Đóng khung và decode:** thêm BOS/EOS theo tác vụ; ghép byte khi decode; xử lý special ID theo quy ước. *(Định nghĩa: Bổ sung các token điều khiển hệ thống như `[BOS]` (bắt đầu câu), `[EOS]` (kết thúc câu) để báo hiệu cho mô hình biết khi nào bắt đầu đọc và khi nào kết thúc sinh. Khi chuyển ngược từ ID sang chữ, thực hiện ráp nối các mảng byte về chuỗi UTF-8 hiển thị cho người dùng).*

Trong lab, cặp không được merge qua ranh giới space/non-space hoặc qua tài liệu. Chuỗi nhiều khoảng trắng vẫn được giữ; các byte trong cùng run whitespace có thể được merge. Dấu câu ở cùng non-space run nên có thể ghép cùng chữ. Đây là lựa chọn đơn giản riêng của bài, không mô phỏng regex của mọi tokenizer có sẵn.

**Hai quá trình học độc lập:** fit tokenizer đếm cặp, không dùng backprop; train LM cập nhật embedding/Transformer/head bằng gradient, không tự sửa merge list. Với pretrained model, phải dùng tokenizer tương thích, không fit lại ID tùy ý rồi giữ nguyên weight.

*(Cảnh báo lỗi thực tế - Tokenizer Mismatch Bug: Tokenizer và Mô hình nơ-ron là hai thực thể tách rời hoàn toàn! Tokenizer được "học" (fit) bằng thống kê tần suất xuất hiện thuần túy (thuật toán đếm, chạy trong vài phút trên CPU, không có đạo hàm, không có loss). Sau khi tokenizer đã được chốt (freeze), bộ từ điển và bảng quy tắc merge trở thành "hợp đồng bất biến". Mô hình ngôn ngữ (Transformer) mới bắt đầu được huấn luyện bằng Gradient Descent và Backpropagation để học trọng số ma trận. Nếu sau này bạn sửa đổi dù chỉ 1 quy tắc merge trong tokenizer, thứ tự ánh xạ ID sẽ bị đảo lộn: ví dụ ID 100 trong tokenizer cũ là chữ `'a'`, nhưng trong tokenizer mới lại là chữ `'b'`. Mô hình sẽ đọc nhầm toàn bộ vector embedding và sinh ra những chuỗi ký tự vô nghĩa hoàn toàn).*

## 6. Tính tay BPE: học và áp merge

Corpus tự tạo gồm hai tài liệu `aba`, `aba` và một tài liệu `ac` (tức ba tài liệu). Base byte ID trong lab là `byte+3`, nên `a=100`, `b=101`, `c=102`; PAD=0, BOS=1, EOS=2. Bắt đầu có 259 ID gồm 256 byte và 3 special.

*(Giải thích chi tiết vì sao có các con số này:
- 3 token đặc biệt được gán ID đầu tiên: `PAD = 0`, `BOS = 1`, `EOS = 2`.
- 256 byte giá trị từ $0$ đến $255$ được dời chỉ số bằng công thức `byte + 3`, chiếm các ID từ $3$ đến $258$.
- Mã ASCII của chữ cái thường: `'a'` là 97 $\to$ ID là $97 + 3 = 100$; `'b'` là 98 $\to$ ID là $98 + 3 = 101$; `'c'` là 99 $\to$ ID là $99 + 3 = 102$.
- Tổng kích thước từ điển ban đầu là $3 + 256 = 259$ token).*

### Vòng 1

```text
aba × 2 → [a,b,a] × 2
ac  × 1 → [a,c]
```

Cặp (a,b) xuất hiện 2 lần; (b,a) 2 lần; (a,c) 1 lần. Quy tắc hòa của lab: chọn cặp có tuple ID nhỏ nhất. (100,101) thắng (101,100); tạo ID 259 cho byte `ab`.

*(Giải thích từng bước:
- Trong tài liệu `aba`, ta có hai cặp liền kề: `(a, b)` tức `(100, 101)` và `(b, a)` tức `(101, 100)`. Vì có 2 tài liệu `aba`, mỗi cặp xuất hiện 2 lần.
- Trong tài liệu `ac`, ta có một cặp liền kề: `(a, c)` tức `(100, 102)`, xuất hiện 1 lần.
- Cả hai cặp `(100, 101)` và `(101, 100)` đều có số lần xuất hiện cao nhất (2 lần). Để chọn ra cặp chiến thắng một cách tất định (deterministic), ta áp dụng quy tắc phá vỡ hòa (tie-breaking): so sánh giá trị tuple `(id1, id2)`, cặp nào có số nhỏ hơn sẽ được ưu tiên. Do $(100, 101) < (101, 100)$, cặp `(a, b)` được chọn.
- Hệ thống tạo ra một token mới có ID là 259 đại diện cho chuỗi byte `ab`).*

Sau merge: `[ab,a] × 2`, `[a,c]`. Không ghép ký tự từ cuối tài liệu này sang đầu tài liệu khác.

### Vòng 2

(ab,a) có 2 lần, (a,c) 1 lần. Tạo ID 260 cho `aba`. Vocabulary cuối có **261 mục**, các byte cũ không bị xóa.

*(Giải thích từng bước:
- Lúc này dữ liệu đã trở thành: hai chuỗi `[259, 100]` và một chuỗi `[100, 102]`.
- Cặp `(ab, a)` tức `(259, 100)` xuất hiện 2 lần, vượt trội so với `(a, c)` chỉ có 1 lần.
- Hệ thống tạo ra token mới có ID là 260 đại diện cho chuỗi `aba`.
- Bảng từ vựng mở rộng từ 259 lên 261 mục (gồm 259 mục gốc + 2 mục mới ghép: 259 là `ab`, 260 là `aba`)).*

Encoding sau fit: `aba → [260]`, `ac → [100,102]`, `abac → [260,102]`. Từ mới vẫn encode được bằng merge quen và byte còn lại. Encoding không đếm lại tần suất của input mới để học merge mới.

*(Giải thích quá trình mã hóa từ mới `abac`:
1. Đầu tiên, chuỗi được chuyển thành các byte ID cơ sở: `[100, 101, 100, 102]`.
2. Áp dụng quy tắc merge 1 `(100, 101) -> 259`: hai phần tử đầu được gộp lại $\to$ `[259, 100, 102]`.
3. Áp dụng quy tắc merge 2 `(259, 100) -> 260`: hai phần tử đầu tiếp tục được gộp $\to$ `[260, 102]`.
4. Kết quả thu được mảng ID là `[260, 102]`. Từ mới `abac` chưa từng xuất hiện trong tập huấn luyện nhưng vẫn được mã hóa trơn tru bằng cách tận dụng mảnh quen thuộc `aba` (ID 260) ghép với byte `c` (ID 102) mà không gặp bất kỳ lỗi từ lạ nào).*

Nếu input là `aaaa`, số cặp (a,a) kề nhau là 3 nhưng phép thay cặp không chồng lấn từ trái sang phải chỉ tạo 2 token `aa` trong một lượt. Phải phân biệt thống kê cặp với số phép thay thực hiện.

*(Giải thích chi tiết phép thay thế không chồng lấn - Non-overlapping replacement:
- Trong chuỗi `aaaa`, các vị trí kề nhau là (0,1), (1,2), (2,3). Khi đếm thống kê tần suất, ta thấy cặp `(a,a)` xuất hiện 3 lần.
- Nhưng khi thực hiện thay thế thực tế từ trái sang phải: cặp tại vị trí (0,1) được gộp thành token `aa`, con trỏ duyệt nhảy ngay sang vị trí 2 (bỏ qua vị trí 1 vì ký tự tại vị trí 1 đã bị gộp vào token trước). Tại vị trí (2,3), hai ký tự tiếp theo được gộp thành `aa`.
- Kết quả chuỗi trở thành `[aa, aa]` gồm 2 token, chứ không thể biến thành 3 token chồng lấn lên nhau được).*

```python
from lab import ByteBPE
tok = ByteBPE.fit(['aba', 'aba', 'ac'], steps=2)
assert tok.encode('aba') == [260]
assert tok.encode('abac') == [260, 102]
assert tok.decode(tok.encode('  Xin chào 🙂\n')) == '  Xin chào 🙂\n'
print(tok.merges)  # [(100, 101), (259, 100)]
```

Lab dùng phiên bản BPE byte đơn giản, ưu tiên dễ kiểm. Tốc độ không phù hợp corpus lớn; không nhận dạng special token trong raw text; không có offset mapping hoặc streaming decoder tối ưu. Nguyên lý đếm cặp và áp merge theo thứ tự được đối chiếu với [Hugging Face BPE](https://huggingface.co/learn/llm-course/chapter6/5); ví dụ số và code ở đây được tự xây.

![Hai vòng merge của ví dụ tự tạo](figures/03-bpe.png)

## 7. Token đặc biệt, decode và hợp đồng với checkpoint

| ID của lab | Vai trò | Lưu ý |
|---|---|---|
| PAD=0 | Điền batch cho cùng chiều dài | Chặn key PAD và bỏ nhãn PAD bằng IGNORE, hai việc khác nhau |
| BOS=1 | Bắt đầu tài liệu/prefix | Thêm bằng code khi tạo batch, không phải chuỗi chữ tự được hiểu |
| EOS=2 | Dự đoán kết thúc tài liệu | Là nhãn thật cần chấm loss; không bỏ như PAD |
| IGNORE=−100 | Bỏ vị trí loss | Không nằm trong vocabulary, không đưa vào embedding |
| UNK | Lab không cần | Vì mọi UTF-8 byte đều có trong base vocabulary |

*(Giải thích vai trò và ứng dụng thực tế của từng token đặc biệt:
- **PAD (Padding - Điền khuyết, ID 0):** Trong huấn luyện mini-batch, các câu văn có độ dài khác nhau (câu 5 token, câu 12 token). Để xếp chúng thành một ma trận tensor chữ nhật `[Batch_Size, Max_Length]`, ta phải thêm các số 0 vào đuôi các câu ngắn. Trong cơ chế Attention, ta dùng `padding_mask` để che các số 0 này đi, ngăn không cho các từ nội dung chú ý vào khoảng đệm vô nghĩa.
- **BOS (Beginning of Sequence - Bắt đầu chuỗi, ID 1):** Đóng vai trò như "phát súng lệnh" hoặc tín hiệu mồi. Khi bạn yêu cầu mô hình sinh câu từ đầu mà chưa có văn bản nào trước đó, bạn đưa `BOS` vào để mô hình dựa vào đó dự đoán token đầu tiên.
- **EOS (End of Sequence - Kết thúc chuỗi, ID 2):** Báo hiệu câu văn hoặc tài liệu đã hoàn thành trọn vẹn. Nếu không có `EOS`, mô hình sẽ sinh token liên miên không bao giờ biết dừng cho đến khi cạn kiệt bộ nhớ context! Đây là lý do ta phải chấm điểm loss cho `EOS` lúc huấn luyện: mô hình bắt buộc phải học được thời điểm nào nên dừng lại.
- **IGNORE (`-100` trong PyTorch):** Đây không phải là một token trong từ điển, mà là một cờ hiệu số học đặc biệt báo cho hàm mất mát `nn.CrossEntropyLoss(ignore_index=-100)` biết rằng: "Vị trí này chỉ là số đệm padding, hãy bỏ qua và ĐỪNG tính đạo hàm/gradient tại đây!".
- **UNK (Unknown - Token lạ):** Thường gặp trong các tokenizer cấp độ từ khi gặp từ mới chưa từng xuất hiện. Trong tokenizer byte-level BPE, vì 256 byte cơ bản đã bao phủ 100% dữ liệu nhị phân, thuật toán không bao giờ cần đến `UNK`).*

Raw text `<BOS>` trong lab là các byte chữ và dấu ngoặc, **không** biến thành ID 1. Chat template với role system/user/assistant và control token là hợp đồng riêng của từng mô hình; không có một chuỗi role chung áp dụng mọi tokenizer. Lab chỉ LM tiếp diễn văn bản, chưa huấn luyện hội thoại.

Đảm bảo `decode(encode(text)) == text` cho mọi chuỗi Unicode hợp lệ của Python trong phạm vi UTF-8 strict (không chứa surrogate đơn lẻ). Nhưng một **dãy ID model tự sinh tùy ý** có thể là byte không tạo thành UTF-8 hợp lệ. Cần ghép byte rồi decode; không decode từng byte token một cách độc lập với `errors='ignore'` vì có thể mất dấu và mất dữ liệu.

*(Ví dụ thực tế về hiện tượng mảnh byte dở dang (Partial Byte):
- Trong tiếng Việt, chữ cái `'ạ'` được mã hóa UTF-8 bằng tổ hợp 3 byte: `0xE1, 0xBA, 0xA1`.
- Khi mô hình sinh văn bản từng bước một (autoregressive generation), giả sử ở bước hiện tại nó sinh ra byte `0xE1`. Byte này đứng một mình KHÔNG PHẢI là một ký tự Unicode hợp lệ! Nếu bạn cố tình gọi hàm giải mã `decode(errors='strict')` ngay lập tức, chương trình sẽ văng ngoại lệ `UnicodeDecodeError`.
- Ứng dụng thực tế: Trong các hệ thống Streaming LLM (hiển thị chữ chạy gõ phím trên giao diện web theo thời gian thực), backend phải duy trì một bộ đệm byte (byte buffer) để tích lũy đủ các byte hợp lệ của một ký tự rồi mới giải mã và bắn về giao diện người dùng).*

Tokenizer lab có `raw(ids)` để xem byte; `decode(..., errors='strict')` báo lỗi dữ liệu sai. Khi hiển thị generation, code dùng replacement nếu cần và báo `valid_utf8=False`. Việc thay ký tự để hiển thị không sửa nội dung sinh.

Tokenizer và model giống một giao thức serialization: đổi bảng ID mà giữ weight khiến model đọc nhầm vector. Khi load, lab kiểm hash ordered merge list; version/normalization/pre-tokenizer cũng được kiểm. Lưu tokenizer JSON cùng checkpoint, không chỉ lưu “vocab size=323”. Hai tokenizer có cùng size vẫn có thể có mapping khác.

## 8. Mô hình ngôn ngữ thực sự học gì?

Causal language model học phân phối token tiếp theo khi biết prefix. *(Định nghĩa: "Causal" ở đây mang nghĩa nhân quả hoặc tự hồi quy (Autoregressive) — mô hình chỉ được phép nhìn về quá khứ (các token đã xuất hiện bên trái) để dự đoán token tiếp theo bên phải, tuyệt đối không được nhìn trộm tương lai).* Ví dụ prefix `Dịch vụ api: ` có thể được nối `timeout`, `lỗi kết nối` hoặc một phần byte của chữ khác tùy vocabulary. Mô hình không phải lúc nào cũng dự đoán trọn một từ hay một câu.

Với chuỗi token x₁…xₙ và BOS:

\[
P(x_1,\ldots,x_n,EOS\mid BOS)=\prod_{t=1}^{n+1}P(x_t\mid BOS,x_{<t}),\quad x_{n+1}=EOS.
\]

*(Giải thích chi tiết công thức toán học cho người mới:
- $x_1, \ldots, x_n$: Dãy các token nội dung của câu văn bản.
- $BOS$: Token mồi báo hiệu bắt đầu chuỗi.
- $EOS$: Token báo hiệu kết thúc câu ($x_{n+1} = EOS$).
- $x_{<t}$: Tất cả các token đứng trước vị trí $t$ (ngữ cảnh quá khứ - prefix). Ví dụ tại $t=3$, $x_{<3}$ là $(x_1, x_2)$.
- $P(x_t \mid BOS, x_{<t})$: Xác suất có điều kiện để token tiếp theo là $x_t$, khi đã biết toàn bộ các token phía trước.
- Ký hiệu tích $\prod_{t=1}^{n+1}$: Phép nhân liên tiếp tất cả các xác suất thành phần lại với nhau.
- **Trực giác đời thường:** Xác suất để câu `"Dịch vụ api timeout [EOS]"` xuất hiện chính bằng:
  $P(\text{"Dịch"}\mid \text{BOS}) \times P(\text{"vụ"}\mid \text{"Dịch"}) \times P(\text{"api"}\mid \text{"Dịch vụ"}) \times P(\text{"timeout"}\mid \text{"Dịch vụ api"}) \times P(\text{EOS}\mid \text{"Dịch vụ api timeout"})$).*

Đây là phân rã theo chain rule xác suất, không phải chain rule đạo hàm bài 10. *(Phân biệt rõ ràng: Chain rule trong giải tích bài 10 là quy tắc dây chuyền dùng để tính đạo hàm của hàm hợp $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ phục vụ lan truyền ngược; còn Chain rule ở đây là quy tắc nhân xác suất có điều kiện trong lý thuyết xác suất thống kê: $P(A, B) = P(A) \cdot P(B \mid A)$).* Giả định Markov bậc 1 của bigram chỉ dùng token liền trước; Transformer có thể dùng nhiều token trong context cho phép, không bị giới hạn đúng một token. *(So sánh trực quan: Mô hình Bigram chỉ có "trí nhớ ngắn hạn" 1 từ: để đoán từ tiếp theo, nó chỉ nhìn đúng 1 từ vừa gõ. Trong khi đó, Transformer sở hữu cơ chế Self-Attention cho phép nó "nhìn lại" toàn bộ hàng nghìn token trong quá khứ ($x_{<t}$), giúp nắm bắt ngữ cảnh câu chuyện và cấu trúc ngữ pháp phức tạp từ rất xa).*

Một hệ thống autocomplete gợi ý `return` sau code có xác suất cao không đồng nghĩa đoạn code chạy đúng. Tương tự, câu “đã khắc phục lỗi” dễ xuất hiện trong mẫu ticket không phải bằng chứng lỗi thực đã được khắc phục. Muốn làm trợ lý hỏi đáp cần dữ liệu, instruction tuning/điều khiển tác vụ và đánh giá riêng; LM base không tự có các khả năng ấy. *(Ứng dụng thực tế: Bản chất của Causal LM chỉ là một bộ máy mô phỏng phân phối thống kê của ngôn ngữ — nó là một "công cụ tự động điền từ thông minh" (glorified autocomplete). Khi mô hình sinh ra dòng chữ "Đã hoàn tiền cho khách", điều đó chỉ phản ánh rằng chuỗi ký tự này có xác suất thống kê xuất hiện rất cao sau các câu khiếu nại, chứ hoàn toàn KHÔNG có nghĩa là hệ thống backend đã thực sự thực hiện giao dịch chuyển tiền).*

Encoder masked-LM giấu một số vị trí để dự đoán theo context hai phía là mục tiêu học khác causal next-token. *(Phân biệt Causal LM vs Masked LM: Masked LM (như BERT) che giấu ngẫu nhiên một số từ ở giữa câu và dùng ngữ cảnh ở cả hai bên trái-phải để đoán từ bị giấu, phù hợp cho bài toán hiểu văn bản, trích xuất embedding và phân loại; trong khi Causal LM (như GPT) chỉ nhìn từ trái sang phải để sinh từ tiếp theo, phù hợp cho bài toán tạo sinh văn bản).* Không dùng nguyên công thức likelihood trái→phải của lab để báo PPL cho mọi masked-LM. [Hugging Face perplexity](https://huggingface.co/docs/transformers/main/en/perplexity).

## 9. Ví dụ tính tay xác suất, CE và perplexity

Xét một chuỗi đã tokenize thành hai token nội dung A,B rồi EOS. Model gán:

```text
P(A | BOS) = 1/2
P(B | BOS,A) = 1/4
P(EOS | BOS,A,B) = 1/8
```

Xác suất đúng cả chuỗi có kết thúc =1/64=0.015625. *(Giải thích: Tích xác suất thành phần: $P = \frac{1}{2} \times \frac{1}{4} \times \frac{1}{8} = \frac{1}{64} = 0.015625$. Tuy nhiên, nếu một đoạn văn dài 100 token, việc nhân 100 số nhỏ hơn 1 sẽ dẫn đến hiện tượng tràn số dưới (arithmetic underflow - số quá nhỏ khiến máy tính coi bằng 0). Do đó, trong thực tế ta luôn chuyển sang miền logarit).* Tổng negative log-likelihood (NLL) =ln2+ln4+ln8=ln64. *(Định nghĩa NLL: $\text{NLL} = -\ln(P) = -\sum \ln(p_t)$. Nhờ tính chất $\ln(a \times b) = \ln(a) + \ln(b)$, phép nhân biến thành phép cộng: $\ln(2) + \ln(4) + \ln(8) = \ln(64) \approx 4.158883$. NLL đo lường mức độ "bất ngờ" của mô hình: xác suất đoán càng cao thì NLL càng nhỏ).* CE trung bình chia 3 nhãn thật, bao gồm EOS, không bao gồm BOS:

\[
CE=-\frac13\sum_t\ln p_t=\ln4\approx1.386294\text{ nats/token},
\qquad PPL=\exp(CE)=4.
\]

*(Giải thích chi tiết Cross-Entropy và Perplexity:
- **Cross-Entropy (CE - Hàm mất mát entropy chéo):** Là giá trị NLL trung bình trên mỗi token:
  $CE = \frac{\ln(64)}{3} = \frac{4.158883}{3} = \ln(4) \approx 1.386294 \text{ nats/token}$.
  *Lưu ý về đơn vị:* Khi dùng log tự nhiên ($\ln$, cơ số $e$), đơn vị đo là **nats/token**. Nếu dùng $\log_2$, đơn vị đo là **bits/token**. Trong PyTorch, hàm `F.cross_entropy` mặc định tính theo $\ln$ nên kết quả luôn có đơn vị nats.
- **Perplexity (PPL - Độ bối rối / Độ phân vân):**
  $PPL = \exp(CE) = e^{1.386294} = 4$.
  *Trực giác cực kỳ dễ hiểu cho người mới:* $PPL = 4$ có nghĩa là **tại mỗi bước sinh token, mức độ phân vân của mô hình tương đương với việc phải lựa chọn ngẫu nhiên giữa 4 khả năng có cơ hội ngang nhau!**
  Nếu mô hình hoàn hảo đoán trúng 100% với $p=1$, thì $CE = 0$ và $PPL = e^0 = 1$ (hoàn toàn tự tin, chỉ có đúng 1 lựa chọn chắc chắn). Ngược lại, nếu mô hình hoàn toàn mù tịt và đoán mò ngẫu nhiên trên toàn bộ từ điển $V = 32,000$, thì $PPL = 32,000$ (bối rối tột độ)).*

Không lấy mean xác suất (0.5+0.25+0.125)/3 rồi đảo; phải dùng log hoặc geometric mean. *(Giải thích: Nếu lấy trung bình cộng $(0.5 + 0.25 + 0.125) / 3 = 0.2916$ rồi nghịch đảo $1 / 0.2916 \approx 3.43$, con số này bị sai lệch toán học nghiêm trọng vì xác suất của chuỗi mang tính chất nhân lũy thừa, trung bình tương ứng bắt buộc phải là trung bình nhân).* PPL=4 không có nghĩa model chỉ biết 4 từ hoặc token accuracy=25%; nó là cách biến đổi mức NLL trung bình. Hai model cùng PPL có thể khác top-1 accuracy.

Nếu model uniform trên V class thì mỗi nhãn có xác suất 1/V, CE=lnV, PPL=V. Nếu đã mask một số output class rồi renormalize lúc generation, phân phối lúc sinh đã khác phân phối đầy đủ dùng để tính CE của lab.

**So sánh PPL cần cùng điều kiện:** dữ liệu, tokenizer, nhãn được chấm (có EOS không), context và cách reset tài liệu. Chia cùng văn bản thành token khác làm thay đổi đơn vị; không xếp hạng hai tokenizer chỉ vì PPL/token thấp hơn. *(Ví dụ thực tế vì sao không thể so sánh PPL giữa 2 tokenizer khác nhau:
- Giả sử cùng một đoạn văn bản tiếng Việt có tổng NLL là 30 nats.
- **Mô hình A (dùng tokenizer cấp từ):** Chia đoạn văn thành 10 token $\to$ $CE = 30 / 10 = 3.0 \text{ nats/token} \to PPL = \exp(3.0) \approx 20.08$.
- **Mô hình B (dùng tokenizer cấp byte):** Chẻ vụn đoạn văn thành 30 token $\to$ $CE = 30 / 30 = 1.0 \text{ nats/token} \to PPL = \exp(1.0) \approx 2.71$.
- Nhìn qua, PPL của mô hình B ($2.71$) có vẻ "thông minh" hơn nhiều so với mô hình A ($20.08$). Nhưng thực chất cả hai mô hình gán cùng một xác suất tổng thể cho văn bản! PPL của B thấp hơn chỉ vì mẫu số (số lượng token) lớn gấp 3 lần. Do đó, chỉ so sánh PPL khi hai mô hình dùng CHUNG một tokenizer).* Xem [phụ lục](deep-dive.md) cho ví dụ và giới hạn của đo theo byte. [Định nghĩa PPL và context window](https://huggingface.co/docs/transformers/main/en/perplexity).

*(Ứng dụng thực tế của PPL trong kỹ thuật AI:
1. **Lựa chọn Checkpoint tốt nhất:** Trong quá trình huấn luyện, mô hình nào có Validation PPL thấp nhất thường được chọn làm checkpoint sản xuất.
2. **Lọc dữ liệu rác (Data Filtering):** Dùng một mô hình ngôn ngữ chuẩn tính PPL trên các bài viết cào từ web; những văn bản có PPL cao bất thường (chứa chuỗi ký tự vô nghĩa, spam SEO, lỗi vỡ font) sẽ bị loại bỏ trước khi đưa vào tập huấn luyện.
3. **Phát hiện tấn công Prompt Injection:** Các prompt độc hại hoặc chuỗi jailbreak lạ thường có phân phối token bất thường, khiến PPL vọt lên cao so với văn bản thông thường).*

![Xác suất chuỗi và đơn vị đo](figures/04-likelihood.png)

## 10. Chuyển văn bản thành bài toán next-token

Giả sử một tokenizer minh họa chia `api lỗi` thành `[api,space,lỗi]`:

```text
full:  BOS  api  space  lỗi  EOS
x:     BOS  api  space  lỗi
y:     api  space lỗi   EOS
```

*(Giải thích chi tiết cơ chế Shifting (Dịch chuyển 1 vị trí):
- Bản chất bài toán Next-Token Prediction là: "Nếu tôi cho bạn thấy ngữ cảnh từ đầu đến thời điểm $t$, bạn hãy đoán xem token ở thời điểm $t+1$ là gì?".
- Vì vậy, với một chuỗi đầy đủ gồm $L$ token:
  + Dữ liệu đầu vào ($x$) lấy từ đầu đến áp chót: `full[:-1]` (chiều dài $L-1$).
  + Nhãn mục tiêu ($y$) lấy từ phần tử thứ 2 đến cuối cùng: `full[1:]` (chiều dài $L-1$).
- Khi ghép cặp theo từng chỉ số vị trí:
  + Tại vị trí 0: Đầu vào là `BOS` $\to$ Nhãn cần đoán là `api`.
  + Tại vị trí 1: Đầu vào là `api` (với ngữ cảnh `[BOS, api]`) $\to$ Nhãn cần đoán là `space`.
  + Tại vị trí 2: Đầu vào là `space` (với ngữ cảnh `[BOS, api, space]`) $\to$ Nhãn cần đoán là `lỗi`.
  + Tại vị trí 3: Đầu vào là `lỗi` (với ngữ cảnh `[BOS, api, space, lỗi]`) $\to$ Nhãn cần đoán là `EOS`).*

Đây là minh họa shift; tokenizer byte BPE thực tế của lab có thể chia khác. `batch()` tự thêm BOS/EOS, lấy full[:-1] và full[1:]. Khác task conditional copy ở bài 18, **lab này chấm tất cả token nội dung và EOS**, không bỏ loss ở prompt train. Self-supervised learning nghĩa là nhãn được lấy từ chính dữ liệu văn bản; không có nghĩa mọi nội dung văn bản là sự thật hoặc không cần dữ liệu. *(Định nghĩa Self-Supervised Learning (Học tự giám sát): Là phương pháp học có giám sát nhưng nhãn được sinh ra hoàn toàn tự động từ chính cấu trúc tự nhiên của dữ liệu mà không cần con người gắn nhãn thủ công từng câu. Đây chính là bí quyết giúp các mô hình ngôn ngữ lớn có thể nuốt trọn hàng nghìn tỷ từ trên Internet mà không tốn chi phí thuê người dán nhãn).*

Mỗi tài liệu là một hàng, right padding; causal mask giữ đường chéo, padding mask chặn PAD key; labels PAD được thay IGNORE. Không nối các tài liệu qua nhau trong lab nên không có attention xuyên tài liệu. Đừng shift thêm lần nữa nếu một API LM khác đã tự shift labels bên trong; `LanguageModel` của bài này không tự tính loss hoặc shift.

*(Giải thích hoạt động của các Mask trong PyTorch:
- **Causal Mask (Mặt nạ nhân quả):** Là ma trận tam giác trên chứa giá trị $-\infty$ ở các vị trí tương lai. Khi cộng ma trận này vào ma trận Attention Scores trước hàm Softmax, trọng số chú ý tới các token tương lai sẽ bằng 0, đảm bảo mô hình không thể "nhìn trộm đáp án" phía trước.
- **Padding Mask (Mặt nạ đệm):** Đánh dấu các vị trí padding `PAD=0` trong batch để cơ chế Attention không lãng phí tài nguyên tính toán với các token vô nghĩa.
- **Loss Mask qua IGNORE=-100:** Tất cả các vị trí padding trong nhãn $y$ được gán giá trị `-100`. Khi tính loss qua `F.cross_entropy(..., ignore_index=-100)`, PyTorch tự động bỏ qua các vị trí này, đảm bảo gradient chỉ được cập nhật dựa trên các token nội dung thật và token `EOS`).*

`cross_entropy` nhận raw logits `[N,V]`, class labels `[N]` và ignore_index; không softmax trước. Đánh giá lấy tổng loss chia tổng nhãn thật, rồi mới exponentiate. Không trung bình PPL từng batch. [CrossEntropyLoss 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html).

![Shift input và nhãn cho language model](figures/05-shift.png)

## 11. Ví dụ ứng dụng: context và dữ liệu cần giữ

Giả sử một ứng dụng đặt giới hạn minh họa 100 token tổng, template/role dùng 10, câu hỏi 20 và dự trù output 30; còn tối đa 40 cho tài liệu. Các số này giả định, không phải quota của một sản phẩm cụ thể. Một đoạn log “chỉ 40 từ” vẫn có thể vượt 40 token, nhất là ID dài, mã nguồn hoặc nhiều byte Unicode.

*(Ứng dụng thực tế: Quản lý ngân sách ngữ cảnh (Context Window Budgeting) trong hệ thống RAG và Agent:
- Khi xây dựng ứng dụng AI trong doanh nghiệp, bạn luôn bị ràng buộc bởi giới hạn context window của mô hình (ví dụ 8,192 token).
- Công thức phân bổ ngân sách: $\text{Tổng Context} = \text{System Prompt} + \text{Lịch sử chat} + \text{Tài liệu RAG tra cứu} + \text{Câu hỏi người dùng} + \text{Ngân sách sinh câu trả lời (Max New Tokens)}$.
- Nếu bạn không tính toán kỹ số token của tài liệu tra cứu (Retrieval Context), câu trả lời của mô hình sẽ bị cắt cụt giữa chừng (`finish_reason = 'length'`) hoặc API văng lỗi quá tải ngữ cảnh `ContextWindowExceededError`).*

Cách xử lý hữu ích là đo bằng đúng tokenizer sau khi thêm template; chọn thông tin cần giữ, chừa output budget và ghi rõ chính sách cắt. Cắt bừa đuôi có thể làm mất phần stack trace gốc hoặc câu hỏi; cắt bừa đầu có thể làm mất định nghĩa biến. Nhiều token hơn không tự là nhiều thông tin hữu ích hơn.

*(Các cạm bẫy khi cắt gọt văn bản (Truncation Policies):
- Cắt chuỗi theo ký tự (`text[:500]`): Rất nguy hiểm vì có thể cắt ngang giữa chừng một byte UTF-8 của tiếng Việt, làm hỏng chuỗi giải mã.
- Cắt chuỗi theo từ (`text.split()[:100]`): Phá vỡ định dạng khoảng trắng và thụt dòng của code.
- Cắt bừa ở đuôi: Trong file log hệ thống, thông tin quan trọng nhất thường là dòng gây lỗi (Root Cause Exception) nằm ở tận cùng của Stack Trace. Nếu cắt cụt đuôi, mô hình sẽ hoàn toàn không thấy nguyên nhân gây lỗi.
- **Giải pháp chuẩn:** Luôn dùng chính Tokenizer của mô hình để mã hóa thành ID, đếm độ dài token thực tế, và áp dụng chiến lược cắt có chủ đích (ví dụ: giữ nguyên System Prompt và câu hỏi người dùng, chỉ cắt bớt các dòng log ở giữa)).*

Đối với completion, `encode(prefix + suffix)` không luôn bằng `encode(prefix) + encode(suffix)` do merge có thể đi qua ranh giới ghép trong một non-space run. Ví dụ tokenizer tính tay: `encode('ab')=[259]`, `encode('a')=[100]`, nhưng `encode('aba')=[260]`. Khi sinh, model nối token vào ID prefix cố định; không retokenize toàn bộ chuỗi văn bản sau mỗi bước để mong “làm đẹp” token.

*(Giải thích hiện tượng bất đối xứng ranh giới token (Token Boundary Mismatch):
- Khi bạn tokenize riêng `'ab'`, nó ra `[259]`. Khi tokenize riêng `'a'`, nó ra `[100]`. Nếu bạn ghép hai mảng ID này lại, bạn được `[259, 100]`.
- Nhưng nếu bạn ghép chuỗi văn bản thành `'aba'` rồi mới tokenize, thuật toán BPE nhận diện được quy tắc merge vòng 2 và mã hóa thành một token duy nhất `[260]`.
- Ứng dụng thực tế: Trong các hệ thống Prompt Caching hoặc sinh văn bản tiếp diễn (continuation), việc ghép chuỗi ở ranh giới giữa Prompt và Output có thể tạo ra các token không tối ưu nếu bạn không cẩn thận với khoảng trắng và ranh giới từ).*

## 12. Lab: byte BPE và Transformer LM trên văn bản backend

### Dữ liệu và kiến trúc

Tạo 192 câu từ 8 tên dịch vụ × 4 loại vấn đề × 6 hành động:

```text
Dịch vụ api: timeout khi đọc hồ sơ.
Dịch vụ auth: thiếu quyền khi xác thực.
```

Tất cả là câu tự tạo, không phải sự cố được xác nhận. Shuffle seed 19, chia nguyên câu 128 train /32 validation /32 test, không trùng câu đầy đủ. Các mảnh template, dịch vụ và vấn đề được chia sẻ: đây là tổng quát hóa tổ hợp hẹp, không phải kiểm chứng trên log thực ngoài miền. Với corpus thật, phải kiểm nguồn/tài liệu/thời gian và trùng gần, không chỉ shuffle từng dòng.

BPE học **64 merge chỉ trên train**; base byte không học từ test vì tập byte được định nghĩa trước. Vocabulary=323. D=32,H=4,F=64, một pre-norm causal block và final LN như bài 18; **learned position embedding** 128×32 thay cho sinusoidal để thấy đây là cấu hình kiến trúc riêng. Không weight tying, không dropout, head 323 class. Toàn mô hình có 33,699 tham số.

*(Giải thích các siêu tham số kiến trúc cho người mới:
- $D = 32$ (Embedding dimension / Model dimension): Kích thước của vector biểu diễn mỗi token. Mỗi token được đặc trưng hóa bằng 32 con số thực.
- $H = 4$ (Attention heads): Số đầu chú ý hoạt động song song trong cơ chế Multi-Head Attention, mỗi đầu có kích thước $d_k = 32 / 4 = 8$. Mỗi đầu chú ý có thể tập trung vào một mối quan hệ ngữ nghĩa khác nhau (ví dụ: một đầu chú ý nhìn vào tên dịch vụ, một đầu chú ý nhìn vào loại lỗi).
- $F = 64$ (Feed-forward dimension): Chiều ẩn của mạng nơ-ron truyền thẳng (FFN) nằm sau lớp Attention, thường gấp 2 đến 4 lần $D$.
- Context limit $= 128$: Chiều dài tối đa của một chuỗi câu (bao gồm cả BOS và EOS) mà mô hình có thể tiếp nhận trong một lượt forward.
- **33,699 tham số:** Tổng số lượng trọng số học được (weights và biases) của toàn bộ mô hình nhỏ này).*

BPE không normalize; mọi câu độc lập; không truncation trong dữ liệu này. Mỗi lần forward từ chối input vượt context 128. Fit tokenizer trước, freeze merge rồi xây embedding/head theo size mới; tokenizer fit lại sẽ đòi model tương thích mới.

### Huấn luyện và artifact

30 epoch, batch 16 (8 update/epoch, tổng 240), Adam lr=0.003, clip norm 1, seed model 2026. Chọn best bằng validation CE; test chỉ đo sau khi chọn best. Checkpoint là inference artifact, không có optimizer/RNG để resume. Run lưu tokenizer.json, data.json, best.pt, metrics.json và report.json chứa config/môi trường/metric/mẫu sinh. Chỉ tải checkpoint tin cậy; dùng weights_only=True.

### Kết quả đã chạy

| Phép đo | Validation | Test |
|---|---:|---:|
| CE, nats/token | 0.412029 | 0.416568 |
| Perplexity | 1.509878 | 1.516747 |
| Top-1 token accuracy, teacher forcing | 84.4681% | 83.2589% |
| Số nhãn gồm EOS | 940 | 896 |

*(Giải thích ý nghĩa của các chỉ số kết quả:
- **CE (0.412 val / 0.416 test):** Mất mát trung bình rất nhỏ (chỉ ~0.41 nats/token), chứng tỏ mô hình học rất khớp mẫu câu backend có sẵn.
- **Perplexity (1.51 val / 1.52 test):** $PPL \approx 1.51$ nghĩa là tại mỗi bước sinh, mô hình hầu như chỉ phân vân giữa 1 và 2 lựa chọn (độ tự tin rất cao do dữ liệu backend tổng hợp có cấu trúc khuôn mẫu lặp lại).
- **Top-1 Token Accuracy (84.4% val / 83.2% test) dưới Teacher Forcing:**
  *Định nghĩa Teacher Forcing:* Khi đánh giá, ở mỗi bước mô hình luôn được cung cấp token thật của đề bài làm đầu vào để dự đoán token kế tiếp, chứ không phải dùng token tự nó vừa sinh ra. 83.2% nghĩa là cứ 100 vị trí thì mô hình đoán trúng phóc token thật tới hơn 83 lần).*

Best theo validation CE là **epoch 30**; số tham số 33,699 và cấu hình được lưu trong `report.json`. Hai baseline đếm token trên train, dùng cùng tokenizer/split/nhãn: unigram add-one có test PPL=29.002664, bigram add-one=14.031948. Đây là baseline đơn giản, smoothing +1 trên toàn 323 class khá mạnh với corpus nhỏ; chênh lệch không chứng minh Transformer luôn hơn mọi mô hình n-gram hoặc tokenizer khác.

Train có 7,423 byte UTF-8, encode thành 3,444 token nội dung (không BOS/EOS), khoảng 2.1553 byte/token. Đây là tỷ số của tokenizer trên corpus cụ thể, không phải công thức đổi từ/byte sang token dùng chung.

Output greedy chạy thật:

```text
Prompt: Dịch vụ api:
Tiếp diễn: phản hồi chậm khi đọc cấu vụ.

Prompt: Dịch vụ auth:
Tiếp diễn: thiếu quyền khi đọc đọc cấu vụ.
```

Prompt thực có một space sau dấu hai chấm, continuation có newline cuối. Cả hai sinh EOS và UTF-8 hợp lệ, nhưng `đọc cấu vụ` bị trộn sai và `đọc đọc` bị lặp. **Không sửa output để làm minh họa đẹp hơn.** PPL thấp trên template dễ đoán vẫn có thể đi kèm văn bản sinh lỗi. *(Giải thích nguyên nhân lỗi sinh văn bản thực tế:
- **Hiện tượng "đọc cấu vụ":** Là sự pha trộn sai giữa hai cụm từ có sẵn trong dữ liệu là "đọc cấu hình" và "gọi dịch vụ". Khi mô hình gặp các cụm từ có xác suất ngang ngửa nhau, cơ chế chọn tham lam (greedy) có thể chọn nhầm mảnh ghép ở giữa.
- **Hiện tượng lặp từ ("đọc đọc"):** Bệnh kinh niên của các mô hình ngôn ngữ nhỏ khi dùng chiến lược sinh tham lam (Greedy Search), do xác suất tự hồi quy rơi vào vòng lặp cực tiểu cục bộ. Để khắc phục điều này trong thực tế, người ta áp dụng kỹ thuật phạt tần suất (frequency penalty) và lấy mẫu xác suất (sampling với temperature / top-p) sẽ học ở bài 21).* Không có một continuation duy nhất đúng cho prompt mở; không báo exact match với một câu tùy chọn như bài copy.

![Đường học và PPL trên cùng tokenizer](figures/06-training.png)

## 13. Inference và decoder của tokenizer

`generate` encode prompt một lần, thêm BOS, chạy model ở eval/no_grad, lấy logits hàng cuối, chọn greedy rồi nối ID. Dừng EOS hoặc hết 32 token mới. PAD và BOS bị chặn trước argmax; đây là ràng buộc cấu trúc rõ ràng, không phải argmax hoàn toàn không ràng buộc. Các ID byte/merge khác vẫn có thể tạo UTF-8 sai; xem valid_utf8 trong report.

*(Giải thích từng bước của chu trình Inference (Suy luận sinh văn bản):
1. **Mã hóa Prompt:** Người dùng truyền vào chuỗi `"Dịch vụ api: "`. Tokenizer mã hóa thành dãy token ID và thêm `BOS` vào đầu chuỗi.
2. **Forward Pass:** Đưa dãy ID qua mô hình Transformer (ở chế độ `eval` và tắt tính đạo hàm `torch.no_grad()`).
3. **Lấy Logits cuối cùng:** Mô hình tính toán và trả về ma trận logits, ta chỉ trích xuất hàng cuối cùng ứng với vị trí token mới nhất (vector độ dài 323).
4. **Áp đặt ràng buộc cấu trúc (Structural Constraints Masking):** Gán `logits[PAD] = -inf` và `logits[BOS] = -inf`. Vì sao? Vì trong quá trình sinh tiếp văn bản, mô hình tuyệt đối không được phép sinh ra token đệm `PAD` hoặc token mồi `BOS` ở giữa câu!
5. **Chọn Token (Greedy Search):** Dùng hàm `argmax()` để chọn token có điểm logit cao nhất.
6. **Kiểm tra dừng:** Nếu token vừa chọn là `EOS`, lập tức kết thúc vòng lặp sinh. Nếu chưa, nối token mới vào mảng ID và tiếp tục quay lại bước 2.
7. **Giải mã (Decode):** Ráp nối các mảng byte của tất cả token đã sinh và giải mã thành chuỗi ký tự UTF-8 cho người dùng).*

Không có KV cache, temperature, top-p hoặc beam search trong lab; bài 21 dự kiến đi sâu decoding. Greedy không phải cách tối đa hóa xác suất cả chuỗi trong mọi trường hợp. Dự đoán token đầu có xác suất cao nhất có thể dẫn tới phần tiếp diễn yếu hơn một nhánh đầu ít xác suất hơn. *(Trực giác về nhược điểm của Greedy Search: Thuật toán tham lam (greedy) chỉ chọn từ có xác suất cao nhất tại thời điểm tức thời ("ăn xổi ở thì"). Tuy nhiên, đôi khi chọn một từ có xác suất thấp hơn một chút ở hiện tại lại mở ra một câu văn cực kỳ trôi chảy, thông minh ở các bước tiếp theo. Đó là lý do các hệ thống LLM thương mại thường dùng Temperature hoặc Top-p Sampling thay vì Greedy thuần túy).*

Giới hạn context kiểm theo token input trước từng bước: với prompt gồm p token nội dung, có BOS nên input đầu dài p+1; để thực hiện m bước sinh cần input dài nhất p+m. Output token cuối chưa cần chạy lại forward. Chính sách ở lab là báo lỗi khi budget vượt, không tự cắt prompt và mất thông tin âm thầm.

## 14. Chạy và tự kiểm

Đã chạy Python 3.14.6, torch 2.15.0 trên macOS arm64 CPU một thread; NumPy 2.5.4 và Matplotlib 3.11.2 dùng vẽ hình. Không cần GPU hay tài khoản. Từ thư mục bài 19:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab.py --mode tokenizer
.venv/bin/python lab.py --mode hand
.venv/bin/python lab.py --mode all --out /tmp/lesson19-run-01
```

`--mode train` chỉ train, `--epochs 2` kiểm pipeline ngắn. Output phải mới; bỏ --out sẽ in đường dẫn tạm. Windows dùng `.venv\Scripts\python`; phiên bản ghim cần wheel phù hợp nền tảng. Seed không đảm bảo giống từng bit khi đổi môi trường.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lab import train, Config, load, generate
with TemporaryDirectory() as folder:
    run = Path(folder) / 'run'
    report = train(run, Config(epochs=2))
    model, tok, cfg, checkpoint = load(run)
    print(generate(model, tok, 'Dịch vụ api: ', max_new=16))
    assert tok.decode(tok.encode('GET /orders\n  504 🙂')) == 'GET /orders\n  504 🙂'
```

Sáu hình có sẵn PNG/SVG; tái tạo bằng [make_figures.py](make_figures.py):

```bash
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python make_figures.py --run /tmp/lesson19-run-01
```

## 15. Tiêu chí hoàn thành và học tiếp

Làm [16 bài tập](exercises.md) trước khi xem [lời giải](solutions.md). Bạn cần tự giải thích được: vì sao token không bằng từ; tại sao space và Unicode phải được giữ theo hợp đồng; vì sao tokenizer cùng size chưa chắc tương thích; vì sao EOS cần học nhưng PAD cần bỏ loss; tại sao PPL thấp chưa chứng minh câu sinh đúng; vì sao chia dữ liệu phải trước fit tokenizer.

[Phụ lục](deep-dive.md) nối các ý này với thiết kế backend, perplexity và kiểm artifact. **Bài tiếp theo:** [Bài 20 — Tiền huấn luyện và tinh chỉnh](../20-tien-huan-luyen-tinh-chinh/README.md), nối LM với SFT, loss mask, transfer và kiểm regression. [Nguồn đã đọc](../SOURCES.md).
