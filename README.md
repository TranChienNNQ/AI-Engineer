# Agent nghiên cứu tự điều phối — phiên bản 2.1

## Bộ tài liệu đang biên soạn

[AI Engineer 2026 — từ nền tảng đến hệ thống ứng dụng](courses/ai-engineer-2026/README.md): dành cho người có nền backend, học Deep Learning từ cơ bản. Có đề cương toàn bộ và bài 1–28 kèm thực hành, bài tập, lời giải; bài 2–28 có hình minh họa. Các bài chưa viết được ghi rõ trong đề cương.

Bộ điều phối đã bổ sung [quy trình biên soạn tài liệu](.agents/skills/learn-domain/references/workflows/course-authoring.md). Khi đã chọn học theo tài liệu, yêu cầu “viết bài tiếp theo” sẽ cập nhật bộ tài liệu và mục lục.

Bạn chỉ cần nói điều muốn học. Agent tự hiểu mục tiêu, chọn kỹ năng nghiên cứu phù hợp, kết hợp chúng và trả kết quả. Bạn không cần nhớ tên skill hoặc dùng lệnh gọi skill.

## Cập nhật 2.1

Agent chuyển sang nguồn phù hợp khác sớm hơn khi gặp trang bị chặn hoặc không có nội dung. Nó không lặp cùng URL bị từ chối; trong một nhóm truy cập thất bại, chỉ thử thêm tối đa một cách có căn cứ trước khi dừng. Nếu bạn yêu cầu đúng một tài liệu cụ thể, agent nói rõ giới hạn thay vì tự thay bằng nguồn khác.

Nếu đã cài v2, cập nhật hai tệp trong `.agents/skills/learn-domain/`: `SKILL.md` và `references/workflows/source-discovery.md`, rồi mở chat mới. Bạn cũng có thể thay toàn bộ thư mục `learn-domain` bằng bản 2.1 nếu không có chỉnh sửa riêng. `AGENTS.md` dùng cùng cấu hình với v2.

Đã kiểm thử bốn tình huống, gồm công cụ truy xuất mô phỏng có log tự động và web thật. Xem [báo cáo kiểm thử 2.1](TEST-REPORT.md) để biết kết quả và giới hạn.

## Cách nói chuyện

Ví dụ bắt đầu:

> Tôi muốn hiểu về ngành bán dẫn từ đầu. Giải thích cho tôi bức tranh chung rồi hướng dẫn học từng phần.

Sau đó bạn có thể nói:

> Phần này khó hiểu quá, cho tôi ví dụ gần gũi hơn.

> Hai tài liệu này nói khác nhau. Cái nào có cơ sở hơn?

> Tôi có 20 phút mỗi ngày. Giúp tôi sắp xếp cách học.

> Lưu lại những gì đã học và câu hỏi tôi còn vướng.

> Tiếp tục từ chỗ hôm trước.

Agent tự chọn cách làm cho từng yêu cầu. Chỉ hỏi lại khi thiếu thông tin thực sự quan trọng; không yêu cầu bạn quyết định kỹ thuật nghiên cứu hay thứ tự gọi skill.

## Import một lần

1. Giải nén gói. Chép `.agents/` và `AGENTS.md` vào gốc dự án dùng cho agent này. Hai mục phải nằm cạnh nhau. Trên macOS, nhấn `Command + Shift + .` để hiện thư mục ẩn nếu cần.
2. Nếu dự án đã có `AGENTS.md`, tích hợp hướng dẫn agent vào tệp đó, giữ các quy tắc không liên quan. Phần chỉ dẫn tự điều phối trong tệp này cần được giữ để agent mặc định tiếp nhận yêu cầu tự nhiên.
3. Mở dự án đó trong Codex và bắt đầu chat mới. Nhập yêu cầu như ví dụ trên. Nếu skill mới chưa được nhận diện, khởi động lại Codex.

Nếu **đã import bản 1**: thay thư mục `.agents/skills/learn-domain/` bằng bản trong gói này và cập nhật `AGENTS.md`. Tám skill rời do bản 1 cung cấp không còn được bản 2 sử dụng; bạn có thể chuyển chúng ra ngoài thư mục skill sau khi kiểm tra không có chỉnh sửa riêng. Tên của chúng: `research-brief`, `domain-map`, `source-discovery`, `evidence-check`, `research-synthesis`, `learning-path`, `concept-tutor`, `research-notes`. Không xóa các skill khác của bạn. Nếu chưa import bản 1, bỏ qua bước chuyển đổi này.

Codex hỗ trợ tự chọn skill theo mô tả và đọc hướng dẫn dự án từ `AGENTS.md`; xem [cơ chế skill chính thức](https://learn.chatgpt.com/docs/build-skills) và [hướng dẫn dự án](https://learn.chatgpt.com/docs/agent-configuration/agents-md). Bản 2 sử dụng cả hai: hướng dẫn dự án định tuyến yêu cầu phù hợp đến bộ điều phối, bộ điều phối đọc quy trình nội bộ cần thiết.

## Agent tự điều phối như thế nào?

```text
Bạn mô tả nhu cầu bằng lời bình thường
                   ↓
Agent hiểu yêu cầu cùng ngữ cảnh đang học
                   ↓
Tự chọn một hoặc vài kỹ năng nội bộ cần thiết
                   ↓
Thực hiện → đối chiếu kết quả → điều chỉnh nếu cần
                   ↓
Trả lời đúng nhu cầu và tiếp tục cùng bạn
```

| Bạn muốn | Agent tự thực hiện |
|---|---|
| Học một lĩnh vực mới | Xác định phạm vi, lập bản đồ, tìm nguồn, giải thích bài đầu |
| Hiểu một khái niệm | Giải thích trực tiếp, đưa ví dụ và sửa hiểu nhầm |
| Kiểm tra một nhận định | Truy nguồn, đánh giá bằng chứng và giới hạn |
| So sánh hai phương pháp | Chọn tiêu chí, đối chiếu nguồn, giải thích khác biệt |
| Tổng hợp tài liệu | Đọc nội dung, kết nối ý theo câu hỏi của bạn |
| Học có lịch trình | Chia chặng, hoạt động và tiêu chí tiến bộ phù hợp thời gian |
| Lưu hoặc tiếp tục | Đọc/lưu hồ sơ khi cần, xác định điểm dừng và bước học tiếp |

Đây là các quyết định nội bộ, không phải menu bạn phải chọn. Không có yêu cầu gọi đủ tất cả kỹ năng cho mọi câu hỏi.

## Bên trong gói

```text
AGENTS.md                              Hướng dẫn vai trò và tự điều phối
.agents/skills/learn-domain/
  SKILL.md                             Một đầu vào được Codex nhận diện
  agents/openai.yaml                   Cho phép tự kích hoạt theo yêu cầu
  references/
    workflows/                         Các quy trình nội bộ, đọc khi cần
    evidence-ledger.md                  Mẫu sổ bằng chứng
    topic-note.md                       Mẫu hồ sơ học
    quality-rubric.md                   Tiêu chí đánh giá chất lượng
```

Các quy trình được đóng gói thành tài liệu nội bộ, không đăng ký thành các skill riêng trong danh sách của Codex. Tên kỹ thuật `learn-domain` được giữ để chuyển đổi từ bản 1; bạn không cần dùng tên này khi trò chuyện.

Đây là cấu hình cho agent trong Codex, không phải một ứng dụng chạy nền độc lập. Khả năng tìm web, đọc tệp và lưu hồ sơ phụ thuộc công cụ và quyền của môi trường. Agent phải nói rõ khi thiếu công cụ, không giả vờ đã research hoặc đã lưu.

## Kiểm tra sau import

Thử một yêu cầu tự nhiên rồi một câu tiếp nối, chẳng hạn “Tôi muốn hiểu UX research từ đầu” và “Cho ví dụ dễ hiểu hơn”. Agent cần tự chọn cách làm, giữ đúng chủ đề, không đưa menu skill và không yêu cầu bạn gõ lệnh.

Xem [các tình huống thử](EVALUATION.md) để đánh giá sâu hơn. Việc kiểm tra định dạng và liên kết tệp không thay thế kiểm thử hành vi trong chat mới.
