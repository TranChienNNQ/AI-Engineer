"""Original synthetic backend documents and graded query judgments. No real incidents."""
DOCS = [
 ('D01','Retry khi gọi upstream','Retry request timeout do lỗi tạm thời. Dùng exponential backoff và jitter, giới hạn số lần thử và deadline. Không retry lỗi quyền truy cập.'),
 ('D02','Idempotency cho tạo đơn','Idempotency key ngăn tạo đơn trùng khi request được gửi lại. Lưu kết quả theo khóa và payload; retry không bảo đảm hành động chỉ xảy ra một lần.'),
 ('D03','Access token hết hạn','HTTP 401 khi access token hết hạn. Xác thực người dùng bằng refresh token hợp lệ rồi cấp access token mới. Không ghi token bí mật vào log.'),
 ('D04','Scope và quyền truy cập','HTTP 403 khi người dùng đã xác thực nhưng thiếu scope hoặc permission. Kiểm tra quyền truy cập tài nguyên, không lặp refresh token như cách sửa mọi lỗi.'),
 ('D05','Cạn connection pool','Database connection pool hết slot khi kết nối không được trả lại. Theo dõi active connection, timeout lấy kết nối và đóng connection sau request.'),
 ('D06','Deadlock giao dịch SQL','Deadlock database xảy ra khi transaction chờ khóa lẫn nhau. Rút ngắn giao dịch, thống nhất thứ tự lấy khóa và retry toàn transaction khi phù hợp.'),
 ('D07','Cache invalidation sau cập nhật','Dữ liệu cache cũ sau khi cập nhật hồ sơ. Invalidate cache key hoặc cập nhật cache theo chính sách nhất quán. TTL dài có thể giữ thông tin cũ.'),
 ('D08','Redis TTL','Redis cache lưu giá trị với TTL để hết hạn tự động. Theo dõi cache hit miss và chọn thời gian sống theo độ mới dữ liệu; TTL không thay thế mọi invalidation.'),
 ('D09','ERR_BILL_042 hoàn tiền','ERR_BILL_042 nghĩa là refund bị từ chối do không tìm thấy giao dịch gốc. Kiểm payment reference của yêu cầu hoàn tiền, không tạo capture thay thế.'),
 ('D10','ERR_BILL_043 thu tiền','ERR_BILL_043 nghĩa là capture vượt số tiền đã authorize. Kiểm số tiền thu và trạng thái payment; không dùng quy trình refund của ERR_BILL_042.'),
 ('D11','ERR_DB_017 serialization','ERR_DB_017 là serialization failure của transaction. Retry toàn giao dịch với dữ liệu đọc lại; khác lỗi cạn connection pool.'),
 ('D12','HTTP 504 gateway','HTTP 504 khi gateway không nhận phản hồi upstream kịp thời. Kiểm timeout ở proxy, thời gian xử lý backend và deadline xuyên request.'),
 ('D13','HTTP 429 giới hạn gọi','HTTP 429 khi quá giới hạn request. Tôn trọng Retry-After, giảm tốc độ gọi và dùng backoff. Retry dồn dập có thể làm rate limit nặng hơn.'),
 ('D14','HTTP 503 quá tải','HTTP 503 khi dịch vụ tạm không sẵn sàng hoặc quá tải. Kiểm queue, tài nguyên worker và giảm tải; phân biệt lỗi quá tải với thiếu quyền.'),
 ('D15','Chữ ký webhook','Webhook signature xác minh nội dung thông báo bằng secret. Kiểm chữ ký trên raw body và timestamp; JSON parse rồi serialize lại có thể đổi byte ký.'),
 ('D16','Cập nhật search index','Search index cần đồng bộ với phiên bản tài liệu. Khi thay analyzer hoặc embedding model phải tính lại representation và rebuild index theo version.'),
 ('D17','Pagination ổn định','Pagination danh sách cần thứ tự ổn định theo thời gian và ID. Cursor hỗ trợ tiếp tục đọc; page size không là giới hạn context của mô hình ngôn ngữ.'),
 ('D18','Runbook nội bộ nhóm blue','ERR_BILL_042 refund payment reference secret nội bộ nhóm blue. Bản tài liệu này chỉ được chọn khi caller được cấp scope team-blue.'),
 ('D19','Token bucket','Token bucket giới hạn tốc độ request bằng ngân sách token nạp theo thời gian. Token bucket cho rate limit khác access token dùng để xác thực người dùng.'),
 ('D20','Queue giao hàng lặp','Consumer queue có thể nhận cùng thông điệp nhiều lần. Xử lý idempotent bằng message ID để tránh cập nhật đơn trùng; theo dõi retry và dead letter queue.')
]

def documents():
    return [dict(id=i,title=t,text=s,scope='team-blue' if i=='D18' else 'public') for i,t,s in DOCS]

QUERIES = [
 ('Q01','dev','identifier','ERR_BILL_042',{'D09':3,'D10':1}),
 ('Q02','dev','lexical','HTTP 401 access token',{'D03':3}),
 ('Q03','dev','paraphrase','gửi lại yêu cầu không tạo đơn trùng',{'D02':3,'D20':2}),
 ('Q04','dev','lexical','Redis TTL cache',{'D08':3,'D07':1}),
 ('Q05','dev','paraphrase','database chờ khóa lẫn nhau',{'D06':3}),
 ('Q06','dev','lexical','webhook raw body chữ ký',{'D15':3}),
 ('Q07','test','identifier','ERR_DB_017',{'D11':3}),
 ('Q08','test','identifier','ERR_BILL_043 capture',{'D10':3}),
 ('Q09','test','lexical','HTTP 429 Retry-After',{'D13':3,'D19':1}),
 ('Q10','test','lexical','token bucket giới hạn tốc độ',{'D19':3,'D13':2}),
 ('Q11','test','paraphrase','đã xác thực nhưng không được phép đọc tài nguyên',{'D04':3}),
 ('Q12','test','paraphrase','hồ sơ mới rồi nhưng vẫn đọc thông tin cũ',{'D07':3,'D08':1}),
 ('Q13','test','paraphrase','kết nối không được trả lại làm hết slot',{'D05':3}),
 ('Q14','test','lexical','gateway upstream không phản hồi kịp thời',{'D12':3,'D01':1}),
 ('Q15','test','lexical','consumer thông điệp trùng message ID',{'D20':3,'D02':1}),
 ('Q16','test','lexical','đổi embedding model rebuild index',{'D16':3}),
 ('Q17','test','no_answer','thuật toán quang tử zzz42',{}),
 ('Q18','test','no_answer','cache trên sao Hỏa',{})
]

def queries():
    return [dict(id=i,split=s,slice=g,text=t,relevance=r) for i,s,g,t,r in QUERIES]
