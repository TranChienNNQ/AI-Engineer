"""Original synthetic runbooks. Evidence spans refer to canonical NFC source text."""
import unicodedata

# source, source title, section, paragraph, canonical evidence, 2 train queries, dev, test.
TASKS = [
('S01','Thanh toán','Idempotency',
 'Client có thể gửi lại yêu cầu tạo đơn vì mất phản hồi. Idempotency key gắn với payload giúp nhận diện thao tác cũ. Lưu kết quả theo khóa để không tạo thêm đơn khi gửi lại. Cùng khóa nhưng khác payload phải báo xung đột.',
 'Lưu kết quả theo khóa để không tạo thêm đơn khi gửi lại',
 'gửi lại tạo đơn dùng idempotency key thế nào','mất phản hồi tạo đơn tránh tạo đơn trùng bằng khóa',
 'client retry tạo đơn mà không muốn sinh đơn thứ hai','tạo đơn xong mất response rồi gửi lại phải làm gì'),
('S01','Thanh toán','Refund',
 'ERR_BILL_042 xảy ra khi yêu cầu refund không tham chiếu được giao dịch gốc. Mã này khác ERR_BILL_043 của thao tác capture. Kiểm payment reference của giao dịch gốc trước khi hoàn tiền. Không tạo capture mới để thay refund bị từ chối.',
 'Kiểm payment reference của giao dịch gốc trước khi hoàn tiền',
 'ERR_BILL_042 refund cần kiểm tra gì','hoàn tiền không tìm được giao dịch gốc payment reference',
 'refund bị từ chối mã ERR_BILL_042 xử lý ra sao','ERR_BILL_042 khác capture cần xem payment reference nào'),
('S02','Xác thực','HTTP 401',
 'Access token có thời hạn để giới hạn phiên đăng nhập. HTTP 401 có thể xuất hiện khi token hết hạn. Dùng refresh token hợp lệ để cấp access token mới. Không ghi giá trị token bí mật vào log hoặc gửi cho người khác.',
 'Dùng refresh token hợp lệ để cấp access token mới',
 'HTTP 401 access token hết hạn phải refresh thế nào','token đăng nhập hết hạn cấp access token mới',
 'đăng nhập gặp 401 vì access token quá hạn','access token hết hạn rồi gọi API bị 401 cần làm gì'),
('S02','Xác thực','HTTP 403',
 'Người dùng có thể đăng nhập thành công nhưng chưa được cấp mọi quyền. HTTP 403 ở ví dụ này là thiếu permission của tài nguyên. Kiểm scope được cấp cho người dùng trước khi cho đọc tài nguyên. Refresh token không tự bổ sung quyền còn thiếu.',
 'Kiểm scope được cấp cho người dùng trước khi cho đọc tài nguyên',
 'HTTP 403 thiếu scope cần kiểm quyền nào','đã xác thực nhưng thiếu permission tài nguyên',
 '403 dù token hợp lệ thì refresh có sửa quyền không','người dùng đã login nhưng không có quyền xem tài nguyên'),
('S03','Bộ nhớ đệm','TTL',
 'Cache Redis giúp giảm số lượt đọc dữ liệu gốc. Mỗi cache key có thể mang thời gian sống TTL. Chọn TTL theo độ mới dữ liệu mà ứng dụng chấp nhận. TTL ngắn làm tăng miss và tải đọc; TTL dài có thể giữ giá trị cũ lâu hơn.',
 'Chọn TTL theo độ mới dữ liệu mà ứng dụng chấp nhận',
 'Redis cache chọn TTL theo độ mới thế nào','TTL dài hay ngắn ảnh hưởng cache miss',
 'đặt thời gian sống cache dựa vào điều kiện gì','cache Redis muốn dữ liệu mới hơn thì chọn TTL ra sao'),
('S03','Bộ nhớ đệm','Invalidation',
 'Sau khi cập nhật hồ sơ, màn hình có thể tiếp tục đọc bản cache cũ. Chờ hết TTL không đáp ứng mọi yêu cầu nhất quán. Invalidate cache key liên quan sau khi cập nhật dữ liệu gốc. Cần phối hợp thứ tự ghi để tránh nạp lại bản cũ.',
 'Invalidate cache key liên quan sau khi cập nhật dữ liệu gốc',
 'cập nhật hồ sơ nhưng cache cũ phải invalidate key','sửa dữ liệu gốc rồi xóa cache liên quan',
 'hồ sơ đã sửa mà màn hình vẫn lấy cache cũ','sau update dữ liệu cần vô hiệu hóa cache key nào'),
('S04','Cơ sở dữ liệu','Connection pool',
 'Database pool giới hạn số connection được sử dụng đồng thời. Request chờ lấy kết nối khi không còn slot trống. Trả connection về pool sau khi kết thúc request kể cả khi có lỗi. Theo dõi active connection và thời gian chờ trước khi tăng pool.',
 'Trả connection về pool sau khi kết thúc request kể cả khi có lỗi',
 'connection pool hết slot vì không trả kết nối','request lỗi vẫn cần đóng trả database connection',
 'pool cạn vì request giữ connection quá lâu','kết thúc request bị exception có cần trả connection về pool'),
('S04','Cơ sở dữ liệu','Deadlock',
 'Deadlock xảy ra khi các transaction giữ khóa và chờ nhau thành vòng. Tăng connection pool không giải quyết thứ tự khóa. Giữ thứ tự lấy khóa nhất quán và retry toàn transaction khi phù hợp. Rút ngắn giao dịch để giảm thời gian giữ khóa.',
 'Giữ thứ tự lấy khóa nhất quán và retry toàn transaction khi phù hợp',
 'deadlock transaction chờ khóa vòng xử lý thế nào','retry toàn transaction và thống nhất thứ tự khóa',
 'hai giao dịch giữ khóa lẫn nhau thành deadlock','SQL deadlock nên thống nhất thứ tự khóa hay tăng pool'),
('S05','Giới hạn gọi','HTTP 429',
 'HTTP 429 báo client đã vượt giới hạn request của dịch vụ. Retry tức thì có thể làm tình trạng nặng hơn. Tôn trọng Retry-After và giảm tốc độ gọi khi gặp HTTP 429. Nếu tiếp tục thử phải giữ tổng deadline và số lần thử hữu hạn.',
 'Tôn trọng Retry-After và giảm tốc độ gọi khi gặp HTTP 429',
 'HTTP 429 Retry-After cần giảm tốc độ gọi','rate limit quá nhiều request không retry dồn',
 'dịch vụ trả 429 kèm Retry-After thì chờ thế nào','vượt hạn mức gọi API HTTP 429 cần tôn trọng header nào'),
('S05','Giới hạn gọi','Token bucket',
 'Token bucket lưu ngân sách cho phép request mới đi qua. Token được nạp lại theo thời gian để giới hạn tốc độ trung bình. Khi bucket cạn phải chờ nạp token trước khi nhận thêm request. Token ở đây không phải access token dùng để xác thực.',
 'Khi bucket cạn phải chờ nạp token trước khi nhận thêm request',
 'token bucket cạn ngân sách request thì làm gì','nạp token theo thời gian để giới hạn tốc độ',
 'bucket hết token có nên refresh access token không','ngân sách token bucket hết thì request tiếp theo phải chờ gì'),
('S06','Webhook','Chữ ký',
 'Webhook gửi thông báo với chữ ký nhằm kiểm tra tính toàn vẹn của payload. Parse rồi serialize JSON có thể thay đổi byte gốc. Kiểm signature trên raw body trước khi parse JSON. Chọn secret đúng phiên bản và không log secret trong trường hợp lỗi.',
 'Kiểm signature trên raw body trước khi parse JSON',
 'webhook signature cần raw body trước parse JSON','JSON serialize lại làm sai chữ ký webhook',
 'xác minh webhook thất bại sau khi parse JSON','đọc raw body hay JSON serialize lại để kiểm chữ ký webhook'),
('S06','Webhook','Replay',
 'Một thông báo có chữ ký hợp lệ vẫn có thể bị gửi lại nhiều lần. Chữ ký không tự bảo đảm sự kiện mới. Kiểm timestamp trong cửa sổ cho phép và chống lặp bằng event ID. Chính sách lưu ID cần phù hợp thời gian replay mà hệ thống chấp nhận.',
 'Kiểm timestamp trong cửa sổ cho phép và chống lặp bằng event ID',
 'webhook replay chữ ký đúng vẫn phải kiểm timestamp','chặn thông báo lặp bằng event ID và cửa sổ thời gian',
 'webhook ký hợp lệ nhưng bị phát lại sự kiện cũ','chống replay webhook cần timestamp và định danh sự kiện nào'),
('S07','Hàng đợi','Thông điệp trùng',
 'Consumer có thể nhận lại một message khi quá trình acknowledge bị gián đoạn. Queue giao nhiều lần không đồng nghĩa nghiệp vụ được phép chạy nhiều lần. Dùng message ID để bỏ qua tác động đã xử lý trước đó. Lưu dấu xử lý phải phối hợp với cập nhật nghiệp vụ.',
 'Dùng message ID để bỏ qua tác động đã xử lý trước đó',
 'consumer nhận message trùng dùng message ID thế nào','queue giao lại thông điệp tránh cập nhật nghiệp vụ lần hai',
 'consumer thấy cùng message ID được giao hai lần','ack thất bại khiến queue gửi lại cần tránh tác động trùng'),
('S07','Hàng đợi','Dead letter',
 'Một message có payload sai có thể thất bại lặp đi lặp lại. Retry vô hạn làm tốn worker và cản công việc mới. Chuyển message sang dead letter queue sau giới hạn retry đã đặt. Ghi nguyên nhân để điều tra và chỉ replay sau khi sửa vấn đề.',
 'Chuyển message sang dead letter queue sau giới hạn retry đã đặt',
 'message lỗi lặp nhiều lần nên vào dead letter queue','queue không retry vô hạn khi payload sai',
 'worker cứ xử lý lại message hỏng mà không tiến triển','message vượt số lần retry cần chuyển sang queue nào'),
('S08','Search index','Đổi model',
 'Vector trong index được tạo bằng một embedding model và quy ước tiền xử lý cụ thể. Hai model cùng số chiều chưa chắc cùng hệ tọa độ. Encode lại chunk khi thay model rồi triển khai index cùng phiên bản query encoder. Không ghép query vector mới với document vector cũ.',
 'Encode lại chunk khi thay model rồi triển khai index cùng phiên bản query encoder',
 'đổi embedding model cần encode lại document vector','query encoder mới không trộn vector cũ dù cùng dimension',
 'model embedding mới vẫn 768 chiều có dùng index cũ không','thay model tạo vector cần rebuild gì để query tương thích'),
('S08','Search index','Scope',
 'Ứng viên truy xuất có thể chứa tài liệu thuộc nhiều scope. Lọc response cuối quá muộn nếu văn bản riêng đã gửi sang model. Lọc quyền trước khi đưa ứng viên sang reranker hoặc bước tạo câu trả lời. Quyền phải được suy ra từ danh tính do server xác thực.',
 'Lọc quyền trước khi đưa ứng viên sang reranker hoặc bước tạo câu trả lời',
 'lọc scope trước khi gửi chunk cho reranker','tài liệu riêng không được đưa sang model rồi mới lọc',
 'lọc output cuối có đủ để bảo vệ tài liệu riêng tư không','quyền truy cập chunk phải kiểm trước hay sau reranking'),
('S09','Gọi upstream','Deadline',
 'Một request có nhiều attempt chia sẻ cùng thời hạn phản hồi. Timeout từng attempt không thay thế giới hạn tổng thời gian. Trừ thời gian đã dùng khỏi tổng deadline trước mỗi lần retry. Dừng khi budget còn lại không đủ cho lần thử có ích.',
 'Trừ thời gian đã dùng khỏi tổng deadline trước mỗi lần retry',
 'retry nhiều attempt cần trừ tổng deadline còn lại','timeout mỗi lần không là tổng budget request',
 'mỗi lần thử có timeout nhưng tổng request vẫn kéo dài','deadline còn ít sau attempt lỗi có nên retry tiếp không'),
('S09','Gọi upstream','Backoff',
 'Lỗi upstream tạm thời có thể được xử lý bằng một số lần thử lại. Nhiều client retry đồng thời sẽ làm tải dồn thành đợt. Dùng exponential backoff kèm jitter để giãn các lần retry. Chỉ retry khi điều kiện nghiệp vụ và tính idempotent cho phép.',
 'Dùng exponential backoff kèm jitter để giãn các lần retry',
 'retry upstream đồng loạt cần backoff và jitter','exponential backoff tránh dồn tải khi gọi lại',
 'nhiều client cùng thử lại làm upstream quá tải theo đợt','giãn retry bằng backoff có nên thêm jitter không'),
]


def dataset():
    sources={}
    queries=[]
    for i,(sid,title,heading,text,quote,t1,t2,dev,test) in enumerate(TASKS,1):
        source=sources.setdefault(sid,dict(id=sid,title=title,version='1',scope='public',text='',sections=[]))
        text=unicodedata.normalize('NFC',text)
        if source['text']:source['text']+='\n\n'
        start=len(source['text']);source['text']+=text
        section_id=f'{sid}.{len(source["sections"])+1}'
        source['sections'].append(dict(id=section_id,title=heading,start=start,end=len(source['text'])))
        assert text.count(quote)==1,(section_id,quote)
        a=start+text.index(quote);evidence=dict(source_id=sid,start=a,end=a+len(quote),quote=quote)
        for split,qs in [('train',[t1,t2]),('dev',[dev]),('test',[test])]:
            for j,q in enumerate(qs,1):
                queries.append(dict(id=f'{split}-{i:02}-{j}',text=q,split=split,section_id=section_id,evidence=evidence.copy()))
    # A separate restricted source never enters the default public index or training vocabulary.
    text='ERR_BILL_042 payment reference secret riêng của nhóm blue. Không phát hành tài liệu nội bộ này ra public.'
    sources['S10']=dict(id='S10',title='Nội bộ blue',version='1',scope='team-blue',text=text,
                        sections=[dict(id='S10.1',title='Runbook riêng',start=0,end=len(text))])
    for i,text in enumerate(['zzquasar999 nebula999','cache trên sao Hỏa'],1):
        queries.append(dict(id=f'test-none-{i}',text=text,split='test',section_id=None,evidence=None))
    return list(sources.values()),queries
