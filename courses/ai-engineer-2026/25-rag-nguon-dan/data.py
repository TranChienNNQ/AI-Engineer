"""Synthetic policies/runbooks and reference annotations, never production facts."""
import hashlib
import unicodedata

# id, version, title, scope, approval, from (inclusive), to (exclusive), paragraphs.
SPECS = [
('P01','1','Policy idempotency','public','approved','2026-01-01','2026-09-01',[
 ('Thời hạn',[('key_24','Theo policy phiên bản 1, idempotency key được lưu 24 giờ.'),
               ('payload','Cùng idempotency key nhưng khác payload phải bị từ chối.')])]),
('P01','2','Policy idempotency','public','approved','2026-09-01',None,[
 ('Thời hạn',[('key_72','Theo policy phiên bản 2, idempotency key được lưu 72 giờ.'),
               ('payload','Cùng idempotency key nhưng khác payload phải bị từ chối.')])]),
('P02','1','Retry upstream','public','approved','2026-01-01',None,[
 ('Điều kiện',[('retry_temp','Chỉ retry lỗi upstream tạm thời khi thao tác đáp ứng điều kiện idempotent.'),
              ('no_permission','Không retry lỗi thiếu quyền bằng cách gọi lại liên tục.')]),
 ('Budget',[('deadline','Mọi attempt retry phải chia sẻ cùng một tổng deadline.'),
            ('backoff','Dùng exponential backoff kèm jitter để giãn các lần retry.')])]),
('P03','1','Xác thực API','public','approved','2026-01-01',None,[
 ('HTTP 401',[('refresh','Dùng refresh token hợp lệ để cấp access token mới khi access token hết hạn.'),
              ('no_log','Không ghi giá trị token bí mật vào log.')]),
 ('HTTP 403',[('scope','Kiểm scope được cấp cho người dùng trước khi cho đọc tài nguyên bị lỗi HTTP 403.'),
              ('no_grant','Refresh token không tự bổ sung quyền còn thiếu.')])]),
('P04','1','Cache hồ sơ','public','approved','2026-01-01',None,[
 ('TTL',[('cache_ttl','Cache hồ sơ của dịch vụ demo có TTL 300 giây.'),
         ('ttl_limit','TTL không thay thế mọi thao tác cache invalidation.')]),
 ('Invalidation',[('invalidate','Invalidate cache key liên quan sau khi cập nhật dữ liệu gốc.'),
                  ('write_order','Phối hợp thứ tự ghi để tránh nạp lại bản cache cũ.')])]),
('P05','1','Consumer queue','public','approved','2026-01-01',None,[
 ('Thông điệp trùng',[('message_id','Dùng message ID để bỏ qua tác động nghiệp vụ đã xử lý trước đó.'),
                      ('delivery','Consumer có thể nhận lại cùng một thông điệp nhiều lần.')]),
 ('Dead letter',[('dlq','Chuyển message sang dead letter queue sau giới hạn retry đã đặt.')])]),
('P06','1','Webhook','public','approved','2026-01-01',None,[
 ('Chữ ký',[('raw_body','Kiểm signature trên raw body trước khi parse JSON.'),
            ('json_bytes','Parse rồi serialize JSON có thể thay đổi byte được ký.')]),
 ('Replay',[('replay','Kiểm timestamp trong cửa sổ cho phép và chống lặp bằng event ID.')])]),
('P07','1','Upload runbook A','public','approved','2026-10-01',None,[
 ('Giới hạn',[('upload_20','Theo runbook A, upload của dịch vụ demo có giới hạn 20 MB.')])]),
('P08','1','Upload runbook B','public','approved','2026-10-01',None,[
 ('Giới hạn',[('upload_25','Theo runbook B, upload của dịch vụ demo có giới hạn 25 MB.')])]),
('P09','1','Đề xuất chưa duyệt','public','draft','2026-10-01',None,[
 ('Đề xuất',[('draft','Đề xuất lưu idempotency key vô hạn chưa được phê duyệt.')])]),
('P10','1','Runbook nội bộ blue','team-blue','approved','2026-01-01',None,[
 ('Đối soát',[('secret','Mã đối soát nội bộ của nhóm blue là BLUE-DEMO-917.')])]),
]


def dataset():
    sources=[];facts={}
    for sid,version,title,scope,approval,since,until,paragraphs in SPECS:
        text='';regions=[]
        for heading,entries in paragraphs:
            if text:text+='\n\n'
            start=len(text)
            for j,(fid,sentence) in enumerate(entries):
                sentence=unicodedata.normalize('NFC',sentence)
                if j:text+=' '
                a=len(text);text+=sentence
                fact=facts.setdefault(fid,dict(text=sentence,spans=[]))
                assert fact['text']==sentence
                fact['spans'].append(dict(source_id=sid,version=version,start=a,end=len(text)))
            regions.append(dict(title=heading,start=start,end=len(text)))
        sources.append(dict(source_id=sid,version=version,title=title,scope=scope,approval=approval,
                            valid_from=since,valid_to=until,text=text,regions=regions,
                            source_hash=hashlib.sha256(text.encode()).hexdigest()))
    queries=[
      ('D01','dev','idempotency key lưu bao nhiêu giờ theo policy phiên bản 2',['key_72'],'answer','2026-11-04'),
      ('D02','dev','webhook kiểm signature trên raw body hay JSON',['raw_body'],'answer','2026-11-04'),
      ('D03','dev','lỗi HTTP 403 cần kiểm scope quyền đọc tài nguyên',['scope'],'answer','2026-11-04'),
      ('D04','dev','TTL cache hồ sơ bao lâu và invalidate cache sau cập nhật thế nào',['cache_ttl','invalidate'],'answer','2026-11-04'),
      ('T01','test','hiện tại idempotency key được lưu bao nhiêu giờ',['key_72'],'answer','2026-11-04'),
      ('T02','test','ngày 2026-06-01 policy idempotency key lưu bao nhiêu giờ',['key_24'],'answer','2026-06-01'),
      ('T03','test','idempotency key lưu bao nhiêu giờ và cùng key khác payload xử lý thế nào',['key_72','payload'],'answer','2026-11-04'),
      ('T04','test','retry lỗi upstream tạm thời khi nào và có retry lỗi thiếu quyền không',['retry_temp','no_permission'],'answer','2026-11-04'),
      ('T05','test','access token hết hạn cần làm gì và HTTP 403 cần kiểm scope nào',['refresh','scope'],'answer','2026-11-04'),
      ('T06','test','cập nhật dữ liệu gốc rồi invalidate cache key thế nào',['invalidate'],'answer','2026-11-04'),
      ('T07','test','consumer nhận thông điệp trùng dùng message ID để làm gì',['message_id'],'answer','2026-11-04'),
      ('T08','test','webhook kiểm signature raw body và chống replay bằng timestamp thế nào',['raw_body','replay'],'answer','2026-11-04'),
      ('T09','test','upload dịch vụ demo có giới hạn bao nhiêu MB theo runbook',['upload_20','upload_25'],'conflict','2026-11-04'),
      ('T10','test','idempotency key lưu bao nhiêu giờ và các attempt retry chia sẻ deadline nào',['key_72','deadline'],'answer','2026-11-04'),
      ('T11','test','zzquasar999 nebula999',[],'abstain','2026-11-04'),
      ('T12','test','TTL cache trên sao Hỏa',[],'abstain','2026-11-04'),
    ]
    queries=[dict(id=i,split=s,text=t,required=r,expected_status=status,as_of=date) for i,s,t,r,status,date in queries]
    return sources,facts,queries


def claim_catalog(facts):
    # Closed reference annotations; NOT a general natural-language entailment judge.
    catalog={f['text']:dict(support_sets=[{fid}],entails={fid}) for fid,f in facts.items()}
    catalog['Idempotency key được lưu 3 ngày theo policy phiên bản 2.']=dict(support_sets=[{'key_72'}],entails={'key_72'})
    catalog['Key được lưu 72 giờ và mọi attempt retry chia sẻ cùng tổng deadline.']=dict(support_sets=[{'key_72','deadline'}],entails={'key_72','deadline'})
    catalog['Mọi lỗi đều có thể retry vô hạn.']=dict(support_sets=[],entails=set())
    return catalog
