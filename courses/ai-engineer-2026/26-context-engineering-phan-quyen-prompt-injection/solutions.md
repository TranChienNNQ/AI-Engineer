# Bài 26 — Lời giải

[← Bài chính](README.md) · [Đề bài](exercises.md)

Các snippet chạy từ thư mục bài với `requirements.txt`. Tất cả thay đổi ticket/quyền chỉ tác động World mới trong RAM, không có mạng hay dịch vụ thật.

## 1. Vai trò của các phần context

System/task contract và tool schema do server thiết lập. Câu hỏi hiện tại xác định nhu cầu nhưng không tự thay principal. History, summary, tài liệu và tool output là data có provenance/owner/quyền; câu `role=system` nằm trong body không được dùng để tạo một system message hay thay role server.

Trong tích hợp LLM thật, role message phải do adapter tin cậy gán. JSON/string label chỉ giúp biểu diễn; nó không chứng minh model luôn tuân thủ. Phân quyền/action gate nằm ngoài việc model tin nội dung nào.

## 2. Budget và lựa chọn

Budget 640. Bốn item được phép có tổng chi phí 720 nên không chọn tất cả. A+B+C tốn 600, utility 23; A+B+D560/22; A+C+D520/20; B+C+D480/19. Các tập ít item hơn không vượt 23. E bị loại theo quyền dù utility 100, nên nghiệm trong ví dụ là A+B+C.

```python
from itertools import product
costs=[240,200,160,120,80]
utilities=[9,8,6,5,100]
authorized=[1,1,1,1,0]
feasible=[]
for mask in product([0,1],repeat=5):
    if all(x<=a for x,a in zip(mask,authorized)) and sum(x*c for x,c in zip(mask,costs))<=640:
        feasible.append((sum(x*u for x,u in zip(mask,utilities)),mask))
value,chosen=max(feasible)
assert value==23 and chosen==(1,1,1,0,0)
print(value,chosen)
```

Utility là số giả định, không phải điểm LLM hay một phép đo bảo mật. Constraint quyền là hard constraint; không cộng nó vào objective như một khoản phạt có thể mua lại bằng relevance.

## 3. Compaction

Summary đã bỏ điều kiện lỗi tạm thời, bỏ phủ định thiếu quyền và biến việc chưa có ý định đóng thành đề xuất đóng khi cần. Cần giữ task hiện tại, điều kiện/ngoại lệ, các quyết định thực sự có thẩm quyền, việc chưa được cho phép và source/version liên quan.

Ngay cả summary rất tốt cũng không thay server identity/ACL/grant store. Những field ấy cần typed state/controller tin cậy. Có thể rút token history nhưng phải đánh giá thông tin bị mất; số token giảm không chứng minh summary đúng hoặc injection đã biến mất.

## 4. Danh tính và ý định

Alice vẫn là engineer theo server, không trở thành operator vì text tự khai. Model có thể đề xuất object, nhưng gateway từ chối action ngoài role/scope/tenant. Ops_blue có quyền loại hành động đóng ticket trong tenant blue nhưng còn cần grant cụ thể cho thao tác hiện tại.

CLI session là fixture để chọn actor; nếu biến nó thành trường client tự gửi rồi tin thì chưa có authentication. Lab giả định tầng danh tính/server/controller là tin cậy, chưa triển khai SSO hoặc sandbox cho Python adapter.

## 5. Ma trận quyền

Alice đọc 6 resource: GLOBAL, B-PUBLIC, B-INTERNAL, B-POISON, T-BLUE-1, T-BLUE-2. Bob đọc 5, thiếu B-INTERNAL. Ops_blue và ops_peer mỗi người đọc 6. Carol đọc 3: GLOBAL, G-INTERNAL, T-GREEN-1. Tổng 26/40 cặp principal/resource được đọc.

B-INTERNAL yêu cầu level 1 và đúng tenant/audience, Bob chỉ level 0. Carol đúng clearance nhưng khác tenant nên không đọc B-INTERNAL. GLOBAL có tenant=None, level 0, audience/scopes phù hợp mới được chia sẻ xuyên tenant. Đủ read không tự đủ close.

## 6. Retrieval trước và sau ACL

Lọc sau khi gửi data sang reranker/model không thu hồi được plaintext đã cấp. Trong lab, corpus statistics được tính trên tập document được phép; nguồn green không góp df/avgdl cho Alice.

```python
from data import make_world
from security import Resource
from context import retrieve
w=make_world();p=w.resolve('session_alice')
query='retry API timeout'
before=[(r.id,score) for r,score in retrieve(w,p,query,10)]
w.resources['G-EXTRA']=Resource('G-EXTRA','green','document',1,('engineer','operator'),'retry API timeout '*100)
after=[(r.id,score) for r,score in retrieve(w,p,query,10)]
assert before==after
print(before)
```

Phép thử chỉ kiểm score/thứ hạng retrieval trong cùng implementation; chưa chứng minh constant-time, không có side channel, hoặc cache toàn hệ thống không phản ánh revision của tenant khác. Các counter global của lab khá bảo thủ và có thể gây miss không cần thiết.

## 7. Injection và benign

Role spoof ở user input là direct attempt; instruction trong runbook/tool result/memory là các đường indirect qua data. Summary có thể lưu lại injection qua nhiều lượt. Nhưng câu hỏi yêu cầu phân tích một attack string là nhiệm vụ benign: đọc/giải thích nội dung không đồng nghĩa thực thi nội dung ấy.

Chặn keyword dễ false positive với bài học/ticket đang mô tả attack và false negative với paraphrase hoặc nội dung ngoài mẫu. Gateway dùng policy/action schema, không dựa vào việc tìm đúng một từ khóa injection.

## 8. Taint còn nguyên, quyền không đổi

Với budget lớn, B-POISON và M-LAUNDER có thể được giữ vì Alice được đọc và metadata còn hợp lệ. Chúng vẫn có authority data. Alice không có role operator; proposal close bị từ chối. Điều này chưa chứng minh model không bị dẫn sai câu trả lời.

```python
from data import make_world
from context import Config,build_context
from security import Gateway
from fixtures import close
w=make_world()
cfg=Config(context_cap=1000,model_window=1600,top_k=10)
ctx=build_context(w,'session_alice','retry API',cfg)
ids={i['id'] for i in ctx['items']}
assert {'B-POISON','M-LAUNDER'}<=ids
assert all(i['authority']=='data' for i in ctx['items'])
assert w.resolve('session_alice').roles==('engineer',)
assert Gateway(w).execute('session_alice',close())['decision']=='deny'
```

## 9. Summary trộn nguồn

Alice là owner nhưng không đọc được G-INTERNAL, nên M-MIXED không được cấp. `Readers(S)⊆Readers(A)∩Readers(B)∩OwnerScope(S)`. Bỏ origin B khỏi metadata mà vẫn giữ thông tin B là che provenance, không declassify dữ liệu.

Lineage bảo thủ giúp xác định khả năng cấp nội dung theo nguồn/snapshot; nó không kiểm summary diễn đạt đúng hay có instruction độc hại. Nếu cần share summary rộng hơn, phải có transformation/policy phù hợp và kiểm lại nội dung, không tự nâng authority.

## 10. Cache và revoke

Alice cùng query/config/revision có cache hit. Bob/Carol có fingerprint khác nên không reuse entry Alice. Sau revoke, key đổi và dependency không còn quyền; M-ALICE phụ thuộc B-INTERNAL cũng bị loại. Revalidation vẫn cần vì có thể quên invalidation hoặc cache payload đã stale.

```python
from data import make_world
from context import Config,ContextCache,build_context
w=make_world();cache=ContextCache()
cfg=Config(context_cap=1000,model_window=1600,include_records=False)
old=build_context(w,'session_alice','retry nội bộ',cfg,cache)
assert 'B-INTERNAL' in {i['id'] for i in old['items']}
epoch=w.acl_revision
w.revoke('alice','B-INTERNAL')
w.acl_revision=epoch  # Test only: deliberately simulate a missed invalidation event.
new=build_context(w,'session_alice','retry nội bộ',cfg,cache)
assert not new['cache_hit']
assert 'B-INTERNAL' not in {i['id'] for i in new['items']}
```

Sau disclosure không thể làm bên nhận quên plaintext. Cần policy theo thời điểm, retention/cancellation/egress phù hợp; code này chỉ recheck lúc build/hit/action trong một process.

## 11. Schema chưa là permission

Proposal đúng JSON/schema nhưng sai tenant bị policy từ chối. Roles thừa, unknown tool, bool version hoặc malformed JSON bị parse/schema từ chối trước dispatch. Bốn case đó bị cả schema-only và guarded executor chặn; schema có ích nhưng thiếu ACL/grant thì 13 case còn lại vẫn có event không được phép.

Code không eval/exec model text, không gọi shell/HTTP/email. Hai tool map sang các hàm in-memory rõ ràng. Nếu thêm tool mới cần thêm contract/policy/effect tests, không chỉ thêm vào prompt để model tự dùng.

## 12. Grant

Hết hạn 130:129 còn valid,130 bị từ chối vì điều kiện `now < expires_at`. Target/reason thay đổi làm hash proposal khác. Subject khác dù cùng role/tenant cũng không khớp binding. Replay bị chặn vì used=true; principal/ACL/resource revision đổi cũng làm grant không còn hợp lệ.

Hash chỉ so nội dung, không cấp quyền. Một token giả không có entry server. Controller cấp grant là trusted-only, không được expose như model tool. UI review/authenticated intent chưa có trong lab; fixtures gọi controller trực tiếp để mô phỏng bước ấy.

```python
from data import make_world
from security import Gateway
from fixtures import close
w=make_world();g=Gateway(w)
raw=close();token=g.issue_after_review('session_ops_blue',raw,now=100,ttl=30)
assert g.execute('session_ops_blue',raw,now=130,grant=token)['reason']=='grant_invalid'
assert not g.execute('session_ops_peer',raw,now=129,grant=token)['effect']
assert not g.execute('session_ops_blue',close(reason='Đã sửa sau review'),now=129,grant=token)['effect']
assert w.resources['T-BLUE-1'].status=='open'
```

## 13. TOCTOU và concurrent calls

Update version hoặc revoke trước execute phải bị kiểm live, không dùng quyết định cũ ở review. Trong cùng World, consume grant + đổi ticket nằm trong một lock nên chỉ có một effect.

```python
from concurrent.futures import ThreadPoolExecutor
from data import make_world
from security import Gateway
from fixtures import close
w=make_world();g=Gateway(w);raw=close()
token=g.issue_after_review('session_ops_blue',raw,now=0)
with ThreadPoolExecutor(max_workers=8) as pool:
    outcomes=list(pool.map(lambda _:g.execute('session_ops_blue',raw,now=1,grant=token),range(8)))
assert sum(o['effect'] for o in outcomes)==1
assert w.resources['T-BLUE-1'].version==2
```

RLock không có durability hoặc phối hợp giữa nhiều process/World. Production phải chọn datastore transaction/conditional write, shared nonce state và idempotency phù hợp; test thread này không chứng nhận distributed execution đúng.

## 14. Mẫu số và residual

`.4×.1=.04` theo chain rule vì số thứ hai là điều kiện theo proposal ngoài quyền; không cần độc lập. Đây là giả định, chưa đo từ model. Benign 5/6 và attack 2/12 là hai mẫu số khác nhau của ví dụ tay.

Report thực tế có 17 attack case,5 benign; schema-only có 13 event ngoài quyền/ý định, gateway 0; benign 5/5 ở cả hai.6 confidentiality check có unsafe exposure 6, guard 0. Source version/freshness và authority-retained báo riêng. Answer-only BANANA_OVERRIDE còn sai và không đi qua tool gateway.

0/17 là kết quả regression của proposal cố định; model thật chưa được sample, chưa có phân bố tấn công/traffic và không thể suy risk 0 hay chống injection 100%. Cần đánh giá cả utility và harmful outcome, không chỉ số lần bị deny.

## 15. Audit

Audit giữ subject/tenant/action/resource/decision/reason/revision/hash, đủ để lần theo vì sao action bị từ chối trong lab. Nó không giữ bearer grant, raw payload, credential hoặc body nguồn. Log production cần access/retention/integrity phù hợp; audit list trong RAM không là log chống sửa.

Report nghiên cứu có raw synthetic proposal/payload để học và đối chiếu, khác production logging. Khi dùng LLM thật phải ghi model/revision, sampling, task/attack surface, goal predicate, budget/reset state và cả benign utility; không dùng tỷ lệ scripted gateway như tỷ lệ model chống attack.

## 16. Sản phẩm đạt yêu cầu

Người khác phải chạy lại được policy matrix, context/source IDs, cache/revoke, proposal validation, grant binding/replay và effect trace. Báo rõ cái gì thực sự bị chặn, cái gì chỉ là dữ liệu vẫn còn trong prompt, cái gì chưa đo và residual nào còn sai.

Không cần gọi LLM để kiểm invariant deterministic. Nếu thêm LLM, giữ cùng boundary và thêm phép đo hành vi; không bỏ gate chỉ vì model có vẻ ngoan trên vài prompt. Nếu tắt mọi action để có ít event xấu, vẫn phải báo phần task hợp lệ bị mất.
