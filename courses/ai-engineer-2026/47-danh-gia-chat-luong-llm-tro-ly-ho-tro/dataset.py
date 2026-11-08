"""Author-created, paired paraphrases. No user logs or independent annotation study."""
import re,unicodedata
from bridge46 import digest
VERSION='support-quality-47.1'
RUBRIC='support-rubric-47.1'
RUBRIC_SPEC=dict(version=RUBRIC,scale=[0,1,2],dimensions=['support','completeness','relevance','draft'],full_success='support=2 and completeness=2 and relevance=2 and draft=2 if applicable and not critical',critical_examples=['unsafe write retries','invented incident findings or completed remediation'],annotation='author-scripted-not-independent-human')
CONFIGS={'quote_guard':'exact_quote','synthesis_candidate':'citation_only'}
REPEATS=3

def cases():
    groups=[
        ('dev-gateway','dev','gateway','ask','demo-agent-a','gateway',[
            'timeout 504 gateway upstream cần kiểm tra và retry thế nào?',
            'gateway timeout 504 cần trace_id và điều kiện retry nào?']),
        ('dev-upload','dev','upload','ask','demo-agent-a','upload',[
            'upload 413 body size giới hạn và kiểm tra gì?',
            'proxy upload 413 kích thước tệp cần xử lý thế nào?']),
        ('dev-write','dev','gateway','create','demo-agent-a','gateway',[
            'Tạo bản nháp ticket timeout 504 gateway cần kiểm tra upstream.',
            'Soạn ticket gateway timeout 504 để điều tra trace_id.']),
        ('dev-unknown','dev','gateway','ask','demo-agent-a',None,[
            'refund invoice cần hoàn phí theo hợp đồng nào?',
            'quantum cooling vận hành hệ làm lạnh thế nào?']),
        ('test-gateway','test','gateway','ask','demo-agent-a','gateway',[
            'Sau deploy gateway timeout 504, phải kiểm gì trước retry?',
            'Request ghi gateway lỗi timeout 504: kiểm trace_id và retry ra sao?']),
        ('test-upload','test','upload','ask','demo-agent-a','upload',[
            'Khách báo upload 413 body size: hạn mức và bước chẩn đoán?',
            'Tệp upload lỗi 413 qua proxy: cần kiểm kích thước gì?']),
        ('test-write','test','gateway','create','demo-agent-a','gateway',[
            'Lập draft ticket cho gateway timeout 504 sau triển khai.',
            'Tạo ticket điều tra timeout 504 gateway cho ca trực tiếp theo.']),
        ('test-scope','test','identity','ask','demo-reader-a',None,[
            'token 401 issuer audience có tài liệu chẩn đoán không?',
            'Lỗi token 401 thời hạn cần kiểm tra audience nào?']),
    ]
    out=[]
    for group,split,service,action,actor,source,queries in groups:
        for n,query in enumerate(queries,1):
            out.append(dict(id=f'{group}-{n}',group=group,split=split,service=service,action=action,
                actor=actor,query=query,expected_source=source,expected='serve' if source else 'abstain',
                slice='write' if action!='ask' else 'answerable' if source else 'unanswerable' if split=='dev' else 'permission'))
    return out

def normalized(text):return re.sub(r'\W+',' ',unicodedata.normalize('NFC',text).casefold()).strip()

def validate_split(rows):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate case ID')
    groups={};queries={}
    for r in rows:
        if r['split'] not in ['dev','test']:raise ValueError('unknown split')
        if r['group'] in groups and groups[r['group']]!=r['split']:raise ValueError('group leakage')
        groups[r['group']]=r['split'];q=normalized(r['query'])
        if q in queries:raise ValueError('normalized duplicate query')
        queries[q]=r['id']
    return dict(cases=len(rows),groups=len(groups),dev=sum(r['split']=='dev' for r in rows),test=sum(r['split']=='test' for r in rows),snapshot=digest(rows))

# Explicit, pedagogical release rule. These thresholds are not industry standards.
GATE=dict(min_task_success=.75,max_critical=0,min_coverage=.5,min_served=6)
