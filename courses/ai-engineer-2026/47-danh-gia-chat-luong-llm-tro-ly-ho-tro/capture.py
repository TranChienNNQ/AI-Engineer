"""Record actual lesson46 workflow outputs with programmed, explicitly labeled responses."""
import json
from pathlib import Path
from typing import Literal
import httpx
from pydantic import Field,model_validator
from bridge46 import SupportEngine,ResponsesAdapter,Clock,envelope,reply_for,request,documents,TOKENS,retrieve,digest
from dataset import CONFIGS,RUBRIC
from scoring import Strict,content

class Record(Strict):
    record_version:Literal['47.1']
    item_id:str
    config:Literal['quote_guard','synthesis_candidate']
    case_id:str
    group:str
    split:Literal['dev','test']
    repeat:int=Field(ge=0,le=2)
    query:str
    action:Literal['ask','create']
    expected:Literal['serve','abstain']
    expected_source:str|None
    slice:str
    origin:Literal['scripted_fixture']
    release:str
    state:str
    code:str
    served:bool
    answer:str
    citations:list[dict]
    draft:dict|None
    evidence:list[dict]
    usage:dict|None
    transport_calls:int=Field(ge=0)
    ticket_count:int
    receipt_count:int
    variant:str
    content_hash:str
    record_hash:str
    @model_validator(mode='after')
    def coherent(self):
        d=self.model_dump()
        if self.served!=(self.state in ['answered','proposed']):raise ValueError('served/state mismatch')
        if self.served and (not self.answer or not self.citations):raise ValueError('served needs answer/citations')
        if (self.draft is not None)!=(self.state=='proposed'):raise ValueError('draft/state mismatch')
        if self.content_hash!=digest(content(d)):raise ValueError('content hash mismatch')
        if self.record_hash!=digest({k:v for k,v in d.items() if k!='record_hash'}):raise ValueError('record hash mismatch')
        return self

class Program:
    def __init__(self,case,repeat):self.case=case;self.repeat=repeat;self.calls=0;self.variant='bypass'
    def __call__(self,req):
        self.calls+=1
        # A transient retry occurs before the same programmed answer, not a new sample.
        if self.case['service']=='gateway' and self.repeat==0 and self.calls==1:
            return httpx.Response(503,json={'error':{'message':'fixture transient'}},headers={'retry-after':'.05'})
        payload=json.loads(json.loads(req.content)['input'][1]['content']);reply=reply_for(payload)
        action=payload['intent_ceiling'];service=payload['service'];self.variant='full'
        if self.case['id']=='test-upload-2' and self.repeat==2:
            self.variant='incomplete';return httpx.Response(200,json=envelope(reply,status='incomplete'))
        if action!='ask':
            reply['ticket_draft']=dict(title='Điều tra timeout 504 gateway',body='Người dùng báo: '+payload['query']+' Cần kiểm tra upstream và trace_id; chưa xác minh nguyên nhân.')
            if self.repeat==1:
                self.variant='invented_draft';reply['ticket_draft']['body']='Đã xác minh nguyên nhân là database hỏng và đã khôi phục toàn bộ dữ liệu.'
            if self.repeat==2:
                self.variant='partial';q=payload['evidence'][0]['text'].split('. ')[0]+'.';reply['claims'][0].update(text=q,quote=q)
        elif service=='gateway' and self.repeat==1:
            self.variant='paraphrase';reply['claims'][0]['text']='Kiểm tra upstream và trace_id; chỉ retry request đọc hoặc request ghi có idempotency key, tránh retry khi chưa biết kết quả ghi.'
        elif service=='gateway' and self.repeat==2:
            self.variant='unsafe';reply['claims'][0]['text']='Retry mọi request ghi vô hạn, không cần idempotency key.'
        elif service=='upload' and self.repeat==1:
            self.variant='partial';q=payload['evidence'][0]['text'].split('. ')[0]+'.';reply['claims'][0].update(text=q,quote=q)
        return httpx.Response(200,json=envelope(reply))

def capture(directory,case,config,repeat):
    program=Program(case,repeat);clock=Clock()
    adapter=ResponsesAdapter('fixture-model','fixture-credential-not-real',transport=httpx.MockTransport(program),clock=clock,sleep=clock.sleep)
    path=Path(directory)/config/(case['id']+f'-{repeat}.sqlite');path.parent.mkdir(parents=True,exist_ok=True)
    engine=SupportEngine(path,adapter,CONFIGS[config])
    try:
        req=request(case['query'],action=case['action'],service=case['service'],key='capture')
        evidence,_=retrieve(documents(),TOKENS[case['actor']],case['service'],case['query'])
        result=engine.ask(case['actor'],req)
        draft=engine.read_proposal('demo-reviewer-a',result['proposal_id'])['payload'] if result['proposal_id'] else None
        if draft:draft={k:draft[k] for k in ['title','body']}
        row=dict(record_version='47.1',item_id=digest([config,case['id'],repeat])[:24],config=config,
            case_id=case['id'],group=case['group'],split=case['split'],repeat=repeat,query=case['query'],
            action=case['action'],expected=case['expected'],expected_source=case['expected_source'],slice=case['slice'],
            origin='scripted_fixture',release=result['release'],state=result['state'],code=result['code'],served=result['state'] in ['answered','proposed'],
            answer=result['answer'],citations=result['citations'],draft=draft,evidence=evidence,usage=result['stats']['llm'],transport_calls=program.calls,
            ticket_count=engine.store.db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0],
            receipt_count=engine.store.db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0],variant=program.variant)
        row['content_hash']=digest(content(row));row['record_hash']=digest(row)
        return Record.model_validate(row).model_dump()
    finally:engine.close()

def author_rating(record):
    """Annotations for the particular authored texts; never an automatic semantic judge."""
    if not record['served']:return None
    variant=record['variant'];bad=variant in ['unsafe','invented_draft']
    return dict(item_id=record['item_id'],content_hash=record['content_hash'],rubric=RUBRIC,
        support=0 if variant=='unsafe' else 2,completeness=0 if variant=='unsafe' else 1 if variant=='partial' else 2,
        relevance=2,draft=None if record['action']=='ask' else 0 if variant=='invented_draft' else 2,
        critical=bad,reason={'unsafe':'Khuyến nghị retry ghi vô hạn trái điều kiện idempotency trong nguồn.',
            'invented_draft':'Draft bịa nguyên nhân và tuyên bố đã khôi phục, user/nguồn không xác nhận.',
            'partial':'Quote đúng nhưng thiếu bước chẩn đoán hoặc điều kiện retry bắt buộc.',
            'paraphrase':'Giữ đủ upstream, trace_id và điều kiện retry; câu chữ khác nguồn.',
            'full':'Giữ đầy đủ hướng dẫn trong nguồn; draft nếu có chỉ báo triệu chứng, chưa bịa kết quả.'}[variant],
        provenance='author-scripted-not-independent-human')
