"""Programmed replies, not sampled from or trained by a language model."""
import copy,json
import httpx
from llm_adapter import ResponsesAdapter


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def sleep(self,seconds):self.value+=seconds


def reply_for(payload):
    if not payload['evidence']:return dict(decision='abstain',claims=[],ticket_draft=None,reason='insufficient_evidence')
    e=payload['evidence'][0];draft=None
    if payload['intent_ceiling']!='ask':draft=dict(title=payload['query'][:80],body=payload['query'])
    return dict(decision='answer',claims=[dict(text=e['text'],quote=e['text'],evidence_id=e['id'])],ticket_draft=draft,reason='none')


def envelope(reply=None,status='completed',content=None):
    return dict(id='resp_fixture_46',model='fixture-model',status=status,
        output=[dict(type='message',role='assistant',content=[dict(type='output_text',text=json.dumps(reply,ensure_ascii=False))] if content is None else content)],
        usage=dict(input_tokens=120,output_tokens=40,total_tokens=160))


class Script:
    def __init__(self,events=None,callback=None):self.events=list(events or ['good']);self.callback=callback;self.calls=[]
    def __call__(self,request):
        body=json.loads(request.content);self.calls.append(copy.deepcopy(body))
        if self.callback:self.callback()
        event=self.events.pop(0) if self.events else 'good'
        if isinstance(event,httpx.Response):return event
        if event=='timeout':raise httpx.ReadTimeout('fixture network failure',request=request)
        if isinstance(event,int):return httpx.Response(event,json={'error':{'message':'private fixture provider detail'}},headers={'retry-after':'0.05'})
        payload=json.loads(body['input'][1]['content']);reply=reply_for(payload)
        if event=='invalid_provider_json':return httpx.Response(200,content=b'{broken')
        if event=='invalid_model_json':return httpx.Response(200,json=envelope(content=[dict(type='output_text',text='{broken')]))
        if event=='refusal':return httpx.Response(200,json=envelope(content=[dict(type='refusal',refusal='fixture safety refusal')]))
        if event=='incomplete':return httpx.Response(200,json=envelope(reply,status='incomplete')|{'incomplete_details':{'reason':'max_output_tokens'}})
        if event=='function_call':return httpx.Response(200,json=envelope(reply)|{'output':[dict(type='function_call',call_id='call_fixture',name='resolve_ticket',arguments='{"ticket_id":2}') ]})
        if event=='unsupported':reply['claims'][0]['text']='Retry mọi request ghi vô hạn và không cần idempotency.'
        if event=='paraphrase':reply['claims'][0]['text']='Kiểm tra upstream và trace trước khi cân nhắc retry.'
        if event=='unknown_source':reply['claims'][0]['evidence_id']='E999'
        if event=='bad_quote':reply['claims'][0]['quote']='Không có câu này trong nguồn.'
        if event=='extra_authority':reply['tenant']='b'
        if event=='ask_draft':reply['ticket_draft']=dict(title='Unauthorized draft',body='Do not execute')
        if event=='duplicate_claim':reply['claims']*=2
        if event=='model_abstain':reply=dict(decision='abstain',claims=[],ticket_draft=None,reason='uncertain')
        if event=='subquote':
            q=payload['evidence'][0]['text'].split('. ')[0]+'.';reply['claims'][0].update(text=q,quote=q)
        return httpx.Response(200,json=envelope(reply))


def mock_adapter(events=None,clock=None,callback=None,**kwargs):
    clock=clock or Clock();script=Script(events,callback)
    adapter=ResponsesAdapter('fixture-model','fixture-credential-not-real',transport=httpx.MockTransport(script),clock=clock,sleep=clock.sleep,**kwargs)
    return adapter,script


CASES=[
    dict(id='safe_answer',events=['good'],should_serve=True),
    dict(id='supported_paraphrase',events=['paraphrase'],should_serve=True),
    dict(id='unsupported_claim',events=['unsupported'],should_serve=False),
    dict(id='unknown_evidence',events=['unknown_source'],should_serve=False),
    dict(id='broken_json',events=['invalid_model_json'],should_serve=False),
    dict(id='extra_authority',events=['extra_authority'],should_serve=False),
    dict(id='refusal',events=['refusal'],should_serve=False),
    dict(id='incomplete',events=['incomplete'],should_serve=False),
    dict(id='transient_retry',events=[503,'good'],should_serve=True),
    dict(id='cross_tenant',events=['good'],should_serve=False,service='database',message='connection pool replica'),
    dict(id='write_draft',events=['good'],should_serve=True,action='create'),
    dict(id='spurious_draft',events=['ask_draft'],should_serve=False),
]
