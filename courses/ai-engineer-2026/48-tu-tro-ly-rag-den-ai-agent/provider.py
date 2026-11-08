"""Bounded Responses tool protocol; HTTPX MockTransport is required by default."""
import copy,json,math,re,time
import httpx
from agent_contract import TOOLS,TERMINAL_SCHEMA,loads
ENDPOINT='https://api.openai.com/v1/responses'
class ProviderError(Exception):
    def __init__(self,code,meta):self.code=code;self.meta=meta;super().__init__(code)

def usage(body):
    u=body.get('usage')
    if not isinstance(u,dict) or any(type(u.get(k)) is not int or u[k]<0 for k in ['input_tokens','output_tokens']):return None
    return {k:u[k] for k in ['input_tokens','output_tokens']}

class ToolAdapter:
    def __init__(self,model,key,transport=None,mode='mock',clock=time.monotonic,sleep=time.sleep,max_attempts=2,total_seconds=8.,attempt_seconds=4.,max_output_tokens=1024):
        if not isinstance(model,str) or not model.strip() or len(model)>120 or not isinstance(key,str) or not key.isascii() or not key or any(ord(c)<33 or ord(c)>126 for c in key):raise ValueError('model and credential required')
        if mode not in ['mock','live'] or mode=='mock' and transport is None:raise ValueError('mock needs transport')
        if type(max_attempts) is not int or not 1<=max_attempts<=3 or any(type(x) not in [int,float] or not math.isfinite(x) or x<=0 for x in [total_seconds,attempt_seconds]) or type(max_output_tokens) is not int or not 64<=max_output_tokens<=2048:raise ValueError('invalid provider budget')
        self.model=model;self._key=key;self.mode=mode;self.clock=clock;self.sleep=sleep;self.max_attempts=max_attempts;self.total_seconds=total_seconds;self.attempt_seconds=attempt_seconds;self.max_output_tokens=max_output_tokens
        self.client=httpx.Client(transport=transport,trust_env=False,follow_redirects=False)
    def close(self):self.client.close()
    def identity(self):return dict(provider='responses-tools-http-v1',model=self.model,mode=self.mode,max_attempts=self.max_attempts,total_seconds=self.total_seconds,attempt_seconds=self.attempt_seconds,max_output_tokens=self.max_output_tokens,input_chars=32000,response_bytes=32768,store=False,parallel_tool_calls=False)
    def generate(self,history):
        start=self.clock();attempts=[]
        def meta():return dict(mode=self.mode,requested_model=self.model,attempts=copy.deepcopy(attempts),known_input_tokens=sum(x['usage']['input_tokens'] for x in attempts if x['usage']),known_output_tokens=sum(x['usage']['output_tokens'] for x in attempts if x['usage']),unknown_usage_attempts=sum(x['usage'] is None for x in attempts),elapsed_ms=round((self.clock()-start)*1000,3),usage_source='scripted_fixture' if self.mode=='mock' else 'provider_reported_partial')
        def fail(code):raise ProviderError(code,meta())
        if len(json.dumps(history,ensure_ascii=False))>32000:fail('input_char_budget')
        payload=dict(model=self.model,input=history,tools=TOOLS,parallel_tool_calls=False,store=False,include=['reasoning.encrypted_content'],max_output_tokens=self.max_output_tokens,text=dict(format=dict(type='json_schema',name='investigation_result',strict=True,schema=TERMINAL_SCHEMA)))
        for n in range(1,self.max_attempts+1):
            remaining=self.total_seconds-(self.clock()-start)
            if remaining<=0:fail('provider_deadline')
            attempt=dict(number=n,status='started',usage=None);attempts.append(attempt);retry=False;delay=.05*2**(n-1)
            try:
                with self.client.stream('POST',ENDPOINT,headers={'Authorization':'Bearer '+self._key},json=payload,timeout=min(remaining,self.attempt_seconds)) as response:
                    attempt['status']=str(response.status_code)
                    if response.status_code in [429,500,502,503,504]:retry=True
                    elif response.status_code!=200:fail('provider_http_error')
                    else:
                        chunks=[];size=0
                        for chunk in response.iter_bytes():
                            size+=len(chunk)
                            if size>32768:fail('response_byte_budget')
                            if self.clock()-start>=self.total_seconds:fail('provider_deadline')
                            chunks.append(chunk)
                        try:body=loads(b''.join(chunks))
                        except (ValueError,UnicodeDecodeError,RecursionError):fail('invalid_provider_json')
                        if not isinstance(body,dict):fail('invalid_provider_envelope')
                        attempt['usage']=usage(body)
                        for name in ['id','model']:
                            value=body.get(name);attempt['response_'+name]=value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}',value) else None
                        if self.clock()-start>=self.total_seconds:fail('provider_deadline')
                        if body.get('status')!='completed':fail('provider_incomplete')
                        items=body.get('output')
                        if not isinstance(items,list) or not items:fail('invalid_provider_envelope')
                        calls=[];texts=[]
                        for item in items:
                            if not isinstance(item,dict):fail('invalid_provider_envelope')
                            if item.get('type')=='reasoning':continue # Opaque protocol item, preserved for the next request.
                            if item.get('type')=='function_call':
                                if item.get('status','completed')!='completed':fail('incomplete_tool_call')
                                if not isinstance(item.get('call_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',item['call_id']):fail('invalid_call_id')
                                if not isinstance(item.get('name'),str) or len(item['name'])>64 or not isinstance(item.get('arguments'),str) or len(item['arguments'])>2000:fail('invalid_tool_call')
                                calls.append(dict(call_id=item['call_id'],name=item['name'],arguments=item['arguments']));continue
                            if item.get('type')!='message' or item.get('role')!='assistant' or not isinstance(item.get('content'),list):fail('unexpected_output_item')
                            for part in item['content']:
                                if not isinstance(part,dict):fail('invalid_provider_envelope')
                                if part.get('type')=='refusal':fail('model_refusal')
                                if part.get('type')!='output_text' or not isinstance(part.get('text'),str):fail('unexpected_output_content')
                                texts.append(part['text'])
                        if calls:
                            if len(calls)!=1 or texts:fail('ambiguous_or_parallel_output')
                            return dict(kind='tool',call=calls[0],items=copy.deepcopy(items)),meta()
                        if len(texts)!=1:fail('expected_terminal_text')
                        return dict(kind='terminal',text=texts[0],items=copy.deepcopy(items)),meta()
            except httpx.TransportError:attempt['status']='transport_error';retry=True
            if not retry:fail('provider_error')
            if n==self.max_attempts:fail('provider_retry_exhausted')
            if self.clock()-start+delay>=self.total_seconds:fail('provider_deadline')
            self.sleep(delay)
        fail('provider_retry_exhausted')
