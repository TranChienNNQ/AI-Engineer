"""HTTP Responses adapter: no network by default; tests inject HTTPX MockTransport."""
import json,time,math,re
import httpx
from contracts import WIRE_SCHEMA
from bridge45 import digest
ENDPOINT='https://api.openai.com/v1/responses'


class ProviderError(Exception):
    def __init__(self,code,meta):self.code=code;self.meta=meta;super().__init__(code)


def usage(body):
    u=body.get('usage')
    if not isinstance(u,dict):return None
    if any(type(u.get(k)) is not int or u[k]<0 for k in ['input_tokens','output_tokens']):return None
    return {k:u[k] for k in ['input_tokens','output_tokens']}


class ResponsesAdapter:
    def __init__(self,model,key,transport=None,clock=time.monotonic,sleep=time.sleep,max_attempts=2,total_seconds=10.,attempt_seconds=5.,max_output_tokens=768,mode='mock'):
        if not isinstance(model,str) or not model.strip() or len(model)>120 or not isinstance(key,str) or not key or not key.isascii() or any(ord(c)<33 or ord(c)>126 for c in key):raise ValueError('model and API credential required')
        if type(max_attempts) is not int or not 1<=max_attempts<=3 or any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in [total_seconds,attempt_seconds]) or type(max_output_tokens) is not int or not 64<=max_output_tokens<=2048:raise ValueError('adapter budgets')
        if mode not in ['mock','live'] or mode=='mock' and transport is None:raise ValueError('mock mode requires an injected transport')
        self.model=model;self._key=key;self.clock=clock;self.sleep=sleep;self.max_attempts=max_attempts;self.total_seconds=total_seconds;self.attempt_seconds=attempt_seconds;self.max_output_tokens=max_output_tokens;self.mode=mode
        self.client=httpx.Client(transport=transport,trust_env=False,follow_redirects=False)

    def close(self):self.client.close()
    def identity(self):return dict(provider='responses-http-v1',mode=self.mode,model=self.model,endpoint=ENDPOINT,max_attempts=self.max_attempts,total_seconds=self.total_seconds,attempt_seconds=self.attempt_seconds,max_output_tokens=self.max_output_tokens,max_response_bytes=16384)

    def generate(self,messages):
        start=self.clock();attempts=[]
        def meta():
            known=[x['usage'] for x in attempts if x['usage'] is not None]
            return dict(mode=self.mode,model=self.model,attempts=attempts,unknown_usage_attempts=sum(x['usage'] is None for x in attempts),
                known_input_tokens=sum(x['input_tokens'] for x in known),known_output_tokens=sum(x['output_tokens'] for x in known),
                usage_source='scripted_fixture' if self.mode=='mock' else 'provider_reported_partial',prompt_hash=digest(messages),elapsed_ms=round((self.clock()-start)*1000,3))
        def fail(code):raise ProviderError(code,meta())
        payload=dict(model=self.model,input=messages,max_output_tokens=self.max_output_tokens,
            text=dict(format=dict(type='json_schema',name='support_reply',strict=True,schema=WIRE_SCHEMA)))
        if len(json.dumps(messages,ensure_ascii=False))>6000:fail('input_char_budget')
        for n in range(1,self.max_attempts+1):
            remaining=self.total_seconds-(self.clock()-start)
            if remaining<=0:fail('provider_deadline')
            attempt=dict(number=n,status='started',usage=None);attempts.append(attempt);retry=False;delay=.05*2**(n-1)
            try:
                with self.client.stream('POST',ENDPOINT,headers={'Authorization':'Bearer '+self._key},json=payload,
                                        timeout=min(self.attempt_seconds,remaining)) as response:
                    attempt['status']=str(response.status_code)
                    if response.status_code in [429,500,502,503,504]:
                        retry=True
                        try:delay=min(1.,max(delay,float(response.headers.get('retry-after','0'))))
                        except ValueError:pass
                    elif response.status_code!=200:fail('provider_http_error')
                    else:
                        chunks=[];size=0
                        for chunk in response.iter_bytes():
                            size+=len(chunk)
                            if size>16384:fail('response_byte_budget')
                            if self.clock()-start>=self.total_seconds:fail('provider_deadline')
                            chunks.append(chunk)
                        try:body=json.loads(b''.join(chunks))
                        except (ValueError,UnicodeDecodeError,RecursionError):fail('invalid_provider_json')
                        if not isinstance(body,dict):fail('invalid_provider_envelope')
                        attempt['usage']=usage(body)
                        for field in ['id','model']:
                            value=body.get(field)
                            attempt['response_'+field]=value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,128}',value) else None
                        if self.clock()-start>=self.total_seconds:fail('provider_deadline')
                        if body.get('status')!='completed':fail('provider_incomplete')
                        output=body.get('output')
                        if not isinstance(output,list):fail('invalid_provider_envelope')
                        texts=[]
                        for item in output:
                            if not isinstance(item,dict):fail('invalid_provider_envelope')
                            if item.get('type')=='reasoning':continue
                            if item.get('type')!='message' or item.get('role')!='assistant' or not isinstance(item.get('content'),list):fail('unexpected_output_item')
                            for content in item['content']:
                                if not isinstance(content,dict):fail('invalid_provider_envelope')
                                if content.get('type')=='refusal':fail('model_refusal')
                                if content.get('type')!='output_text' or not isinstance(content.get('text'),str):fail('unexpected_output_content')
                                texts.append(content['text'])
                        if len(texts)!=1:fail('expected_one_structured_output')
                        return texts[0],meta()
            except httpx.TransportError:
                attempt['status']='transport_error';retry=True
            if not retry:fail('provider_error')
            if n==self.max_attempts:fail('provider_retry_exhausted')
            if self.clock()-start+delay>=self.total_seconds:fail('provider_deadline')
            self.sleep(delay)
        fail('provider_retry_exhausted')
