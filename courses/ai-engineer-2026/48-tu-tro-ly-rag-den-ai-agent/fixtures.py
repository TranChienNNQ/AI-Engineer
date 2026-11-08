"""Responses-shaped fixtures that react to actual observations, never LLM samples."""
import copy,json
import httpx
from provider import ToolAdapter

class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def sleep(self,seconds):self.value+=seconds

def task(goal='triage',question='gateway timeout 504 upstream trace_id',ticket_id=1,service='gateway'):
    return dict(ticket_id=ticket_id,service=service,goal=goal,question=question)
def terminal(obs,kind='answer',reason='completed'):
    if kind=='handoff':return dict(kind=kind,reason=reason,claims=[],health_observation_id=None)
    source=next((o for o in obs if o['tool']=='search_runbooks' and o['ok'] and o['data']['evidence']),None)
    health=next((o for o in obs if o['tool']=='read_health' and o['ok']),None)
    return dict(kind='answer',reason='completed',claims=[] if source is None else [dict(observation_id=source['id'],quote=source['data']['evidence'][0]['quote'])],health_observation_id=health['id'] if health else None)
def response(items,status='completed'):
    return dict(id='resp_fixture_48',model='fixture-model',status=status,output=items,usage=dict(input_tokens=100,output_tokens=30))
def message(obj):return dict(type='message',role='assistant',status='completed',content=[dict(type='output_text',text=json.dumps(obj,ensure_ascii=False))])

class AdaptiveScript:
    def __init__(self,profile='adaptive',callback=None):self.profile=profile;self.requests=[];self.callback=callback
    def __call__(self,req):
        body=json.loads(req.content);self.requests.append(copy.deepcopy(body));n=len(self.requests)
        if self.callback:self.callback(n)
        if self.profile=='retry_503' and n==1:return httpx.Response(503,json={'error':{'message':'fixture transient'}})
        if self.profile=='http400':return httpx.Response(400,json={'error':{'message':'fixture private error'}})
        if self.profile=='timeouts':raise httpx.ReadTimeout('fixture timeout',request=req)
        if self.profile=='huge':return httpx.Response(200,content=b'x'*32769)
        if self.profile=='invalid_provider_json':return httpx.Response(200,content=b'{bad')
        t=json.loads(body['input'][1]['content']);obs=[json.loads(i['output']) for i in body['input'] if i.get('type')=='function_call_output']
        def call(name,args,call_id=None):return dict(type='function_call',id=f'fc_{n}',call_id=call_id or f'call_{n}',name=name,arguments=json.dumps(args,ensure_ascii=False),status='completed')
        if not obs:item=call('read_ticket',dict(ticket_id=t['ticket_id']))
        elif self.profile=='write_after_ticket':item=call('resolve_ticket',dict(ticket_id=2))
        elif self.profile=='loop':item=call('read_ticket',dict(ticket_id=t['ticket_id']))
        elif not obs[-1]['ok']:item=message(terminal(obs,'handoff','tool_unavailable'))
        elif any(o['tool']=='search_runbooks' and not o['data']['evidence'] for o in obs):item=message(terminal(obs,'handoff','need_info'))
        elif self.profile=='health_first' and t['goal']=='triage' and not any(o['tool']=='read_health' for o in obs):item=call('read_health',{})
        elif not any(o['tool']=='search_runbooks' for o in obs):item=call('search_runbooks',dict(query=t['question']))
        elif t['goal']=='triage' and not any(o['tool']=='read_health' for o in obs) and self.profile!='missing_health':item=call('read_health',{})
        elif any(o['tool']=='read_health' and o['data']['status']=='unknown' for o in obs):item=message(terminal(obs,'handoff','need_info'))
        else:
            final=terminal(obs)
            if self.profile=='bad_quote':final['claims'][0]['quote']='Retry mọi request ghi vô hạn.'
            if self.profile=='bad_observation':final['claims'][0]['observation_id']='O999'
            if self.profile=='extra_terminal':final['approved']=True
            item=message(final)
        if not obs:
            if self.profile=='unknown_tool':item=call('resolve_ticket',dict(ticket_id=1))
            if self.profile=='invalid_args':item=call('read_ticket',dict(ticket_id=1.0))
            if self.profile=='extra_authority':item=call('read_ticket',dict(ticket_id=1,tenant='b'))
            if self.profile=='cross_target':item=call('read_ticket',dict(ticket_id=2))
            if self.profile=='search_first':item=call('search_runbooks',dict(query=t['question']))
            if self.profile=='premature':item=message(dict(kind='answer',reason='completed',claims=[dict(observation_id='O2',quote='x')],health_observation_id=None))
            if self.profile=='invalid_terminal':item=message(dict(kind='answer',reason='completed'))
            if self.profile=='refusal':item=dict(type='message',role='assistant',content=[dict(type='refusal',refusal='fixture refusal')])
        if self.profile=='duplicate_id' and len(obs)==1 and item['type']=='function_call':item['call_id']=obs[0]['call_id']
        items=[dict(type='reasoning',id=f'rs_{n}',summary=[],encrypted_content=f'fixture-opaque-{n}'),item]
        if self.profile=='parallel':items.append(call('read_health',{}))
        if self.profile=='mixed':items.append(message(terminal(obs,'handoff','uncertain')))
        return httpx.Response(200,json=response(items,status='incomplete' if self.profile=='incomplete' else 'completed'))

def mock(profile='adaptive',callback=None,clock=None,**kwargs):
    clock=clock or Clock();script=AdaptiveScript(profile,callback)
    adapter=ToolAdapter('fixture-model','fixture-credential-not-real',transport=httpx.MockTransport(script),clock=clock,sleep=clock.sleep,**kwargs)
    return adapter,script,clock

CASES=[
 dict(id='triage',profile='adaptive',state='answered',code='ok',reads=3),
 dict(id='retry_only',profile='adaptive',goal='retry',state='answered',code='ok',reads=2),
 dict(id='health_first',profile='health_first',state='answered',code='ok',reads=3),
 dict(id='health_unknown',profile='adaptive',health_status='unknown',state='handoff',code='need_info',reads=3),
 dict(id='health_unavailable',profile='adaptive',health_down=True,state='handoff',code='tool_unavailable',reads=3),
 dict(id='no_source',profile='adaptive',question='quantum cooling',state='handoff',code='need_info',reads=2),
 dict(id='search_unavailable',profile='adaptive',search_down=True,state='handoff',code='tool_unavailable',reads=2),
 dict(id='reader_health_denied',profile='adaptive',actor='demo-reader-a',state='blocked',code='health_scope_denied',reads=2),
 dict(id='cross_tenant',profile='adaptive',ticket_id=2,service='database',state='blocked',code='ticket_not_found',reads=0),
 dict(id='cross_target',profile='cross_target',state='blocked',code='target_scope_denied',reads=0),
 dict(id='unknown_tool',profile='unknown_tool',state='blocked',code='unknown_tool',reads=0),
 dict(id='invalid_args',profile='invalid_args',state='blocked',code='invalid_tool_arguments',reads=0),
 dict(id='extra_authority',profile='extra_authority',state='blocked',code='invalid_tool_arguments',reads=0),
 dict(id='search_first',profile='search_first',state='blocked',code='read_ticket_first',reads=0),
 dict(id='loop',profile='loop',state='stopped',code='repeated_tool_call',reads=1),
 dict(id='duplicate_id',profile='duplicate_id',state='failed',code='duplicate_call_id',reads=1),
 dict(id='premature',profile='premature',state='failed',code='missing_ticket_observation',reads=0),
 dict(id='bad_quote',profile='bad_quote',state='failed',code='quote_not_unique_or_present',reads=3),
 dict(id='missing_health',profile='missing_health',state='failed',code='missing_health_observation',reads=2),
 dict(id='parallel',profile='parallel',state='failed',code='ambiguous_or_parallel_output',reads=0),
 dict(id='mixed',profile='mixed',state='failed',code='ambiguous_or_parallel_output',reads=0),
 dict(id='incomplete',profile='incomplete',state='failed',code='provider_incomplete',reads=0),
 dict(id='refusal',profile='refusal',state='handoff',code='model_refusal',reads=0),
 dict(id='invalid_terminal',profile='invalid_terminal',state='failed',code='invalid_terminal_schema',reads=0),
 dict(id='transient_retry',profile='retry_503',state='answered',code='ok',reads=3),
 dict(id='tool_budget',profile='adaptive',max_tools=1,state='stopped',code='tool_budget',reads=1),
 dict(id='turn_budget',profile='adaptive',max_turns=2,state='stopped',code='turn_budget',reads=2),
]
