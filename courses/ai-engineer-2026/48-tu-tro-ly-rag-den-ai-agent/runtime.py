import copy,hashlib,json,math,time
from pathlib import Path
from bridge import Fault,digest,source_hash,validate
from agent_contract import Terminal,tool_args,initial,identity,loads
from provider import ProviderError

REASONS=dict(need_info='Cần thêm dữ liệu để tiếp tục.',tool_unavailable='Công cụ chưa cung cấp được dữ liệu cần thiết.',out_of_scope='Yêu cầu nằm ngoài phạm vi được cấp.',uncertain='Chưa đủ cơ sở kết luận.')

def implementation_hash():
    root=Path(__file__).parent
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['bridge.py','agent_contract.py','provider.py','tools.py','runtime.py']})

class Agent:
    def __init__(self,world,adapter,max_turns=6,max_tools=3,total_seconds=30.,clock=time.monotonic):
        if type(max_turns) is not int or not 1<=max_turns<=10 or type(max_tools) is not int or not 0<=max_tools<=6 or type(total_seconds) not in [int,float] or not math.isfinite(total_seconds) or total_seconds<=0:raise ValueError('runtime budgets')
        self.world=world;self.adapter=adapter;self.max_turns=max_turns;self.max_tools=max_tools;self.total_seconds=total_seconds;self.clock=clock
        self.manifest=dict(base_source=source_hash(),implementation=implementation_hash(),contracts=identity(),adapter=adapter.identity(),corpus=world.core.manifest['corpus_hash'],backend=copy.deepcopy(world.core.manifest),health=digest(world.health),health_down=world.health_down,search_down=world.search_down,max_turns=max_turns,max_tools=max_tools,total_seconds=total_seconds,schema='48.1')
        self.release=digest(self.manifest)[:16];self.used=False;self.history=[]
    def close(self):self.adapter.close();self.world.close()
    def integrity(self):
        if any(getattr(self,k)!=self.manifest[k] for k in ['max_turns','max_tools','total_seconds']) or self.world.core.manifest!=self.manifest['backend']:raise Fault(503,'runtime_config_changed')
        if self.manifest['health']!=digest(self.world.health) or self.manifest['health_down']!=self.world.health_down or self.manifest['search_down']!=self.world.search_down:raise Fault(503,'world_release_changed')
        if self.manifest['base_source']!=source_hash() or self.manifest['implementation']!=implementation_hash() or self.manifest['contracts']!=identity() or self.manifest['adapter']!=self.adapter.identity():raise Fault(503,'agent_release_changed')
    def finish(self,terminal,task,token,obs):
        p=self.world.current(token,task,obs)
        if terminal.kind=='handoff':
            if terminal.reason=='completed' or terminal.claims or terminal.health_observation_id is not None:raise Fault(503,'invalid_handoff')
            return dict(state='handoff',code=terminal.reason,answer=REASONS[terminal.reason],citations=[])
        if terminal.reason!='completed' or not terminal.claims:raise Fault(503,'invalid_final')
        if not any(o['tool']=='read_ticket' and o['ok'] for o in obs):raise Fault(503,'missing_ticket_observation')
        byid={o['id']:o for o in obs};citations=[];texts=[];seen=set()
        for claim in terminal.claims:
            o=byid.get(claim.observation_id)
            if not o or o['tool']!='search_runbooks' or not o['ok']:raise Fault(503,'unknown_evidence_observation')
            parents=[e for e in o['data']['evidence'] if e['quote'].count(claim.quote)==1]
            if len(parents)!=1:raise Fault(503,'quote_not_unique_or_present')
            e=parents[0];start=e['start']+e['quote'].find(claim.quote);cite=e|dict(start=start,end=start+len(claim.quote),quote=claim.quote)
            if (claim.observation_id,claim.quote) in seen:raise Fault(503,'duplicate_claim')
            seen.add((claim.observation_id,claim.quote))
            if not validate([cite],[cite],self.world.core.docs,p,task['service'],self.world.core.revoked):raise Fault(409,'source_no_longer_allowed')
            citations.append(cite);texts.append(f"{claim.quote} [{cite['doc_id']}@v{cite['version']}]")
        if task['goal']=='triage':
            health=byid.get(terminal.health_observation_id)
            if not health or health['tool']!='read_health' or not health['ok']:raise Fault(503,'missing_health_observation')
            if health['data']['status']=='unknown':raise Fault(503,'health_unknown_requires_handoff')
            texts.append(f"Health snapshot {health['data']['observed_at']}: {health['data']['status']}; chưa xác định nguyên nhân gốc.")
        elif terminal.health_observation_id is not None:raise Fault(503,'unexpected_health_reference')
        return dict(state='answered',code='ok',answer='\n'.join(texts),citations=citations)
    def run(self,token,raw):
        if self.used:raise ValueError('one run per Agent instance; no durable resume in lesson48')
        self.used=True;start=self.clock();obs=[];trace=[];metas=[];turns=0;reserved=0;ids=set();signatures=set();task=None
        self.history=[]
        def deadline():
            if self.clock()-start>=self.total_seconds:raise Fault(429,'run_deadline')
        try:
            task=self.world.bind(token,raw);self.history=initial(task)
            while True:
                self.integrity();self.world.current(token,task,obs);deadline()
                if turns>=self.max_turns:raise Fault(429,'turn_budget')
                turns+=1
                try:decision,meta=self.adapter.generate(self.history)
                except ProviderError as err:metas.append(err.meta);raise
                metas.append(meta);self.integrity();self.world.current(token,task,obs);deadline()
                self.history.extend(copy.deepcopy(decision['items']))
                if decision['kind']=='terminal':
                    try:terminal=Terminal.model_validate(loads(decision['text']))
                    except (ValueError,TypeError,RecursionError):raise Fault(503,'invalid_terminal_schema') from None
                    result=self.finish(terminal,task,token,obs);trace.append(dict(turn=turns,kind=terminal.kind,code=result['code']));break
                call=decision['call'];call_id=call['call_id']
                if call_id in ids:raise Fault(503,'duplicate_call_id')
                ids.add(call_id)
                try:args=tool_args(call['name'],call['arguments'])
                except (ValueError,TypeError,RecursionError):raise Fault(403,'unknown_tool' if call['name'] not in ['read_ticket','search_runbooks','read_health'] else 'invalid_tool_arguments') from None
                signature=digest([call['name'],args])
                if signature in signatures:raise Fault(429,'repeated_tool_call')
                if reserved>=self.max_tools:raise Fault(429,'tool_budget')
                if call['name']!='read_ticket' and not any(o['tool']=='read_ticket' and o['ok'] for o in obs):raise Fault(403,'read_ticket_first')
                signatures.add(signature);reserved+=1
                data=self.world.dispatch(call['name'],args,task,token);deadline()
                o=dict(id=f'O{len(obs)+1}',call_id=call_id,tool=call['name'],args=args,**data);obs.append(o)
                trace.append(dict(turn=turns,kind='tool',name=call['name'],call_id=call_id,observation_id=o['id'],code=o['code']))
                self.history.append(dict(type='function_call_output',call_id=call_id,output=json.dumps(o,ensure_ascii=False,sort_keys=True)))
        except ProviderError as err:result=dict(state='handoff' if err.code=='model_refusal' else 'failed',code=err.code,answer='',citations=[])
        except Fault as err:result=dict(state='stopped' if err.status==429 else 'blocked' if err.status in [401,403,404] else 'failed',code=err.code,answer='',citations=[])
        # Revalidate before exporting content; revocation cannot undo bytes already sent to a provider.
        if task is not None:
            try:self.world.current(token,task,obs)
            except Fault as err:obs=[];result=dict(state='blocked',code=err.code,answer='',citations=[])
        return result|dict(release=self.release,turns=turns,tool_reservations=reserved,actual_reads=self.world.actual_reads,observations=obs,trace=trace,model_calls=len(metas),usage=metas,
            known_input_tokens=sum(m['known_input_tokens'] for m in metas),known_output_tokens=sum(m['known_output_tokens'] for m in metas),unknown_usage_attempts=sum(m['unknown_usage_attempts'] for m in metas),effects=self.world.effects(),live_api_called=self.adapter.mode=='live',scope='read-only synthetic world; exact observed quotes, not semantic diagnosis verification')
