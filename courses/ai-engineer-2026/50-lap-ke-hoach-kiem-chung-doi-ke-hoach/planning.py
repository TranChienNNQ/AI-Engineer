import copy,json,hashlib
from pathlib import Path
from bridge50 import MCPAgent,ProviderError,Fault,digest,source_hash,loads
from plan_guard import PlanGuard,PlanError
from plan_contract import PROMPT,identity
from verification import verify,discrepancy
from policy50 import PlannerScript

def implementation_hash():
 root=Path(__file__).parent
 return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge50.py','plan_contract.py','verification.py','plan_guard.py','policy50.py','planning.py']})

class PlanningPort:
 def __init__(self,profile='adaptive',max_revisions=3):self.guard=PlanGuard(max_revisions);self.script=PlannerScript(profile);self.profile=profile;self.mode='fixture_plan';self.turns=0;self.trace=[];self.closed=False
 def identity(self):return dict(port='scripted-plan-proposal-v1',profile=self.profile,max_revisions=self.guard.max_revisions,contracts=identity())
 def close(self):self.closed=True
 def generate(self,history):
  self.turns+=1
  meta=dict(mode=self.mode,attempts=[],known_input_tokens=0,known_output_tokens=0,unknown_usage_attempts=0,elapsed_ms=0,usage_source='not_applicable_scripted_port')
  obs=[loads(i['output']) for i in history if i.get('type')=='function_call_output'];task=loads(history[1]['content'])
  view=dict(instructions=PROMPT,task=task,observations=obs,plan=self.guard.current(),plan_hash=None if not self.guard.history else self.guard.history[-1]['hash'],step_states=self.guard.states(obs),checks=verify(obs))
  # Explicit planning view replaces, rather than silently reusing, the old one-shot search-stop prompt.
  if len(json.dumps(view,ensure_ascii=False))>16000:raise ProviderError('planning_context_budget',meta)
  try:p=self.guard.authorize(self.script.propose(view),obs,task)
  except PlanError as e:
   self.trace.append(dict(turn=self.turns,status='rejected',code=str(e)));raise ProviderError('plan_'+str(e),meta) from None
  action=p['action'];self.trace.append(dict(turn=self.turns,status='accepted',revision=self.guard.current()['revision'],plan_hash=self.guard.history[-1]['hash'],action=copy.deepcopy(action),step_states=self.guard.states(obs),checks=verify(obs)))
  if action['kind']=='tool':
   call=dict(call_id=f'plan_call_{self.turns}',name=action['tool'],arguments=json.dumps(action['args'],ensure_ascii=False))
   item=dict(type='function_call',status='completed',**call)
   return dict(kind='tool',call=call,items=[item]),meta
  text=json.dumps(action['terminal'],ensure_ascii=False);item=dict(type='message',role='assistant',content=[dict(type='output_text',text=text)])
  return dict(kind='terminal',text=text,items=[item]),meta

class PlannedAgent(MCPAgent):
 def __init__(self,world,port,**kwargs):
  super().__init__(world,port,max_turns=kwargs.pop('max_turns',8),max_tools=kwargs.pop('max_tools',5),**kwargs)
  self.manifest.update(schema50='50.1',base49=source_hash(),implementation50=implementation_hash(),planning=port.identity())
  self.release=digest(self.manifest)[:16]
 def integrity(self):
  super().integrity()
  if self.manifest['base49']!=source_hash() or self.manifest['implementation50']!=implementation_hash() or self.manifest['planning']!=self.adapter.identity():raise Fault(503,'planning_release_changed')
 def finish(self,terminal,task,token,obs):
  if terminal.kind=='answer':
   checks=verify(obs);required=['ticket_observed','runbook_available']+(['health_known'] if task['goal']=='triage' else [])
   if any(next(c['status'] for c in checks if c['check']==k)!='supported' for k in required):raise Fault(503,'verification_incomplete')
   if discrepancy(obs):raise Fault(503,'unresolved_discrepancy')
  return super().finish(terminal,task,token,obs)
 def run(self,token,raw):
  result=super().run(token,raw)
  if result['code']=='plan_revision_budget':result['state']='stopped'
  redacted=result['state']=='blocked'
  return result|dict(plans=[] if redacted else copy.deepcopy(self.adapter.guard.history),planning_trace=[] if redacted else copy.deepcopy(self.adapter.trace),step_states={} if redacted else self.adapter.guard.states(result['observations']),verification=[] if redacted else verify(result['observations']),planning_export_redacted=redacted,planning_turns=self.adapter.turns,provider_http_attempts=0,scope='structured scripted planner + deterministic fixture verifier + real MCP execution; no LLM sampling/causal diagnosis')
