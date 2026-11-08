"""Observation-dependent scripted proposals, not an LLM, solver or private chain-of-thought."""
import copy
from bridge50 import digest,terminal
from verification import report_from,discrepancy
from plan_contract import PROMPT

def initial_plan(task):
 steps=[dict(id='S1',tool='read_ticket',args=dict(ticket_id=task['ticket_id']),depends_on=[],postcondition='ticket_observed'),dict(id='S2',tool='search_runbooks',args=dict(query=task['question']),depends_on=['S1'],postcondition='runbook_available')]
 if task['goal']=='triage':steps.append(dict(id='S3',tool='read_health',args={},depends_on=['S2'],postcondition='health_known'))
 return dict(intent='investigate',revision=1,parent_hash=None,trigger='initial',based_on=[],steps=steps)
def tool_action(step):return dict(kind='tool',step_id=step['id'],tool=step['tool'],args=copy.deepcopy(step['args']),terminal=None,conclusion='needs_investigation')
def end_action(obs,kind='answer',reason='completed',conclusion='guidance'):
 return dict(kind=kind,step_id=None,tool=None,args=None,terminal=terminal(obs,kind,reason),conclusion=conclusion)
def revised(plan,parent,trigger,refs,intent='investigate'):
 out=copy.deepcopy(plan);out.update(revision=plan['revision']+1,parent_hash=parent,trigger=trigger,based_on=refs,intent=intent);return out

class PlannerScript:
 def __init__(self,profile='adaptive'):self.profile=profile;self.contexts=[]
 def propose(self,view):
  self.contexts.append(copy.deepcopy(view));t=view['task'];obs=view['observations'];old=view['plan'];new=None
  if old is None:
   new=initial_plan(t);action=tool_action(new['steps'][0])
   if self.profile=='cycle':new['steps'][0]['depends_on']=['S3']
   if self.profile=='unknown_tool':new['steps'][0]['tool']='resolve_ticket'
   if self.profile=='status_forgery':new['steps'][0]['status']='done'
   if self.profile=='unplanned':action['step_id']='S99'
   if self.profile=='premature':action=end_action(obs)
   if self.profile=='skip_dependency':action=tool_action(new['steps'][-1])
   return dict(plan=new,action=action)
  checks=view['checks'];searches=[o for o in obs if o['tool']=='search_runbooks'];last=obs[-1] if obs else None
  if self.profile=='thrash' and old['revision']==2:
   empty=next(o for o in obs if o['tool']=='search_runbooks' and not o['data']['evidence'])
   new=revised(old,view['plan_hash'],'missing_evidence',[empty['id']]);new['steps'].append(dict(id='S5',tool='search_runbooks',args=dict(query='gateway upstream'),depends_on=['S1'],postcondition='runbook_available'))
   return dict(plan=new,action=tool_action(new['steps'][-1]))
  if last and not last['ok']:
   new=revised(old,view['plan_hash'],'tool_failure',[last['id']],'request_clarification')
   return dict(plan=new,action=end_action(obs,'handoff','tool_unavailable','needs_investigation'))
  if discrepancy(obs):
   if self.profile=='ignore_discrepancy':return dict(plan=None,action=end_action(obs))
   refs=next(c['evidence'] for c in checks if c['check']=='report_consistency')
   new=revised(old,view['plan_hash'],'discrepancy',refs,'request_clarification')
   return dict(plan=new,action=end_action(obs,'handoff','uncertain','needs_investigation'))
  if last and last['tool']=='read_health' and last['data']['status']=='unknown':
   new=revised(old,view['plan_hash'],'missing_evidence',[last['id']],'request_clarification')
   return dict(plan=new,action=end_action(obs,'handoff','need_info','needs_investigation'))
  if last and last['tool']=='search_runbooks' and not last['data']['evidence']:
   rep=report_from(obs);hint=None if rep is None else rep['symptom']
   if not hint or hint in [o['args']['query'] for o in searches]:
    new=revised(old,view['plan_hash'],'missing_evidence',[last['id']],'request_clarification')
    return dict(plan=new,action=end_action(obs,'handoff','need_info','needs_investigation'))
   new=revised(old,view['plan_hash'],'missing_evidence',[last['id']]);step=dict(id='S4',tool='search_runbooks',args=dict(query=hint),depends_on=['S1'],postcondition='runbook_available')
   if self.profile=='repeated_query':step['args']=copy.deepcopy(last['args'])
   # Keep the executed prefix, insert an alternative search, change only unexecuted health dependencies.
   for s in new['steps']:
    if s['tool']=='read_health':s['depends_on']=['S4']
   new['steps'].append(step)
   if self.profile=='stale_parent':new['parent_hash']='0'*64
   if self.profile=='unknown_evidence':new['based_on']=['O999']
   if self.profile=='no_reason':new['based_on']=['O1']
   if self.profile=='erase_history':
    new['steps'][1]['args']=copy.deepcopy(step['args']);new['steps'].pop()
    for s in new['steps']:
     if s['tool']=='read_health':s['depends_on']=['S2']
   return dict(plan=new,action=tool_action(step))
  ready=[s for s in old['steps'] if view['step_states'][s['id']]=='ready']
  if ready:return dict(plan=None,action=tool_action(ready[0]))
  action=end_action(obs)
  if self.profile=='claim_root':action['conclusion']='root_cause_confirmed'
  return dict(plan=None,action=action)
