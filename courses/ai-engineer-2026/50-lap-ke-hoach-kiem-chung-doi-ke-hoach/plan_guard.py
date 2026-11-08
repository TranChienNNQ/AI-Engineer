import copy,json
from bridge50 import digest,tool_args
from plan_contract import Plan,Proposal,signature
from verification import postcondition,verify,discrepancy

class PlanError(ValueError):pass
EXPECTED={'read_ticket':'ticket_observed','search_runbooks':'runbook_available','read_health':'health_known'}

class PlanGuard:
 def __init__(self,max_revisions=3):
  if type(max_revisions) is not int or not 1<=max_revisions<=4:raise ValueError('revision budget')
  self.max_revisions=max_revisions;self.history=[];self.used_triggers=set()
 def current(self):return None if not self.history else copy.deepcopy(self.history[-1]['plan'])
 def states(self,obs,plan=None):
  plan=plan or self.current()
  if not plan:return {}
  status={}
  for s in plan['steps']:
   matched=[o for o in obs if signature(o)==signature(s)]
   status[s['id']]=postcondition(matched[-1]) if matched else 'pending'
  for s in plan['steps']:
   if status[s['id']]=='pending' and all(status[d]=='satisfied' for d in s['depends_on']):status[s['id']]='ready'
  return status
 def accept(self,raw,obs,task):
  try:plan=Plan.model_validate(raw).model_dump()
  except (ValueError,TypeError):raise PlanError('invalid_plan_schema') from None
  ids=[s['id'] for s in plan['steps']]
  if len(ids)!=len(set(ids)):raise PlanError('duplicate_step_id')
  byid={s['id']:s for s in plan['steps']};sigs=[]
  for s in plan['steps']:
   try:s['args']=tool_args(s['tool'],json.dumps(s['args']))
   except (ValueError,TypeError):raise PlanError('invalid_plan_arguments') from None
   if s['postcondition']!=EXPECTED[s['tool']]:raise PlanError('postcondition_mismatch')
   if len(s['depends_on'])!=len(set(s['depends_on'])) or any(d not in byid for d in s['depends_on']):raise PlanError('unknown_dependency')
   if s['tool']=='read_ticket' and s['args']['ticket_id']!=task['ticket_id'] or s['tool']=='read_health' and task['goal']!='triage':raise PlanError('plan_scope_denied')
   sigs.append(signature(s))
  if len(sigs)!=len(set(sigs)):raise PlanError('duplicate_plan_action')
  visiting=set();done=set()
  def visit(id_):
   if id_ in visiting:raise PlanError('cyclic_plan')
   if id_ in done:return
   visiting.add(id_)
   for d in byid[id_]['depends_on']:visit(d)
   visiting.remove(id_);done.add(id_)
  for id_ in ids:visit(id_)
  def ancestors(id_):
   direct=byid[id_]['depends_on'];return set(direct).union(*(ancestors(d) for d in direct)) if direct else set()
  for s in plan['steps']:
   if s['tool']!='read_ticket' and not any(byid[d]['tool']=='read_ticket' for d in ancestors(s['id'])):raise PlanError('ticket_prerequisite_missing')
  obsids={o['id'] for o in obs}
  if len(plan['based_on'])!=len(set(plan['based_on'])) or any(i not in obsids for i in plan['based_on']):raise PlanError('unknown_plan_evidence')
  old=self.current()
  if old is None:
   if plan['revision']!=1 or plan['parent_hash'] is not None or plan['trigger']!='initial' or plan['based_on']:raise PlanError('invalid_initial_plan')
  else:
   if len(self.history)>=self.max_revisions:raise PlanError('revision_budget')
   if plan['revision']!=old['revision']+1 or plan['parent_hash']!=self.history[-1]['hash']:raise PlanError('stale_plan_parent')
   if plan['trigger']=='initial' or not plan['based_on']:raise PlanError('missing_revision_trigger')
   trigger=(plan['trigger'],tuple(sorted(plan['based_on'])))
   if trigger in self.used_triggers:raise PlanError('reused_revision_trigger')
   referenced=[o for o in obs if o['id'] in plan['based_on']]
   relevant=(plan['trigger']=='missing_evidence' and any(postcondition(o) in ['empty','unknown'] for o in referenced) or
             plan['trigger']=='tool_failure' and any(not o['ok'] for o in referenced) or
             plan['trigger']=='discrepancy' and discrepancy(obs) and set(next(r['evidence'] for r in verify(obs) if r['check']=='report_consistency'))<=set(plan['based_on']))
   if not relevant:raise PlanError('unjustified_replan')
   for previous in old['steps']:
    if any(signature(o)==signature(previous) for o in obs):
     if previous['id'] not in byid or byid[previous['id']]!=previous:raise PlanError('rewrote_executed_step')
   if digest(plan['steps'])==digest(old['steps']) and plan['intent']==old['intent']:raise PlanError('no_plan_change')
  entry=dict(plan=copy.deepcopy(plan),hash=digest(plan),accepted_after=[o['id'] for o in obs],states=self.states(obs,plan))
  self.history.append(entry)
  if old is not None:self.used_triggers.add((plan['trigger'],tuple(sorted(plan['based_on']))))
  return copy.deepcopy(entry)
 def authorize(self,proposal,obs,task):
  try:p=Proposal.model_validate(proposal).model_dump()
  except (ValueError,TypeError):raise PlanError('invalid_proposal_schema') from None
  if p['plan'] is not None:self.accept(p['plan'],obs,task)
  if not self.history:raise PlanError('missing_plan')
  action=p['action']
  if action['kind']!='tool' and action['terminal'].get('kind')!=action['kind']:raise PlanError('terminal_kind_mismatch')
  if action['conclusion']=='root_cause_confirmed':raise PlanError('unverified_conclusion')
  if action['kind']=='tool':
   s=next((s for s in self.current()['steps'] if s['id']==action['step_id']),None)
   if s is None or s['tool']!=action['tool'] or s['args']!=action['args']:raise PlanError('action_not_in_plan')
   if self.states(obs)[s['id']]!='ready':raise PlanError('step_not_ready')
  return p
