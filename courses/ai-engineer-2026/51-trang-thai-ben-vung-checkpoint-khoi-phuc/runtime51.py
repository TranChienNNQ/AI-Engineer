"""Durable single-agent loop: commit intent, reserve attempt, execute, commit result."""
import copy,hashlib,os,sqlite3
from pathlib import Path
from bridge51 import (digest,source_hash,PlanGuard,PlanError,PlannerScript,verify,PlannedAgent,PlanningPort,Terminal,Fault,pc)
from contracts51 import State,Limits,FINAL,VERSION,operation_id,recovery_rule,canonical
from journal51 import JournalError

def implementation_hash():
 root=Path(__file__).parent
 return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge51.py','contracts51.py','journal51.py','runtime51.py','worker.py']})
def restore(state,limits,task):
 g=PlanGuard(limits['revisions']);obs=state['observations']
 for entry in state['plans']:
  seen=obs[:len(entry['accepted_after'])]
  if [o['id'] for o in seen]!=entry['accepted_after']:raise JournalError('plan_evidence_prefix')
  if g.accept(entry['plan'],seen,task)!=entry:raise JournalError('plan_checkpoint_changed')
 return g

def probe_append(path,run_id,op_id,attempt,epoch,data):
 # Separate test instrument. Agent recovery NEVER consults this acknowledgement ledger.
 db=sqlite3.connect(path,isolation_level=None)
 try:
  db.execute('CREATE TABLE IF NOT EXISTS acknowledgements(n INTEGER PRIMARY KEY,run_id TEXT,op_id TEXT,attempt INTEGER,epoch INTEGER,result_hash TEXT)')
  db.execute('INSERT INTO acknowledgements(run_id,op_id,attempt,epoch,result_hash) VALUES(?,?,?,?,?)',(run_id,op_id,attempt,epoch,digest(data)))
 finally:db.close()

class DurableAgent:
 def __init__(self,journal,world,run_id,task,token,limits=None,expires_at=160,probe=None,crash_at=None,crash_step='S2',pause_after=None):
  import re
  if re.fullmatch(r'[a-zA-Z0-9_-]{1,60}',run_id) is None:raise ValueError('run id')
  self.journal=journal;self.world=world;self.run_id=run_id;self.token=token
  self.task=world.bind(token,task);principal=world.principal(token);self.actor=principal['tenant']+':'+principal['id']
  self.limits=Limits.model_validate(limits or {}).model_dump();self.expires_at=expires_at
  if type(expires_at) is not int:raise ValueError('absolute deadline must be integer seconds')
  self.gate=PlannedAgent(world,PlanningPort(max_revisions=self.limits['revisions']))
  rows=[dict(r) for r in world.core.store.db.execute('SELECT * FROM tickets ORDER BY id')]
  self.binding=dict(version=VERSION,task=copy.deepcopy(self.task),principal=dict(id=principal['id'],tenant=principal['tenant']),limits=self.limits,expires_at=expires_at,
                    base50=source_hash(),implementation51=implementation_hash(),gate_manifest=self.gate.manifest,world_seed=digest(rows))
  self.probe=probe;self.crash_at=crash_at;self.crash_step=crash_step;self.pause_after=pause_after
 def current(self,state):
  self.gate.integrity();self.world.current(self.token,self.task,state['observations'])
  p=self.world.principal(self.token)
  for o in state['observations']:
   if o['tool']=='read_ticket' and o['ok']:
    t=self.world.core.store.ticket(self.task['ticket_id'],p)
    if o['data']!={k:t[k] for k in ['id','title','body','status','version']}:raise Fault(409,'ticket_snapshot_changed')
 def hook(self,point,state):
  step=state['pending']['action']['step_id'] if state['pending'] else None
  if self.crash_at==point and (point in ['planning_reserved','terminal_committed'] or step==self.crash_step):os._exit(83)
 def create(self):return self.journal.create(self.run_id,self.actor,self.binding)
 def run(self,owner,now=100,takeover_epoch=None):
  if type(now) is not int:raise ValueError('business clock')
  # Authentication and binding checked before claiming or exposing checkpoint content.
  initial=self.journal.load(self.run_id,self.actor)
  if initial['header']['binding']!=canonical(self.binding):raise JournalError('run_binding_changed')
  s=initial['state'];self.current(s)
  handle=self.journal.claim(self.run_id,self.actor,self.binding,owner,takeover_epoch)
  def save(kind,release=False,crash=False):
   nonlocal handle
   handle=self.journal.save(self.run_id,self.actor,handle,s,kind,release,crash)
  def end(state,code):
   s.update(phase=state,pending=None,result=dict(state=state,code=code,answer='',citations=[]));save('finished',True)
  try:
   # A durable final can be returned again after delivery acknowledgement was lost.
   if s['phase'] in FINAL:
    save('terminal_read',True);return self.export(s,handle)
   if now>=self.expires_at:end('stopped','run_expired');return self.export(s,handle)
   if s['phase']=='paused':s['phase']='ready';save('resumed')
   if s['phase']=='planning':s['phase']='ready';save('abandoned_planning')
   while s['phase'] not in FINAL:
    self.current(s)
    if s['phase']=='ready':
     if s['turns']>=self.limits['turns']:end('stopped','turn_budget');break
     s['turns']+=1;s['phase']='planning';save('planning_reserved');self.hook('planning_reserved',s)
     g=restore(s,self.limits,self.task)
     view=dict(instructions=pc.PROMPT,task=self.task,observations=s['observations'],plan=g.current(),plan_hash=g.history[-1]['hash'] if g.history else None,step_states=g.states(s['observations']),checks=verify(s['observations']))
     if len(canonical(view))>16000:raise JournalError('planning_context_budget')
     proposal=g.authorize(PlannerScript().propose(view),s['observations'],self.task);s['plans']=copy.deepcopy(g.history)
     action=proposal['action']
     if action['kind']!='tool':
      result=self.gate.finish(Terminal.model_validate(action['terminal']),self.task,self.token,s['observations'])
      s.update(phase=result['state'],result=result,pending=None);save('finished',True,crash=self.crash_at=='terminal_transaction');self.hook('terminal_committed',s);break
     s.update(phase='pending',pending=dict(op_id=operation_id(self.run_id,action['step_id']),action=action,attempts=0));save('intent_committed');self.hook('intent_committed',s)
    pending=s['pending'];action=pending['action']
    g=restore(s,self.limits,self.task)
    g.authorize(dict(plan=None,action=action),s['observations'],self.task)
    if pending['op_id']!=operation_id(self.run_id,action['step_id']):raise JournalError('operation_identity')
    if recovery_rule(action['tool'])!='bounded_read_retry':end('handoff','reconcile_required');break
    if pending['attempts']>=self.limits['per_operation']:end('stopped','operation_attempt_budget');break
    if s['attempts']>=self.limits['attempts']:end('stopped','tool_attempt_budget');break
    pending['attempts']+=1;s['attempts']+=1;save('attempt_reserved');self.hook('attempt_reserved',s)
    self.current(s)
    data=self.world.dispatch(action['tool'],action['args'],self.task,self.token)
    if self.probe:probe_append(self.probe,self.run_id,pending['op_id'],pending['attempts'],handle['epoch'],data)
    self.hook('after_response',s)
    o=dict(id=f"O{len(s['observations'])+1}",call_id=pending['op_id'],tool=action['tool'],args=action['args'],**data,
           durable_provenance=dict(op_id=pending['op_id'],attempt=pending['attempts'],epoch=handle['epoch']))
    s['observations'].append(o);s.update(phase='ready',pending=None);save('observation_committed');
    if self.crash_at=='observation_committed' and action['step_id']==self.crash_step:os._exit(83)
    if self.pause_after==len(s['observations']):s['phase']='paused';save('paused',True);break
  except PlanError as e:end('stopped' if str(e)=='revision_budget' else 'failed','plan_'+str(e))
  except Fault as e:end('blocked' if e.status in [401,403,404] else 'failed',e.code)
  self.current(s) # Final/export revalidation. A denied export never returns old content.
  return self.export(s,handle)
 def export(self,s,handle):
  self.current(s)
  return dict(run_id=self.run_id,state=s['phase'],result=s['result'],turns=s['turns'],attempt_reservations=s['attempts'],observations=copy.deepcopy(s['observations']),plans=copy.deepcopy(s['plans']),pending=copy.deepcopy(s['pending']),
              verification=verify(s['observations']),checkpoint_revision=handle['revision'],owner_epoch=handle['epoch'],binding_hash=digest(self.binding),provider_http_attempts=0)
