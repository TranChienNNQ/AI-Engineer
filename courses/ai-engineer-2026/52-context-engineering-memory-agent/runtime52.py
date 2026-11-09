"""Durable loop using lesson51's State/Journal, with a new explicit context consumer."""
import copy,hashlib,sqlite3,os
from pathlib import Path
from bridge52 import (DurableAgent,restore,canonical,digest,FINAL,operation_id,probe_append,PlanError,Fault,Terminal)
from contracts52 import ContextPolicy
from context52 import compose,validate_bundle,propose,ContextError
from memory52 import MemoryError

def implementation_hash():
 root=Path(__file__).parent
 return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge52.py','contracts52.py','memory52.py','context52.py','runtime52.py','worker.py']})

def selection_id(recalled):return digest([dict(id=m['id'],payload=m['payload'],refs=m['refs'],expires_at=m['expires_at']) for m in recalled['items']])

def record_context(path,run_id,turn,bundle):
 db=sqlite3.connect(path,isolation_level=None)
 try:
  db.execute('CREATE TABLE IF NOT EXISTS context_receipts(n INTEGER PRIMARY KEY,run_id TEXT,turn INTEGER,receipt TEXT)')
  db.execute('INSERT INTO context_receipts(run_id,turn,receipt) VALUES(?,?,?)',(run_id,turn,canonical(bundle['receipt'])))
 finally:db.close()

class ContextAgent(DurableAgent):
 def __init__(self,*args,memory,policy=None,now=100,**kwargs):
  super().__init__(*args,**kwargs);self.memory=memory;self.policy=ContextPolicy.model_validate(policy or {}).model_dump();self.now=now
  self.principal=self.world.principal(self.token);recalled=self.recall()
  self.binding.update(schema52='52.1',implementation52=implementation_hash(),context_policy=self.policy,memory_epoch=recalled['epoch'],memory_selection=selection_id(recalled))
 def recall(self):return self.memory.recall(self.principal,self.task['service'],self.task['question'],self.now)
 def current(self,state):
  super().current(state)
  if self.binding['implementation52']!=implementation_hash():raise MemoryError('context_release_changed')
  recalled=self.recall()
  if recalled['epoch']!=self.binding['memory_epoch'] or selection_id(recalled)!=self.binding['memory_selection']:raise MemoryError('memory_snapshot_changed')
 def run(self,owner,now=100,takeover_epoch=None,pause_after=None,crash_after_intent=False,before_consume=None):
  self.now=now;initial=self.journal.load(self.run_id,self.actor)
  if initial['header']['binding']!=canonical(self.binding):raise MemoryError('run_binding_changed')
  s=initial['state'];self.current(s);handle=self.journal.claim(self.run_id,self.actor,self.binding,owner,takeover_epoch)
  def save(kind,release=False):
   nonlocal handle
   handle=self.journal.save(self.run_id,self.actor,handle,s,kind,release)
  def stop(state,code):s.update(phase=state,pending=None,result=dict(state=state,code=code,answer='',citations=[]));save('finished',True)
  try:
   if s['phase'] in FINAL:save('terminal_read',True);return self.export(s,handle)
   if now>=self.expires_at:stop('stopped','run_expired');return self.export(s,handle)
   if s['phase'] in ['planning','paused']:s['phase']='ready';save('resumed')
   while s['phase'] not in FINAL:
    self.current(s)
    if s['phase']=='ready':
     if s['turns']>=self.limits['turns']:stop('stopped','turn_budget');break
     s['turns']+=1;s['phase']='planning';save('planning_reserved')
     g=restore(s,self.limits,self.task);recalled=self.recall();bundle=compose(self.task,s,g,recalled,self.policy)
     if before_consume:before_consume(bundle,s)
     self.current(s);validate_bundle(bundle,self.task,s,g,self.recall(),self.policy)
     if self.probe:record_context(self.probe,self.run_id,s['turns'],bundle)
     # The planner gets only the projected packet. Full observations stay behind host gates.
     proposal=g.authorize(propose(bundle['packet']),s['observations'],self.task);s['plans']=copy.deepcopy(g.history);action=proposal['action']
     self.current(s)
     if action['kind']!='tool':
      result=self.gate.finish(Terminal.model_validate(action['terminal']),self.task,self.token,s['observations']);s.update(phase=result['state'],result=result,pending=None);save('finished',True);break
     s.update(phase='pending',pending=dict(op_id=operation_id(self.run_id,action['step_id']),action=action,attempts=0));save('intent_committed')
     if crash_after_intent:os._exit(83) # Trusted subprocess test hook only.
    pending=s['pending'];action=pending['action'];g=restore(s,self.limits,self.task);g.authorize(dict(plan=None,action=action),s['observations'],self.task)
    if pending['op_id']!=operation_id(self.run_id,action['step_id']):raise ContextError('operation_identity')
    if pending['attempts']>=self.limits['per_operation'] or s['attempts']>=self.limits['attempts']:stop('stopped','tool_attempt_budget');break
    pending['attempts']+=1;s['attempts']+=1;save('attempt_reserved');self.current(s)
    data=self.world.dispatch(action['tool'],action['args'],self.task,self.token)
    if self.probe:probe_append(self.probe,self.run_id,pending['op_id'],pending['attempts'],handle['epoch'],data)
    s['observations'].append(dict(id=f"O{len(s['observations'])+1}",call_id=pending['op_id'],tool=action['tool'],args=action['args'],**data,durable_provenance=dict(op_id=pending['op_id'],attempt=pending['attempts'],epoch=handle['epoch'])))
    s.update(phase='ready',pending=None);save('observation_committed')
    if pause_after==len(s['observations']):s['phase']='paused';save('paused',True);break
  except ContextError as e:stop('failed',str(e))
  except PlanError as e:stop('failed','plan_'+str(e))
  except Fault as e:stop('blocked' if e.status in [401,403,404] else 'failed',e.code)
  self.current(s);return self.export(s,handle)
 def export(self,state,handle):
  return super().export(state,handle)|dict(context_policy=self.policy,memory_epoch=self.binding['memory_epoch'],scope='scripted planner consumes an explicit packet; real MCP reads; durable local journal and scoped reviewed memory; no LLM')
