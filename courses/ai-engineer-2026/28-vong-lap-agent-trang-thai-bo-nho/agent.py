"""Host-owned loop with reserved budgets, persisted checkpoints and read-only recovery."""
import copy
from contracts import VERSION,TERMINAL,decision,digest
from memory import build_view,refresh,required
from world import ToolError
from policy import AdaptivePolicy


class SimulatedCrash(Exception):pass
class Busy(Exception):pass


class Agent:
    def __init__(self,store,world,run_id,actor='alice',policy=None):
        self.store=store;self.world=world;self.run_id=run_id;self.actor=actor;self.policy=policy or AdaptivePolicy()

    def snapshot(self,now=0):
        s=self.store.load(self.run_id,self.actor,self.world)
        refresh(s,self.world,now)
        a=s['answer']
        if a and (now>=a['valid_until'] or not self.world.valid_refs(self.actor,a['refs'])):
            s['answer']=None;s['reason']='answer_stale_or_not_visible'
        if s['pending']:
            s['pending']=dict(kind='tool',name=s['pending']['name'])  # Do not export raw model args from a historical checkpoint.
        return s  # View is revalidated; the historical DB checkpoint is not erased.

    def stop(self,s,status,reason):
        s['status']=status;s['reason']=reason;s['pending']=None
        return self.store.save(s,dict(type='stopped',status=status,reason=reason))

    def step(self,*,now=0,recover=False,crash_after=None):
        s=self.store.load(self.run_id,self.actor,self.world)
        if s['model_id']!=self.policy.model_id or s['schema_version']!=VERSION:raise ValueError('runtime_version_mismatch')
        if s['status'] in TERMINAL:return self.snapshot(now)
        if s['status']=='PENDING_TOOL':
            if not recover:raise Busy('explicit_recovery_required')
            if s['tool_reservations']>=s['limits']['max_calls']:
                self.stop(s,'LIMIT_REACHED','tool_budget');return self.snapshot(now)
            refresh(s,self.world,now);s['tool_reservations']+=1
            s=self.store.save(s,dict(type='tool_recovery_reserved',tool=s['pending']['name']))
            self.execute_pending(s,now,crash_after);return self.snapshot(now)
        if s['status']=='THINKING':
            if not recover:raise Busy('explicit_recovery_required')
            s['status']='READY';s=self.store.save(s,dict(type='model_recovery',note='Old reservation remains charged'))
        refresh(s,self.world,now)
        if s['turns']>=s['limits']['max_turns']:
            self.stop(s,'LIMIT_REACHED','turn_budget');return self.snapshot(now)
        try:view,cost=build_view(s,self.store,self.world,now)
        except ValueError:
            self.stop(s,'LIMIT_REACHED','context_cap');return self.snapshot(now)
        if s['context_units']+cost>s['limits']['max_context_units']:
            self.stop(s,'LIMIT_REACHED','context_budget');return self.snapshot(now)
        s['turns']+=1;s['context_units']+=cost;s['status']='THINKING'
        s=self.store.save(s,dict(type='model_reserved',turn=s['turns'],charged_units=cost))
        if crash_after=='model_reserved':raise SimulatedCrash('after_model_reservation')
        try:d=decision(self.policy.choose(copy.deepcopy(view)))
        except Exception:
            self.stop(s,'FAILED','invalid_model_decision');return self.snapshot(now)
        if d['kind']=='handoff':
            self.stop(s,'HANDOFF',d['reason']);return self.snapshot(now)
        if d['kind']=='final':
            refresh(s,self.world,now)
            refs=[r for o in s['observations'].values() for r in o['refs']]
            expected={r['id'] for r in refs}
            if not all(name in s['observations'] for name in required(s)) or set(d['source_ids'])!=expected:
                self.stop(s,'FAILED','insufficient_or_wrong_evidence');return self.snapshot(now)
            s['answer']=dict(text=d['answer'],source_ids=d['source_ids'],refs=refs,
                             valid_until=min(o['valid_until'] for o in s['observations'].values()),authority='model_output')
            s['status']='DONE';s['reason']=None
            self.store.save(s,dict(type='final',status='DONE',source_count=len(expected)))
            return self.snapshot(now)
        if s['tool_reservations']>=s['limits']['max_calls']:
            self.stop(s,'LIMIT_REACHED','tool_budget');return self.snapshot(now)
        s['tool_reservations']+=1;s['pending']=d;s['status']='PENDING_TOOL'
        s=self.store.save(s,dict(type='tool_reserved',tool=d['name'],reservation=s['tool_reservations']))
        if crash_after=='tool_reserved':raise SimulatedCrash('after_tool_reservation')
        self.execute_pending(s,now,crash_after);return self.snapshot(now)

    def execute_pending(self,s,now,crash_after):
        c=s['pending'];name=c['name']
        try:
            o=self.world.execute(self.actor,s['task'],c,s['observations'],now)
            if crash_after=='tool_result':raise SimulatedCrash('after_read_before_checkpoint')
            s['observations'][name]=o;s['reason']=None
            fingerprint=digest([o['data'],o['refs']]);event=dict(type='observation',tool=name)
        except ToolError as exc:
            s['observations'].pop(name,None);s['reason']=str(exc)
            fingerprint=digest(['error',str(exc)]);event=dict(type='tool_error',tool=name,code=str(exc))
        key=digest([name,c['args']]);previous=s['repeats'].get(key)
        repeats=previous['repeats']+1 if previous and previous['fingerprint']==fingerprint else 0
        s['repeats'][key]=dict(fingerprint=fingerprint,repeats=repeats)
        s['recent']=(s['recent']+[event])[-2:];s['pending']=None
        s['status']='STALLED' if repeats>=s['limits']['max_repeats'] else 'READY'
        if s['status']=='STALLED':s['reason']='repeated_no_new_observation'
        self.store.save(s,event|dict(status=s['status']))

    def run(self,*,now=0,max_steps=30,recover=False):
        if type(max_steps) is not int or max_steps<1:raise ValueError('positive_steps_required')
        for _ in range(max_steps):
            s=self.step(now=now,recover=recover);recover=False
            if s['status'] in TERMINAL:return s
        return self.snapshot(now)  # Bounded driver can return a still-READY checkpoint.

    def cancel(self):
        s=self.store.load(self.run_id,self.actor,self.world)
        if s['status'] not in TERMINAL:self.stop(s,'CANCELLED','controller_cancelled')
