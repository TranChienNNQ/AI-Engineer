import copy
from bridge import Engine45,Fault,digest,retrieve,validate
from agent_contract import Task

HEALTH=dict(gateway=dict(status='degraded',observed_at='2026-11-08T02:00:00Z',detail='Synthetic snapshot: upstream latency elevated; root cause not established.'),database=dict(status='ok',observed_at='2026-11-08T02:00:00Z',detail='Synthetic snapshot, not production telemetry.'))

class World:
    def __init__(self,path,health_status='degraded',health_down=False,search_down=False):
        if health_status not in ['degraded','ok','unknown']:raise ValueError('health status')
        self.core=Engine45(path);self.health=copy.deepcopy(HEALTH);self.health['gateway']['status']=health_status;self.health_down=health_down;self.search_down=search_down;self.actual_reads=0
    def close(self):self.core.close()
    def principal(self,token):return self.core.principal(token)
    def bind(self,token,raw):
        task=Task.model_validate(raw).model_dump();p=self.principal(token);self.core.integrity()
        self.core.store.ticket(task['ticket_id'],p) # Authorization preflight, not a tool observation.
        if {1:'gateway',2:'database'}.get(task['ticket_id'])!=task['service']:raise Fault(403,'task_service_mismatch')
        return task
    def dispatch(self,name,args,task,token):
        p=self.principal(token);self.core.integrity()
        if name=='read_ticket':
            if args['ticket_id']!=task['ticket_id']:raise Fault(403,'target_scope_denied')
            self.actual_reads+=1;t=self.core.store.ticket(args['ticket_id'],p)
            return dict(ok=True,data={k:t[k] for k in ['id','title','body','status','version']},code='ok')
        if name=='search_runbooks':
            self.actual_reads+=1
            if self.search_down:return dict(ok=False,data=None,code='unavailable')
            evidence,stats=retrieve(self.core.docs,p,task['service'],args['query'],self.core.manifest['min_terms'],self.core.manifest['context_chars'],self.core.revoked)
            return dict(ok=True,data=dict(evidence=evidence),code='ok')
        if name=='read_health':
            if task['goal']!='triage' or p['role'] not in ['agent','reviewer']:raise Fault(403,'health_scope_denied')
            self.actual_reads+=1
            if self.health_down:return dict(ok=False,data=None,code='unavailable')
            snap=copy.deepcopy(self.health[task['service']]);snap['service']=task['service'];snap['content_hash']=digest(snap)
            return dict(ok=True,data=snap,code='ok')
        raise Fault(403,'unknown_tool')
    def current(self,token,task,observations):
        p=self.principal(token);self.core.integrity();self.core.store.ticket(task['ticket_id'],p)
        for obs in observations:
            if not obs['ok']:continue
            if obs['tool']=='search_runbooks':
                e=obs['data']['evidence']
                if e and not validate(e,e,self.core.docs,p,task['service'],self.core.revoked):raise Fault(409,'source_no_longer_allowed')
            if obs['tool']=='read_health':
                if p['role'] not in ['agent','reviewer']:raise Fault(403,'health_scope_denied')
                snap=copy.deepcopy(self.health[task['service']]);snap['service']=task['service']
                if digest(snap)!=obs['data']['content_hash']:raise Fault(409,'health_snapshot_changed')
        return p
    def effects(self):
        db=self.core.store.db
        return {k:db.execute('SELECT COUNT(*) FROM '+k).fetchone()[0] for k in ['tickets','runs','proposals','receipts','audit']}
