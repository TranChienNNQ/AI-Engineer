"""Synthetic ticket service + trusted review controller, all in one process/RAM.
Fault timings are modeled milliseconds, NOT wall-clock network measurements.
"""
import copy
from dataclasses import dataclass
import secrets
from threading import RLock
from contracts import digest


class ToolError(Exception):
    def __init__(self,code,retryable=False,elapsed_ms=0,uncertain=False):
        super().__init__(code);self.code=code;self.retryable=retryable
        self.elapsed_ms=elapsed_ms;self.uncertain=uncertain


@dataclass
class Fault:
    code:str
    elapsed_ms:int=20


class Backend:
    def __init__(self):
        self.lock=RLock()
        self.people={'ops_blue':dict(tenant='blue',write=True,active=True,revision=1),
                     'reader_blue':dict(tenant='blue',write=False,active=True,revision=1),
                     'ops_green':dict(tenant='green',write=True,active=True,revision=1)}
        self.tickets={
            'T-BLUE-1':dict(tenant='blue',service='checkout',category='technical',version=1,notes=[]),
            'T-BLUE-2':dict(tenant='blue',service='billing',category='billing',version=1,notes=[]),
            'T-GREEN-1':dict(tenant='green',service='checkout',category='technical',version=1,notes=[])}
        self.kb={('blue','checkout'):[dict(id='KB-BLUE-RETRY',version=1,text='Chỉ retry lỗi tạm thời; kiểm tra idempotency trước khi retry write.')],
                 ('green','checkout'):[dict(id='KB-GREEN',version=1,text='GREEN_PRIVATE_27: runbook riêng tenant green.')]}
        self.health={('blue','checkout'):'degraded',('green','checkout'):'healthy'}
        self.faults={};self.grants={};self.operations={};self.effects=0
        self.latency={'get_ticket':40,'lookup_runbook':80,'get_health':120,'append_note':60}

    def principal(self,subject):
        p=self.people.get(subject)
        if p is None or not p['active']:raise ToolError('not_authenticated')
        return p

    def ticket(self,p,tid):
        r=self.tickets.get(tid)
        if r is None or r['tenant']!=p['tenant']:raise ToolError('not_permitted')
        return r

    def set_faults(self,name,*codes):
        self.faults[name]=[Fault(code) if isinstance(code,str) else code for code in codes]

    def source_refs(self,subject,service):
        p=self.principal(subject)
        return [dict(id=s['id'],version=s['version'],body_hash=digest(s['text'])) for s in self.kb.get((p['tenant'],service),[])]

    def issue_review(self,subject,write_call,source_refs,now=0,ttl=500):
        """Trusted coordinator only, not an available model tool; UI/auth intent are assumed."""
        if type(ttl) is not int or ttl<=0:raise ValueError('positive_ttl_required')
        with self.lock:
            p=self.principal(subject);a=write_call['arguments'];r=self.ticket(p,a['ticket_id'])
            if write_call['name']!='append_note' or not p['write']:raise ToolError('not_permitted')
            if r['version']!=a['expected_version']:raise ToolError('stale_version')
            if not source_refs or source_refs!=self.source_refs(subject,r['service']):raise ToolError('stale_evidence')
            token=secrets.token_urlsafe(24)
            self.grants[token]=dict(subject=subject,tenant=p['tenant'],revision=p['revision'],
                request_hash=digest([write_call['name'],a]),source_refs=copy.deepcopy(source_refs),expires=now+ttl,used=False)
            return token

    def execute(self,subject,c,*,key=None,grant=None,now=0):
        with self.lock:
            p=self.principal(subject);name=c['name'];a=c['arguments']
            if name in {'get_ticket','append_note'}:r=self.ticket(p,a['ticket_id'])
            else:
                if not any(t['tenant']==p['tenant'] and t['service']==a['service'] for t in self.tickets.values()):raise ToolError('not_permitted')
            fault=self.faults.get(name,[])
            f=fault.pop(0) if fault else None
            if f and f.code in {'unavailable','timeout_before'}:
                # Simulator knows no effect occurred; real remote timeout need not have this knowledge.
                raise ToolError(f.code,retryable=True,elapsed_ms=f.elapsed_ms,uncertain=name=='append_note')
            elapsed=self.latency[name]
            if name=='get_ticket':data=dict(ticket_id=a['ticket_id'],service=r['service'],category=r['category'],version=r['version'])
            elif name=='lookup_runbook':data=dict(sources=copy.deepcopy(self.kb.get((p['tenant'],a['service']),[])))
            elif name=='get_health':data=dict(service=a['service'],state=self.health.get((p['tenant'],a['service']),'unknown'))
            elif name=='append_note':
                if not p['write']:raise ToolError('not_permitted')
                if not isinstance(key,str) or not key:raise ToolError('idempotency_key_required')
                request_hash=digest([name,a]);slot=(subject,p['tenant'],key)
                previous=self.operations.get(slot)
                if previous:
                    if previous['request_hash']!=request_hash:raise ToolError('idempotency_conflict')
                    return copy.deepcopy(previous['result']),elapsed
                g=self.grants.get(grant) if isinstance(grant,str) else None
                if not g or g['used'] or g['subject']!=subject or g['tenant']!=p['tenant'] or g['revision']!=p['revision'] or now>=g['expires'] or g['request_hash']!=request_hash:
                    raise ToolError('review_required')
                if a['expected_version']!=r['version']:raise ToolError('stale_version')
                if g['source_refs']!=self.source_refs(subject,r['service']):raise ToolError('stale_evidence')
                # No await/network between effect and operation receipt in this in-memory lock.
                note_id=f'N-{self.effects+1}';r['notes'].append(dict(id=note_id,text=a['note']))
                r['version']+=1;g['used']=True;self.effects+=1
                data=dict(ticket_id=a['ticket_id'],note_id=note_id,version=r['version'])
                self.operations[slot]=dict(request_hash=request_hash,result=copy.deepcopy(data))
                if f and f.code=='timeout_after':raise ToolError('response_lost',elapsed_ms=elapsed,uncertain=True)
            else:raise ToolError('unknown_tool')
            if f and f.code=='bad_output':data={'unexpected':'invalid contract'}
            return data,elapsed

    def reconcile(self,subject,c,key):
        """Authoritative local receipt lookup. Missing receipt stays unknown; no new write."""
        with self.lock:
            p=self.principal(subject);self.ticket(p,c['arguments']['ticket_id'])
            if not p['write']:raise ToolError('not_permitted')
            entry=self.operations.get((subject,p['tenant'],key))
            if entry is None:return None
            if entry['request_hash']!=digest([c['name'],c['arguments']]):raise ToolError('idempotency_conflict')
            return copy.deepcopy(entry['result'])
