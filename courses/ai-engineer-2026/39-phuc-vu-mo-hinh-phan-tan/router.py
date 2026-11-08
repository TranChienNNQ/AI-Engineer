"""Small synchronous coordinator over real local worker processes.
One physical in-flight slot per worker, no waiting queue, explicit tick/poll, bounded history.
Timeout/cancel finish the client record; slot remains busy until reply or worker loss.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from model import weights,digest,validate,REVISION
from transport import start,stop


@dataclass
class Slot:
    process:object
    conn:object
    ready:dict
    epoch:int=1
    draining:bool=False
    healthy:bool=True
    busy:str|None=None


class Router:
    def __init__(self,replicas=2,max_records=64):
        if type(replicas) is not int or not 1<=replicas<=4 or type(max_records) is not int or max_records<1:raise ValueError('router limits')
        self.w=weights();self.model_hash=digest(self.w);self.slots=[];self.records={};self.max_records=max_records
        self.counters=dict(accepted=0,published=0,late=0,stale_epoch=0,bad_reply=0,timeouts=0,cancelled=0,worker_lost=0)
        try:
            for _ in range(replicas):
                p,c,r=start(self.w,'dp',model_hash=self.model_hash);self.slots.append(Slot(p,c,r))
        except BaseException:self.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        for r in self.records.values():
            if self._terminal(r,'CANCELLED'):self.counters['cancelled']+=1
        for slot in self.slots:stop(slot.process,slot.conn)
        self.slots=[]
    def submit(self,request_id,x,timeout=2.,replica=None,revision=REVISION,hold=False,crash=False,reply_revision=None):
        if not isinstance(request_id,str) or not request_id or request_id in self.records:raise ValueError('unique request id')
        if len(self.records)>=self.max_records:raise RuntimeError('HISTORY_FULL')
        x=validate(x).copy()
        if not math.isfinite(timeout) or timeout<=0 or revision!=REVISION:raise ValueError('deadline/revision')
        self.tick()
        eligible=[i for i,s in enumerate(self.slots) if s.healthy and not s.draining and s.process.is_alive()]
        if replica is not None:
            if type(replica) is not int or replica not in eligible:raise RuntimeError('UNAVAILABLE')
            eligible=[replica]
        available=[i for i in eligible if self.slots[i].busy is None]
        if not available:raise RuntimeError('OVERLOADED' if eligible else 'UNAVAILABLE')
        i=available[0];slot=self.slots[i];deadline=time.monotonic()+timeout
        record=dict(id=request_id,status='RUNNING',replica=i,epoch=slot.epoch,revision=revision,deadline=deadline,rows=len(x),started=False,y=None,publications=0)
        self.records[request_id]=record;slot.busy=request_id;self.counters['accepted']+=1
        try:slot.conn.send(dict(type='RUN',id=request_id,x=x,deadline=deadline,hold=hold,crash=crash,reply_revision=reply_revision))
        except (OSError,EOFError):self._lost(slot)
        return record
    def _terminal(self,r,status,y=None):
        if r['status']!='RUNNING':return False
        r['status']=status;r['y']=None if y is None else np.asarray(y).tolist();r['publications']+=1;self.counters['published']+=1
        return True
    def _lost(self,slot):
        slot.healthy=False
        if slot.busy is not None:
            r=self.records[slot.busy]
            if self._terminal(r,'WORKER_LOST'):self.counters['worker_lost']+=1
            slot.busy=None
    def accept(self,worker_index,msg):
        """Also exposed for a synthetic stale-frame fencing check in fixtures."""
        slot=self.slots[worker_index]
        if msg.get('epoch')!=slot.epoch:self.counters['stale_epoch']+=1;return False
        if msg.get('id')!=slot.busy or msg.get('revision')!=REVISION or msg.get('type') not in ['STARTED','RESULT']:
            self.counters['bad_reply']+=1
            if slot.busy is not None:self._terminal(self.records[slot.busy],'BAD_REPLY')
            slot.busy=None;slot.healthy=False
            if slot.process.is_alive():slot.process.terminate();slot.process.join(2)
            return False
        r=self.records[slot.busy]
        if msg['type']=='STARTED':r['started']=True;return True
        if r['status']!='RUNNING':self.counters['late']+=1
        elif time.monotonic()>=r['deadline'] or msg.get('status')=='EXPIRED':
            if self._terminal(r,'TIMEOUT'):self.counters['timeouts']+=1
        elif msg.get('status')=='OK':
            y=np.asarray(msg.get('y'))
            if y.shape!=(r['rows'],2) or not np.isfinite(y).all():
                self._terminal(r,'BAD_REPLY');self.counters['bad_reply']+=1;slot.healthy=False
            else:self._terminal(r,'OK',y)
        else:
            self._terminal(r,'BAD_REPLY');self.counters['bad_reply']+=1;slot.healthy=False
        slot.busy=None;return True
    def tick(self):
        now=time.monotonic()
        for r in self.records.values():
            if r['status']=='RUNNING' and now>=r['deadline']:
                if self._terminal(r,'TIMEOUT'):self.counters['timeouts']+=1
        for i,s in enumerate(self.slots):
            if s.healthy:
                try:
                    while s.conn.poll():self.accept(i,s.conn.recv())
                except (OSError,EOFError):self._lost(s)
                if not s.process.is_alive():self._lost(s)
    def until(self,predicate,timeout=5.):
        end=time.monotonic()+timeout
        while not predicate():
            self.tick()
            if time.monotonic()>=end:raise TimeoutError('fixture wait timed out')
            time.sleep(.001)
        return True
    def release(self,worker_index):
        s=self.slots[worker_index]
        if s.busy is None or not s.healthy:raise ValueError('no held worker')
        s.conn.send(dict(type='RELEASE',id=s.busy))
    def cancel(self,request_id):
        if self._terminal(self.records[request_id],'CANCELLED'):self.counters['cancelled']+=1
    def drain(self,worker_index):self.slots[worker_index].draining=True
    def restart(self,worker_index):
        s=self.slots[worker_index]
        if s.busy is not None:raise ValueError('cannot replace an occupied slot; finish/lose it first')
        stop(s.process,s.conn);epoch=s.epoch+1
        p,c,r=start(self.w,'dp',epoch=epoch,model_hash=self.model_hash);self.slots[worker_index]=Slot(p,c,r,epoch)
    def snapshot(self):
        return dict(records=[{k:v for k,v in r.items() if k not in ['deadline']} for r in self.records.values()],counters=dict(self.counters),
                    workers=[dict(epoch=s.epoch,healthy=s.healthy,draining=s.draining,busy=s.busy,weight_bytes=s.ready['weight_bytes'],pid=s.ready['pid']) for s in self.slots])
