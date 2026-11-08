"""Deterministic inter-process barriers exercise failure paths without latency benchmarks."""
import os
import numpy as np
from router import Router
from model import forward,weights,REVISION


def run():
    checks=[]
    def check(name,ok):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    def rejected(name,fn,error,text=None):
        try:fn()
        except error as e:
            if text is not None and text not in str(e):raise
            check(name,True);return
        raise AssertionError(name+' accepted')
    x=np.array([[1,2,0,-1]],dtype=np.float32);ref=forward(x,weights()).tolist();summaries=[]
    with Router() as r:
        check('two actual child processes',len({s.ready['pid'] for s in r.slots})==2 and all(s.ready['pid']!=os.getpid() for s in r.slots))
        check('both ready model hashes',all(s.ready['model_hash']==r.model_hash and s.ready['revision']==REVISION for s in r.slots))
        a=r.submit('A',x,hold=True);b=r.submit('B',x,hold=True)
        r.until(lambda:a['started'] and b['started'])
        check('two replicas own independent requests',(a['replica'],b['replica'])==(0,1))
        rejected('bounded physical admission',lambda:r.submit('overflow',x),RuntimeError,'OVERLOADED')
        rejected('duplicate logical id',lambda:r.submit('A',x),ValueError)
        r.release(0);r.release(1);r.until(lambda:a['status']!='RUNNING' and b['status']!='RUNNING')
        check('replica numerical reference',a['y']==b['y']==ref)
        check('single outcome per request',a['publications']==b['publications']==1)
        summaries.append(dict(name='replica_admission',statuses=[a['status'],b['status']],rejected='OVERLOADED'))
        # Timeout is real monotonic time; held worker cannot complete until explicitly released.
        t=r.submit('timeout',x,timeout=.03,replica=0,hold=True)
        r.until(lambda:t['status']=='TIMEOUT')
        check('timeout keeps physical slot',r.slots[0].busy=='timeout')
        rejected('cannot overbook timed-out worker',lambda:r.submit('blocked',x,replica=0),RuntimeError,'OVERLOADED')
        r.release(0);r.until(lambda:r.slots[0].busy is None)
        check('late timeout response fenced',t['status']=='TIMEOUT' and t['publications']==1 and r.counters['late']==1)
        summaries.append(dict(name='timeout_late',status=t['status'],slot_held_until_reply=True,late_count=r.counters['late']))
        c=r.submit('cancel',x,replica=0,hold=True);r.until(lambda:c['started']);r.cancel('cancel')
        check('cancel keeps physical slot',r.slots[0].busy=='cancel')
        r.release(0);r.until(lambda:r.slots[0].busy is None)
        check('late cancel result ignored',c['status']=='CANCELLED' and c['publications']==1 and r.counters['late']==2)
        summaries.append(dict(name='cancel_late',status=c['status'],slot_held_until_reply=True))
        d=r.submit('draining',x,replica=0,hold=True);r.until(lambda:d['started']);r.drain(0)
        other=r.submit('other',x);check('drain routes new work elsewhere',other['replica']==1)
        r.release(0);r.until(lambda:d['status']=='OK' and other['status']=='OK')
        check('drain finishes existing job',r.slots[0].draining and d['y']==ref)
        rejected('draining worker rejects admission',lambda:r.submit('not-on-drain',x,replica=0),RuntimeError,'UNAVAILABLE')
        r.restart(0);check('restart increments epoch',r.slots[0].epoch==2 and not r.slots[0].draining)
        fenced=r.accept(0,dict(type='RESULT',id='draining',epoch=1,revision=REVISION,status='OK',y=x))
        check('synthetic stale epoch frame ignored',not fenced and r.counters['stale_epoch']==1 and r.slots[0].busy is None)
        summaries.append(dict(name='drain_restart',old_status=d['status'],new_epoch=r.slots[0].epoch,synthetic_stale_frame_ignored=True))
        lost=r.submit('crash',x,replica=0,crash=True);r.until(lambda:lost['status']=='WORKER_LOST')
        check('process loss is observed',not r.slots[0].healthy and lost['publications']==1)
        surviving=r.submit('survives',x);r.until(lambda:surviving['status']=='OK')
        check('replica failure does not require all workers',surviving['replica']==1 and surviving['y']==ref)
        r.restart(0);check('crashed worker restored through readiness',r.slots[0].healthy and r.slots[0].epoch==3)
        summaries.append(dict(name='worker_crash',status=lost['status'],survivor_status=surviving['status'],replacement_epoch=r.slots[0].epoch))
        bad=r.submit('bad-revision',x,replica=0,reply_revision='mlp-v0');r.until(lambda:bad['status']=='BAD_REPLY')
        check('wrong revision output not published as success',bad['y'] is None and bad['publications']==1 and not r.slots[0].healthy)
        r.restart(0)
        rejected('request revision pin enforced',lambda:r.submit('wrong-pin',x,revision='mlp-v0'),ValueError)
        rejected('nonfinite input rejected',lambda:r.submit('nan',[[1,2,float('nan'),4]]),ValueError)
        rejected('invalid timeout rejected',lambda:r.submit('negative-timeout',x,timeout=0),ValueError)
        rejected('restart while busy forbidden',lambda:(r.submit('held',x,replica=0,hold=True),r.restart(0)),ValueError)
        r.until(lambda:r.records['held']['started']);r.release(0);r.until(lambda:r.records['held']['status']=='OK')
        check('bounded history cap is explicit',len(r.records)<=r.max_records)
        check('no occupied slots at end',all(s.busy is None for s in r.slots))
        summaries.append(dict(name='revision_guard',status=bad['status'],replacement_epoch=r.slots[0].epoch))
        snapshot=r.snapshot();pids=[s.process for s in r.slots]
    check('children joined after context',all(not p.is_alive() for p in pids))
    with Router(replicas=1,max_records=1) as r:
        one=r.submit('only',x);r.until(lambda:one['status']=='OK')
        rejected('history bounded independent of inflight',lambda:r.submit('new',x),RuntimeError,'HISTORY_FULL')
    closing=Router(replicas=1);pending=closing.submit('closing',x,hold=True);closing.until(lambda:pending['started']);process=closing.slots[0].process;closing.close()
    check('shutdown terminates pending client record',pending['status']=='CANCELLED' and pending['publications']==1)
    check('shutdown joins held child',not process.is_alive())
    return dict(checks=len(checks),passed=len(checks),names=checks,scenarios=summaries,snapshot=snapshot,
                children_joined=True,scope='actual local spawn/Pipe workers; hold/crash/reply override are fault fixtures; stale epoch frame is synthetic; no HTTP or automatic retry')
