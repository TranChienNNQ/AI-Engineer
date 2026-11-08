"""Deterministic discrete-event FIFO, modeled milliseconds; not wall-clock performance."""
import heapq
import math
from fractions import Fraction
from collections import deque
from measurements import summarize


def simulate(arrivals_ms,service_ms=20,workers=2,queue_cap=4,slo_ms=40,horizon_ms=0):
    if type(workers) is not int or workers<1 or type(queue_cap) is not int or queue_cap<0:raise ValueError('capacity')
    if any(type(v) not in {int,float,Fraction} or not math.isfinite(v) or v<0 for v in [service_ms,slo_ms,horizon_ms]+list(arrivals_ms)) or service_ms==0 or list(arrivals_ms)!=sorted(arrivals_ms):raise ValueError('time')
    def exact(v):return v if isinstance(v,Fraction) else Fraction(str(v))
    arrivals_ms=[exact(v) for v in arrivals_ms];service_ms=exact(service_ms);slo_ms=exact(slo_ms);horizon_ms=exact(horizon_ms)
    events=[(t,1,i) for i,t in enumerate(arrivals_ms)];heapq.heapify(events)
    rows=[dict(id=i,planned_ms=t,sent_ms=t,done_ms=None,status=None,start_ms=None) for i,t in enumerate(arrivals_ms)]
    queue=deque();active=0;peak_queue=0;busy_ms=0
    def start(i,t):
        nonlocal active,busy_ms
        active+=1;busy_ms+=service_ms;rows[i]['start_ms']=t
        heapq.heappush(events,(t+service_ms,0,i))
    while events:
        t,kind,i=heapq.heappop(events)
        if kind==0:
            rows[i].update(done_ms=t,status=200);active-=1
            if queue:start(queue.popleft(),t)
        elif active<workers:start(i,t)
        elif len(queue)<queue_cap:queue.append(i);peak_queue=max(peak_queue,len(queue))
        else:rows[i].update(done_ms=t,status=503)
    window=max([r['done_ms'] for r in rows]+[1,horizon_ms])
    window=max(window,1)
    result=dict(kind='deterministic rational-time simulation; completion before arrival at same timestamp',
                config=dict(service_ms=service_ms,workers=workers,queue_cap=queue_cap,slo_ms=slo_ms),
                rows=rows,summary=summarize(rows,window,slo_ms),peak_queue=peak_queue,
                worker_utilization=busy_ms/(workers*window))
    def serializable(value):
        if isinstance(value,Fraction):return float(value)
        if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
        if isinstance(value,list):return [serializable(v) for v in value]
        return value
    return serializable(result)


def scenarios():
    sweep={str(rate):simulate([Fraction(i*1000,rate) for i in range(40)],horizon_ms=Fraction(40000,rate)) for rate in [20,60,100,140,200]}
    return dict(sweep=sweep,
                closed_one=simulate([i*100 for i in range(10)],service_ms=100,workers=1,queue_cap=20,slo_ms=200),
                open_same_demand=simulate([i*20 for i in range(10)],service_ms=100,workers=1,queue_cap=20,slo_ms=200),
                bounded_burst=simulate([0]*8,service_ms=20,workers=2,queue_cap=2,slo_ms=30))
