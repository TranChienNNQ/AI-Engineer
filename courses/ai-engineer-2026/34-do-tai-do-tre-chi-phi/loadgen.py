"""Bounded async load generation with absolute planned times and explicit generator drops."""
import asyncio
from time import perf_counter
from dataclasses import dataclass,asdict
import httpx
from measurements import summarize
from serving import valid_response

PAYLOAD={'latency_ratio':.6,'error_ratio':0.0}


@dataclass(frozen=True)
class Config:
    mode:str='open'
    count:int=24
    rate:float=40.0
    concurrency:int=1
    max_inflight:int=16
    max_lag_ms:float=25.0
    request_timeout_ms:float=500.0
    slo_ms:float=40.0

    def validate(self):
        if self.mode not in {'open','closed'}:raise ValueError('mode')
        for value,lo,hi in [(self.count,1,96),(self.concurrency,1,8),(self.max_inflight,1,32)]:
            if type(value) is not int or not lo<=value<=hi:raise ValueError('bounded integer setting')
        import math
        for value,lo,hi in [(self.rate,1,500),(self.max_lag_ms,0,1000),(self.request_timeout_ms,1,1000),(self.slo_ms,0,1000)]:
            if type(value) not in {int,float} or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('bounded numeric setting')


async def run(client,config,emitted=None):
    config.validate();origin=perf_counter();rows=[]
    def clock():return (perf_counter()-origin)*1000
    async def send(i,planned):
        sent=clock();status='client_error';http_status=None
        if config.mode=='open' and sent-planned>config.max_lag_ms:
            rows.append(dict(id=i,planned_ms=planned,sent_ms=None,done_ms=sent,status='generator_drop',http_status=None))
            return
        try:
            async with asyncio.timeout(config.request_timeout_ms/1000):
                response=await client.post('/v1/predict',json=PAYLOAD)
                http_status=response.status_code;status=http_status
                if http_status==200:
                    try:ok=valid_response(response.json(),PAYLOAD)
                    except (ValueError,TypeError):ok=False
                    if not ok:status='invalid_response'
        except TimeoutError:status='total_timeout'
        except httpx.TimeoutException:status='httpx_timeout'
        except httpx.TransportError:status='transport_error'
        rows.append(dict(id=i,planned_ms=planned,sent_ms=sent,done_ms=clock(),status=status,http_status=http_status))
    if config.mode=='closed':
        indices=iter(range(config.count))
        async def user():
            for i in indices:await send(i,clock())
        await asyncio.gather(*(user() for _ in range(config.concurrency)))
        horizon=0
    else:
        tasks=[];pending=set()
        for i in range(config.count):
            planned=i*1000/config.rate
            await asyncio.sleep(max(0,(planned-clock())/1000))
            now=clock();pending={t for t in pending if not t.done()}
            if len(pending)>=config.max_inflight or now-planned>config.max_lag_ms:
                rows.append(dict(id=i,planned_ms=planned,sent_ms=None,done_ms=now,status='generator_drop',http_status=None))
            else:
                task=asyncio.create_task(send(i,planned));tasks.append(task);pending.add(task)
        if emitted is not None:emitted.set()
        if tasks:await asyncio.gather(*tasks)
        horizon=config.count*1000/config.rate
    window=max(horizon,max(r['done_ms'] for r in rows),.001);rows.sort(key=lambda r:r['id'])
    return dict(config=asdict(config),rows=rows,summary=summarize(rows,window,config.slo_ms),
                window_rule='first planned instant through max(planned horizon, last client terminal); includes drain, excludes warmup')
