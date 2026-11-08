"""Bounded toy inference service, with cooperative fake delay and real numeric forward."""
import asyncio
import math
from time import perf_counter
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field

RELEASE='toy-load-v1'


class Features(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    latency_ratio:float=Field(ge=0,le=1,allow_inf_nan=False)
    error_ratio:float=Field(ge=0,le=1,allow_inf_nan=False)


def forward(latency,error):return 1/(1+math.exp(-(4*latency+error-2)))


class Service:
    def __init__(self,workers=2,queue_cap=4,delay=.015,gate=None):
        if type(workers) is not int or workers<1 or type(queue_cap) is not int or queue_cap<0 or not 0<=delay<=1:raise ValueError('config')
        self.workers=workers;self.queue_cap=queue_cap;self.delay=delay;self.gate=gate
        self.semaphore=asyncio.Semaphore(workers);self.entered=asyncio.Event()
        self.outstanding=0;self.active=0;self.peak_active=0;self.peak_waiting=0;self.rows=[]

    def reset(self):
        if self.outstanding:raise ValueError('reset while busy')
        self.rows=[];self.peak_active=0;self.peak_waiting=0

    async def idle(self):
        async with asyncio.timeout(2):
            while self.outstanding:await asyncio.sleep(.001)

    async def predict(self,features):
        entered=perf_counter()
        if self.outstanding>=self.workers+self.queue_cap:
            self.rows.append(dict(status=503,queue_ms=None,run_ms=None))
            return JSONResponse({'code':'overloaded'},status_code=503)
        self.outstanding+=1;status='cancelled';queue_ms=run_ms=None
        self.peak_waiting=max(self.peak_waiting,max(0,self.outstanding-self.workers))
        try:
            async with self.semaphore:
                start=perf_counter();queue_ms=(start-entered)*1000
                self.active+=1;self.peak_active=max(self.peak_active,self.active);self.entered.set()
                try:
                    if self.gate is not None:await self.gate.wait()
                    await asyncio.sleep(self.delay)
                    probability=forward(features.latency_ratio,features.error_ratio)
                    run_ms=(perf_counter()-start)*1000;status=200
                    return dict(release=RELEASE,score=probability,priority=int(probability>=.5),server_queue_ms=queue_ms,server_run_ms=run_ms)
                finally:self.active-=1
        finally:
            self.outstanding-=1;self.rows.append(dict(status=status,queue_ms=queue_ms,run_ms=run_ms))


def create_app(service):
    app=FastAPI()
    @app.post('/v1/predict')
    async def predict(features:Features):return await service.predict(features)
    return app


def valid_response(value,payload):
    if not isinstance(value,dict) or set(value)!={'release','score','priority','server_queue_ms','server_run_ms'}:return False
    if value['release']!=RELEASE or type(value['priority']) is not int:return False
    for name in ['score','server_queue_ms','server_run_ms']:
        if type(value[name]) not in {int,float} or not math.isfinite(value[name]) or value[name]<0:return False
    expected=forward(payload['latency_ratio'],payload['error_ratio'])
    return abs(value['score']-expected)<1e-12 and value['priority']==int(expected>=.5)
