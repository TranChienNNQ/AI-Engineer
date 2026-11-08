"""Two instrumented ASGI apps and a pre-existing queue worker: explicit W3C propagation."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio
import math
import time
import contextvars
import httpx
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel,ConfigDict,Field,field_validator
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind
from telemetry import Telemetry,extract,inject,error,attrs

MODEL_VERSION='toy-v1';FEATURE_VERSION='features-v1'
API='lesson32-api';MODEL='lesson32-model'


class Features(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    latency_ratio:float=Field(ge=0,le=1)
    error_ratio:float=Field(ge=0,le=1)

    @field_validator('latency_ratio','error_ratio',mode='before')
    @classmethod
    def numeric(cls,value):
        if type(value) not in {int,float}:raise ValueError('numeric required')
        return value


class TraceMiddleware:
    def __init__(self,app,telemetry,service):self.app=app;self.telemetry=telemetry;self.service=service

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        t=self.telemetry;route=scope['path'] if scope['path'] in {'/v1/predict','/score'} else 'unmatched'
        method=scope['method'] if scope['method'] in {'GET','POST','DELETE'} else 'OTHER'
        headers={k.decode('latin1'):v.decode('latin1') for k,v in scope['headers']}
        status=500;started=time.monotonic()
        async def observed_send(message):
            nonlocal status
            if message['type']=='http.response.start':status=message['status']
            await send(message)
        with t.span(self.service,method+' '+route,context=extract(headers),kind=SpanKind.SERVER,
                    attributes=attrs({'app.route':route,'app.method':method,'app.model_version':MODEL_VERSION,
                                      'app.feature_version':FEATURE_VERSION})) as span:
            try:await self.app(scope,receive,observed_send)
            except asyncio.CancelledError:
                error(span,'caller_cancelled');raise
            except Exception:
                error(span,'unexpected');raise
            finally:
                span.set_attribute('app.status_code',status)
                if status>=500:error(span,scope.get('lesson32.error_code','unavailable'))
                t.metrics.completed(self.service,route,status,time.monotonic()-started)
                t.log(self.service,route,status,span)


@dataclass
class Message:
    features:Features
    carrier:dict
    future:asyncio.Future
    waiting_span:object
    enqueued_at:float


class Runtime:
    def __init__(self,telemetry=None,preserve_context=True,gate=None,fail_model=False):
        self.telemetry=telemetry or Telemetry();self.preserve_context=preserve_context
        self.gate=gate;self.fail_model=fail_model;self.queue=asyncio.Queue(maxsize=1)
        self.entered=asyncio.Event();self.worker_task=None;self.client=None;self.seq=0
        self.predictions=[];self.call_count=0  # Bounded fixture workload, not a production store.
        self.model_app=FastAPI();self.api_app=FastAPI(lifespan=self.lifespan)
        self.model_app.add_middleware(TraceMiddleware,telemetry=self.telemetry,service=MODEL)
        self.api_app.add_middleware(TraceMiddleware,telemetry=self.telemetry,service=API)

        @self.api_app.exception_handler(RequestValidationError)
        @self.model_app.exception_handler(RequestValidationError)
        async def invalid(request,exc):return JSONResponse(status_code=422,content={'code':'invalid_input'})

        @self.model_app.post('/score')
        async def score(body:Features):
            with self.telemetry.span(MODEL,'model.forward') as span:
                self.call_count+=1;self.entered.set()
                if self.gate is not None:await self.gate.wait()
                await asyncio.sleep(.002)
                if self.fail_model:
                    error(span,'model_unavailable')
                    raise RuntimeError('SECRET32_EXCEPTION_DO_NOT_EXPORT')
                p=1/(1+math.exp(-(2*body.latency_ratio+3*body.error_ratio-2)))
                return dict(score=p,model_version=MODEL_VERSION,feature_version=FEATURE_VERSION)

        @self.api_app.post('/v1/predict')
        async def predict(body:Features,request:Request):
            if self.queue.full():
                request.scope['lesson32.error_code']='queue_full';raise HTTPException(503,'queue_full')
            t=self.telemetry
            waiting=t.tracers[API].start_span('queue.wait')
            future=asyncio.get_running_loop().create_future()
            with t.span(API,'queue.publish',kind=SpanKind.PRODUCER) as span:
                span.add_event('queue.admitted',attributes={'app.queue_depth':self.queue.qsize()})
                message=Message(body,inject(),future,waiting,time.monotonic())
                self.queue.put_nowait(message)
            result=await asyncio.shield(future)
            if result.get('error'):
                request.scope['lesson32.error_code']='model_unavailable';raise HTTPException(503,result['error'])
            self.seq+=1;prediction_id=f'live-{self.seq:04}'
            # Quality ledger is separate from sampled telemetry; only synthetic features here.
            self.predictions.append(dict(prediction_id=prediction_id,model_version=MODEL_VERSION,
                 feature_version=FEATURE_VERSION,feature_value=body.latency_ratio,score=result['score']))
            return dict(prediction_id=prediction_id,**result)

    @asynccontextmanager
    async def lifespan(self,app):
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.model_app,raise_app_exceptions=False),
                                      base_url='http://model.local')
        # Created before requests. It cannot inherit a future request's context automatically.
        self.worker_task=asyncio.create_task(self.worker(),context=contextvars.Context())
        try:yield
        finally:
            try:await asyncio.wait_for(self.queue.join(),2)
            finally:
                self.worker_task.cancel();await asyncio.gather(self.worker_task,return_exceptions=True)
                while not self.queue.empty():
                    pending=self.queue.get_nowait();pending.waiting_span.end()
                    if not pending.future.done():pending.future.set_result({'error':'shutdown'})
                    self.queue.task_done()
                await self.client.aclose();self.telemetry.close()

    async def worker(self):
        while True:
            message=await self.queue.get()
            try:
                message.waiting_span.end();self.telemetry.metrics.queue_wait.observe(time.monotonic()-message.enqueued_at)
                context=extract(message.carrier) if self.preserve_context else Context()
                with self.telemetry.span(API,'job.process',context=context,kind=SpanKind.CONSUMER) as process:
                    with self.telemetry.span(API,'model.call',kind=SpanKind.CLIENT) as call:
                        try:
                            response=await self.client.post('/score',json=message.features.model_dump(),headers=inject())
                            if response.status_code!=200:
                                error(call,'model_unavailable');error(process,'model_unavailable')
                                result={'error':'model_unavailable'}
                            else:result=response.json()
                        except Exception:
                            error(call,'unexpected');error(process,'unexpected');result={'error':'model_unavailable'}
                if not message.future.done():message.future.set_result(result)
            except asyncio.CancelledError:
                if not message.future.done():message.future.set_result({'error':'shutdown'})
                raise
            finally:self.queue.task_done()


@asynccontextmanager
async def client_for(runtime=None):
    runtime=runtime or Runtime()
    async with runtime.api_app.router.lifespan_context(runtime.api_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.api_app),base_url='http://api.local') as client:
            yield client,runtime
