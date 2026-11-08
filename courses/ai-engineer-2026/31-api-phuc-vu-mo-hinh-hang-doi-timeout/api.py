"""Local teaching API. In-memory jobs have no authentication or durable delivery."""
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel,ConfigDict,Field,field_validator
from model import Features
from service import Service,AdmissionError


class PredictIn(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    latency_ratio: Annotated[float,Field(ge=0,le=1)]
    error_ratio: Annotated[float,Field(ge=0,le=1)]
    budget_ms: Annotated[int,Field(ge=10,le=2000)]=500

    @field_validator('latency_ratio','error_ratio',mode='before')
    @classmethod
    def numeric(cls,value):
        if type(value) not in {int,float}:raise ValueError('numeric features required; bool/string rejected')
        return value


def create_app(service=None):
    svc=service or Service()
    @asynccontextmanager
    async def lifespan(app):
        await svc.start()
        try:yield
        finally:await svc.close()

    app=FastAPI(title='Lesson 31 local model service',version='31.0',lifespan=lifespan)
    app.state.service=svc

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request:Request,exc:RequestValidationError):
        # Never echo raw input, including non-finite numbers that JSONResponse cannot serialize.
        return JSONResponse(status_code=422,content=dict(code='invalid_input',
                            errors=[dict(loc=list(e['loc']),type=e['type']) for e in exc.errors()]))

    @app.exception_handler(AdmissionError)
    async def admission_error(request:Request,exc:AdmissionError):
        return JSONResponse(status_code=503,content=dict(code=exc.code),headers={'Retry-After':'1'})

    @app.get('/livez')
    async def live():return dict(live=True)

    @app.get('/readyz')
    async def ready():
        return JSONResponse(status_code=200 if svc.accepting else 503,content=dict(ready=svc.accepting))

    @app.post('/v1/predict')
    async def predict(body:PredictIn):
        job=svc.submit(Features(body.latency_ratio,body.error_ratio),body.budget_ms)
        result=await svc.wait(job)
        if result['state']=='SUCCEEDED':status=200
        elif result['state']=='FAILED':status=500
        else:status=503  # Local unavailable/deadline, not falsely labeled an upstream 504.
        return JSONResponse(status_code=status,content=result,headers={'Cache-Control':'no-store'})

    @app.post('/v1/jobs',status_code=202)
    async def submit_job(body:PredictIn):
        job=svc.submit(Features(body.latency_ratio,body.error_ratio),body.budget_ms)
        return JSONResponse(status_code=202,content=svc.snapshot(job),
                            headers={'Location':f'/v1/jobs/{job.job_id}','Cache-Control':'no-store'})

    @app.get('/v1/jobs/{job_id}')
    async def get_job(job_id:str):
        data=svc.get(job_id)
        if data is None:raise HTTPException(404,'job_unknown_or_evicted')
        return JSONResponse(content=data,headers={'Cache-Control':'no-store'})

    @app.delete('/v1/jobs/{job_id}')
    async def cancel_job(job_id:str):
        data=svc.cancel(job_id)
        if data is None:raise HTTPException(404,'job_unknown_or_evicted')
        return JSONResponse(content=data,headers={'Cache-Control':'no-store'})

    return app


app=create_app()
