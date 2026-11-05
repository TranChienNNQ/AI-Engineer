"""Local read-only proposal API. No routing side effect, keys or real model service."""
from uuid import uuid4
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contracts import RouteRequest, Success, ErrorBody
from service import RealClock, Policy, execute, demo_provider, ServiceFailure


def create_app(provider_factory=demo_provider, clock_factory=RealClock, policy=Policy()):
    app=FastAPI(title='Lesson 22: route proposals',version='1.0.0')

    @app.get('/health')
    def health(): return {'status':'ok','provider':'local-rule-demo'}

    @app.post('/v1/route-proposals',response_model=Success,
              responses={502:{'model':ErrorBody},503:{'model':ErrorBody},504:{'model':ErrorBody}})
    def propose(request: RouteRequest):
        request_id=str(uuid4());clock=clock_factory();provider=provider_factory(request,clock)
        try:
            result,_=execute(request,provider,clock,policy,request_id)
            return result
        except ServiceFailure as exc:
            error=ErrorBody(request_id=request_id,error=exc.code,attempts=exc.attempts)
            return JSONResponse(status_code=exc.status,content=error.model_dump())
    return app

app=create_app()
