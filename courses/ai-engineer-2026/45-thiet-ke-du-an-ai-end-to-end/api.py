"""Local API; fixture bearer tokens are not production authentication."""
import sqlite3
from typing import Literal
from fastapi import FastAPI,Request,Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from pydantic import BaseModel,ConfigDict,Field,model_validator,field_validator
from pipeline import Engine
from store import Fault


class Strict(BaseModel):model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
class RunIn(Strict):
    request_key:str=Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    action:Literal['ask','create','note','resolve']='ask'
    service:Literal['gateway','upload','identity','database']='gateway'
    message:str=Field(min_length=1,max_length=1000)
    ticket_id:int|None=Field(default=None,ge=1)
    expected_version:int|None=Field(default=None,ge=1)
    max_steps:int=Field(default=8,ge=1,le=10)
    @field_validator('message')
    @classmethod
    def nonblank(cls,v):
        if not v.strip():raise ValueError('blank message')
        return v
    @model_validator(mode='after')
    def ticket_fields(self):
        if self.action in ['note','resolve']:
            if self.ticket_id is None or self.expected_version is None:raise ValueError('ticket and version required')
        elif self.ticket_id is not None or self.expected_version is not None:raise ValueError('unexpected target fields')
        return self
class CommitIn(Strict):request_key:str=Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
class Citation(Strict):
    doc_id:str;version:int;start:int;end:int;quote:str;content_hash:str
class RunOut(Strict):
    run_id:int;release:str;service:str;state:Literal['answered','abstained','proposed','failed'];code:str;answer:str
    citations:list[Citation];proposal_id:int|None;http_status:int;stats:dict;trace:list[dict];steps:int


def create_app(engine):
    app=FastAPI(title='Lesson 45 support assistant baseline',version='45.0');app.state.engine=engine;bearer=HTTPBearer(auto_error=False)
    def token(credentials:HTTPAuthorizationCredentials|None=Depends(bearer)):
        if credentials is None or credentials.scheme.casefold()!='bearer':raise Fault(401,'unauthorized')
        engine.principal(credentials.credentials);return credentials.credentials
    @app.exception_handler(Fault)
    async def fault_handler(request:Request,e:Fault):return JSONResponse(status_code=e.status,content=dict(code=e.code),headers={'Cache-Control':'no-store'})
    @app.exception_handler(sqlite3.OperationalError)
    async def db_handler(request:Request,e:sqlite3.OperationalError):return JSONResponse(status_code=503,content=dict(code='store_unavailable'))
    @app.exception_handler(RequestValidationError)
    async def invalid(request:Request,e:RequestValidationError):return JSONResponse(status_code=422,content=dict(code='invalid_input',errors=[dict(loc=list(x['loc']),type=x['type']) for x in e.errors()]))
    @app.get('/livez')
    def live():return dict(live=True)
    @app.get('/readyz')
    def ready():
        engine.integrity();return dict(ready=True,release=engine.release)
    @app.post('/v1/runs',response_model=RunOut)
    def run(body:RunIn,access=Depends(token)):
        result=engine.ask(access,body.model_dump());RunOut.model_validate(result)
        return JSONResponse(status_code=result['http_status'],content=result,headers={'Cache-Control':'no-store'})
    @app.get('/v1/runs/{id_}',response_model=RunOut)
    def get_run(id_:int,access=Depends(token)):
        return engine.read_run(access,id_)
    @app.get('/v1/tickets/{id_}')
    def get_ticket(id_:int,access=Depends(token)):
        with engine.store.lock:return engine.store.ticket(id_,engine.principal(access))
    @app.get('/v1/proposals/{id_}')
    def get_proposal(id_:int,access=Depends(token)):
        return engine.read_proposal(access,id_)
    @app.post('/v1/proposals/{id_}/approve')
    def approve(id_:int,access=Depends(token)):return engine.review(access,id_)
    @app.post('/v1/proposals/{id_}/execute')
    def execute(id_:int,body:CommitIn,access=Depends(token)):return engine.commit(access,id_,body.request_key)
    return app
