from typing import Literal
from pydantic import Field,model_validator
from bridge51 import Strict,digest
VERSION='51.1'
FINAL={'answered','handoff','failed','blocked','stopped'}
class Limits(Strict):
 turns:int=Field(default=8,ge=1,le=12)
 attempts:int=Field(default=6,ge=0,le=12)
 per_operation:int=Field(default=2,ge=1,le=3)
 revisions:int=Field(default=3,ge=1,le=4)
class Pending(Strict):
 op_id:str
 action:dict
 attempts:int=Field(ge=0,le=3)
class State(Strict):
 version:str=VERSION
 phase:Literal['ready','planning','pending','paused','answered','handoff','failed','blocked','stopped']='ready'
 turns:int=Field(default=0,ge=0,le=12)
 attempts:int=Field(default=0,ge=0,le=12)
 plans:list[dict]=Field(default_factory=list,max_length=4)
 observations:list[dict]=Field(default_factory=list,max_length=6)
 pending:Pending|None=None
 result:dict|None=None
 @model_validator(mode='after')
 def invariants(self):
  if self.version!=VERSION:raise ValueError('schema version')
  if (self.phase=='pending')!=(self.pending is not None):raise ValueError('pending phase')
  if (self.phase in FINAL)!=(self.result is not None):raise ValueError('terminal phase')
  if self.result and self.result.get('state')!=self.phase:raise ValueError('result phase')
  if [o.get('id') for o in self.observations]!=[f'O{i+1}' for i in range(len(self.observations))]:raise ValueError('observation sequence')
  return self

def canonical(x):
 import json
 return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def operation_id(run_id,step_id):return run_id+':'+step_id

def recovery_rule(tool):
 # Policy belongs to this application, not inferred from a model/MCP annotation.
 return 'bounded_read_retry' if tool in ['read_ticket','search_runbooks','read_health'] else 'reconcile_or_review'
