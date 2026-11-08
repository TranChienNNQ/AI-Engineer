from typing import Literal
from pydantic import Field,model_validator
from bridge50 import Strict,digest
VERSION='50.1'
PROMPT='''Propose a short typed plan and one next action for read-only technical investigation. A plan is tentative, not evidence or authority. Use only read_ticket, search_runbooks and read_health in the assigned task scope. Steps carry dependencies and observable postconditions. Never mark steps complete yourself. Replan only from observed missing evidence, tool failure or a report/snapshot discrepancy, binding the parent hash and triggering observation IDs. Never change executed steps, fabricate observations, reset budgets or confirm a root cause from health alone. If a search is empty you may change its query using the symptom actually reported in the ticket, once within remaining budget. On a discrepancy or insufficient data, handoff honestly. The host verifies plans, progress, current permissions and final evidence. Return no private chain-of-thought, only structured steps/reason codes/action.'''

class Step(Strict):
 id:str=Field(pattern='^S[1-9][0-9]?$')
 tool:Literal['read_ticket','search_runbooks','read_health']
 args:dict
 depends_on:list[str]=Field(max_length=4)
 postcondition:Literal['ticket_observed','runbook_available','health_known']
class Plan(Strict):
 intent:Literal['investigate','request_clarification']
 revision:int=Field(ge=1,le=4)
 parent_hash:str|None
 trigger:Literal['initial','missing_evidence','tool_failure','discrepancy']
 based_on:list[str]=Field(max_length=8)
 steps:list[Step]=Field(min_length=1,max_length=5)
class Action(Strict):
 kind:Literal['tool','answer','handoff']
 step_id:str|None
 tool:str|None
 args:dict|None
 terminal:dict|None
 conclusion:Literal['guidance','root_cause_confirmed','needs_investigation']
 @model_validator(mode='after')
 def shape(self):
  if self.kind=='tool':
   if self.step_id is None or self.tool is None or self.args is None or self.terminal is not None:raise ValueError('tool action shape')
  elif self.step_id is not None or self.tool is not None or self.args is not None or self.terminal is None:raise ValueError('terminal action shape')
  return self
class Proposal(Strict):
 plan:Plan|None
 action:Action

def signature(step):return digest([step['tool'],step['args']])
def identity():return dict(version=VERSION,prompt=digest(PROMPT),schema=digest(Proposal.model_json_schema()),verifier='closed-fixture-postconditions-v1')
