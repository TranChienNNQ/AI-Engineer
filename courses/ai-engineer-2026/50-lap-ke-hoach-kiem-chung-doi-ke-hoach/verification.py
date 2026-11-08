"""Closed, deterministic checks. Not a general semantic or causal verifier."""
from pydantic import Field
from typing import Literal
from bridge50 import Strict,loads

class TicketReport(Strict):
 symptom:str=Field(min_length=1,max_length=200)
 reported_health:Literal['ok','degraded','unknown']
 report_at:str

def report_from(obs):
 ticket=next((o for o in obs if o['tool']=='read_ticket' and o['ok']),None)
 if ticket is None:return None
 try:return TicketReport.model_validate(loads(ticket['data']['body'])).model_dump()
 except (ValueError,TypeError,RecursionError):return None

def postcondition(observation):
 if not observation['ok']:return 'tool_error'
 if observation['tool']=='read_ticket':return 'satisfied'
 if observation['tool']=='search_runbooks':return 'satisfied' if observation['data']['evidence'] else 'empty'
 if observation['tool']=='read_health':return 'unknown' if observation['data']['status']=='unknown' else 'satisfied'
 return 'unknown'

def verify(obs):
 rows=[]
 for name,tool in [('ticket_observed','read_ticket'),('runbook_available','search_runbooks'),('health_known','read_health')]:
  selected=[o for o in obs if o['tool']==tool]
  # A later successful search can satisfy the obligation; earlier empty result remains in history.
  status='unknown' if not selected else 'supported' if any(postcondition(o)=='satisfied' for o in selected) else postcondition(selected[-1])
  rows.append(dict(check=name,status=status,evidence=[o['id'] for o in selected]))
 rep=report_from(obs);health=next((o for o in reversed(obs) if o['tool']=='read_health' and o['ok']),None)
 consistency='unknown';refs=[]
 if rep and health:
  refs=[next(o['id'] for o in obs if o['tool']=='read_ticket' and o['ok']),health['id']]
  if rep['report_at']!=health['data']['observed_at']:consistency='not_comparable'
  elif rep['reported_health']=='unknown' or health['data']['status']=='unknown':consistency='unknown'
  else:consistency='consistent' if rep['reported_health']==health['data']['status'] else 'discrepancy'
 rows.append(dict(check='report_consistency',status=consistency,evidence=refs))
 rows.append(dict(check='root_cause_established',status='unknown',evidence=[],reason='No causal diagnostic tool exists in this lab.'))
 return rows

def discrepancy(obs):return next(r for r in verify(obs) if r['check']=='report_consistency')['status']=='discrepancy'
