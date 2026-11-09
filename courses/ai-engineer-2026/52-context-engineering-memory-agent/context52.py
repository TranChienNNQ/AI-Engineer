"""Deterministic projection/packing, byte accounting, and source-linked working summary."""
import copy
from bridge52 import digest,loads,verify,PlannerScript,TOOLS,pc
from contracts52 import INSTRUCTIONS,ContextPolicy,byte_count,identity
class ContextError(ValueError):pass

def project(obs):
 out=[]
 for o in obs:
  q={k:copy.deepcopy(o[k]) for k in ['id','tool','args','ok','code']}
  if not o['ok']:q['data']=None
  elif o['tool']=='read_ticket':q['data']=dict(body=o['data']['body'])
  elif o['tool']=='search_runbooks':q['data']=dict(evidence=copy.deepcopy(o['data']['evidence']))
  else:q['data']={k:copy.deepcopy(o['data'][k]) for k in ['status','observed_at']}
  out.append(dict(observation=q,origin=dict(observation_id=o['id'],content_hash=digest(o))))
 return out

def compose(task,state,guard,recall,policy=None):
 policy=ContextPolicy.model_validate(policy or {}).model_dump();available=policy['window_bytes']-policy['reserve_bytes']
 packet=dict(version='52.1',instructions=INSTRUCTIONS,tool_contracts=copy.deepcopy(TOOLS),response_contract=pc.Proposal.model_json_schema(),task=copy.deepcopy(task),plan=guard.current(),plan_hash=guard.history[-1]['hash'] if guard.history else None,
             step_states=guard.states(state['observations']),checks=verify(state['observations']),counters=dict(turns=state['turns'],attempts=state['attempts']),
             observation_mode=policy['mode'],working=project(state['observations']) if policy['mode']=='compact' else copy.deepcopy(state['observations']),memory_data=[])
 mandatory=byte_count(packet)
 if mandatory>available:raise ContextError('required_context_budget')
 omitted=[]
 for item in recall['items']:
  if len(packet['memory_data'])>=policy['max_memories']:omitted.append(dict(id=item['id'],reason='memory_count_budget'));continue
  candidate=packet|dict(memory_data=packet['memory_data']+[copy.deepcopy(item)])
  if byte_count(candidate)>available:omitted.append(dict(id=item['id'],reason='memory_byte_budget'))
  else:packet=candidate
 return dict(packet=packet,epoch=recall['epoch'],receipt=dict(context_hash=digest(packet),input_bytes=byte_count(packet),mandatory_bytes=mandatory,available_bytes=available,
                  memory_ids=[m['id'] for m in packet['memory_data']],omitted=omitted,policy=digest(policy),compiler=identity()))

def validate_bundle(bundle,task,state,guard,recall,policy):
 if compose(task,state,guard,recall,policy)!=bundle:raise ContextError('context_projection_changed')

def propose(packet):
 # This scripted consumer sees ONLY the composed packet, not raw hidden host state.
 obs=[x['observation'] for x in packet['working']] if packet['observation_mode']=='compact' else packet['working']
 view=dict(instructions=packet['instructions'],task=packet['task'],observations=obs,plan=packet['plan'],plan_hash=packet['plan_hash'],step_states=packet['step_states'],checks=packet['checks'])
 proposal=PlannerScript().propose(view)
 if packet['plan'] is None:
  hint=next((m for m in packet['memory_data'] if m['kind']=='hint'),None)
  if hint:proposal['plan']['steps'][1]['args']['query']=hint['payload']['query']
 return proposal
