import copy,json,re
from typing import Literal
from pydantic import Field,model_validator
from jsonschema import Draft202012Validator
from bridge49 import Strict,Fault,digest
VERSION='49.1'
ALIAS='support_a'
MAP={'read_ticket':'ticket_read','search_runbooks':'runbook_search','read_health':'service_health'}
DESCRIPTIONS={
 'ticket_read':'Read only the ticket assigned to this deployment task. Returns its current version; never modifies or resolves it.',
 'runbook_search':'Search approved runbooks in this deployment service and identity scope. Empty evidence is a successful search, not a transport failure.',
 'service_health':'Read a synthetic service health snapshot for triage only. Unknown health is not a root cause or proof of remediation.'}

class Envelope(Strict):
 contract_version:Literal['49.1']
 ok:bool
 code:Literal['ok','unavailable']
 data:dict|None
 @model_validator(mode='after')
 def consistency(self):
  if self.ok!=(self.code=='ok') or self.ok!=(self.data is not None):raise ValueError('inconsistent envelope')
  return self
class Ticket(Strict):
 id:int=Field(ge=1)
 title:str=Field(min_length=1,max_length=120)
 body:str=Field(max_length=2000)
 status:Literal['open','resolved']
 version:int=Field(ge=1)
class Evidence(Strict):
 doc_id:str=Field(min_length=1,max_length=40)
 version:int=Field(ge=1)
 start:int=Field(ge=0)
 end:int=Field(ge=1)
 quote:str=Field(min_length=1,max_length=600)
 content_hash:str=Field(pattern='^[a-f0-9]{64}$')
class Search(Strict):evidence:list[Evidence]=Field(max_length=2)
class Health(Strict):
 status:Literal['ok','degraded','unknown']
 observed_at:str=Field(min_length=1,max_length=40)
 detail:str=Field(max_length=400)
 service:Literal['gateway','database']
 content_hash:str=Field(pattern='^[a-f0-9]{64}$')
MODELS={'read_ticket':Ticket,'search_runbooks':Search,'read_health':Health}

class HostError(Exception):
 def __init__(self,code,layer='host',rpc_code=None):super().__init__(code);self.code=code;self.layer=layer;self.rpc_code=rpc_code

def safe_schema(schema):
 if not isinstance(schema,dict) or len(json.dumps(schema))>12000:raise HostError('invalid_schema')
 def visit(node):
  if isinstance(node,dict):
   if '$id' in node or '$dynamicRef' in node:raise HostError('unsupported_reference')
   if '$ref' in node and (not isinstance(node['$ref'],str) or not node['$ref'].startswith('#/')):raise HostError('external_schema_reference')
   if '$schema' in node and node['$schema']!='https://json-schema.org/draft/2020-12/schema':raise HostError('unsupported_dialect')
   for value in node.values():visit(value)
  elif isinstance(node,list):
   for value in node:visit(value)
 visit(schema)
 try:Draft202012Validator.check_schema(schema)
 except Exception:raise HostError('invalid_schema') from None

def catalog_rows(pages,max_tools=8):
 result={}
 for page in pages:
  for row in page:
   if row['name'] in result:raise HostError('duplicate_tool_name')
   result[row['name']]=copy.deepcopy(row)
   if len(result)>max_tools:raise HostError('catalog_tool_budget')
 if len(json.dumps(result,ensure_ascii=False).encode())>48000:raise HostError('catalog_byte_budget')
 if set(result)!=set(MAP.values()):raise HostError('unexpected_catalog')
 if any(not isinstance(r.get('description'),str) or len(r['description'])>800 for r in result.values()):raise HostError('description_budget')
 for row in result.values():safe_schema(row.get('inputSchema'));safe_schema(row.get('outputSchema'))
 return dict(sorted(result.items()))

def qualified(alias,remote):
 if not re.fullmatch(r'[a-z][a-z0-9_]{0,24}',alias):raise HostError('invalid_alias')
 return alias+'__'+remote

def merge_routes(routes):
 out={}
 for alias,catalog in routes:
  for remote in catalog:
   name=qualified(alias,remote)
   if name in out:raise HostError('duplicate_route')
   out[name]=(alias,remote)
 return out

def validated_output(logical,reply,catalog_row):
 raw=reply.model_dump(by_alias=True,exclude_none=True)
 if len(json.dumps(raw,ensure_ascii=False).encode())>20000:raise HostError('tool_result_byte_budget','result')
 if reply.result_type!='complete':raise HostError('unsupported_result_type','result')
 if reply.is_error:raise HostError('tool_execution_error','tool')
 data=reply.structured_content
 if data is None:raise HostError('structured_content_required','result')
 if not Draft202012Validator(catalog_row['outputSchema']).is_valid(data):raise HostError('remote_output_schema','result')
 try:
  env=Envelope.model_validate(data)
  parsed=None if env.data is None else MODELS[logical].model_validate(env.data).model_dump()
 except (ValueError,TypeError):raise HostError('local_output_contract','result') from None
 # The structured representation is canonical. Free text is not concatenated into model context.
 return dict(ok=env.ok,code=env.code,data=parsed)
