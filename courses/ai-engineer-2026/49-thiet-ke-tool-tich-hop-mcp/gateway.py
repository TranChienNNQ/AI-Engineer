import copy,hashlib
from pathlib import Path
from bridge49 import Agent48,World48,Fault,digest,validate,source_hash,tool_args
from contracts49 import HostError,MAP

def implementation_hash():
 root=Path(__file__).parent
 return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge49.py','contracts49.py','mcp_server.py','mcp_link.py','gateway.py']})

class MCPWorld(World48):
 def __init__(self,path,link,**kwargs):super().__init__(path,**kwargs);self.link=link
 def dispatch(self,name,args,task,token):
  p=self.principal(token);self.core.integrity()
  if name=='read_ticket' and args['ticket_id']!=task['ticket_id']:raise Fault(403,'target_scope_denied')
  if name=='read_health' and (task['goal']!='triage' or p['role'] not in ['agent','reviewer']):raise Fault(403,'health_scope_denied')
  try:
   result=self.link.call(name,args)
  except HostError as err:
   if err.code=='tool_execution_error':result=dict(ok=False,code='mcp_tool_error',data=None)
   else:raise Fault(503,err.code) from None
  finally:self.actual_reads=self.link.sent # Remote attempts, not confirmed backend executions.
  if result['ok']:
   data=result['data']
   if name=='read_ticket':
    t=self.core.store.ticket(task['ticket_id'],p);expected={k:t[k] for k in ['id','title','body','status','version']}
    if data!=expected:raise Fault(503,'ticket_result_identity')
   if name=='search_runbooks':
    e=data['evidence']
    if e and not validate(e,e,self.core.docs,p,task['service'],self.core.revoked):raise Fault(503,'source_result_identity')
   if name=='read_health':
    snap=copy.deepcopy(self.health[task['service']]);snap['service']=task['service']
    if data['content_hash']!=digest(snap) or digest({k:v for k,v in data.items() if k!='content_hash'})!=data['content_hash']:raise Fault(503,'health_result_identity')
  return result|dict(mcp_provenance=dict(server_alias=self.link.alias,remote_name=MAP[name],catalog_hash=self.link.catalog_hash,call_sequence=self.link.sent))

class MCPAgent(Agent48):
 def __init__(self,world,adapter,**kwargs):
  super().__init__(world,adapter,**kwargs)
  self.manifest.update(schema49='49.1',base48=source_hash(),implementation49=implementation_hash(),mcp_catalog=world.link.catalog_hash,mcp_alias=world.link.alias,mcp_protocol=world.link.protocol,mcp_limits=dict(timeout=world.link.timeout,max_requests=world.link.max_requests),mcp_deployment=copy.deepcopy(world.link.deployment))
  self.release=digest(self.manifest)[:16]
 def integrity(self):
  super().integrity()
  if self.manifest['base48']!=source_hash() or self.manifest['implementation49']!=implementation_hash() or self.world.link.catalog_hash!=self.manifest['mcp_catalog'] or self.world.link.alias!=self.manifest['mcp_alias'] or self.world.link.protocol!=self.manifest['mcp_protocol'] or self.manifest['mcp_limits']!=dict(timeout=self.world.link.timeout,max_requests=self.world.link.max_requests) or self.manifest['mcp_deployment']!=self.world.link.deployment:raise Fault(503,'mcp_release_changed')
 def run(self,token,raw):
  result=super().run(token,raw)
  return result|dict(scope='real MCP client/server; scripted model; synthetic read-only backend',read_count_scope='MCP tools/call attempts sent, not confirmed reads',mcp=self.world.link.snapshot())
