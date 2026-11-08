import asyncio,copy,json,sys,math
from contextlib import asynccontextmanager,ExitStack
from pathlib import Path
from anyio.from_thread import start_blocking_portal
from mcp import Client,StdioServerParameters,MCPError
from jsonschema import Draft202012Validator
from contracts49 import MAP,ALIAS,HostError,catalog_rows,qualified,validated_output
from bridge49 import digest,tool_args
from mcp_server import build_server

@asynccontextmanager
async def connect(path,transport='memory',goal='triage',actor='demo-agent-a',health_status='degraded',fault=None):
 world=None
 if transport=='memory':target,world,_=build_server(path,goal,actor,health_status,fault)
 elif transport=='stdio':
  if fault is not None:raise ValueError('faults use in-process server')
  root=Path(__file__).resolve().parent
  target=StdioServerParameters(command=sys.executable,args=[str(root/'mcp_server.py'),'--db',str(Path(path).resolve()),'--goal',goal,'--actor',actor,'--health-status',health_status],cwd=root,env={'PYTHONIOENCODING':'utf-8'})
 else:raise ValueError('memory or stdio required')
 try:
  async with Client(target,mode='auto',read_timeout_seconds=3) as client:yield client,target if transport=='memory' else None
 finally:
  if world is not None:world.close()

class Link:
 def __init__(self,path,transport='memory',goal='triage',actor='demo-agent-a',health_status='degraded',fault=None,timeout=.5,max_requests=24,alias=ALIAS):
  qualified(alias,'test')
  if type(max_requests) is not int or not 2<=max_requests<=64 or type(timeout) not in [int,float] or not math.isfinite(timeout) or timeout<=0:raise ValueError('link budgets')
  self.deployment=dict(transport=transport,goal=goal,actor=actor,health_status=health_status,fault=fault);self.alias=alias;self.timeout=timeout;self.max_requests=max_requests;self.ops=0;self.sent=0;self.trace=[];self.closed=False;self.catalog={};self.catalog_hash=None;self.stack=ExitStack()
  try:
   self.portal=self.stack.enter_context(start_blocking_portal())
   self.client,self.server=self.stack.enter_context(self.portal.wrap_async_context_manager(connect(path,transport,goal,actor,health_status,fault)))
   self.protocol=self.client.protocol_version
   if self.protocol!='2026-07-28' or self.client.server_capabilities.tools is None:raise HostError('unsupported_protocol_or_capability')
   self.catalog=self.portal.call(self._catalog);self.catalog_hash=digest(self.catalog)
  except BaseException:self.stack.close();raise
 def close(self):
  if not self.closed:self.stack.close();self.closed=True
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 async def _request(self,label,operation):
  if self.ops>=self.max_requests:raise HostError('mcp_request_budget')
  self.ops+=1
  try:
   async with asyncio.timeout(self.timeout):result=await operation()
  except TimeoutError:self.trace.append(dict(operation=label,status='local_timeout'));raise HostError('mcp_timeout','timeout') from None
  except MCPError as err:self.trace.append(dict(operation=label,status='protocol_error',rpc_code=err.code));raise HostError('mcp_protocol_error','protocol',err.code) from None
  self.trace.append(dict(operation=label,status='received'));return result
 async def _catalog(self):
  pages=[];cursor=None;seen=set()
  for _ in range(3):
   page=await self._request('tools/list',lambda:self.client.list_tools(cursor=cursor,cache_mode='refresh'))
   pages.append([t.model_dump(by_alias=True,exclude_none=True) for t in page.tools]);cursor=page.next_cursor
   if cursor is None:return catalog_rows(pages)
   if cursor in seen:raise HostError('catalog_cursor_cycle')
   seen.add(cursor)
  raise HostError('catalog_page_budget')
 async def _call(self,logical,args):
  if self.closed:raise HostError('closed_link')
  if digest(self.catalog)!=self.catalog_hash:raise HostError('local_catalog_changed')
  if logical not in MAP:raise HostError('tool_not_allowed')
  try:args=tool_args(logical,json.dumps(args,ensure_ascii=False))
  except (ValueError,TypeError):raise HostError('local_input_contract') from None
  if digest(await self._catalog())!=self.catalog_hash:raise HostError('catalog_changed')
  remote=MAP[logical];row=self.catalog[remote];wire={} if logical=='read_health' else dict(request=args)
  if not Draft202012Validator(row['inputSchema']).is_valid(wire):raise HostError('remote_input_schema')
  async def invoke():
   self.sent+=1
   return await self.client.call_tool(remote,wire)
  result=await self._request('tools/call:'+qualified(self.alias,remote),invoke)
  return validated_output(logical,result,row)
 def call(self,logical,args):
  if self.closed:raise HostError('closed_link')
  return self.portal.call(self._call,logical,copy.deepcopy(args))
 def audit(self):
  async def read():
   result=await self._request('resources/read:lab-audit',lambda:self.client.read_resource('lab://audit'))
   if len(result.contents)!=1:raise HostError('audit_resource_shape')
   return json.loads(result.contents[0].text)
  return self.portal.call(read)
 def snapshot(self):return dict(protocol=self.protocol,alias=self.alias,catalog_hash=self.catalog_hash,ops=self.ops,tools_calls_sent=self.sent,trace=copy.deepcopy(self.trace),closed=self.closed)
