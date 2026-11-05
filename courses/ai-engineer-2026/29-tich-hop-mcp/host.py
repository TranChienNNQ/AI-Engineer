"""MCP client transport + host policy. Catalog review is a trusted-controller helper, not a UI."""
import asyncio
import copy
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import sys
import httpx2
from jsonschema import Draft202012Validator
from mcp import Client,StdioServerParameters,MCPError
from mcp.client.streamable_http import streamable_http_client
from domain import Ticket,SearchResult,Document,digest
from server import build_server

ROOT=Path(__file__).resolve().parent
SUPPORTED={'2026-07-28','2025-11-25'}
ALLOWED={'get_ticket','search_runbooks'}


class HostError(Exception):
    def __init__(self,code,layer='host',rpc_code=None):
        super().__init__(code);self.code=code;self.layer=layer;self.rpc_code=rpc_code


def strict_json(text,max_bytes=6000):
    if not isinstance(text,str) or len(text.encode())>max_bytes:raise HostError('payload_too_large')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise HostError('duplicate_json_key')
            d[k]=v
        return d
    def bad_constant(value):raise HostError('nonstandard_number')
    try:return json.loads(text,object_pairs_hook=pairs,parse_constant=bad_constant)
    except (ValueError,TypeError) as exc:raise HostError('invalid_json') from exc


def safe_schema(schema):
    if not isinstance(schema,dict) or len(json.dumps(schema).encode())>16000:raise HostError('invalid_schema')
    def visit(node):
        if isinstance(node,dict):
            if '$id' in node or '$dynamicRef' in node:raise HostError('unsupported_schema_reference')
            if '$ref' in node and (not isinstance(node['$ref'],str) or not node['$ref'].startswith('#/')):raise HostError('external_schema_reference')
            if '$schema' in node and node['$schema']!='https://json-schema.org/draft/2020-12/schema':raise HostError('unsupported_dialect')
            for child in node.values():visit(child)
        elif isinstance(node,list):
            for child in node:visit(child)
    visit(schema)
    try:Draft202012Validator.check_schema(schema)
    except Exception as exc:raise HostError('invalid_schema') from exc


@asynccontextmanager
async def connect(transport='stdio',workspace='blue',mode='auto',fault=None):
    if transport=='stdio':
        if fault:raise ValueError('Fault injection uses the in-memory server')
        params=StdioServerParameters(command=sys.executable,args=[str(ROOT/'server.py'),'--workspace',workspace],cwd=ROOT,
                                     env={'PYTHONIOENCODING':'utf-8'})
        async with Client(params,mode=mode,read_timeout_seconds=3) as client:yield client,None
    elif transport=='memory':
        server,backend=build_server(workspace,fault)
        async with Client(server,mode=mode,read_timeout_seconds=3) as client:yield client,backend
    elif transport=='http-asgi':
        if mode!='auto':raise ValueError('HTTP-ASGI demo uses modern auto discovery')
        server,backend=build_server(workspace,fault)
        app=server.streamable_http_app(stateless_http=True,json_response=True)
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),base_url='http://127.0.0.1:8000') as http:
                async with Client(streamable_http_client('http://127.0.0.1:8000/mcp',http_client=http),read_timeout_seconds=3) as client:yield client,backend
    else:raise ValueError('unknown_transport')


class GuardedHost:
    def __init__(self,client,alias='backend_blue',ticket_id='T-BLUE-1',max_ops=8,timeout=.5):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}',alias):raise ValueError('invalid_alias')
        if type(max_ops) is not int or max_ops<1 or timeout<=0:raise ValueError('invalid_budget')
        self.client=client;self.alias=alias;self.ticket_id=ticket_id;self.max_ops=max_ops;self.timeout=timeout
        self.catalog={};self.reviewed_hash=None;self.ops=0;self.trace=[];self.allowed_uris={};self.sent=0

    async def review_catalog(self):
        """Trusted controller reviews this fixed local deployment; not automatic trust for arbitrary servers."""
        if self.client.protocol_version not in SUPPORTED:raise HostError('unsupported_protocol')
        if self.client.server_capabilities.tools is None:raise HostError('tools_not_advertised')
        result=await self.client.list_tools()
        if result.next_cursor:raise HostError('pagination_not_implemented')
        catalog={}
        for tool in result.tools:
            if tool.name in catalog:raise HostError('duplicate_tool_name')
            catalog[tool.name]=tool.model_dump(by_alias=True,exclude_none=True)
        if set(catalog)!=ALLOWED:raise HostError('unexpected_catalog')
        for tool in catalog.values():
            safe_schema(tool['inputSchema']);safe_schema(tool.get('outputSchema'))
        self.catalog=copy.deepcopy(catalog);self.reviewed_hash=digest(catalog)
        return self.descriptors()

    def descriptors(self):
        return [dict(name=self.alias+'__'+name,server_alias=self.alias,remote_name=name,
                     input_schema=copy.deepcopy(meta['inputSchema']),authority='tool_descriptor') for name,meta in sorted(self.catalog.items())]

    async def refresh(self):
        if self.reviewed_hash is None:raise HostError('catalog_not_reviewed')
        latest=await self.client.list_tools()
        if latest.next_cursor:raise HostError('pagination_not_implemented')
        names=[t.name for t in latest.tools]
        if len(names)!=len(set(names)):raise HostError('duplicate_tool_name')
        data={t.name:t.model_dump(by_alias=True,exclude_none=True) for t in latest.tools}
        if digest(data)!=self.reviewed_hash:raise HostError('catalog_changed_review_required')

    async def request(self,label,operation):
        if self.ops>=self.max_ops:raise HostError('operation_budget')
        self.ops+=1;self.sent+=1
        try:
            async with asyncio.timeout(self.timeout):result=await operation()
        except TimeoutError as exc:
            self.trace.append(dict(operation=label,status='local_timeout'))
            raise HostError('timeout','local_timeout') from exc
        except MCPError as exc:
            self.trace.append(dict(operation=label,status='protocol_error',rpc_code=exc.code))
            raise HostError('mcp_error','protocol',exc.code) from exc
        self.trace.append(dict(operation=label,status='received'))
        return result

    async def call(self,qualified,args):
        prefix=self.alias+'__'
        if not qualified.startswith(prefix):raise HostError('unknown_server_alias')
        name=qualified[len(prefix):]
        if name not in ALLOWED:raise HostError('tool_not_allowed')
        if not isinstance(args,dict) or set(args)!={'request'}:raise HostError('invalid_outer_arguments')
        if not self.catalog:raise HostError('catalog_not_reviewed')
        if not Draft202012Validator(self.catalog[name]['inputSchema']).is_valid(args):raise HostError('invalid_arguments')
        if name=='get_ticket' and args['request']['ticket_id']!=self.ticket_id:raise HostError('outside_task')
        await self.refresh()
        result=await self.request('tools/call:'+name,lambda:self.client.call_tool(name,args))
        if result.is_error:raise HostError('tool_execution_error','tool')
        if result.result_type!='complete':raise HostError('unsupported_result_type')
        data=result.structured_content
        if not Draft202012Validator(self.catalog[name]['outputSchema']).is_valid(data):raise HostError('invalid_output')
        if name=='get_ticket':
            parsed=Ticket.model_validate(data)
            if parsed.ticket_id!=self.ticket_id:raise HostError('wrong_result_resource')
        else:
            parsed=SearchResult.model_validate(data)
            if len(parsed.hits)>args['request'].get('limit',2):raise HostError('too_many_results')
            for hit in parsed.hits:
                if hit.uri!='kb://runbooks/'+hit.doc_id or not re.fullmatch(r'RB-[A-Z]+-[A-Z]+',hit.doc_id):raise HostError('unexpected_resource_uri')
                self.allowed_uris[hit.uri]=dict(doc_id=hit.doc_id,version=hit.version,body_hash=hit.body_hash)
        return dict(server_alias=self.alias,tool=name,authority='data',data=copy.deepcopy(data))

    async def read_document(self,uri):
        expected=self.allowed_uris.get(uri)
        if expected is None:raise HostError('resource_not_selected')
        result=await self.request('resources/read',lambda:self.client.read_resource(uri))
        if len(result.contents)!=1:raise HostError('unexpected_resource_count')
        item=result.contents[0]
        if str(item.uri)!=uri or getattr(item,'mime_type',None)!='application/json':raise HostError('resource_identity_mismatch')
        data=strict_json(getattr(item,'text',None))
        try:doc=Document.model_validate(data)
        except Exception as exc:raise HostError('invalid_document') from exc
        if doc.doc_id!=expected['doc_id'] or doc.version!=expected['version'] or doc.body_hash!=expected['body_hash']:raise HostError('resource_snapshot_changed')
        if doc.body_hash!=digest(doc.text):raise HostError('body_hash_mismatch')
        return dict(uri=uri,authority='data',server_alias=self.alias,data=doc.model_dump())

    async def prompt(self):
        result=await self.request('prompts/get',lambda:self.client.get_prompt('triage_ticket',{'ticket_id':self.ticket_id}))
        return dict(authority='external_template',server_alias=self.alias,
                    messages=[m.model_dump(by_alias=True,exclude_none=True) for m in result.messages])


def merge_descriptors(hosts):
    aliases=[h.alias for h in hosts]
    if len(aliases)!=len(set(aliases)):raise HostError('duplicate_server_alias')
    return [item for host in hosts for item in host.descriptors()]
