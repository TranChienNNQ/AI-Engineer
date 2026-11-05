"""Small sequential raw stdio client to inspect MCP 2026 JSON-RPC frames, not a production SDK."""
import asyncio
import json
import sys
from pathlib import Path
from host import strict_json

VERSION='2026-07-28'


def metadata(version=VERSION):
    return {'io.modelcontextprotocol/protocolVersion':version,
            'io.modelcontextprotocol/clientInfo':dict(name='course29-wire',version='29.0'),
            'io.modelcontextprotocol/clientCapabilities':{}}


async def wire_demo():
    path=Path(__file__).resolve().parent/'server.py'
    process=await asyncio.create_subprocess_exec(sys.executable,str(path),'--workspace','blue',
                 stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,limit=65537)
    transcript=[]
    async def rpc(method,params=None,version=VERSION):
        rid=len(transcript)+1;params=dict(params or {});params['_meta']=metadata(version)
        request=dict(jsonrpc='2.0',id=rid,method=method,params=params)
        process.stdin.write((json.dumps(request,ensure_ascii=False)+'\n').encode());await process.stdin.drain()
        async with asyncio.timeout(3):line=await process.stdout.readline()
        if not line.endswith(b'\n') or len(line)>65536:raise ValueError('invalid_stdio_frame')
        response=strict_json(line.decode(),65536)
        if response.get('jsonrpc')!='2.0' or response.get('id')!=rid or ('result' in response)==('error' in response):raise ValueError('bad_response_correlation')
        transcript.append(dict(request=request,response=response,request_bytes=len((json.dumps(request,ensure_ascii=False)+'\n').encode()),response_bytes=len(line)))
        return response
    try:
        await rpc('server/discover')
        await rpc('tools/list')
        await rpc('tools/call',dict(name='get_ticket',arguments=dict(request=dict(ticket_id='T-BLUE-1'))))
        await rpc('resources/list')
        await rpc('resources/templates/list')
        await rpc('resources/read',dict(uri='kb://runbooks/RB-BLUE-RETRY'))
        await rpc('prompts/list')
        await rpc('prompts/get',dict(name='triage_ticket',arguments=dict(ticket_id='T-BLUE-1')))
        await rpc('tools/call',dict(name='get_ticket',arguments=dict(request=dict(ticket_id='T-GREEN-1'))))
        await rpc('tools/call',dict(name='does_not_exist',arguments={}))
        await rpc('course/unknown_method')
        await rpc('server/discover',version='1900-01-01')
        await rpc('tools/list')  # A bad request version must not poison later valid per-request metadata.
    finally:
        process.stdin.close()
        try:await asyncio.wait_for(process.wait(),3)
        except TimeoutError:process.kill();await process.wait()
    return dict(protocol_version=VERSION,transport='stdio, real child process',requests=len(transcript),
                transcript=transcript,exit_code=process.returncode,
                note='A limited sequential wire observer, no retry/multiplexing/subscriptions/MRTR')
