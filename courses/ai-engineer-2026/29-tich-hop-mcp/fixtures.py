"""Real MCP SDK exchanges + host boundary checks; no LLM performance benchmark."""
import copy
import json
from collections import Counter
from mcp import Client,MCPError
from mcp.types import ToolAnnotations
from agent import demo
from domain import digest
from host import connect,GuardedHost,HostError,safe_schema,strict_json,merge_descriptors
from server import build_server


async def _evaluate():
    rows=[]
    def record(name,observed,expected,layer):
        rows.append(dict(name=name,observed=json.loads(json.dumps(observed)),expected=json.loads(json.dumps(expected)),passed=observed==expected,layer=layer))
    for transport in ['memory','stdio','http-asgi']:
        r=await demo(transport)
        record('happy_'+transport,(r['protocol_version'],len(r['documents']),r['logical_data_operations']),('2026-07-28',2,5),'integration')
    r=await demo('memory',mode='legacy')
    record('legacy_sdk',r['protocol_version'],'2025-11-25','compatibility')
    async with connect('memory') as (client,backend):
        host=GuardedHost(client);await host.review_catalog()
        cases=[('unknown_alias','other__get_ticket',{'request':{'ticket_id':'T-BLUE-1'}},'unknown_server_alias'),
               ('host_unknown_write','backend_blue__append_note',{'request':{}},'tool_not_allowed'),
               ('host_outer_actor','backend_blue__get_ticket',{'request':{'ticket_id':'T-BLUE-1'},'actor':'carol'},'invalid_outer_arguments'),
               ('host_nested_actor','backend_blue__get_ticket',{'request':{'ticket_id':'T-BLUE-1','actor':'carol'}},'invalid_arguments'),
               ('host_boolean_limit','backend_blue__search_runbooks',{'request':{'query':'retry','limit':True}},'invalid_arguments'),
               ('host_limit_out_of_range','backend_blue__search_runbooks',{'request':{'query':'retry','limit':4}},'invalid_arguments'),
               ('host_cross_task','backend_blue__get_ticket',{'request':{'ticket_id':'T-GREEN-1'}},'outside_task')]
        for name,tool,args,expected in cases:
            before=len(backend.reads)
            try:await host.call(tool,args);code='accepted'
            except HostError as exc:code=exc.code
            record(name,(code,len(backend.reads)-before),(expected,0),'host_before_dispatch')
        for name,args in [('server_cross_workspace',{'request':{'ticket_id':'T-GREEN-1'}}),('server_nested_extra',{'request':{'ticket_id':'T-BLUE-1','actor':'carol'}})]:
            r=await client.call_tool('get_ticket',args);record(name,r.is_error,True,'server_tool')
        r=await client.call_tool('get_ticket',{'request':{'ticket_id':'T-BLUE-1'},'actor':'carol'})
        record('sdk_ignores_extra_outer',(r.is_error,r.structured_content['ticket_id']),(False,'T-BLUE-1'),'sdk_observed_behavior')
        r=await client.call_tool('missing_tool',{})
        record('sdk_unknown_tool_result',r.is_error,True,'sdk_observed_behavior')
        try:await client.get_prompt('triage_ticket',{});code=None
        except MCPError as exc:code=exc.code
        record('prompt_missing_argument',code,-32603,'sdk_observed_behavior')
        rows[-1]['documentation_example_code']=-32602
        rows[-1]['note']='SDK 2.3.0 observed behavior differs from the current SDK documentation example; not a protocol rule.'
        try:await client.read_resource('file:///course29-not-a-real-file');code=None
        except MCPError as exc:code=exc.code
        record('unknown_resource_uri',code,-32602,'protocol')
        try:await host.read_document('https://collector.invalid/collect');code='accepted'
        except HostError as exc:code=exc.code
        record('host_unselected_uri',code,'resource_not_selected','host_before_dispatch')
        hit=await host.call('backend_blue__search_runbooks',{'request':{'query':'checkout retry','limit':3}})
        record('search_scope_leak','GREEN_PRIVATE_29' in json.dumps(hit),False,'server_data')
        poison=await host.read_document('kb://runbooks/RB-BLUE-POISON')
        record('poison_retained_as_data',('[SYSTEM]' in poison['data']['text'],poison['authority']),(True,'data'),'host_data_authority')
        backend.docs['RB-BLUE-RETRY']['version']+=1
        try:await host.read_document('kb://runbooks/RB-BLUE-RETRY');code='accepted'
        except HostError as exc:code=exc.code
        record('source_version_changed',code,'resource_snapshot_changed','host_output')
        backend.docs['RB-BLUE-RETRY']['workspace']='green'
        try:await host.read_document('kb://runbooks/RB-BLUE-RETRY');code='accepted'
        except HostError as exc:code=exc.layer
        record('source_scope_revoked',code,'protocol','server_data')
    for fault,expected in [('wrong_ticket','wrong_result_resource'),('unavailable','tool_execution_error'),('slow','timeout')]:
        async with connect('memory',fault=fault) as (client,_):
            host=GuardedHost(client,timeout=.02 if fault=='slow' else .5);await host.review_catalog()
            try:await host.call('backend_blue__get_ticket',{'request':{'ticket_id':'T-BLUE-1'}});code='accepted'
            except HostError as exc:code=exc.code
            record(fault,code,expected,'host_output_or_local')
    async with connect('memory',fault='oversized_resource') as (client,_):
        host=GuardedHost(client);await host.review_catalog()
        await host.call('backend_blue__search_runbooks',{'request':{'query':'retry','limit':3}})
        try:await host.read_document('kb://runbooks/RB-BLUE-RETRY');code='accepted'
        except HostError as exc:code=exc.code
        record('oversized_resource',code,'payload_too_large','host_output')
    server,_=build_server()
    async with Client(server) as client:
        host=GuardedHost(client);await host.review_catalog();server.remove_tool('search_runbooks')
        try:await host.refresh();code='accepted'
        except HostError as exc:code=exc.code
        record('catalog_drift',code,'catalog_changed_review_required','host_catalog')
    async with connect('memory') as (client,_):
        host=GuardedHost(client,max_ops=1);await host.review_catalog();await host.prompt()
        try:await host.call('backend_blue__get_ticket',{'request':{'ticket_id':'T-BLUE-1'}});code='accepted'
        except HostError as exc:code=exc.code
        record('operation_budget',code,'operation_budget','host_budget')
    for name,func,expected in [
       ('external_schema_ref',lambda:safe_schema({'type':'object','$ref':'https://collector.invalid/schema'}),'external_schema_reference'),
       ('duplicate_resource_key',lambda:strict_json('{"v":1,"v":2}'),'duplicate_json_key'),
       ('non_json_nan',lambda:strict_json('{"v":NaN}'),'nonstandard_number')]:
        try:func();code='accepted'
        except HostError as exc:code=exc.code
        record(name,code,expected,'host_parser_unit')
    async with connect('memory','blue') as (blue,_),connect('memory','green') as (green,_):
        a=GuardedHost(blue,alias='blue');b=GuardedHost(green,alias='green',ticket_id='T-GREEN-1')
        await a.review_catalog();await b.review_catalog()
        descriptors=merge_descriptors([a,b])
        record('server_name_collision',(blue.server_info.name==green.server_info.name,len({d['name'] for d in descriptors})),(True,4),'host_routing')
        b.alias='blue'
        try:merge_descriptors([a,b]);code='accepted'
        except HostError as exc:code=exc.code
        record('duplicate_alias',code,'duplicate_server_alias','host_routing')
    return dict(kind='MCP integration and deterministic host checks, no LLM benchmark',cases=len(rows),
                passed=sum(r['passed'] for r in rows),layers=dict(Counter(r['layer'] for r in rows)),rows=rows,
                residual=dict(name='answer_only_injection',scripted_response='BANANA_OVERRIDE',semantic_correct=False,
                              blocked=False,note='No semantic verifier or LLM attack evaluation in this lab'))


def hand_demo():
    return dict(hosts=3,servers=4,pairwise_integrations=3*4,idealized_adapters=3+4,
                model_turns_example=3,mcp_tool_calls_example=2,resource_reads_example=2,
                data_operations_example=5,discovery_and_listing_not_counted_as_data_operations=True,
                hypothetical_latency_ms=dict(discovery=10,list_tools=10,get_ticket=20,search=30,read1=20,read2=20),
                sequential_example_ms=10+10+20+30+20+20,
                note='Integration counts and latency are illustrative, not measured performance or total wire request counts')


async def evaluate():
    # The deliberate negative cases can log SDK tracebacks; keep production server logging intact.
    import logging
    logger=logging.getLogger('mcp');handlers,propagate=logger.handlers,logger.propagate
    logger.handlers=[logging.NullHandler()];logger.propagate=False
    try:return await _evaluate()
    finally:logger.handlers=handlers;logger.propagate=propagate
