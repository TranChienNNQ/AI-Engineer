"""A fixed small host workflow consumes MCP-backed tools/resources, without an LLM."""
from host import connect,GuardedHost


async def demo(transport='stdio',workspace='blue',mode='auto'):
    ticket='T-BLUE-1' if workspace=='blue' else 'T-GREEN-1'
    alias='backend_'+workspace
    async with connect(transport,workspace,mode) as (client,_):
        host=GuardedHost(client,alias=alias,ticket_id=ticket)
        tools=await host.review_catalog()
        resources=await client.list_resources();templates=await client.list_resource_templates();prompts=await client.list_prompts()
        prompt=await host.prompt()
        t=await host.call(alias+'__get_ticket',{'request':{'ticket_id':ticket}})
        query=t['data']['service']+' '+t['data']['issue']
        hits=await host.call(alias+'__search_runbooks',{'request':{'query':query,'limit':2}})
        docs=[await host.read_document(hit['uri']) for hit in hits['data']['hits']]
        sources=[dict(uri=d['uri'],version=d['data']['version'],body_hash=d['data']['body_hash']) for d in docs]
        # Scripted summary, not a semantic correctness assertion about the sources.
        answer=f"{ticket}: {t['data']['service']} / {t['data']['issue']}. Tìm được {len(docs)} nguồn; chưa xác nhận root cause hoặc sửa ticket."
        return dict(transport=transport,workspace=workspace,protocol_version=client.protocol_version,
                    server_info=client.server_info.model_dump(),server_capabilities=client.server_capabilities.model_dump(by_alias=True,exclude_none=True),catalog_hash=host.reviewed_hash,
                    tools=tools,resources=[str(r.uri) for r in resources.resources],
                    resource_templates=[str(r.uri_template) for r in templates.resource_templates],
                    prompt_names=[p.name for p in prompts.prompts],rendered_prompt=prompt,
                    ticket=t,search=hits,documents=docs,sources=sources,answer=answer,
                    logical_data_operations=host.ops,trace=host.trace,
                    note='Real SDK/protocol; scripted host workflow, no LLM or writable business tools')
