import asyncio,json
from pathlib import Path
import httpx
from bridge45 import request,digest
from fixtures import CASES,mock_adapter
from support import SupportEngine,app_for


def suite(directory,policy):
    rows=[];known_in=known_out=unknown=attempt_count=0
    for case in CASES:
        adapter,script=mock_adapter(case['events']);e=SupportEngine(directory/(case['id']+'.sqlite'),adapter,policy)
        req=request(case.get('message','timeout 504'),action=case.get('action','ask'),key=case['id'],service=case.get('service','gateway'))
        out=e.ask('demo-agent-a',req);served=out['state'] in ['answered','proposed'];meta=out['stats']['llm']
        if meta:
            known_in+=meta['known_input_tokens'];known_out+=meta['known_output_tokens'];unknown+=meta['unknown_usage_attempts'];attempt_count+=len(meta['attempts'])
        rows.append(dict(id=case['id'],expected_serve=case['should_serve'],served=served,correct=served==case['should_serve'],state=out['state'],code=out['code'],calls=len(script.calls),steps=out['steps'],semantic_check=out['stats'].get('semantic_check'),proposal_id=out['proposal_id']))
        e.close()
    tp=sum(x['served'] and x['expected_serve'] for x in rows);fp=sum(x['served'] and not x['expected_serve'] for x in rows);fn=sum(not x['served'] and x['expected_serve'] for x in rows)
    return dict(cases=len(rows),correct=sum(x['correct'] for x in rows),accuracy=sum(x['correct'] for x in rows)/len(rows),served=sum(x['served'] for x in rows),unsafe_served=fp,unnecessarily_rejected=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        attempt_count=attempt_count,known_input_tokens=known_in,known_output_tokens=known_out,unknown_usage_attempts=unknown,usage_source='scripted_fixture_not_provider_bill',rows=rows)


async def api_demo(directory):
    adapter,script=mock_adapter();e=SupportEngine(directory/'api.sqlite',adapter)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_for(e)),base_url='http://lesson46.local',trust_env=False) as client:
        h={'Authorization':'Bearer demo-agent-a'};rh={'Authorization':'Bearer demo-reviewer-a'}
        a=await client.post('/v1/runs',headers=h,json=request(action='create',key='api-create'))
        pid=a.json()['proposal_id'];preview=await client.get(f'/v1/proposals/{pid}',headers=rh)
        review=await client.post(f'/v1/proposals/{pid}/approve',headers=rh);done=await client.post(f'/v1/proposals/{pid}/execute',headers=h,json={'request_key':'write-once'})
        again=await client.post(f'/v1/proposals/{pid}/execute',headers=h,json={'request_key':'write-once'})
        replay=await client.post('/v1/runs',headers=h,json=request(action='create',key='api-create'))
        result=dict(propose_status=a.status_code,preview_status=preview.status_code,review_status=review.status_code,execute_status=done.status_code,receipt_replay_exact=done.json()==again.json(),run_replay_exact=a.json()==replay.json(),provider_calls=len(script.calls),release=e.release,receipt=done.json())
    e.close();return result


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    results={p:suite(out/p,p) for p in ['exact_quote','citation_only']};api=asyncio.run(api_demo(out))
    a,s=mock_adapter();e=SupportEngine(out/'project.sqlite',a);req=request(action='create',key='durable')
    proposed=e.ask('demo-agent-a',req);pid=proposed['proposal_id'];preview=e.read_proposal('demo-reviewer-a',pid);e.review('demo-reviewer-a',pid);receipt=e.commit('demo-agent-a',pid,'effect')
    manifest=e.manifest;release=e.release;e.close()
    a,s=mock_adapter(['unsupported']);e=SupportEngine(out/'project.sqlite',a);replay=e.ask('demo-agent-a',req);restored=e.commit('demo-agent-a',pid,'effect');calls=len(s.calls)
    audit=[dict(x) for x in e.store.db.execute('SELECT kind,entity,actor,code FROM audit ORDER BY id')];e.close()
    (out/'release.json').write_text(json.dumps(dict(id=release,manifest=manifest),ensure_ascii=False,indent=2)+'\n')
    return dict(policy_comparison=results,default_policy='exact_quote',policy_selected_by='predeclared quote identity gate, not a model benchmark or tuning on this suite',
        fixture_snapshot=digest(CASES),api=api,manifest=manifest,release=release,
        durable=dict(run_exact=replay==proposed,receipt_exact=restored==receipt,provider_calls_after_reopen=calls,receipt=receipt,reviewed_payload_hash=preview['payload_hash']),audit=audit,
        live_api_called=False,scope='HTTPX MockTransport programmed responses through actual HTTP adapter; ASGI integration and lesson45 SQLite gateway; no real LLM sampling, model quality/price/latency benchmark, remote ticket or live provider call')
