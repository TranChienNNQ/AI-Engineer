"""Tiny authored fixture evaluation, not a human/LLM usefulness benchmark."""
import asyncio,json
from pathlib import Path
import httpx
from corpus import DEV,TEST,digest
from pipeline import Engine,request
from api import create_app
from store import Fault


class Clock:
    def __init__(self):self.value=1000.
    def __call__(self):return self.value


def score_rows(engine,rows):
    out=[]
    for row in rows:
        result=engine.ask(row['token'],request(row['query'],key=row['id'],service=row['service']))
        actual=result['citations'][0]['doc_id'] if result['citations'] else None
        match=actual==row['expected'] and result['state'] in ['answered','abstained']
        out.append(dict(id=row['id'],expected=row['expected'],actual=actual,exact=match,answerable=row['expected'] is not None,state=result['state'],steps=result['steps'],packed_chars=result['stats'].get('packed_chars',0),release=result['release']))
    answered=sum(x['actual'] is not None for x in out);correct=sum(x['exact'] and x['actual'] is not None for x in out);possible=sum(x['answerable'] for x in out)
    return dict(cases=len(out),exact=sum(x['exact'] for x in out)/len(out),answer_rate=answered/len(out),conditional_source_accuracy=correct/answered if answered else None,answerable_recall=correct/possible if possible else None,rows=out)


async def api_demo(engine):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)),base_url='http://lesson45.local',trust_env=False) as client:
        headers={'Authorization':'Bearer demo-agent-a'}
        a=await client.post('/v1/runs',headers=headers,json=request(action='create',key='api-create'))
        pid=a.json()['proposal_id'];review=await client.post(f'/v1/proposals/{pid}/approve',headers={'Authorization':'Bearer demo-reviewer-a'})
        commit=await client.post(f'/v1/proposals/{pid}/execute',headers=headers,json={'request_key':'api-execute'})
        again=await client.post(f'/v1/proposals/{pid}/execute',headers=headers,json={'request_key':'api-execute'})
        ticket=await client.get(f"/v1/tickets/{commit.json()['ticket_id']}",headers=headers)
        unauthorized=await client.post('/v1/runs',json=request(key='no-token'))
        forbidden=await client.get('/v1/tickets/2',headers=headers)
        return dict(propose_status=a.status_code,review_status=review.status_code,commit_status=commit.status_code,retry_exact=commit.json()==again.json(),ticket_status=ticket.json()['status'],ticket_version=ticket.json()['version'],unauthenticated=unauthorized.status_code,cross_tenant_ticket=forbidden.status_code)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dev={};trials=[]
    for minimum in [1,2]:
        e=Engine(out/f'dev-{minimum}.sqlite',min_terms=minimum);dev[str(minimum)]=score_rows(e,DEV);e.close();trials.append(dict(min_terms=minimum,dev_exact=dev[str(minimum)]['exact']))
    selected=max(trials,key=lambda t:t['dev_exact'])['min_terms']
    e=Engine(out/'project.sqlite',min_terms=selected,clock=Clock());test=score_rows(e,TEST)
    p=e.ask('demo-agent-a',request(action='create',key='lost-response'))['proposal_id'];e.review('demo-reviewer-a',p)
    try:e.commit('demo-agent-a',p,'lost-key',fault='after_commit')
    except Fault as err:lost=err.code
    manifest=e.manifest;release=e.release;e.close()
    e=Engine(out/'project.sqlite',min_terms=selected,clock=Clock());receipt=e.commit('demo-agent-a',p,'lost-key');retry=e.commit('demo-agent-a',p,'lost-key')
    recovered=dict(injected_fault=lost,receipt=receipt,retry_exact=receipt==retry,ticket_count=e.store.db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0],receipt_count=e.store.db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0])
    api=asyncio.run(api_demo(e));audit=[dict(x) for x in e.store.db.execute('SELECT kind,entity,actor,code FROM audit ORDER BY id')]
    (out/'release.json').write_text(json.dumps(dict(id=release,manifest=manifest),ensure_ascii=False,indent=2)+'\n');e.close()
    # Acceptance gate values are authored targets for this fixture suite, not production SLO estimates.
    return dict(development=dev,trials=trials,selected_min_terms=selected,test=test,always_abstain_exact=sum(x['expected'] is None for x in TEST)/len(TEST),
        release=release,manifest=manifest,eval_snapshot=digest(dict(dev=DEV,test=TEST)),recovery=recovered,api=api,audit=audit,
        gate=dict(source_exact=test['exact']==1.,api_contract=api['commit_status']==200,recovery=recovered['retry_exact']),
        scope='authored 6 dev/8 test cases; extractive BM25 baseline, explicit workflow actions, local SQLite effects/receipts, HTTPX ASGI in-process; no LLM/embedding/vector DB/external ticket system, traffic load test, cloud deployment or production auth')
