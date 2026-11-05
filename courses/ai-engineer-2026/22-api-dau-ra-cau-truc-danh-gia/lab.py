"""Contract/API fault fixtures and reproducible evaluation, not LLM quality measurement."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from fastapi.testclient import TestClient
from api import create_app
from contracts import RouteRequest, ContractError, parse_json, validate_candidate, OUTPUT_VALIDATOR, export_schema
from service import *


def fixtures():
    good=proposal_json('NET')
    def case(name,expected='NET',raw=good,finish='stop',events=None,text='event=timeout.',slice='clean'):
        return dict(id=name,slice=slice,expected=expected,
                    request=dict(schema_version='1',ticket_id=name,lines=[dict(id='L1',text=text)]),
                    events=events or [Event(reply=Reply(raw,finish))])
    return [
        case('net-ok'),case('auth-ok','AUTH',proposal_json('AUTH'),text='event=denied.'),
        case('data-ok','DATA',proposal_json('DATA'),text='event=deadlock.'),
        case('schema-valid-wrong-team',raw=proposal_json('AUTH'),slice='semantic'),
        case('unknown-abstain','ABSTAIN',proposal_json(None),text='không đủ dữ kiện',slice='ambiguous'),
        case('ambiguous-forced-route','ABSTAIN',text='event=timeout và event=denied.',slice='ambiguous'),
        case('malformed',raw='{"decision":',slice='format'),
        case('markdown-fence',raw='```json\n'+good+'\n```',slice='format'),
        case('missing-team',raw='{"schema_version":"1","decision":"route","evidence_ids":["L1"]}',slice='schema'),
        case('extra-key',raw=good[:-1]+',"action":"close_ticket"}',slice='schema'),
        case('unknown-evidence',raw=proposal_json('NET',['L9']),slice='grounding'),
        case('contradictory-fields',raw='{"schema_version":"1","decision":"route","team":null,"evidence_ids":["L1"]}',slice='schema'),
        case('valid-json-but-truncated',finish='length',slice='transport'),
        case('provider-refusal',raw=None,finish='refusal',slice='transport'),
        case('timeout-twice',events=[Event(duration_ms=500),Event(duration_ms=500)],slice='transport'),
        case('retry-recovers',events=[Event(kind='rate_limit',duration_ms=20,retry_after_ms=100),Event(reply=Reply(good),duration_ms=30)],slice='transport'),
        case('unavailable-twice',events=[Event(kind='transient',duration_ms=10),Event(kind='transient',duration_ms=10)],slice='transport'),
        case('duplicate-json-key',raw=good[:-1]+',"team":"AUTH"}',slice='format'),
        case('json-root-string',raw='"NET"',slice='schema'),
        case('evidence-does-not-support','AUTH',good,text='timeout đã được loại trừ; event=denied.',slice='semantic')
    ]


def evaluate():
    rows=[]
    for item in fixtures():
        request=RouteRequest.model_validate(item['request']);clock=FakeClock();provider=ScriptedProvider(item['events'],clock)
        # Stage metrics inspect final scripted reply only if its finish is stop; failures count in denominator.
        row=dict(id=item['id'],slice=item['slice'],expected=item['expected'],complete=False,json_valid=False,schema_valid=False,
                 accepted=False,correct=False,predicted=None,error=None,attempts=0,simulated_elapsed_ms=0.)
        try:
            response,trace=execute(request,provider,clock,request_id=item['id'])
            row.update(accepted=True,predicted=response.result.team if response.result.decision=='route' else 'ABSTAIN',
                       attempts=response.metrics.attempts,usage_complete=response.metrics.usage_complete)
        except ServiceFailure as exc:
            row.update(error=exc.code,attempts=exc.attempts,usage_complete=False);trace=exc.trace
        final_event=item['events'][provider.calls-1]
        if final_event.reply is not None and final_event.reply.finish=='stop' and final_event.duration_ms<400:
            row['complete']=True
            try:
                obj=parse_json(final_event.reply.content);row['json_valid']=True
                row['schema_valid']=OUTPUT_VALIDATOR.is_valid(obj)
            except ContractError:pass
        row['correct']=row['accepted'] and row['predicted']==row['expected']
        row['simulated_elapsed_ms']=clock.now_ms();row['trace']=trace
        rows.append(row)
    n=len(rows);accepted=sum(r['accepted'] for r in rows);routes=[r for r in rows if r['accepted'] and r['predicted']!='ABSTAIN']
    matrix_labels=['AUTH','DATA','NET','ABSTAIN'];columns=matrix_labels+['ERROR']
    matrix=[[sum(r['expected']==truth and (r['predicted'] or 'ERROR')==prediction for r in rows) for prediction in columns] for truth in matrix_labels]
    summary={key:sum(bool(r[key]) for r in rows)/n for key in ['complete','json_valid','schema_valid','accepted','correct']}
    summary.update(samples=n,accepted_count=accepted,correct_count=sum(r['correct'] for r in rows),
                   route_coverage=len(routes)/n,route_accuracy=sum(r['correct'] for r in routes)/len(routes) if routes else None,
                   accepted_accuracy=sum(r['correct'] for r in rows)/accepted if accepted else None,
                   observed_attempts=sum(r['attempts'] for r in rows))
    slices={name:dict(n=sum(r['slice']==name for r in rows),correct=sum(r['correct'] for r in rows if r['slice']==name),
                       accepted=sum(r['accepted'] for r in rows if r['slice']==name)) for name in sorted({r['slice'] for r in rows})}
    fixture_data=[dict(item,events=[asdict(e) for e in item['events']]) for item in fixtures()]
    digest=hashlib.sha256(json.dumps(fixture_data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    report=dict(kind='deterministic contract/fault fixtures; NOT real model evaluation',summary=summary,slices=slices,
                confusion=dict(rows=matrix_labels,columns=columns,counts=matrix),rows=rows,fixture_sha256=digest,
                versions={name:version(name) for name in ['fastapi','pydantic','httpx','jsonschema','uvicorn']},python=platform.python_version())
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


def api_demo():
    request=dict(schema_version='1',ticket_id='demo-1',lines=[dict(id='L1',text='event=timeout.')])
    with TestClient(create_app()) as client:
        response=client.post('/v1/route-proposals',json=request)
        assert response.status_code==200 and response.json()['result']['team']=='NET'
        bad=client.post('/v1/route-proposals',json=dict(request,extra='not allowed'))
        assert bad.status_code==422
        assert client.get('/openapi.json').status_code==200
    def timeout_factory(req,clock):return ScriptedProvider([Event(duration_ms=500),Event(duration_ms=500)],clock)
    with TestClient(create_app(timeout_factory,FakeClock)) as client:
        timeout=client.post('/v1/route-proposals',json=request)
        assert timeout.status_code==504
        assert timeout.json()['error']=='upstream_timeout'
    print('PASS HTTP 200/422/504, response contract and OpenAPI (in-process ASGI transport).')
    return dict(ok=200,invalid_request=422,upstream_timeout=504)


def schema_demo():
    request=RouteRequest(schema_version='1',ticket_id='demo',lines=[dict(id='L1',text='event=timeout.')])
    accepted=validate_candidate(proposal_json('AUTH'),request)
    assert accepted.team=='AUTH'  # Schema + ID existence cannot detect wrong task label.
    failures={}
    for name,raw in [('markdown','```json\n'+proposal_json()+'\n```'),('unknown-evidence',proposal_json('NET',['L9'])),
                     ('null-team','{"schema_version":"1","decision":"route","team":null,"evidence_ids":["L1"]}')]:
        try:validate_candidate(raw,request)
        except ContractError as exc:failures[name]=exc.code
        else:raise AssertionError('Invalid candidate accepted')
    print(json.dumps(dict(schema_valid_but_wrong_team=accepted.team,failures=failures),indent=2))
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['schema','api','eval','all'],default='schema');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.out and a.out.exists():raise FileExistsError('Choose a new output directory')
    report={}
    if a.mode in ('schema','all'):report['schema']=schema_demo()
    if a.mode in ('api','all'):report['api']=api_demo()
    if a.mode in ('eval','all'):report['evaluation']=evaluate()
    if a.out:
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        export_schema(a.out/'output-schema.json')
        print('Output:',a.out)
if __name__=='__main__':main()
