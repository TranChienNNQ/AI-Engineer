"""Actual SDK/ASGI examples plus separate synthetic quality replay and hand calculations."""
import asyncio
from collections import Counter
from math import ceil
from telemetry import Telemetry,Histogram
from serving import Runtime,client_for,API,MODEL

PAYLOAD=dict(latency_ratio=.5,error_ratio=.5)
HEADERS=[{'traceparent':'00-'+'1'*32+'-'+'2'*16+'-01'},
         {'traceparent':'00-'+'3'*32+'-'+'4'*16+'-01'}]


def capture(runtime,responses):
    t=runtime.telemetry;spans=list(t.exporter.rows);by_id={s['span_id']:s for s in spans}
    edges=Counter((s['service'],s['name'],by_id[s['parent_id']]['name'] if s['parent_id'] in by_id else ('REMOTE' if s['parent_id'] else 'ROOT')) for s in spans)
    return dict(statuses=[x.status_code for x in responses],predictions=runtime.predictions,
                trace_count=len({s['trace_id'] for s in spans}),span_count=len(spans),
                edges=[dict(service=s,name=n,parent=p,count=count) for (s,n,p),count in sorted(edges.items())],
                spans=spans,logs=list(t.logs),metrics=t.metrics.snapshot(),
                exporter=dict(evicted=t.exporter.evicted,failed_spans=t.exporter.failed_spans),
                worker_done=runtime.worker_task.done(),provider_closed=t.closed)


async def trace_demo(preserve_context=True,sampled=True,export_fail=False,fail_model=False):
    runtime=Runtime(Telemetry(sampled=sampled,export_fail=export_fail),preserve_context=preserve_context,fail_model=fail_model)
    async with client_for(runtime) as (client,_):
        first=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD,headers=HEADERS[0]))
        await asyncio.wait_for(runtime.entered.wait(),1)
        second=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD,headers=HEADERS[1]))
        responses=await asyncio.gather(first,second)
    return capture(runtime,responses)


async def incident_demo():
    gate=asyncio.Event();runtime=Runtime(gate=gate)
    async with client_for(runtime) as (client,_):
        first=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD))
        await asyncio.wait_for(runtime.entered.wait(),1)
        second=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD))
        async def queued():
            while runtime.queue.qsize()!=1:await asyncio.sleep(0)
        try:
            await asyncio.wait_for(queued(),1)
            rejected=await client.post('/v1/predict',json=PAYLOAD)
        finally:gate.set()
        accepted=await asyncio.gather(first,second)
    return capture(runtime,[*accepted,rejected])


async def trace_scenarios():
    return dict(normal=await trace_demo(),broken=await trace_demo(preserve_context=False),
                unsampled=await trace_demo(sampled=False),exporter_failure=await trace_demo(export_fail=True),
                model_failure=await trace_demo(fail_model=True),overload=await incident_demo())


def nearest_rank(values,q):
    if not values or not 0<q<=1:raise ValueError('nonempty values and 0 < q <= 1 required')
    return sorted(values)[ceil(q*len(values))-1]


def hand_demo():
    values=[.01,.02,.03,.04,.5];h=Histogram((.02,.05,.1))
    for x in values:h.observe(x)
    return dict(latency_seconds=values,mean_seconds=sum(values)/len(values),p95_nearest_rank=nearest_rank(values,.95),
                histogram=h.snapshot(),critical_path_ms=max(40,60)+10,sum_work_ms=40+60+10,
                label_example=dict(early_correct=6,early_labeled=6,total_predictions=10,mature_correct=6),
                cardinality_example=dict(routes=3,models=2,status_classes=3,base_series=3*2*3,
                                         with_10000_prediction_ids=3*2*3*10000),
                histogram_example_note='5 latency values and parallel path are hypothetical, not measured from the trace demo')


def stable_facts(report):
    """Compare reproducible relationships/counts; trace IDs, timestamps and timing buckets vary."""
    out={}
    for name,row in report['tracing'].items():
        out[name]={k:row[k] for k in ['statuses','predictions','trace_count','span_count','edges','exporter','worker_done','provider_closed']}
        out[name]['request_counts']=row['metrics']['requests']
        out[name]['queue_wait_count']=row['metrics']['queue_wait_seconds']['count']
    return dict(tracing=out,quality=report['quality'],evaluation=report['evaluation'],hand=report['hand'],metadata=report['metadata'])
