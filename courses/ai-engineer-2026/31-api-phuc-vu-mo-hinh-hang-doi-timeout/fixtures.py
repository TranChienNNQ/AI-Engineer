"""API/lifecycle regression, not load testing, accuracy evaluation, or production SLO proof."""
import asyncio
import math
import httpx
from api import create_app
from demos import scenarios,local_client,entered,PAYLOAD
from model import Backend,Features,Model
from service import Service,AdmissionError


async def evaluate(results=None):
    r=results or await scenarios();rows=[]
    def record(name,predicate):
        if not predicate:raise AssertionError(name)
        rows.append(dict(name=name,passed=True))
    h=r['happy']
    record('numeric_forward_matches_hand',math.isclose(h['prediction']['result']['priority_score'],1/(1+math.exp(-.5))))
    record('sync_prediction_200',h['predict_status']==200 and h['prediction']['state']=='SUCCEEDED')
    record('async_accept_then_get_result',h['job_submit_status']==202 and h['job_get_status']==200 and h['job']['state']=='SUCCEEDED')
    record('load_once_for_two_requests',h['load_count']==1)
    o=r['overload']
    record('bounded_queue_sheds_load',o['statuses']==[200,202,503] and o['rejected_code']=='queue_full')
    record('retry_after_is_hint',o['retry_after']=='1')
    record('queue_full_not_readiness_failure',o['ready_while_full']==200)
    record('physical_concurrency_bound',o['at_full']['busy']==1 and o['at_full']['queued']==1 and o['after']['peak_busy']==1)
    for key,code in [('expired_in_queue','request_deadline'),('queue_wait_timeout','queue_timeout')]:
        x=r[key]
        record(key,x['expired']['state']=='EXPIRED' and x['expired']['code']==code)
        record(key+'_no_inference',x['model_calls']==['j0001'])
        record(key+'_tombstone_retains_slot',x['retry_before_drain']=='queue_full' and x['at_expiry']['queued']==1 and x['after']['queued']==0)
    x=r['expired_running']
    record('running_deadline_uses_real_timer',x['status']==503 and x['body']['state']=='EXPIRED' and x['body']['code']=='request_deadline')
    record('cooperative_backend_cancelled',x['cancelled']==['j0001'] and not x['finished'] and x['stats']['busy']==0)
    record('cancel_queued_not_dispatched',r['cancel_queued']['calls']==['j0001'] and r['cancel_queued']['cancelled_job']['state']=='CANCELLED')
    record('cancel_running_backend_unwinds',r['cancel_running']['cancelled_backend']==['j0001'])
    record('caller_cancel_propagated_explicitly',r['caller_cancel']['cancelled_job']['code']=='caller_cancelled' and r['caller_cancel']['cancelled_backend']==['j0001'])
    record('worker_survives_inference_error',r['failure']['statuses']==[500,200] and r['failure']['worker_alive'])
    x=r['shutdown']
    record('shutdown_stops_accepting',not x['stats']['accepting'])
    record('shutdown_cancels_and_drains',all(j['code']=='shutdown' for j in x['jobs']) and x['stats']['queued']==x['stats']['busy']==0)
    record('shutdown_releases_model_workers',x['all_workers_stopped'] and x['model_released'])
    x=r['thread']
    record('thread_continues_after_caller_timeout',x['at_timeout']==dict(caller_timed_out=True,thread_finished=False,task_done=False))
    record('thread_joined_before_exit',x['after']==dict(thread_finished=True,task_done=True))
    # Actual API validation, including non-finite JSON without reflecting raw input.
    async with local_client() as (client,svc):
        for name,payload in [('bool',PAYLOAD|{'latency_ratio':True}),('string',PAYLOAD|{'error_ratio':'0.5'}),
                             ('range',PAYLOAD|{'error_ratio':2}),('extra',PAYLOAD|{'actor':'admin'}),
                             ('budget_bool',PAYLOAD|{'budget_ms':True}),('budget_range',PAYLOAD|{'budget_ms':2001})]:
            response=await client.post('/v1/predict',json=payload)
            record('validation_'+name,response.status_code==422 and response.json()['code']=='invalid_input')
        response=await client.post('/v1/predict',content='{"latency_ratio":NaN,"error_ratio":0.5}',headers={'Content-Type':'application/json'})
        record('validation_nonfinite_sanitized',response.status_code==422 and 'NaN' not in response.text)
        record('validation_before_admission',svc.counts['attempts']==0 and not svc.jobs)
        record('unknown_job_404',(await client.get('/v1/jobs/missing')).status_code==404)
        record('unknown_cancel_404',(await client.delete('/v1/jobs/missing')).status_code==404)
    svc=Service();app=create_app(svc)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://lesson31') as client:
        record('live_before_ready',(await client.get('/livez')).status_code==200 and (await client.get('/readyz')).status_code==503)
        record('reject_before_lifespan',(await client.post('/v1/jobs',json=PAYLOAD)).json()['code']=='not_ready')
    # Retention is bounded; terminal eviction gives a 404, not persistent storage.
    svc=Service(record_limit=2)
    async with local_client(svc) as (client,_):
        for _ in range(3):await client.post('/v1/predict',json=PAYLOAD)
        record('terminal_records_evicted',svc.stats()['records']==2 and svc.counts['evicted']==1 and (await client.get('/v1/jobs/j0001')).status_code==404)
        data=svc.get('j0003');data['result']['label']='tampered'
        record('snapshot_copy',svc.get('j0003')['result']['label']=='high')
        last=svc.get('j0003');svc.cancel('j0003')
        record('terminal_state_not_overwritten',svc.get('j0003')==last)
    gate=asyncio.Event();b=Backend(gate=gate);svc=Service(b,record_limit=1)
    await svc.start()
    try:
        j=svc.submit(Features(.5,.5));await entered(b)
        try:svc.submit(Features(0,0));code=None
        except AdmissionError as e:code=e.code
        record('active_records_not_evicted',code=='record_capacity' and svc.get(j.job_id)['state']=='RUNNING')
    finally:gate.set();await svc.close()
    # More than one cooperative worker, verified by barriers rather than elapsed-time ratios.
    gate=asyncio.Event();b=Backend(gate=gate);svc=Service(b,workers=2,queue_size=2)
    await svc.start()
    try:
        a=svc.submit(Features(0,0));await entered(b)
        z=svc.submit(Features(1,1));await entered(b)
        record('two_workers_active',svc.busy==svc.peak_busy==2)
        gate.set();await svc.wait(a);await svc.wait(z)
    finally:await svc.close()
    record('two_workers_clean_shutdown',all(t.done() for t in svc.tasks))
    bad=0
    for config in [dict(queue_size=0),dict(workers=True),dict(record_limit=0)]:
        try:Service(**config)
        except ValueError:bad+=1
    record('zero_size_not_accidental_unbounded_queue',bad==3)
    class Broken(Backend):
        async def load(self):raise RuntimeError('synthetic load error')
    b=Broken();svc=Service(b)
    try:await svc.start()
    except RuntimeError:pass
    record('startup_failure_never_ready',not svc.accepting and svc.tasks==[] and b.model is None)
    return dict(kind='ASGI/async serving regression, not model quality or a load benchmark',checks=len(rows),passed=len(rows),rows=rows)
