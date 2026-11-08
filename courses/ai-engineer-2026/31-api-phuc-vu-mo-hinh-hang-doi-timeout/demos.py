"""Async/HTTP scenarios: gate handshakes force order; timers use actual event-loop time."""
from contextlib import asynccontextmanager
import asyncio
import threading
import math
import httpx
from api import create_app
from model import Backend,Features,ARTIFACT_HASH
from service import Service,AdmissionError

PAYLOAD=dict(latency_ratio=.5,error_ratio=.5,budget_ms=500)


async def entered(backend):
    value=await asyncio.wait_for(backend.started.get(),1)
    backend.started.task_done();return value


@asynccontextmanager
async def local_client(service=None):
    app=create_app(service)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://lesson31') as client:
            yield client,app.state.service


async def happy_demo():
    async with local_client() as (client,svc):
        response=await client.post('/v1/predict',json=PAYLOAD)
        created=await client.post('/v1/jobs',json=PAYLOAD);job_id=created.json()['job_id']
        await svc.wait(svc.jobs[job_id])  # Harness synchronization; public clients poll GET.
        fetched=await client.get(created.headers['Location'])
        return dict(predict_status=response.status_code,prediction=response.json(),
                    job_submit_status=created.status_code,job_get_status=fetched.status_code,
                    job=fetched.json(),load_count=svc.backend.load_count,stats=svc.stats())


async def overload_demo():
    gate=asyncio.Event();backend=Backend(gate=gate);svc=Service(backend,queue_size=1)
    async with local_client(svc) as (client,_):
        first=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD));await entered(backend)
        second=await client.post('/v1/jobs',json=PAYLOAD)
        third=await client.post('/v1/predict',json=PAYLOAD)
        ready=await client.get('/readyz');at_full=svc.stats()
        gate.set();first_response=await first;await asyncio.wait_for(svc.queue.join(),1)
        return dict(statuses=[first_response.status_code,second.status_code,third.status_code],
                    rejected_code=third.json()['code'],retry_after=third.headers.get('Retry-After'),
                    ready_while_full=ready.status_code,at_full=at_full,after=svc.stats(),
                    model_calls=list(backend.calls))


async def queued_timeout_demo(queue_timeout=False):
    gate=asyncio.Event();backend=Backend(gate=gate)
    svc=Service(backend,queue_size=1,queue_wait_ms=20 if queue_timeout else 500)
    await svc.start()
    try:
        first=svc.submit(Features(.5,.5),1000);await entered(backend)
        second=svc.submit(Features(.5,.5),1000 if queue_timeout else 20)
        expired=await asyncio.wait_for(svc.wait(second),1)
        try:svc.submit(Features(0,0));rejection=None
        except AdmissionError as exc:rejection=exc.code
        at_expiry=svc.stats();gate.set();await svc.wait(first);await svc.queue.join()
        return dict(expired=expired,retry_before_drain=rejection,at_expiry=at_expiry,
                    after=svc.stats(),model_calls=list(backend.calls))
    finally:await svc.close()


async def running_timeout_demo():
    gate=asyncio.Event();backend=Backend(gate=gate);svc=Service(backend)
    async with local_client(svc) as (client,_):
        response=await client.post('/v1/predict',json=PAYLOAD|{'budget_ms':20})
        await asyncio.wait_for(svc.queue.join(),1)
        return dict(status=response.status_code,body=response.json(),stats=svc.stats(),
                    started=list(backend.calls),finished=list(backend.finished),cancelled=list(backend.cancelled))


async def cancel_demo(running=False,caller=False):
    gate=asyncio.Event();backend=Backend(gate=gate);svc=Service(backend,queue_size=1)
    async with local_client(svc) as (client,_):
        if caller:
            task=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD));job_id=await entered(backend)
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass
            snapshot=svc.get(job_id)
        else:
            first=svc.submit(Features(.5,.5));await entered(backend)
            target=first if running else svc.submit(Features(.5,.5))
            response=await client.delete('/v1/jobs/'+target.job_id);snapshot=response.json()
        gate.set();await asyncio.wait_for(svc.queue.join(),1)
        return dict(cancelled_job=snapshot,stats=svc.stats(),calls=list(backend.calls),
                    cancelled_backend=list(backend.cancelled))


async def failure_demo():
    backend=Backend(fail_ids={'j0001'});svc=Service(backend)
    async with local_client(svc) as (client,_):
        first=await client.post('/v1/predict',json=PAYLOAD)
        second=await client.post('/v1/predict',json=PAYLOAD)
        return dict(statuses=[first.status_code,second.status_code],codes=[first.json()['code'],second.json()['code']],
                    worker_alive=all(not t.done() for t in svc.tasks),stats=svc.stats())


async def shutdown_demo():
    gate=asyncio.Event();backend=Backend(gate=gate);svc=Service(backend,queue_size=1)
    await svc.start();first=svc.submit(Features(.5,.5));await entered(backend)
    second=svc.submit(Features(.5,.5));await svc.close(grace_s=.01)
    return dict(jobs=[svc.get(first.job_id),svc.get(second.job_id)],stats=svc.stats(),
                all_workers_stopped=all(t.done() for t in svc.tasks),model_released=backend.model is None)


async def thread_demo():
    """Caller deadline does not kill a thread; retain the task until physical completion."""
    loop=asyncio.get_running_loop();started=asyncio.Event();release=threading.Event();finished=threading.Event()
    def blocking():
        loop.call_soon_threadsafe(started.set)
        if not release.wait(2):raise RuntimeError('test gate not released')
        finished.set();return 'physical_done'
    work=asyncio.create_task(asyncio.to_thread(blocking));timed_out=False
    try:
        await asyncio.wait_for(started.wait(),1)
        try:
            async with asyncio.timeout(.02):await asyncio.shield(work)
        except TimeoutError:timed_out=True
        at_timeout=dict(caller_timed_out=timed_out,thread_finished=finished.is_set(),task_done=work.done())
    finally:
        release.set();await work
    return dict(at_timeout=at_timeout,after=dict(thread_finished=finished.is_set(),task_done=work.done()),
                note='Standalone to_thread case; thread backend is not plugged into the cooperative service')


def hand_demo():
    return dict(logit=2*.5+3*.5-2,score=1/(1+math.exp(-.5)),artifact_hash=ARTIFACT_HASH,
                service_ms=100,one_worker_capacity_rps=1000/100,
                fifo_burst=[dict(job='A',arrival=0,start=0,finish=100,wait=0),
                            dict(job='B',arrival=10,start=100,finish=200,wait=90),
                            dict(job='C',arrival=20,start=200,finish=300,wait=180)],
                rejected_at_30_40=['D','E'],little_example=dict(admitted_rps=8,mean_s=.5,mean_in_system=4),
                mm1=[dict(arrival_rps=l,service_rps=10,rho=l/10,mean_s=1/(10-l)) for l in [5,8,9,9.5]],
                budget_example=dict(total_ms=500,elapsed_ms=120,reserve_response_ms=30,remaining_for_work_ms=350),
                retry_amplification=3**3,note='Hand calculations, not measured model serving throughput/latency')


async def scenarios():
    return dict(happy=await happy_demo(),overload=await overload_demo(),
                expired_in_queue=await queued_timeout_demo(),queue_wait_timeout=await queued_timeout_demo(True),
                expired_running=await running_timeout_demo(),cancel_queued=await cancel_demo(),
                cancel_running=await cancel_demo(running=True),caller_cancel=await cancel_demo(caller=True),
                failure=await failure_demo(),shutdown=await shutdown_demo(),thread=await thread_demo())
