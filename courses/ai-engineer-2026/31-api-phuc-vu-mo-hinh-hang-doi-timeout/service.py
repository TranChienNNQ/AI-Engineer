"""One-process bounded FIFO serving service with monotonic deadlines and lifecycle."""
from dataclasses import dataclass
from collections import Counter,deque
import asyncio
import copy
from model import Backend,Features


TERMINAL={'SUCCEEDED','EXPIRED','CANCELLED','FAILED'}


class AdmissionError(Exception):
    def __init__(self,code):super().__init__(code);self.code=code


@dataclass
class Job:
    job_id: str
    features: Features
    enqueued_at: float
    deadline: float
    queue_deadline: float
    future: asyncio.Future
    state: str='QUEUED'
    code: str|None=None
    result: dict|None=None
    timer: asyncio.TimerHandle|None=None
    task: asyncio.Task|None=None
    physical_done: bool=False


class Service:
    def __init__(self,backend=None,workers=1,queue_size=2,queue_wait_ms=200,record_limit=16):
        if any(type(v) is not int or v<1 for v in [workers,queue_size,queue_wait_ms,record_limit]):
            raise ValueError('positive integer configuration required')
        self.backend=backend or Backend();self.worker_count=workers;self.queue_wait_ms=queue_wait_ms
        self.queue=asyncio.Queue(maxsize=queue_size);self.record_limit=record_limit
        self.jobs={};self.tasks=[];self.events=deque(maxlen=128);self.counts=Counter()
        self.accepting=False;self.started=False;self.closed=False;self.busy=0;self.peak_busy=0;self.seq=0

    def event(self,kind,job=None,**data):
        self.seq+=1;self.events.append(dict(seq=self.seq,kind=kind,
                                          job_id=job.job_id if job else None,**data))

    async def start(self):
        if self.started or self.closed:raise RuntimeError('service is single-use')
        try:await self.backend.load()
        except Exception:
            await self.backend.close();raise
        self.started=True;self.accepting=True
        self.tasks=[asyncio.create_task(self.worker(),name=f'lesson31-worker-{i}') for i in range(self.worker_count)]
        self.event('ready')

    def finish(self,job,state,code=None,result=None):
        if job.state in TERMINAL:return False
        job.state=state;job.code=code;job.result=copy.deepcopy(result)
        if job.timer:job.timer.cancel();job.timer=None
        self.counts[state]+=1;self.event('terminal',job,state=state,code=code)
        if not job.future.done():job.future.set_result(self.snapshot(job))
        return True

    def expire(self,job,code):
        if self.finish(job,'EXPIRED',code) and job.task:job.task.cancel()

    def submit(self,features,budget_ms=500):
        # No await between readiness/capacity checks and put_nowait: one event-loop thread.
        if not isinstance(features,Features):raise ValueError('Features required')
        if type(budget_ms) is not int or not 10<=budget_ms<=2000:raise ValueError('budget_ms must be integer 10..2000')
        self.counts['attempts']+=1
        if not self.accepting:
            self.counts['rejected']+=1;raise AdmissionError('not_ready')
        if self.queue.full():
            self.counts['rejected']+=1;raise AdmissionError('queue_full')
        while len(self.jobs)>=self.record_limit:
            victim=next((key for key,j in self.jobs.items() if j.state in TERMINAL and j.physical_done),None)
            if victim is None:
                self.counts['rejected']+=1;raise AdmissionError('record_capacity')
            del self.jobs[victim];self.counts['evicted']+=1
        loop=asyncio.get_running_loop();now=loop.time();deadline=now+budget_ms/1000
        q_deadline=min(deadline,now+self.queue_wait_ms/1000)
        job=Job(f'j{self.counts["accepted"]+1:04}',features,now,deadline,q_deadline,loop.create_future())
        self.queue.put_nowait(job);self.jobs[job.job_id]=job;self.counts['accepted']+=1
        code='request_deadline' if deadline<=now+self.queue_wait_ms/1000 else 'queue_timeout'
        job.timer=loop.call_at(q_deadline,self.expire,job,code)
        self.event('admitted',job);return job

    async def wait(self,job):
        try:
            # A caller cancellation must not silently cancel the shared job future.
            return copy.deepcopy(await asyncio.shield(job.future))
        except asyncio.CancelledError:
            self.cancel(job.job_id,'caller_cancelled');raise

    def cancel(self,job_id,code='cancel_requested'):
        job=self.jobs.get(job_id)
        if not job:return None
        if self.finish(job,'CANCELLED',code) and job.task:job.task.cancel()
        return self.snapshot(job)

    async def worker(self):
        loop=asyncio.get_running_loop()
        while True:
            job=await self.queue.get()
            try:
                if job.state in TERMINAL:continue
                now=loop.time()
                if now>=job.queue_deadline:
                    self.expire(job,'request_deadline' if now>=job.deadline else 'queue_timeout');continue
                if job.timer:job.timer.cancel()
                job.timer=loop.call_at(job.deadline,self.expire,job,'request_deadline')
                job.state='RUNNING';self.busy+=1;self.peak_busy=max(self.peak_busy,self.busy)
                self.event('running',job)
                async def invoke():
                    # Recheck at actual task dispatch too, in case the loop was stalled.
                    if loop.time()>=job.deadline:
                        self.finish(job,'EXPIRED','request_deadline');return None
                    return await self.backend.infer(job.features,job.job_id)
                job.task=asyncio.create_task(invoke())
                try:
                    result=await job.task
                    # Deadline can be crossed while the event loop cannot run its timer callback.
                    if loop.time()>=job.deadline:self.expire(job,'request_deadline')
                    else:self.finish(job,'SUCCEEDED',result=result)
                except asyncio.CancelledError:
                    if job.state not in TERMINAL:
                        self.finish(job,'CANCELLED','worker_cancelled')
                        raise
                except Exception:
                    self.finish(job,'FAILED','inference_failed')
                finally:
                    self.busy-=1;job.task=None
            finally:
                job.physical_done=True;self.queue.task_done()

    def snapshot(self,job):
        return dict(job_id=job.job_id,state=job.state,code=job.code,result=copy.deepcopy(job.result))

    def get(self,job_id):
        job=self.jobs.get(job_id);return self.snapshot(job) if job else None

    def stats(self):
        return dict(accepting=self.accepting,queued=self.queue.qsize(),busy=self.busy,
                    peak_busy=self.peak_busy,records=len(self.jobs),workers=self.worker_count,
                    queue_capacity=self.queue.maxsize,counts=dict(self.counts))

    async def close(self,grace_s=.1):
        if self.closed:return
        self.accepting=False;self.event('draining')
        try:
            await asyncio.wait_for(self.queue.join(),grace_s)
        except TimeoutError:
            for job in self.jobs.values():self.cancel(job.job_id,'shutdown')
            # Teaching backend is cooperative: cancellation unwinds active tasks and drains tombstones.
            await asyncio.wait_for(self.queue.join(),1)
        finally:
            for task in self.tasks:task.cancel()
            await asyncio.gather(*self.tasks,return_exceptions=True)
            for job in self.jobs.values():
                if job.timer:job.timer.cancel();job.timer=None
            await self.backend.close();self.closed=True
            self.event('closed')
