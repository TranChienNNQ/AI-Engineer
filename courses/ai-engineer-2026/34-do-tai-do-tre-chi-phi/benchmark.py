"""Real HTTP/TCP loopback only, ephemeral port. No configurable remote target."""
import asyncio
import socket
from contextlib import asynccontextmanager
import httpx
import uvicorn
from serving import Service,create_app
from loadgen import Config,run,PAYLOAD
from measurements import distribution


@asynccontextmanager
async def local_server(service):
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    task=None
    try:
        sock.bind(('127.0.0.1',0));sock.listen(64);sock.setblocking(False)
        port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(create_app(service),host='127.0.0.1',log_level='critical',access_log=False,
                                            lifespan='off',timeout_graceful_shutdown=2))
        task=asyncio.create_task(server.serve(sockets=[sock]))
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():await task;raise RuntimeError('startup failed')
                await asyncio.sleep(.002)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',trust_env=False,
                                     timeout=httpx.Timeout(.5),limits=httpx.Limits(max_connections=32,max_keepalive_connections=32)) as client:
            yield client
    finally:
        if task is not None:
            server.should_exit=True
            try:await asyncio.wait_for(task,3)
            finally:
                if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
        sock.close()


async def benchmark(repeats=2):
    if type(repeats) is not int or not 1<=repeats<=3:raise ValueError('repeats must be 1..3')
    service=Service();runs=[]
    profiles=[('closed-1',Config(mode='closed',concurrency=1)),('closed-4',Config(mode='closed',concurrency=4)),
              ('open-40',Config(rate=40)),('open-250',Config(rate=250,count=48))]
    async with local_server(service) as client:
        for repetition in range(repeats):
            for name,config in profiles:
                for _ in range(3):
                    warm=await client.post('/v1/predict',json=PAYLOAD)
                    if warm.status_code!=200:raise RuntimeError('warmup failed')
                await service.idle();service.reset()
                measured=await run(client,config)
                await service.idle()
                successes=[x for x in service.rows if x['status']==200]
                measured.update(profile=name,repetition=repetition+1,
                                server=dict(terminal_count=len(service.rows),
                                            statuses={str(s):sum(x['status']==s for x in service.rows) for s in {x['status'] for x in service.rows}},
                                            peak_active=service.peak_active,peak_waiting=service.peak_waiting,
                                            outstanding_after_drain=service.outstanding,
                                            queue_ms=distribution(x['queue_ms'] for x in successes),
                                            run_ms=distribution(x['run_ms'] for x in successes)))
                runs.append(measured)
    return dict(kind='measured HTTP TCP loopback, real wall time, one process/event loop; fake cooperative inference delay',
                service=dict(workers=2,queue_cap=4,delay_seconds=.015,release='toy-load-v1'),
                warmup_per_run=3,repeats=repeats,runs=runs,
                scope='No GPU/LLM/TLS/proxy/DNS/production traffic. Load generator and server share CPU/event loop. Not capacity certification.')
