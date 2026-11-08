"""Optional real TCP loopback integration. Binds an ephemeral local port, then closes it."""
import asyncio
import socket
import httpx
import uvicorn
from api import create_app
from model import Backend
from service import Service
from demos import PAYLOAD,entered


async def socket_demo():
    backend=Backend(delay=.06);svc=Service(backend,queue_wait_ms=500)
    app=create_app(svc);sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind(('127.0.0.1',0));sock.listen(32);sock.setblocking(False)
    port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='critical',access_log=False,
                                        lifespan='on',timeout_graceful_shutdown=1))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():await task;raise RuntimeError('server did not start')
                await asyncio.sleep(.005)
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=1) as client:
            ready=await client.get('/readyz')
            prediction=await client.post('/v1/predict',json=PAYLOAD)
            await entered(backend)  # Remove the first job's handshake.
            submitted=await client.post('/v1/jobs',json=PAYLOAD)
            await entered(backend);await asyncio.wait_for(svc.queue.join(),1)
            polled=await client.get(submitted.headers['Location'])
            request=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD|{'budget_ms':1000},
                       timeout=httpx.Timeout(1,read=.02)))
            await entered(backend);read_timeout=False
            try:await request
            except httpx.ReadTimeout:read_timeout=True
            await asyncio.wait_for(svc.queue.join(),1)
            continued=svc.get('j0003')
            return dict(transport='TCP loopback, ephemeral port; no TLS/proxy',
                        statuses=[ready.status_code,prediction.status_code,submitted.status_code,polled.status_code],
                        async_final=polled.json()['state'],client_read_timeout=read_timeout,
                        job_after_client_timeout=continued['state'],model_calls=backend.call_count,
                        note='This endpoint does not auto-cancel on socket disconnect; explicit ASGI task cancellation is a separate check')
    finally:
        server.should_exit=True
        try:await asyncio.wait_for(task,3)
        finally:sock.close()
