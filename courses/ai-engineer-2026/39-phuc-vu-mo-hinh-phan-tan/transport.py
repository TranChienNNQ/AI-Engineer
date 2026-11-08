"""Trusted local IPC helpers. No HTTP/RPC network, NCCL or shared-memory tensor transport."""
import multiprocessing as mp
import os
from multiprocessing.connection import wait
from model import forward,digest,payload_bytes,REVISION
import numpy as np


def worker(conn,payload,kind,epoch,revision,model_hash,downstream=None):
    try:
        if kind=='dp':forward(np.zeros((1,4),dtype=np.float32),payload) # warmup before readiness
        conn.send(dict(type='READY',pid=os.getpid(),epoch=epoch,revision=revision,model_hash=model_hash,
                       shard_hash=digest(payload),weight_bytes=payload_bytes(payload)))
        while True:
            msg=conn.recv()
            if msg['type']=='STOP':break
            if msg['type']!='RUN':raise ValueError('worker protocol')
            conn.send(dict(type='STARTED',id=msg['id'],epoch=epoch,revision=revision))
            if msg.get('crash'):os._exit(17) # intentional fail-stop fixture
            if msg.get('hold'):
                release=conn.recv()
                if release.get('type')=='STOP':break
                if release.get('type')!='RELEASE' or release.get('id')!=msg['id']:raise ValueError('hold protocol')
            x=msg['x']
            if kind=='dp':
                import time
                expired=time.monotonic()>=msg['deadline']
                y=None if expired else forward(x,payload)
                conn.send(dict(type='RESULT',id=msg['id'],epoch=epoch,revision=msg.get('reply_revision') or revision,
                               status='EXPIRED' if expired else 'OK',y=y))
            elif kind=='tp':
                y=np.maximum(x@payload['w1']+payload['b1'],0)@payload['w2']
                conn.send(dict(type='RESULT',id=msg['id'],epoch=epoch,revision=revision,status='OK',y=y))
            elif kind=='pp0':
                h=np.maximum(x@payload['w1']+payload['b1'],0)
                downstream.send(dict(type='STAGE',id=msg['id'],revision=revision,model_hash=model_hash,h=h))
                conn.send(dict(type='RESULT',id=msg['id'],epoch=epoch,revision=revision,status='OK',y=None))
            else:raise ValueError('worker kind')
    except (EOFError,BrokenPipeError):pass
    finally:
        conn.close()
        if downstream is not None:downstream.close()


def final_stage(conn,incoming,payload,epoch,revision,model_hash):
    try:
        conn.send(dict(type='READY',pid=os.getpid(),epoch=epoch,revision=revision,model_hash=model_hash,
                       shard_hash=digest(payload),weight_bytes=payload_bytes(payload)))
        while True:
            msg=incoming.recv()
            if msg['type']=='STOP':break
            if msg['type']!='STAGE' or msg['revision']!=revision or msg['model_hash']!=model_hash:raise ValueError('stage identity')
            y=msg['h']@payload['w2']+payload['b2']
            conn.send(dict(type='RESULT',id=msg['id'],epoch=epoch,revision=revision,status='OK',y=y))
    except (EOFError,BrokenPipeError):pass
    finally:conn.close();incoming.close()


def start(payload,kind,epoch=1,revision=REVISION,model_hash='',downstream=None):
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,payload,kind,epoch,revision,model_hash,downstream));p.start();child.close()
    if not parent.poll(10):stop(p,parent);raise RuntimeError('readiness timeout')
    ready=parent.recv()
    if ready.get('type')!='READY' or ready['epoch']!=epoch or ready['revision']!=revision or ready['model_hash']!=model_hash or ready['shard_hash']!=digest(payload):stop(p,parent);raise RuntimeError('readiness mismatch')
    return p,parent,ready


def receive_result(conn,request_id,revision=REVISION,epoch=1,timeout=10):
    import time
    deadline=time.monotonic()+timeout
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0 or not conn.poll(remaining):raise TimeoutError('IPC response deadline')
        msg=conn.recv()
        if msg.get('id')!=request_id or msg.get('revision')!=revision or msg.get('epoch')!=epoch:raise ValueError('response identity')
        if msg['type']=='RESULT':return msg
        if msg['type']!='STARTED':raise ValueError('response protocol')


def stop(process,conn):
    try:
        if process.is_alive():conn.send(dict(type='STOP'))
    except (OSError,EOFError):pass
    process.join(.5)
    if process.is_alive():process.terminate();process.join(2)
    conn.close()
    if process.is_alive():raise RuntimeError('child did not stop')
