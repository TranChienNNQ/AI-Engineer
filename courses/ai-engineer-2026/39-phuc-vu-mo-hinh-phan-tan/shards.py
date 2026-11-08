"""Real local processes: replicated model, TP partial sum, and PP activation transfer."""
import multiprocessing as mp
import os
import numpy as np
from model import weights,tp_parts,pp_parts,forward,digest,payload_bytes,validate,REVISION
from transport import start,stop,receive_result,final_stage


def demo():
    w=weights();identity=digest(w);x=np.array([[1,2,0,-1],[0,1,2,3],[-1,0,1,2],[3,-1,0,1]],dtype=np.float32)
    expected=forward(x,w);result={};all_pids=[]
    for name,parts in [('replica',[w,w]),('tensor',tp_parts(w))]:
        handles=[]
        try:
            for part in parts:handles.append(start(part,'dp' if name=='replica' else 'tp',model_hash=identity))
            responses=[]
            for i,(p,c,ready) in enumerate(handles):
                c.send(dict(type='RUN',id=f'{name}-{i}',x=x[i::2] if name=='replica' else x,deadline=float('inf')))
            for i,(p,c,ready) in enumerate(handles):responses.append(receive_result(c,f'{name}-{i}')['y'])
            if name=='replica':
                actual=np.empty_like(expected)
                for i,y in enumerate(responses):actual[i::2]=y
            else:actual=responses[0]+responses[1]+w['b2'] # reduce to coordinator; not all-reduce
            if not np.allclose(actual,expected,atol=1e-6):raise AssertionError(name+' numerical mismatch')
            all_pids.extend(h[2]['pid'] for h in handles)
            result[name]=dict(outputs=actual.tolist(),max_error=float(np.abs(actual-expected).max()),
                              worker_weight_bytes=[h[2]['weight_bytes'] for h in handles],
                              coordinator_bias_bytes=w['b2'].nbytes if name=='tensor' else 0,
                              process_ids=[h[2]['pid'] for h in handles],messages='replicas return independent rows' if name=='replica' else 'two partial matrices reduced once; bias added once')
        finally:
            for p,c,_ in handles:stop(p,c)
    ctx=mp.get_context('spawn');incoming,sender=ctx.Pipe(duplex=False);parent,child=ctx.Pipe();parts=pp_parts(w)
    stage1=ctx.Process(target=final_stage,args=(child,incoming,parts[1],1,REVISION,identity));stage1.start();child.close();incoming.close()
    handle0=None
    try:
        if not parent.poll(10):raise TimeoutError('pipeline readiness')
        ready1=parent.recv()
        if ready1['revision']!=REVISION or ready1['model_hash']!=identity or ready1['shard_hash']!=digest(parts[1]):raise ValueError('pipeline readiness identity')
        handle0=start(parts[0],'pp0',model_hash=identity,downstream=sender);p0,c0,ready0=handle0
        responses=[]
        # Real stage-to-stage activation transfer for four one-row microbatches.
        for i in range(len(x)):c0.send(dict(type='RUN',id=f'pp-{i}',x=x[i:i+1]))
        for i in range(len(x)):
            receive_result(c0,f'pp-{i}')
            responses.append(receive_result(parent,f'pp-{i}')['y'])
        actual=np.concatenate(responses);assert np.allclose(actual,expected,atol=1e-6)
        result['pipeline']=dict(outputs=actual.tolist(),max_error=float(np.abs(actual-expected).max()),worker_weight_bytes=[ready0['weight_bytes'],ready1['weight_bytes']],
                                microbatches=len(x),activation_bytes_per_microbatch=4*4,process_ids=[ready0['pid'],ready1['pid']])
        all_pids.extend([ready0['pid'],ready1['pid']]);sender.send(dict(type='STOP'))
    finally:
        if handle0 is not None:stop(handle0[0],handle0[1])
        sender.close();stage1.join(.5)
        if stage1.is_alive():stage1.terminate();stage1.join(2)
        parent.close()
        if stage1.is_alive():raise RuntimeError('pipeline child leak')
    return dict(input=x.tolist(),reference=expected.tolist(),model_hash=identity,model_weight_bytes=payload_bytes(w),
                modes=result,all_children_distinct_from_parent=all(pid!=os.getpid() for pid in all_pids),
                children_joined=True,scope='spawn processes on one CPU host, trusted Pipe serialization; no GPU/NCCL/all-reduce/network cluster or speed measurement')


if __name__=='__main__':
    import json
    print(json.dumps(demo(),indent=2))
