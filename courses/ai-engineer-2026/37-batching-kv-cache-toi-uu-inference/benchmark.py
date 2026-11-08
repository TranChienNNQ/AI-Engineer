"""Local CPU wall time only. Fixed output length, no server/network/GPU claims."""
import statistics
import time
import torch
from model import Config,make_model,pack
from generation import generate


def run(repeats=7):
    if type(repeats) is not int or repeats<3:raise ValueError('at least 3 repeats')
    torch.set_num_threads(1);m=make_model();g=torch.Generator().manual_seed(370)
    prompts=[[1]+torch.randint(3,32,(47,),generator=g).tolist() for _ in range(4)]
    # Ignore EOS for this fixed-work microbenchmark; same ban and argmax across all paths.
    def fixed(logits,step,active):
        scores=logits.clone();scores[:,:3]=-torch.inf;return scores.argmax(-1).tolist()
    configs=[('single_full',False,False,'manual'),('single_cache',True,False,'manual'),
             ('batch_cache',True,True,'manual'),('batch_cache_sdpa',True,True,'sdpa')]
    def call(cache,batched,backend):
        if batched:return generate(m,prompts,16,cache,backend,selector=fixed)['outputs']
        return [generate(m,[p],16,cache,backend,selector=fixed)['outputs'][0] for p in prompts]
    results={name:[] for name,*_ in configs};outputs={}
    for name,cache,batched,backend in configs:
        for _ in range(2):call(cache,batched,backend)
    # Rotate config order to avoid measuring the same path first in every round.
    for r in range(repeats):
        for i in range(len(configs)):
            name,cache,batched,backend=configs[(i+r)%len(configs)]
            start=time.perf_counter_ns();out=call(cache,batched,backend);elapsed=(time.perf_counter_ns()-start)/1e6
            results[name].append(elapsed);outputs[name]=out
    reference=outputs['single_full']
    if not all(v==reference for v in outputs.values()):raise AssertionError('benchmark output mismatch')
    table=[]
    for name,*_ in configs:
        v=results[name];median=statistics.median(v)
        table.append(dict(name=name,median_ms=median,min_ms=min(v),max_ms=max(v),samples_ms=v,
                          output_tokens=64,output_tokens_per_second=64/(median/1000)))
    # Local prefill+first selection TTFT and subsequent selection intervals, one request only.
    ids,lens=pack([prompts[0]]);cache=m.new_cache(['r0'],64);times=[]
    start=time.perf_counter_ns();logits=m(ids,lens,cache,['r0']);token=fixed(logits[:,-1],0,[0])[0];times.append(time.perf_counter_ns())
    for step in range(1,16):
        x,n=pack([[token]]);logits=m(x,n,cache,['r0']);token=fixed(logits[:,0],step,[0])[0];times.append(time.perf_counter_ns())
    return dict(device='cpu',threads=1,repeats=repeats,warmups_per_config=2,prompt_tokens=48,new_tokens=16,requests=4,
                model_parameters=sum(p.numel() for p in m.parameters()),rows=table,outputs_identical=True,
                token_probe=dict(first_token_ms=(times[0]-start)/1e6,itl_ms=[(b-a)/1e6 for a,b in zip(times,times[1:])],
                                 scope='one local cached request; initial pack/cache allocation excluded; decode packing and selection included; no queue/network/tokenizer'),
                scope='untrained TinyDecoder CPU, perf_counter_ns wall time, prefill and cache allocation included in table; forced fixed length; no quality or GPU claim')
