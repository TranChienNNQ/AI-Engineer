"""Analytic examples; counts are work proxies, not latency predictions."""
import math


def kv_bytes(layers,kv_heads,head_dim,tokens,bytes_per_element=2):
    if any(type(v) is not int or v<1 for v in [layers,kv_heads,head_dim,bytes_per_element]) or any(type(t) is not int or t<0 for t in tokens):raise ValueError('memory dimensions')
    return 2*layers*kv_heads*head_dim*sum(tokens)*bytes_per_element


def work(prompt,new):
    if type(prompt) is not int or type(new) is not int or min(prompt,new)<1:raise ValueError('positive P and N')
    n=list(range(prompt,prompt+new))
    # Logical causal query-key pairs per head/layer, not rectangular masked matmul FLOPs.
    return dict(uncached_positions=sum(n),cached_positions=prompt+new-1,
                uncached_causal_pairs=sum(t*(t+1)//2 for t in n),
                cached_causal_pairs=prompt*(prompt+1)//2+sum(n[1:]))


def padding(lengths,batch_size=2):
    if not lengths or any(type(v) is not int or v<1 for v in lengths) or type(batch_size) is not int or batch_size<1:raise ValueError('lengths/batch size')
    def slots(xs):return sum(len(xs[i:i+batch_size])*max(xs[i:i+batch_size]) for i in range(0,len(xs),batch_size))
    return dict(valid=sum(lengths),input_order_dense=slots(lengths),bucketed_dense=slots(sorted(lengths)),
                input_order=list(lengths),bucketed_order=sorted(lengths))


def hand():
    scores=[1.,0.,1.];weights=[math.exp(z)/(2*math.e+1) for z in scores]
    values=[2.,4.,8.];out=sum(p*v for p,v in zip(weights,values))
    return dict(attention=dict(scores=scores,weights=weights,values=values,output=out),work=work(4,3),
                memory=dict(mha_4096_bytes=kv_bytes(32,32,128,[4096]),gqa_4096_bytes=kv_bytes(32,8,128,[4096]),
                            ragged_bytes=kv_bytes(2,2,8,[3,5],4),dense_capacity_bytes=kv_bytes(2,2,8,[16,16],4)),
                padding=padding([2,9,3,8]))
