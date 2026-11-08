"""NumPy teaching algorithms. Python tiling is not a custom native/GPU kernel."""
import math
import numpy as np


def positive(v):
    if type(v) is not int or v<1:raise ValueError('positive integer required')


def matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('finite nonempty matrix required')
    return x


def tiled_matmul(a,b,tile=2):
    a,b=matrix(a),matrix(b);positive(tile)
    m,k=a.shape;kb,n=b.shape
    if k!=kb:raise ValueError('inner dimension mismatch')
    c=np.zeros((m,n));peak=0;products=0
    for i in range(0,m,tile):
        for j in range(0,n,tile):
            acc=np.zeros((min(tile,m-i),min(tile,n-j)))
            for h in range(0,k,tile):
                block=a[i:i+tile,h:h+tile]@b[h:h+tile,j:j+tile]
                acc+=block;peak=max(peak,block.size);products+=1
            c[i:i+tile,j:j+tile]=acc
    return c,dict(tile_products=products,max_product_elements=peak,output_elements=c.size)


def attention_inputs(q,k,v,mask):
    q,k,v=matrix(q),matrix(k),matrix(v)
    if q.shape[1]!=k.shape[1] or k.shape[0]!=v.shape[0]:raise ValueError('Q/K/V shape mismatch')
    if mask is not None:
        if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.shape!=(len(q),len(k)):raise ValueError('boolean [M,N] allowed-mask required')
    return q,k,v


def shifted_exp(s,m):
    delta=np.full_like(s,-np.inf)
    np.subtract(s,m[:,None],out=delta,where=np.isfinite(s)&np.isfinite(m[:,None]))
    return np.exp(delta)


def dense_attention(q,k,v,mask=None):
    q,k,v=attention_inputs(q,k,v,mask)
    s=q@k.T/math.sqrt(q.shape[1])
    if not np.isfinite(s).all():raise ValueError('score overflow')
    if mask is not None:s=np.where(mask,s,-np.inf)
    p=shifted_exp(s,s.max(axis=1));den=p.sum(axis=1)
    p=np.divide(p,den[:,None],out=np.zeros_like(p),where=den[:,None]>0)
    return p@v


def online_attention(q,k,v,mask=None,query_tile=2,key_tile=2):
    q,k,v=attention_inputs(q,k,v,mask);positive(query_tile);positive(key_tile)
    out=np.zeros((len(q),v.shape[1]));peak=0;tiles=0
    for i in range(0,len(q),query_tile):
        qi=q[i:i+query_tile];m=np.full(len(qi),-np.inf);l=np.zeros(len(qi));z=np.zeros((len(qi),v.shape[1]))
        for j in range(0,len(k),key_tile):
            s=qi@k[j:j+key_tile].T/math.sqrt(q.shape[1])
            if not np.isfinite(s).all():raise ValueError('score overflow')
            if mask is not None:s=np.where(mask[i:i+len(qi),j:j+s.shape[1]],s,-np.inf)
            new_m=np.maximum(m,s.max(axis=1));alpha=np.zeros_like(m)
            valid=np.isfinite(m)&np.isfinite(new_m)
            alpha[valid]=np.exp(m[valid]-new_m[valid])
            p=shifted_exp(s,new_m)
            z=alpha[:,None]*z+p@v[j:j+key_tile];l=alpha*l+p.sum(axis=1);m=new_m
            peak=max(peak,s.size);tiles+=1
        out[i:i+len(qi)]=np.divide(z,l[:,None],out=np.zeros_like(z),where=l[:,None]>0)
    return out,dict(score_tiles=tiles,max_score_elements=peak,dense_score_elements=len(q)*len(k),
        scope='largest single score tile; not allocator peak/RSS; full mask if supplied remains owned by caller')


def demo():
    a=np.array([[1,2,3],[4,5,6]],dtype=float);b=np.array([[1,0],[0,1],[1,1]],dtype=float)
    c,t=tiled_matmul(a,b,2)
    rng=np.random.default_rng(40);q=rng.normal(size=(7,4));k=rng.normal(size=(11,4));v=rng.normal(size=(11,3))
    mask=np.arange(11)[None,:]<=np.arange(7)[:,None];mask[3]=False
    y,stats=online_attention(q,k,v,mask,3,4);expected=dense_attention(q,k,v,mask)
    return dict(matmul=dict(output=c.tolist(),stats=t),attention=dict(output=y.tolist(),max_error=float(np.abs(y-expected).max()),stats=stats,all_masked_row=y[3].tolist()))
