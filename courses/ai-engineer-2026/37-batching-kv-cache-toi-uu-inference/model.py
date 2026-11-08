"""Teaching decoder: two layers, GQA, explicit positions and preallocated dense KV.
CPU inference only. Buffers are per request; no shared prefix cache or paged kernel.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
PAD,BOS,EOS=0,1,2


@dataclass(frozen=True)
class Config:
    vocab:int=32
    dim:int=32
    heads:int=4
    kv_heads:int=2
    layers:int=2
    ff:int=64
    max_len:int=128
    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in vars(self).values()):raise ValueError('positive integer config')
        if self.dim%self.heads or self.heads%self.kv_heads or self.vocab<=EOS:raise ValueError('head/vocab shape')


@dataclass
class KVCache:
    keys:list
    values:list
    lengths:torch.Tensor
    request_ids:tuple
    owner:int
    capacity:int
    versions:tuple
    @property
    def reserved_bytes(self):return sum(t.numel()*t.element_size() for t in self.keys+self.values)
    @property
    def used_bytes(self):
        t=self.keys[0]
        return 2*len(self.keys)*int(self.lengths.sum())*t.shape[1]*t.shape[-1]*t.element_size()
    @torch.inference_mode()
    def select(self,indices):
        indices=list(indices)
        if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(self.request_ids) for i in indices):raise ValueError('nonempty unique row selection')
        ix=torch.tensor(indices,dtype=torch.long,device=self.lengths.device)
        return KVCache([t.index_select(0,ix) for t in self.keys],[t.index_select(0,ix) for t in self.values],
                       self.lengths.index_select(0,ix),tuple(self.request_ids[i] for i in indices),self.owner,self.capacity,self.versions)


def pack(sequences):
    if not sequences or any(not row for row in sequences):raise ValueError('nonempty sequences')
    lengths=torch.tensor([len(row) for row in sequences],dtype=torch.long)
    ids=torch.full((len(sequences),int(lengths.max())),PAD,dtype=torch.long)
    for i,row in enumerate(sequences):ids[i,:len(row)]=torch.tensor(row)
    return ids,lengths


def allowed_mask(old,lengths,width,key_width=None):
    """True participates in attention; absolute positions exclude future and PAD keys."""
    total=old+lengths;key=torch.arange(int(total.max()) if key_width is None else key_width,device=old.device)
    query=old[:,None]+torch.arange(width,device=old.device)
    return (key[None,None,:]<total[:,None,None])&(key[None,None,:]<=query[:,:,None])


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;dh=c.dim//c.heads
        self.n1=nn.LayerNorm(c.dim);self.n2=nn.LayerNorm(c.dim)
        self.q=nn.Linear(c.dim,c.dim,bias=False);self.k=nn.Linear(c.dim,c.kv_heads*dh,bias=False);self.v=nn.Linear(c.dim,c.kv_heads*dh,bias=False)
        self.o=nn.Linear(c.dim,c.dim,bias=False)
        self.ff=nn.Sequential(nn.Linear(c.dim,c.ff),nn.GELU(),nn.Linear(c.ff,c.dim))
    def forward(self,x,valid,old,lengths,cache,layer,backend):
        b,t,d=x.shape;c=self.c;dh=d//c.heads;z=self.n1(x)
        q=self.q(z).view(b,t,c.heads,dh).transpose(1,2)
        k=self.k(z).view(b,t,c.kv_heads,dh).transpose(1,2);v=self.v(z).view(b,t,c.kv_heads,dh).transpose(1,2)
        if cache is not None:
            for i in range(b):
                start=int(old[i]);n=int(lengths[i]);cache.keys[layer][i,:,start:start+n]=k[i,:,:n];cache.values[layer][i,:,start:start+n]=v[i,:,:n]
            end=int((old+lengths).max());k=cache.keys[layer][:,:,:end];v=cache.values[layer][:,:,:end]
        # Repeat only transient K/V for this simple implementation; persistent cache stays Hkv.
        k=k.repeat_interleave(c.heads//c.kv_heads,1);v=v.repeat_interleave(c.heads//c.kv_heads,1)
        mask=allowed_mask(old,lengths,t,k.shape[-2])[:,None]
        if backend=='manual':a=(q@k.transpose(-2,-1)/math.sqrt(dh)).masked_fill(~mask,-torch.inf).softmax(-1);out=a@v
        else:out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False)
        x=x+self.o(out.transpose(1,2).reshape(b,t,d));x=x+self.ff(self.n2(x))
        return x.masked_fill(~valid[:,:,None],0)


class TinyDecoder(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.c=c or Config();c=self.c
        self.embedding=nn.Embedding(c.vocab,c.dim,padding_idx=PAD);self.position=nn.Embedding(c.max_len,c.dim)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)]);self.norm=nn.LayerNorm(c.dim);self.head=nn.Linear(c.dim,c.vocab,bias=False)
    def new_cache(self,request_ids,capacity):
        request_ids=tuple(request_ids);c=self.c
        if not request_ids or any(not isinstance(x,str) or not x for x in request_ids) or len(set(request_ids))!=len(request_ids):raise ValueError('unique nonempty request ids')
        if type(capacity) is not int or not 1<=capacity<=c.max_len:raise ValueError('cache capacity')
        w=self.embedding.weight;b=len(request_ids);shape=(b,c.kv_heads,capacity,c.dim//c.heads)
        return KVCache([w.new_zeros(shape) for _ in range(c.layers)],[w.new_zeros(shape) for _ in range(c.layers)],
                       torch.zeros(b,dtype=torch.long,device=w.device),request_ids,id(self),capacity,tuple((id(p),p._version) for p in self.parameters()))
    @torch.inference_mode()
    def forward(self,ids,lengths,cache=None,request_ids=None,backend='manual'):
        c=self.c
        if self.training:raise ValueError('call eval() for this inference-only lab')
        if backend not in ['manual','sdpa']:raise ValueError('attention backend')
        if ids.ndim!=2 or ids.dtype!=torch.long or lengths.dtype!=torch.long or lengths.shape!=(ids.shape[0],) or not ids.numel():raise ValueError('input shape/dtype')
        if ids.device!=self.embedding.weight.device or lengths.device!=ids.device:raise ValueError('device')
        b,t=ids.shape
        if (lengths<1).any() or (lengths>t).any() or (ids<0).any() or (ids>=c.vocab).any():raise ValueError('length/token range')
        valid=torch.arange(t,device=ids.device)[None]<lengths[:,None]
        if (ids[~valid]!=PAD).any() or (ids[valid]==PAD).any():raise ValueError('right PAD only outside valid tokens')
        old=torch.zeros_like(lengths)
        if cache is not None:
            if cache.owner!=id(self) or cache.request_ids!=tuple(request_ids or ()) or len(cache.request_ids)!=b or cache.versions!=tuple((id(p),p._version) for p in self.parameters()):raise ValueError('cache owner/request mapping')
            shape=(b,c.kv_heads,cache.capacity,c.dim//c.heads)
            if len(cache.keys)!=c.layers or len(cache.values)!=c.layers or cache.lengths.shape!=(b,) or cache.lengths.dtype!=torch.long:raise ValueError('cache shape')
            if any(z.shape!=shape or z.dtype!=self.embedding.weight.dtype or z.device!=ids.device for z in cache.keys+cache.values):raise ValueError('cache storage')
            old=cache.lengths.clone()
            if (old<0).any() or (old+lengths>cache.capacity).any():raise ValueError('cache capacity exceeded')
        if (old+lengths>c.max_len).any():raise ValueError('model context exceeded')
        if (ids[old==0,0]!=BOS).any():raise ValueError('prefill starts with BOS')
        pos=old[:,None]+torch.arange(t,device=ids.device);pos=pos.masked_fill(~valid,0)
        x=(self.embedding(ids)+self.position(pos)).masked_fill(~valid[:,:,None],0)
        for i,block in enumerate(self.blocks):x=block(x,valid,old,lengths,cache,i,backend)
        if cache is not None:cache.lengths.copy_(old+lengths)
        return self.head(self.norm(x)).masked_fill(~valid[:,:,None],0)


def make_model(c=None):
    torch.manual_seed(37);return TinyDecoder(c).eval().requires_grad_(False)
