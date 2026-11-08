"""Tiny causal LM with explicit Linear projections so adapter targets are inspectable."""
import copy
import hashlib
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from data import PAD,BOS
from quantization import NF4Linear


@dataclass
class Config:
    dim:int=24
    heads:int=4
    ff:int=48
    max_len:int=32


class TinyLM(nn.Module):
    def __init__(self,vocab,cfg=None):
        super().__init__();self.cfg=cfg or Config();c=self.cfg
        self.embedding=nn.Embedding(vocab,c.dim,padding_idx=PAD);self.position=nn.Embedding(c.max_len,c.dim)
        self.norm1=nn.LayerNorm(c.dim);self.norm2=nn.LayerNorm(c.dim);self.final_norm=nn.LayerNorm(c.dim)
        for name in ['q_proj','k_proj','v_proj','o_proj']:setattr(self,name,nn.Linear(c.dim,c.dim,bias=False))
        self.up_proj=nn.Linear(c.dim,c.ff,bias=False);self.down_proj=nn.Linear(c.ff,c.dim,bias=False)
        self.head=nn.Linear(c.dim,vocab,bias=False)
    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[1]>self.cfg.max_len or ids.shape[1]<1 or not ids[:,0].eq(BOS).all():raise ValueError('BOS batch within context')
        padding=ids.eq(PAD)
        if ((padding.cumsum(1)>0)&~padding).any():raise ValueError('right padding only')
        b,t=ids.shape;c=self.cfg
        x=self.embedding(ids)+self.position(torch.arange(t,device=ids.device));z=self.norm1(x)
        q,k,v=[getattr(self,name)(z).view(b,t,c.heads,c.dim//c.heads).transpose(1,2) for name in ['q_proj','k_proj','v_proj']]
        allowed=torch.ones(t,t,dtype=torch.bool,device=ids.device).tril()[None,None]&~padding[:,None,None,:]
        a=(q@k.transpose(-2,-1)/math.sqrt(c.dim//c.heads)).masked_fill(~allowed,-torch.inf).softmax(-1)
        context=(a@v).transpose(1,2).reshape(b,t,c.dim)
        x=x+self.o_proj(context);x=x+self.down_proj(F.gelu(self.up_proj(self.norm2(x))))
        return self.head(self.final_norm(x))


class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__()
        if type(rank) is not int or rank<1 or alpha<=0:raise ValueError('rank/alpha')
        self.base=base;self.rank=rank;self.alpha=alpha;self.scale=alpha/rank
        for p in base.parameters():p.requires_grad_(False)
        self.A=nn.Parameter(torch.randn(rank,base.in_features)*.02)
        self.B=nn.Parameter(torch.zeros(base.out_features,rank))
    def forward(self,x):return self.base(x)+self.scale*F.linear(F.linear(x,self.A),self.B)


TARGETS=['q_proj','v_proj','up_proj','down_proj','head']


def adapt(base,quantized=False,rank=4,alpha=8,targets=None):
    model=copy.deepcopy(base);targets=TARGETS if targets is None else list(targets)
    names=[name for name,m in model.named_children() if isinstance(m,nn.Linear)]
    if not targets or len(set(targets))!=len(targets) or not set(targets)<=set(names):raise ValueError('target modules')
    for p in model.parameters():p.requires_grad_(False)
    for name in names:
        layer=getattr(model,name)
        if quantized:layer=NF4Linear(layer)
        if name in targets:layer=LoRALinear(layer,rank,alpha)
        setattr(model,name,layer)
    return model


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items() if not k.endswith(('.A','.B'))}


def tensor_bytes(model):return sum(v.numel()*v.element_size() for v in model.state_dict().values())


def merged(model):
    model=copy.deepcopy(model).eval()
    for name,layer in list(model.named_children()):
        if isinstance(layer,LoRALinear):
            b=layer.base;w=b.weight_value() if isinstance(b,NF4Linear) else b.weight.detach()
            dense=nn.Linear(b.in_features,b.out_features,bias=b.bias is not None)
            with torch.no_grad():
                dense.weight.copy_(w+layer.scale*(layer.B@layer.A))
                if b.bias is not None:dense.bias.copy_(b.bias)
            setattr(model,name,dense)
    return model
