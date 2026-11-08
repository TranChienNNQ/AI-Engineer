"""Tiny causal LM adapted from lesson 35, standalone for sequence DPO."""
import hashlib
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from data import PAD,BOS


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


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
