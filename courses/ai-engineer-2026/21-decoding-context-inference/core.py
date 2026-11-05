"""Core carried from lesson 19: byte BPE and one-block causal LM (no new external dependency)."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
PAD, BOS, EOS, IGNORE = 0, 1, 2, -100

class ByteBPE:
    """All 256 bytes, no normalization. Merges stay inside whitespace/nonwhitespace runs."""
    def __init__(self, merges=None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.pieces = {i+3: bytes([i]) for i in range(256)}
        for i, (a,b) in enumerate(self.merges,259):
            self.pieces[i] = self.pieces[a] + self.pieces[b]
    @property
    def size(self): return 259 + len(self.merges)
    @staticmethod
    def segments(text):
        return [[b+3 for b in part.encode('utf-8')] for part in re.findall(r'\s+|\S+',text)]
    @staticmethod
    def merge(seq, pair, new):
        out=[]; i=0
        while i<len(seq):
            if i+1<len(seq) and (seq[i],seq[i+1])==pair:
                out.append(new); i+=2
            else: out.append(seq[i]); i+=1
        return out
    @classmethod
    def fit(cls, texts, steps=64):
        sequences=[seq for text in texts for seq in cls.segments(text)]
        merges=[]
        for new in range(259,259+steps):
            counts=Counter(pair for seq in sequences for pair in zip(seq,seq[1:]))
            if not counts: break
            # Frequency first; numeric pair ascending makes ties deterministic.
            pair=min(counts,key=lambda p:(-counts[p],p))
            merges.append(pair)
            sequences=[cls.merge(seq,pair,new) for seq in sequences]
        return cls(merges)
    def encode(self,text):
        result=[]
        for seq in self.segments(text):
            for new,pair in enumerate(self.merges,259): seq=self.merge(seq,pair,new)
            result.extend(seq)
        return result
    def raw(self,ids):
        result=b''
        for token in ids:
            if token in (PAD,BOS,EOS): continue
            if token not in self.pieces: raise ValueError('Unknown token ID')
            result+=self.pieces[token]
        return result
    def decode(self,ids,errors='strict'): return self.raw(ids).decode('utf-8',errors=errors)
    def save(self,path):
        Path(path).write_text(json.dumps(dict(version=1,normalization='none',pretokenizer='whitespace-runs',merges=self.merges),indent=2)+'\n')
    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text())
        if obj['version']!=1 or obj['normalization']!='none' or obj['pretokenizer']!='whitespace-runs':
            raise ValueError('Unsupported tokenizer config')
        return cls(obj['merges'])
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.merges).encode()).hexdigest()

class Block(nn.Module):
    # Same pre-norm structure as lesson 18, written locally for standalone execution.
    def __init__(self,d,h,ff):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,h,dropout=0.,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.ReLU(),nn.Linear(ff,d))
    def forward(self,x,mask,padding):
        z=self.n1(x)
        x=x+self.attn(z,z,z,attn_mask=mask,key_padding_mask=padding,need_weights=False)[0]
        return x+self.ff(self.n2(x))

@dataclass
class Config:
    dim:int=32
    heads:int=4
    ff:int=64
    max_len:int=128
    epochs:int=30
    batch_size:int=16
    lr:float=.003
    seed:int=2026

class LanguageModel(nn.Module):
    def __init__(self,vocab,cfg):
        super().__init__(); self.cfg=cfg
        self.embedding=nn.Embedding(vocab,cfg.dim,padding_idx=PAD)
        self.position=nn.Embedding(cfg.max_len,cfg.dim)
        self.block=Block(cfg.dim,cfg.heads,cfg.ff)
        self.norm=nn.LayerNorm(cfg.dim); self.head=nn.Linear(cfg.dim,vocab)
    def forward(self,ids):
        if ids.ndim!=2 or not 0<ids.shape[1]<=self.cfg.max_len or not ids[:,0].eq(BOS).all():
            raise ValueError('Expected BOS-started [B,T] within context limit')
        padding=ids.eq(PAD)
        if ((padding.cumsum(1)>0)&~padding).any(): raise ValueError('Right padding only')
        x=self.embedding(ids)*math.sqrt(self.cfg.dim)+self.position(torch.arange(ids.shape[1]))
        mask=torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).triu(1)
        return self.head(self.norm(self.block(x,mask,padding)))
