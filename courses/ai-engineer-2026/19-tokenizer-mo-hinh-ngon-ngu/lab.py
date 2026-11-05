"""Byte BPE + causal Transformer language model. Original synthetic backend text, CPU."""
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import tempfile
import torch
from torch import nn
from torch.nn import functional as F

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


def data(seed=19):
    services=['api','worker','gateway','billing','search','cache','auth','queue']
    issues=['timeout','lỗi kết nối','phản hồi chậm','thiếu quyền']
    actions=['đọc hồ sơ','ghi đơn hàng','gọi dịch vụ','xác thực','đọc cấu hình','cập nhật trạng thái']
    texts=[f'Dịch vụ {s}: {i} khi {a}.\n' for s,i,a in itertools.product(services,issues,actions)]
    random.Random(seed).shuffle(texts)
    return dict(train=texts[:128],validation=texts[128:160],test=texts[160:])


def batch(texts,tok):
    sequences=[[BOS]+tok.encode(s)+[EOS] for s in texts]
    length=max(map(len,sequences))-1
    x=torch.full((len(texts),length),PAD,dtype=torch.long)
    y=torch.full_like(x,IGNORE)
    for i,seq in enumerate(sequences):
        x[i,:len(seq)-1]=torch.tensor(seq[:-1]); y[i,:len(seq)-1]=torch.tensor(seq[1:])
    return x,y

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

@torch.no_grad()
def evaluate(model,texts,tok,cfg):
    model.eval(); total=correct=count=0
    for start in range(0,len(texts),cfg.batch_size):
        x,y=batch(texts[start:start+cfg.batch_size],tok); logits=model(x); valid=y.ne(IGNORE)
        total+=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=IGNORE,reduction='sum').item()
        correct+=int((logits.argmax(-1)[valid]==y[valid]).sum()); count+=int(valid.sum())
    ce=total/count
    return dict(ce=ce,perplexity=math.exp(ce),token_accuracy=correct/count,tokens=count)


def count_baselines(train,test,tok):
    uni=Counter(); bi=Counter(); context=Counter()
    for text in train:
        seq=[BOS]+tok.encode(text)+[EOS]
        uni.update(seq[1:]); bi.update(zip(seq,seq[1:])); context.update(seq[:-1])
    n=sum(uni.values()); v=tok.size
    losses=[0.,0.]; count=0
    for text in test:
        seq=[BOS]+tok.encode(text)+[EOS]
        for a,b in zip(seq,seq[1:]):
            losses[0]-=math.log((uni[b]+1)/(n+v))
            losses[1]-=math.log((bi[a,b]+1)/(context[a]+v)); count+=1
    return {name:dict(ce=loss/count,perplexity=math.exp(loss/count)) for name,loss in zip(['unigram_add1','bigram_add1'],losses)}

@torch.no_grad()
def generate(model,tok,prompt,max_new=32):
    ids=[BOS]+tok.encode(prompt)
    if max_new<1 or len(ids)+max_new-1>model.cfg.max_len: raise ValueError('Prompt plus generation exceeds context budget')
    model.eval(); generated=[]; stopped=False
    for _ in range(max_new):
        logits=model(torch.tensor([ids]))[0,-1].clone()
        logits[PAD]=logits[BOS]=-float('inf')  # Structural constraint, not an unconstrained argmax.
        token=int(logits.argmax()); generated.append(token)
        if token==EOS: stopped=True; break
        ids.append(token)
    raw=tok.raw(generated)
    try: continuation=raw.decode('utf-8'); valid=True
    except UnicodeDecodeError: continuation=raw.decode('utf-8',errors='replace'); valid=False
    return dict(prompt=prompt,continuation=continuation,ids=generated,eos=stopped,valid_utf8=valid)


def load(run):
    tok=ByteBPE.load(run/'tokenizer.json')
    cp=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    if cp['tokenizer_hash']!=tok.fingerprint(): raise ValueError('Tokenizer/checkpoint mismatch')
    cfg=Config(**cp['config']); model=LanguageModel(tok.size,cfg)
    model.load_state_dict(cp['model']); return model.eval(),tok,cfg,cp


def train(out,cfg=Config()):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(cfg.seed)
    if cfg.epochs<1: raise ValueError('epochs must be positive')
    records=data(); tok=ByteBPE.fit(records['train'])
    tok.save(out/'tokenizer.json'); (out/'data.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    model=LanguageModel(tok.size,cfg); opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    rng=random.Random(cfg.seed); rows=[]; best=float('inf')
    for epoch in range(1,cfg.epochs+1):
        order=list(records['train']); rng.shuffle(order); model.train(); total=count=0
        for start in range(0,len(order),cfg.batch_size):
            x,y=batch(order[start:start+cfg.batch_size],tok)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=IGNORE)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            tokens=int(y.ne(IGNORE).sum()); total+=loss.item()*tokens; count+=tokens
        val=evaluate(model,records['validation'],tok,cfg)
        rows.append(dict(epoch=epoch,train_ce=total/count,val_ce=val['ce']))
        if val['ce']<best:
            best=val['ce']; torch.save(dict(model=model.state_dict(),config=asdict(cfg),epoch=epoch,tokenizer_hash=tok.fingerprint()),out/'best.pt')
    model,tok,cfg,cp=load(out)
    report=dict(vocab=tok.size,parameters=sum(p.numel() for p in model.parameters()),best_epoch=cp['epoch'],
                validation=evaluate(model,records['validation'],tok,cfg),test=evaluate(model,records['test'],tok,cfg),
                baselines=count_baselines(records['train'],records['test'],tok),
                tokenizer=dict(train_bytes=sum(len(s.encode()) for s in records['train']),train_tokens=sum(len(tok.encode(s)) for s in records['train'])),
                samples=[generate(model,tok,p,32) for p in ['Dịch vụ api: ','Dịch vụ auth: ']],
                metadata=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=1),config=asdict(cfg))
    (out/'metrics.json').write_text(json.dumps(rows,indent=2)); (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2)); return report


def tokenizer_demo():
    tok=ByteBPE.fit(['aba','aba','ac'],steps=2)
    assert tok.merges==[(100,101),(259,100)]
    assert tok.encode('aba')==[260]
    for text in ['Xin chào!','  api\n\t🙂','<BOS>','e\u0301','é','']:
        assert tok.decode(tok.encode(text))==text
    print('merges:',tok.merges,'aba:',tok.encode('aba'),'ac:',tok.encode('ac'))
    print('é code points / bytes:',len('é'),len('é'.encode()))
    print('PASS: Unicode, whitespace, literal control text roundtrip')


def hand_demo():
    p=[.5,.25,.125]; ce=-sum(math.log(x) for x in p)/3
    assert math.isclose(ce,math.log(4))
    print(dict(sequence_probability=math.prod(p),ce=ce,ppl=math.exp(ce)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['tokenizer','hand','train','all'],default='tokenizer')
    p.add_argument('--out',type=Path); p.add_argument('--epochs',type=int,default=30); args=p.parse_args()
    if args.mode in ('tokenizer','all'): tokenizer_demo()
    if args.mode in ('hand','all'): hand_demo()
    if args.mode in ('train','all'):
        out=args.out or Path(tempfile.mkdtemp(prefix='lesson19-'))/'run'
        train(out,Config(epochs=args.epochs)); print('Run:',out)
if __name__=='__main__': main()
