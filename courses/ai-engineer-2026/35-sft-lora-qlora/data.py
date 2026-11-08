"""Small synthetic instruction task; group split before tokenizer fit."""
import hashlib
import itertools
import json
import random
import re
import torch

PAD,BOS,EOS,USER,ASSISTANT,UNK=0,1,2,3,4,5
IGNORE=-100
EVENTS={'timeout':'NET','reset':'NET','denied':'AUTH','expired':'AUTH','deadlock':'DATA','duplicate':'DATA'}
TEMPLATE='roles-v1: BOS USER prompt ASSISTANT answer EOS; answer-only shifted labels'


def records():
    groups=list(itertools.product(['api','worker','gateway','billing','search','cache','auth','queue'],['read','write','check']))
    random.Random(35).shuffle(groups);splits={}
    for split,items in [('train',groups[:16]),('validation',groups[16:20]),('test',groups[20:])]:
        splits[split]=[dict(group=[s,o],prompt=f'route service {s} op {o} event {e}',answer=label,
                            text=f'service {s} op {o} event {e}') for s,o in items for e,label in EVENTS.items()]
    return splits


class Tokenizer:
    def __init__(self,vocab):
        self.vocab=dict(vocab);self.inverse={v:k for k,v in vocab.items()}
    @staticmethod
    def pieces(text):return re.findall(r'[A-Za-z0-9_-]+|[^\s]',text)
    @classmethod
    def fit(cls,rows):
        words=sorted({p for r in rows for key in ['prompt','answer','text'] for p in cls.pieces(r[key])})
        return cls({word:i+6 for i,word in enumerate(words)})
    @property
    def size(self):return len(self.vocab)+6
    def encode(self,text):return [self.vocab.get(p,UNK) for p in self.pieces(text)]
    def fingerprint(self):return hashlib.sha256(json.dumps(self.vocab,sort_keys=True).encode()).hexdigest()


def example(row,tok,sft=True):
    if sft:
        prefix=[BOS,USER]+tok.encode(row['prompt'])+[ASSISTANT]
        sequence=prefix+tok.encode(row['answer'])+[EOS]
    else:prefix=[];sequence=[BOS]+tok.encode(row['text'])+[EOS]
    x=sequence[:-1];y=sequence[1:]
    if sft:y[:len(prefix)-1]=[IGNORE]*(len(prefix)-1)
    if not any(v!=IGNORE for v in y):raise ValueError('no supervised token')
    return x,y


def batch(rows,tok,sft=True,max_len=32):
    if not rows:raise ValueError('empty batch')
    pairs=[example(r,tok,sft) for r in rows];length=max(len(x) for x,y in pairs)
    if length>max_len:raise ValueError('oversize: do not silently truncate answer')
    x=torch.full((len(rows),length),PAD,dtype=torch.long);y=torch.full_like(x,IGNORE)
    for i,(xx,yy) in enumerate(pairs):x[i,:len(xx)]=torch.tensor(xx);y[i,:len(yy)]=torch.tensor(yy)
    return x,y


def data_hash(splits):return hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
