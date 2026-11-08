"""Synthetic grouped preference pairs; labels are rules, not human feedback."""
import hashlib
import json
import random
import torch
PAD,BOS,EOS,USER,ASSISTANT,UNK=range(6)
IGNORE=-100
TEMPLATE='roles-v1: BOS USER prompt ASSISTANT completion EOS; shifted answer-only sum'
EVENTS={'timeout':'NET','denied':'AUTH','deadlock':'DATA'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def records():
    groups=[(f's{i}',f'op{j}') for i in range(8) for j in range(3)]
    random.Random(35).shuffle(groups)
    result={}
    for split,subset in [('train',groups[:16]),('validation',groups[16:20]),('test',groups[20:])]:
        rows=[]
        for s,o in subset:
            for event,label in EVENTS.items():
                prompt=f'route service {s} op {o} event {event}'
                bad=next(v for v in EVENTS.values() if v!=label)
                for kind,rejected in [('style',f'{label} GUESS'),('route',f'{bad} CHECK')]:
                    rows.append(dict(group=f'{s}/{o}',prompt=prompt,chosen=f'{label} CHECK',rejected=rejected,
                                     kind=kind,label=label,rubric='triage-v1',source='synthetic-rule-v1'))
        result[split]=rows
    validate(result)
    return result


def validate(splits):
    groups=[];prompts=[]
    for split,rows in splits.items():
        if not rows:raise ValueError('empty split')
        groups.append({r['group'] for r in rows});prompts.append({r['prompt'] for r in rows})
        seen=set()
        for r in rows:
            if not all(isinstance(r[k],str) and r[k].strip() for k in ['group','prompt','chosen','rejected','rubric','source']):raise ValueError('empty field')
            if r['chosen'].strip()==r['rejected'].strip():raise ValueError('identical completion')
            key=(r['prompt'],r['chosen'],r['rejected'])
            if key in seen or (key[0],key[2],key[1]) in seen:raise ValueError('duplicate/conflicting pair')
            seen.add(key)
    for i in range(len(groups)):
        for j in range(i):
            if groups[i]&groups[j] or prompts[i]&prompts[j]:raise ValueError('split leakage')


class Tokenizer:
    def __init__(self,rows):
        self.words=['<PAD>','<BOS>','<EOS>','<USER>','<ASSISTANT>','<UNK>']+sorted({w for r in rows for key in ['prompt','chosen','rejected'] for w in r[key].split()})
        self.index={w:i for i,w in enumerate(self.words)}
    def encode(self,text):return [self.index.get(w,UNK) for w in text.split()]
    @property
    def size(self):return len(self.words)
    @property
    def fingerprint(self):return digest(self.words)


def prefix(tok,prompt):return [BOS,USER]+tok.encode(prompt)+[ASSISTANT]


def example(tok,prompt,completion,max_len=32):
    if not completion.strip():raise ValueError('empty completion')
    p=prefix(tok,prompt);seq=p+tok.encode(completion)+[EOS]
    if len(seq)-1>max_len:raise ValueError('context exceeded; no silent truncation')
    x=seq[:-1];y=seq[1:];y[:len(p)-1]=[IGNORE]*(len(p)-1)
    return x,y


def batch(tok,items,max_len=32):
    if not items:raise ValueError('empty batch')
    examples=[example(tok,p,a,max_len) for p,a in items];t=max(len(x) for x,y in examples)
    x=torch.full((len(items),t),PAD,dtype=torch.long);y=torch.full_like(x,IGNORE)
    for i,(xx,yy) in enumerate(examples):x[i,:len(xx)]=torch.tensor(xx);y[i,:len(yy)]=torch.tensor(yy)
    return x,y


def pair_batch(tok,rows):
    return batch(tok,[(r['prompt'],r['chosen']) for r in rows]+[(r['prompt'],r['rejected']) for r in rows])
